---
status: approved
done-when: The dashboard's tps / prompt_tps / cache_hit_pct / spec fields are fed solely by the tamad's /metrics scrape (per-response extraction is gone from the codebase); a single 30s staleness clock is observable (values blank 30s after the last traffic; spec_decoding_active un-sticks); llama.cpp backends run with the managed --metrics flag; the full validation gate (fmt, clippy --all-targets, ssr + csr checks, nextest) passes.
---

# Inference stats from the tamad /metrics scrape (single source of truth)

## Why

Today there are two writers of inference stats with two staleness clocks: the proxy's per-response extraction (`tps`/`prompt_tps`/`cache_hit_pct` — one request's instantaneous rate; vLLM's `tokens_per_second` = `num_gen / (last − scheduled)` and includes the prefill phase; `prompt_tps` needs the user flag `--enable-prompt-tokens-details`) and the tamad's 10s `/metrics` scrape (spec fields only, ADR-0012). vLLM's `/metrics` is on by default and exposes the token counters a 10s windowed diff needs; llama.cpp ≥ b7191 does too via `--metrics`. The scrape becomes the **sole source** for all inference stats; the per-response extraction is retired. See `docs/adr/0014-single-source-inference-stats.md` for the decision record and `docs/research/inference-stats-from-metrics-scrape.md` for the metric-surface evidence.

## Decisions

1. **Scope:** all four fields (`tps`, `prompt_tps`, `cache_hit_pct`, `spec_accept_pct`) + `spec_decoding_active` — the scrape is the sole source.
2. **Retire** the per-response extraction entirely.
3. **Inject `--metrics` as a managed default** for llama.cpp (mirror the vLLM managed-flags pattern).
4. **Single 30s clock:** tamad `STALE_MS` 60s → 30s; `spec_decoding_active` un-sticks after 30s idle (no longer a sticky OR); `cache_hit_pct` gains the 30s blanking.
5. **Rates standardised on the wall-clock window** — `Δtokens / Δt` (measured elapsed between scrapes) for both engines; llama.cpp's seconds counters are not used.
6. **Single branch, phased commits:** (1) wire + parse, (2) merge switch, (3) retire + flag. Lands as one PR.

## Design

### Wire

`ProcessInfo` (`crates/tama-core/proto/tamad.proto`) gains, after field 11:

```proto
optional double tps = 12;
optional double prompt_tps = 13;
optional double cache_hit_pct = 14;
```

- `optional` (proto3 explicit presence) — a `0.0` tps (idle window) is distinguishable from "no observation"; old tamads omit the fields → `None` on the proxy (graceful, same pattern as fields 10-11).
- `ModelRow` (`crates/tama-core/src/proxy/state/rows.rs:88-104`) mirrors the three fields in `row_from`.

### Tamad (`crates/tamad`)

**`vllm_metrics.rs` → `engine_metrics.rs`** — parser generalised to `vllm:` **and** `llamacpp:` prefixes, returning one `EngineCounters` per body:

| Counter | vLLM name | llama.cpp name | Notes |
|---|---|---|---|
| decode tokens | `vllm:generation_tokens_total` | `llamacpp:tokens_predicted_total` | |
| prompt tokens (compute) | `vllm:prompt_tokens_by_source_total{source="local_compute"}` | `llamacpp:prompt_tokens_total` | llama.cpp already excludes cache hits; vLLM's raw `prompt_tokens_total` is cache-inflated — never used as a fallback |
| prompt tokens (cached) | `vllm:prompt_tokens_by_source_total{source="local_cache_hit"}` | `llamacpp:prompt_tokens_cached_total` | |
| spec (3 counters) | `vllm:spec_decode_num_drafts_total` / `_num_draft_tokens_total` / `_num_accepted_tokens_total` (existing) | `llamacpp:spec_decode_*_total` (upstream since 2026-08-05) | |

**`engine_kind` detection** (generalises `is_vllm`): `vllm:` names present → `Vllm`; else `llamacpp:` names present → `Llamacpp`; else `None` (no-op, unchanged behavior for unknown engines). One process is one engine binary — "both present" is not special-cased.

**Per-window observation** (`SpecState` → `EngineState` with `prev: Option<EngineCounters>`):

- `tps` = `Δdecode_tokens / Δt` (Δt = measured elapsed between scrapes, not nominal 10s)
- `prompt_tps` = `Δprompt_computed / Δt` — `None` when the vLLM `by_source` metric is absent (no raw-counter fallback)
- `cache_hit_pct` = `Δprompt_cached / (Δprompt_cached + Δprompt_computed) × 100` — disjoint-set semantics, identical to today's per-response `cache_n / prompt_n`
- `spec_accept_pct` = `Δaccepted_tokens / Δdraft_tokens × 100` (existing logic, prefix-generalised)
- `spec_decoding_active` = the window had spec traffic (`Δdraft_tokens > 0`)
- `last_obs_ms` stamped when the window had *any* traffic; `STALE_MS = 30s` (was 60s) — the single clock; any value older than 30s → `None` on the wire
- Reset-tolerance generalised from `observe()`: *any* counter in `cur` lower than `prev` → whole window `None` (engine restart)

**Scrape loop** (`scrape_spec`, `stats.rs:201-307`): unchanged — 10s per-endpoint throttle, 2s per-scrape timeout, 3s preflighted cumulative budget, whole-body fetch. Adding counters is parse-time only.

### llama.cpp managed flag (`crates/tama-core/src/config/`)

New `llama_cpp_args.rs` mirroring the vLLM managed-flags pattern (`config/vllm_args.rs:16-38`): managed list = `["metrics"]`, appended by `build_full_args` (`config/resolve/mod.rs:227-270`) when absent — idempotent, user args never overridden. Deploy note: existing installs need a backend restart to pick up the new arg.

### Proxy (`crates/tama-core/src/proxy/`)

**Merge** (`merge_tamad_spec_stats` → renamed `merge_tamad_inference_stats`, `server/metrics.rs:33-58`):

- **Overwrite all five fields unconditionally from the row — including `None`.** Inverts today's "skip stale defaults" invariant; safe because the row is itself 30s-gated by the tamad (a `None` means "no traffic for 30s", which must blank the entry), and there is no longer a second writer to protect.
- **Stamp `last_updated_ms` only when the row's `tps` is `Some`.** The merge runs every 2s and touches every entry; an unconditional stamp would make `last_updated_ms` ~equal across entries and `aggregate_inference`'s "latest entry wins" would degenerate to arbitrary. Stamping only on `Some` preserves "most recently *active* backend wins" exactly; a `None` row leaves `last_updated_ms` untouched so the entry ages out naturally.
- Fan-out unchanged: one row's values apply to all config entries (aliases) matching via `resolve_backends_for_model`.

**Aggregate** (`aggregate_inference`, `metrics.rs:66-100`): **unchanged** — latest entry by `last_updated_ms`, 30s `BUCKET_MS` gate. The gate becomes a redundant backstop (the tamad already blanks at 30s; the gate now only fires on edge cases like a wedged row stream).

**`LatestInferenceStats`** (`proxy/types.rs:21-37`): shape unchanged, single writer (the merge). `record_inference_stats` (whole-entry insert, used only by the per-response path) deleted in commit 3; the merge writes fields via `modify_inference_stats`.

**Behavior changes:** `spec_decoding_active` is no longer a sticky OR — the badge flips back to inactive 30s after the last spec traffic. `cache_hit_pct` gains the 30s blanking it doesn't have today. A live-but-idle engine shows its last window's rate for up to 30s, then blanks.

### Commit 3 deletions (no behavior change — the merge already owns the fields)

- `extract_inference_stats` + `extract_llama_cpp_stats` + `extract_vllm_stats` (`forward/stats.rs`, ~170 lines incl. tests)
- Call sites: `forward/request.rs:358`, `forward/sse.rs:25`
- The spec-preservation + borrow-guard code inside `extract_vllm_stats` (the self-deadlock workaround dies with its only caller)
- `MetricsState::record_inference_stats`

Postgres/dashboard plumbing untouched — values flow through the same `MetricSample` / `SystemMetricsRow`.

### Degradation (all graceful, body-driven)

| Engine | tps | prompt_tps | cache | spec |
|---|---|---|---|---|
| vLLM current | ✓ | ✓ | ✓ | ✓ (if spec enabled) |
| vLLM pre-2026 (no `by_source`) | ✓ | — | — | ✓ |
| llama.cpp ≥ 2026-08-05 + `--metrics` | ✓ | ✓ | ✓ | ✓ (if spec enabled) |
| older llama.cpp + `--metrics` | ✓ | ✓ | ✓ | — (no spec counters) |
| llama.cpp without `--metrics` (501) | — | — | — | — |
| pre-b7191 llama.cpp (escaped exposition) | — (parser rejects) | — | — | — |

## Test plan

**Commit 1 — wire + parse (additive, no behavior change):**
- `engine_metrics.rs`: existing tests carry over (vLLM body, exact-name guard, spaced labels, trailing timestamps, reset-tolerance); new tests for `llamacpp:` bodies (unlabelled counters), `engine_kind` detection, and per-window `observe` (Δ/Δt with measured elapsed, idle window → keep last, reset → `None`, `by_source` absent → `prompt_tps`/`cache` `None`)
- Proto round-trip tests for fields 12-14 (explicit presence: `0.0` vs absent)
- `ModelRow` mirror tests; extend the existing mock-engine tick tests with token counters

**Commit 2 — merge switch:**
- Merge tests: overwrite-including-`None`; `last_updated_ms` stamped only on `Some`; alias fan-out
- `STALE_MS` 30s stamping test (value `None` 30s after last traffic)
- Aggregate backstop test (entry ages out via untouched `last_updated_ms`)
- End-to-end: mock engine with counters → tick → row → merge → aggregate → `MetricSample`

**Commit 3 — retire + flag:**
- Deletions compile clean; workspace tests that referenced the deleted code are fixed up
- `llama_cpp_args.rs` managed-flag tests: `--metrics` injected when absent, idempotent when present, user args never overridden
- Full validation gate (fmt / clippy `--all-targets` / ssr + csr checks / nextest)

## Rollout notes

- Existing installs need a backend restart to pick up `--metrics` (tamad reload)
- vLLM pre-2026: tps works, `prompt_tps`/cache `None`; older llama.cpp: spec `None`; pre-b7191 llama.cpp: all `None` — all graceful
