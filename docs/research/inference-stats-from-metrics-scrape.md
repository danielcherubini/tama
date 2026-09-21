---
status: current
last-verified: 2026-09-21
verified-by: web research — vLLM source (main @ 82daf9f5756e1868be0aa751afaec4726beca12a, v0.29.0) + docs.vllm.ai + llama.cpp docs, 2026-09-21
---

# Can inference stats (prefill + decode tok/s) come from the tamad's /metrics scrape?

## Executive Summary

**Yes — tok/s for both prefill and decode can be derived from the same `/metrics` pull the tamad already performs, and the scrape-side cost is ~zero** (the scrape already downloads the entire body; adding counters is parse-time only, plus two `optional double` wire fields). The real cost and the real decisions are on the **proxy side**: the merge semantics (`merge_tamad_spec_stats` is or-merge-only and never touches `tps`/`last_updated_ms`), the 30s-vs-60s staleness clock mismatch, and the question of which source wins when a windowed scraped value and a per-request instantaneous value disagree.

Secondary findings: vLLM's `/metrics` is **on by default** (no flag), so the scrape needs no user config for vLLM — unlike the per-response `prompt_tps`, which requires `--enable-prompt-tokens-details`. llama.cpp **also** serves a Prometheus `/metrics` (opt-in `--metrics`, 501 without it) with token counters and — since upstream 2026-08-05 — the same three spec-decode counters under a `llamacpp:` prefix, so a generalised parser could give llama.cpp tok/s *and* spec stats from the same scrape.

## Findings

### 1. What vLLM's `/metrics` exposes

Verified against vLLM main @ `82daf9f5756e1868be0aa751afaec4726beca12a` (2026-09-21) and v0.29.0 (2026-09-09) — the metric surface is identical between the two.

**Enabling:** there is no `--enable-metrics` flag. Prometheus metrics are **on by default** in the OpenAI-compatible server (`vllm serve`); the `/metrics` route is unconditionally mounted in `vllm/entrypoints/serve/instrumentator/metrics.py`. (The old V0 gauges `vllm:avg_prompt_throughput_toks_per_s` / `vllm:avg_generation_throughput_toks_per_s` were deprecated (PR #2764) and removed (PR #12383).)

**Counters (cumulative; `prometheus_client` appends `_total` on the wire):**

| Metric | Meaning | Condition |
|---|---|---|
| `vllm:prompt_tokens_total` | All prompt (prefill) tokens processed = `computed + local_cache_hit + external_kv_transfer` | always |
| `vllm:prompt_tokens_by_source_total{source=...}` | Breakdown: `local_compute` / `local_cache_hit` / `external_kv_transfer` (sum = total) | always (PR #33290, for P/D disaggregation) |
| `vllm:prompt_tokens_cached_total` | Cached prompt tokens (local + external) | always (PR #33290) |
| `vllm:generation_tokens_total` | All new output tokens, incl. first token of a prefill and all spec-decode **accepted** tokens | always |
| `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total` | Prefix-cache queries/hits **in tokens** | always |
| `vllm:spec_decode_num_drafts_total` / `_num_draft_tokens_total` / `_num_accepted_tokens_total` | Spec-decode bookkeeping (the three the tamad already scrapes) | only when speculative decoding enabled |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total{position=...}` | Per-position accepted counts | same |

**Gauges:** `vllm:num_requests_running`, `vllm:num_requests_waiting` (+ `_by_reason`), `vllm:kv_cache_usage_perc`, `vllm:engine_sleep_state` — all with `model_name` + `engine` labels.

**Histograms** (all `model_name` + `engine` labels; buckets per `vllm/v1/metrics/buckets.py`): `vllm:time_to_first_token_seconds` (0.001…2560 s), `vllm:inter_token_latency_seconds` (0.01…80 s, one sample per streamed output event), `vllm:request_time_per_output_token_seconds` (per-request TPOT = (e2e−TTFT)/(num_gen−1); **recorded as 0 for ≤1-token requests**), `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds` (first SCHEDULED → first NEW_TOKENS; excludes queue time), `vllm:request_decode_time_seconds`, `vllm:request_prompt_tokens` / `vllm:request_generation_tokens` (1-2-5 buckets up to `max_model_len`).

**No pre-computed per-second rates exist in `/metrics`.** Throughput is computed only for the text log (`LoggingStatLogger._update_stats` → "Avg prompt/generation throughput: X tokens/s" over a 5 s window). Note an upstream inconsistency: the *log* prompt throughput uses **compute-only** tokens (`prompt_token_stats.computed`), while the Prometheus `vllm:prompt_tokens` counter uses **total** (incl. cached).

**Labels:** every engine metric carries `model_name` + `engine` (engine index as string; multiple DP engines → multiple series). No per-request or per-LoRA label.

**Histogram precision nuance:** Prometheus histograms export `_bucket` **and** `_sum` **and** `_count`. The `_sum`/`_count` are exact, so a windowed mean `Δsum/Δcount` is computed with **full precision — bucketing loses nothing for a mean**. Bucket boundaries only limit *quantile* estimation (up to one bucket width error).

### 2. Deriving windowed tok/s over a 10 s scrape

| Quantity | Best formula | Precision | Misleading when | Verdict |
|---|---|---|---|---|
| **Decode tok/s** | `Δvllm:generation_tokens_total / Δt` | Exact (integer counter) | idle window (0 ≠ slow), mixed prefill+decode windows, preemptions inflating per-request times | **Good dashboard gauge; not a per-request number** |
| **Prefill tok/s** | `Δvllm:prompt_tokens_by_source_total{source="local_compute"} / Δt` | Exact | high prefix-cache hit rate (raw counter overstates compute), chunked-prefill spikes, P/D-disaggregated external transfer, idle | **Good dashboard gauge — must use the compute-only counter** |

**Caveats for a 10 s-scraping daemon:**

- **Counter resets on engine restart** → naive diff goes negative. Must detect a decrease and discard (the existing `observe()` already does this for the spec counters). `process_start_time_seconds` is available for detection unless `--api-server-count > 1` multiprocess mode.
- **Label aggregation:** one `/metrics` body can contain multiple `model_name`/`engine` label sets; the diff must sum a consistent set.
- **Idle:** Δ = 0 → 0 tok/s. A defined value ("no tokens emitted") but conflates idle with slow. Gate against `vllm:num_requests_running` (and `engine_sleep_state`) so 0-during-no-requests renders distinctly.
- **Mixed windows:** a window with one 10k-token prefill + 40 short decodes yields one blended number. `generation_tokens` mixes first tokens of prefills with decode tokens; it is a good decode-throughput proxy only when decode-dominated.
- **`mean-of-rates` vs `rate-of-means`:** `1/mean(TPOT)` (request-weighted) ≠ `Σtokens/Σtime` (token-weighted). The counter diff is the token-weighted number.
- **≤1-token requests** record 0 into the TPOT histogram, dragging down windowed TPOT means when short/aborted requests are common.
- **Spec decode:** the generation counter counts *accepted* output tokens, so `Δgen/Δt` remains the correct emitted decode throughput; it just decouples from decode *steps*.
- **IPC skew:** counters increment when the frontend receives each `EngineCoreOutput`, not when the GPU finished — negligible against a 10 s window.

**Per-request alternative (context):** vLLM v0.26.0+ can embed per-request metrics in API responses behind `--enable-per-request-metrics` (RFE #40076, PRs #36383, #46768, merged 2026-07-07). Its `metrics.tokens_per_second` = `num_generation_tokens / (last_token_ts − scheduled_ts)` — i.e. it **includes the prefill/TTFT phase** and is not the reciprocal of `mean_itl_ms`. The vLLM team's stated motivation: *"Server-aggregated Prometheus histograms at /metrics are not sufficient for per-user billing, per-request SLA attribution, or latency debugging of individual requests."* For a dashboard gauge of *backend* throughput, the windowed engine rate is the right quantity — it is the same number vLLM prints in its own 5 s log line.

**Version notes:** `vllm:time_per_output_token_seconds` (per-iteration) was deprecated in v0.11 (PR #24110, issue #24015 — "what we are actually measuring is the time between iterations, and a single iteration can produce multiple tokens"), hidden in v0.12, removed in v0.13 (PR #32661). `vllm:iteration_tokens_total` added v0.10 (PR #13288). `prompt_tokens_by_source` / `prompt_tokens_cached` are relatively recent (PR #33290, 2026) — an older pinned vLLM may not have them, in which case the raw `prompt_tokens` counter is the fallback (with the cache-inflation caveat).

### 3. Local codebase fit (tama)

**What exists today** (file:line index at the end):

- Tamad `StatsCollector::tick` runs every 1 s (`crates/tamad/src/server.rs:488`, `spawn_blocking`), calls `scrape_spec` (`stats.rs:201-307`): 10 s per-endpoint throttle, 2 s per-scrape timeout, 3 s preflighted cumulative budget, whole-body fetch (`stats.rs:245-248`), body-driven vLLM detection, windowed `observe()` diffing (`vllm_metrics.rs:148-170`), `STALE_MS = 60 s` blanking.
- Proxy per-response extraction (`crates/tama-core/src/proxy/forward/stats.rs`): llama.cpp `timings` (`predicted_per_second`, `prompt_per_second`, `cache_n/prompt_n`, `draft_n*`) and vLLM `metrics` (`tokens_per_second`, `time_to_first_token_ms`, cache-aware `prompt_tps` **only when `--enable-prompt-tokens-details` is set** — a user flag the codebase never injects). Writes a whole-entry replacement into the per-config-key `inference_stats` watch map, carrying the tamad-merged spec fields across.
- Proxy 2 s metrics loop (`proxy/server/metrics.rs:355-465`): `merge_tamad_spec_stats` (or-merge only; **never touches `tps`/`prompt_tps`/`last_updated_ms`**) then `aggregate_inference` (latest entry by `last_updated_ms`, 30 s `BUCKET_MS` gate: stale → `tps`/`prompt_tps`/`spec_accept_pct` blanked; `cache_hit_pct` ungated; `spec_decoding_active` sticky-OR).
- The tamad **already scrapes llama.cpp backends today** (every ready+alive process); a llama.cpp without `--metrics` answers 501 → no-op.

**Scrape-side cost of adding tok/s: ~zero.** Same HTTP fetch (whole body already downloaded); a few counter names added to the parser; two `optional double` wire fields (12-13) on `ProcessInfo` (proto3 explicit presence → old tamads simply omit them); a few bytes per 1 Hz frame. Budget constants unchanged.

**llama.cpp coverage (bonus):** llama.cpp ≥ b7191 serves Prometheus text at `/metrics` on its API port **only with `--metrics`** (501 otherwise), with:

| Series | Kind | Use |
|---|---|---|
| `llamacpp:tokens_predicted_total` | counter | Δ / Δt or Δ / Δseconds → decode tps |
| `llamacpp:tokens_predicted_seconds_total` | counter | generation-time denominator (true tps, not wall-clock-diluted) |
| `llamacpp:prompt_tokens_total` | counter | prompt tokens **actually prefilled, cache hits excluded** |
| `llamacpp:prompt_seconds_total` | counter | prefill-time denominator → prompt_tps |
| `llamacpp:prompt_tokens_cached_total` | counter | cache hits; ratio = cached/(cached+processed) — disjoint sets, matching per-response `cache_n/prompt_n` semantics |
| `llamacpp:spec_decode_num_drafts_total` / `_draft_tokens_total` / `_accepted_tokens_total` | counters | **same three spec counters as vLLM, different prefix** — added upstream 2026-08-05 (PR #26389) |

Caveats: pre-b7191 builds served the exposition JSON-escaped in quotes (upstream PR #17386) — the existing `split_metric_line` parser safely rejects those (line skipped), so old builds yield "no counters", same as today's 501 no-op. `--metrics` is off by default and not in any tama default args — until a user adds it, llama.cpp tps via scrape is unavailable (the per-response `timings` path works without the flag and stays the fallback).

**The proxy-side semantic decisions (the real work):**

1. **The merge must learn to overwrite `tps`/`prompt_tps` — including `None`.** Today a row at `spec_accept_pct = None` is *skipped* so a merged value survives a tamad blip. For tps, a `None` (no traffic in the window) **must** overwrite, or the last rate lingers forever while `last_updated_ms` keeps the entry artificially fresh. The "or-merge only, never clear" invariant and the "always overwrite tps" requirement are directly opposed; the merge needs field-specific semantics.
2. **The 30 s vs 60 s staleness clock mismatch (the biggest one).** `aggregate_inference` gates tps at `BUCKET_MS = 30 s`; the tamad blanks its observation at `STALE_MS = 60 s`. If the merge re-stamps `last_updated_ms` every 2 s tick, the 30 s gate is effectively bypassed (the entry is always fresh while the row is alive) and the tamad's 60 s becomes the *only* staleness clock — an idle backend would display its last windowed rate for up to 60 s vs today's 30 s blank. If the merge does *not* re-stamp, scraped values are never displayed at all. Reconcile deliberately: align the proxy gate to the tamad's 60 s, or have the tamad stamp the observation timestamp on the wire and the proxy gate against that.
3. **Which source wins.** Recommendation: **the scraped windowed value wins for `tps`/`prompt_tps`** — it is a server-level windowed average (what "backend tok/s" means on a dashboard; the same quantity vLLM logs), it keeps updating across requests, and the per-response value is one request's instantaneous rate (vLLM's `tokens_per_second` even includes the prefill phase). The per-response write extends its existing preservation pattern (which it already does for the spec fields) to carry `tps`/`prompt_tps` across. `cache_hit_pct` can stay per-response or also come from the scrape (`Δprompt_tokens_cached/(Δprompt_tokens_cached+Δprompt_tokens_total)` — same disjoint-set semantics as llama.cpp's `cache_n/prompt_n`).
   - Alternative (more conservative): separate `scrape_tps`/`scrape_prompt_tps` fields + `scrape_updated_ms`; aggregator picks per-response if ≤30 s fresh, else scraped if ≤60 s fresh. Preserves today's exact display semantics at the cost of a new field, a new timestamp, and two-source arbitration.
4. **Keying friction.** The `inference_stats` map is keyed by model config key; the scrape is per physical process. The merge fans one row's value out to *all* config entries matching the row (aliases); per-response writes land only in the *requested* alias's entry. One physical engine can hold several map entries with divergent per-response `tps` but identical scraped `tps`; `aggregate_inference`'s "latest entry wins" then picks whichever alias was last touched.
5. **Display-behaviour change either way.** A 10 s wall-clock diff dips toward 0 between bursts (even while a request is in flight); the per-request value never goes to 0 mid-request. Worth calling out explicitly in the ADR.
6. **Sticky-flag asymmetry.** `spec_decoding_active` is OR'd with no un-stick in the aggregate, but the tamad blanks it at 60 s and the merge never clears it — if tps follows the same merge path, the same "sticky flag outlives the data" asymmetry applies.

**Pre-existing overlap (context):** the proxy already has an on-demand backend `/metrics` passthrough (`proxy/handlers/status.rs:63-140`, `GET /metrics` at `router.rs:142`) that fetches every ready backend's `/metrics` concurrently (5 s timeout) and re-exposes it with injected `{server="name"}` labels — a live proxy→backend read path that runs *against* the ADR-0012 direction and overlaps with what the tamad scrape would now own. Candidate for deprecation in favour of the tamad-scraped values.

## Evidence

| Source | Credibility | Notes |
|---|---|---|
| vLLM source @ main `82daf9f5756e1868be0aa751afaec4726beca12a`: `vllm/v1/metrics/loggers.py`, `vllm/v1/spec_decode/metrics.py`, `vllm/v1/metrics/perf.py`, `vllm/v1/metrics/buckets.py`, `vllm/v1/metrics/stats.py`, `vllm/v1/metrics/prometheus.py`, `vllm/entrypoints/serve/instrumentator/metrics.py`, `vllm/engine/arg_utils.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py` | 1 (official source) | Primary source of truth; v0.29.0 diffed identical |
| [docs.vllm.ai — Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) | 1 (official docs, generated from source) | Table generated by `docs/mkdocs/gen_files/generate_metrics.py` |
| [docs.vllm.ai — metrics design doc](https://github.com/vllm-project/vllm/blob/82daf9f5756e1868be0aa751afaec4726beca12a/docs/design/metrics.md) | 1 | Interval semantics, counter reset on restart, multiprocess caveats, deprecation policy |
| [docs.vllm.ai — speculative decoding acceptance metrics](https://docs.vllm.ai/en/stable/features/speculative_decoding/acceptance_metrics/) | 1 | Per-request ↔ Prometheus spec counter reconciliation; `rate(accepted)/rate(draft)` recommendation |
| vLLM PRs: #46768 / #36383 / RFE #40076 (per-request response metrics, v0.26.0+), #33290 (prompt-token source labels), #13288 (`iteration_tokens_total`), #16665 (spec-decode counters), #24110 / #32661 / issue #24015 (TPOT→ITL deprecation/removal), #2764 / #12383 (old throughput-gauge removal) | 1 | Version availability + semantics rationale |
| llama.cpp `tools/server/README.md` (`--metrics` flag, 501 default) + upstream PR #26389 (spec counters, 2026-08-05) + PR #17386 (exposition format floor b7191) | 1 | llama.cpp /metrics surface |
| Local code: `crates/tamad/src/{stats.rs,vllm_metrics.rs,lifecycle.rs,server.rs}`, `crates/tama-core/src/proxy/{forward/stats.rs,server/metrics.rs,state/rows.rs,types.rs,handlers/status.rs,handlers/metrics.rs}`, `crates/tama-core/proto/tamad.proto`, `docs/adr/0012-host-owned-telemetry-scraping.md` | 1 (own code) | file:line index below |

## Unresolved Contradictions

None between sources. One **semantic tension** to decide by design (not fact): windowed scraped values (dip to ~0 between bursts, 10 s quantised, 60 s blanking) vs per-request values (never 0 mid-request, update on completion, 30 s blanking) — adopting the scrape as the tps source is a display-behaviour change either way.

## Gaps / What Remains Unknown

- Whether `metrics.tokens_per_second` in vLLM *responses* is on by default or only behind `--enable-per-request-metrics` (v0.26.0+) — the flag was confirmed for the extended per-request metrics; the base `metrics` object's availability in the versions the codebase targets was not pinned down. (Matters for how much of today's per-response tps actually works in practice.)
- Exact `llamacpp:` series availability floor vs the versions in the DB (b7191 changed the exposition format; pre-b7191 bodies are safely rejected by the current parser, but the counter *names* in older builds need verification if we want to support them).
- Whether `prompt_tokens_by_source` exists in the specific vLLM versions installed in the DB (added 2026, PR #33290) — older installs fall back to the raw `prompt_tokens` counter with the cache-inflation caveat.

## File:line index (local)

| Item | Location |
|---|---|
| `LatestInferenceStats` | `crates/tama-core/src/proxy/types.rs:21-37` |
| watch map + accessors | `types.rs:516`, `proxy/state/metrics.rs:56-80` |
| per-response extraction (llama.cpp / vLLM) | `proxy/forward/stats.rs:14-171` |
| call sites (non-streaming / SSE) | `proxy/forward/request.rs:358`, `proxy/forward/sse.rs:25` |
| 2 s metrics loop, merge + aggregate | `proxy/server/metrics.rs:355-465` |
| `merge_tamad_spec_stats` (or-merge, never touches tps/last_updated_ms) | `proxy/server/metrics.rs:28-58` |
| `aggregate_inference` (30 s gate, latest-wins) | `proxy/server/metrics.rs:6-11, 66-100` |
| `ModelRow` + `row_from` | `proxy/state/rows.rs:60-104` |
| wire `ProcessInfo` fields 10-11 | `crates/tama-core/proto/tamad.proto:104-119` |
| tamad tick (1 s) | `crates/tamad/src/server.rs:488` |
| `scrape_spec` (throttle, budget, whole-body fetch, stamping) | `crates/tamad/src/stats.rs:201-307` |
| budget constants (10 s / 2 s / 3 s / 60 s) | `crates/tamad/src/vllm_metrics.rs:17-31` |
| pure parser + `observe` + `metrics_url_for` | `crates/tamad/src/vllm_metrics.rs:54-199` |
| ProcessInfo defaults in tamad | `crates/tamad/src/lifecycle.rs:109-110` |
| backend arg building (no `--metrics` / `--enable-prompt-tokens-details` injection) | `crates/tama-core/src/config/resolve/mod.rs:184-270`, `config/vllm_args.rs:16-38` |
| proxy on-demand backend /metrics passthrough | `proxy/handlers/status.rs:63-140`, `proxy/handlers/metrics.rs:195-275`, `proxy/server/router.rs:142` |
| design rationale | `docs/adr/0012-host-owned-telemetry-scraping.md` |
