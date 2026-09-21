---
status: committed
done-when: The dashboard's tps / prompt_tps / cache_hit_pct / spec fields are fed solely by the tamad's /metrics scrape (per-response extraction is gone from the codebase); a single 30s staleness clock is observable (values blank 30s after the last traffic; spec_decoding_active un-sticks); llama.cpp backends run with the managed --metrics flag; the full validation gate (fmt, clippy --all-targets, ssr + csr checks, nextest) passes.
---

# Inference stats from the tamad /metrics scrape — Plan

**Goal:** The tamad's 10s `/metrics` scrape becomes the sole source of inference stats (`tps`, `prompt_tps`, `cache_hit_pct`, `spec_accept_pct`, `spec_decoding_active`); the proxy's per-response extraction is retired.

**Architecture:** The tamad scrape loop generalises from vLLM-only spec counters to vLLM **+** llama.cpp token counters (body-driven engine detection), computes windowed rates (`Δcounters / Δt` over the measured scrape interval, 30s staleness), and stamps three new optional wire fields on the 1 Hz process rows. The proxy's merge overwrites the per-backend inference-stats map unconditionally from the rows (stamping `last_updated_ms` only when `tps` is `Some`); the per-response extraction is deleted. See `docs/adr/0014-single-source-inference-stats.md` (decision record) and `docs/research/inference-stats-from-metrics-scrape.md` (metric-surface evidence) — read both before executing.

**Tech Stack:** Rust workspace (`tama-core` / `tama` / `tamad`), prost/tonic proto (codegen via `build.rs` → `OUT_DIR`), reqwest blocking client, `watch` channels.

## Conventions (apply to every task)

- **TDD order:** write the failing test → run it and confirm it fails with the expected error → implement minimally → confirm it passes → `cargo fmt --all` → clippy → commit. Never skip the red step.
- **Per-task validation** (run in this order after the code compiles):
  1. `cargo fmt --all`
  2. `cargo clippy --workspace --all-targets -- -D warnings`
  3. `cargo nextest run --package <crate>` (affected crate(s))
- **Commit** with the suggested message when the task's acceptance criteria are met. Each task is independently commitable.
- **Do not** touch `crates/tama/dist/` (Trunk build output) or add features beyond what the task specifies (YAGNI).
- The `--enable-prompt-tokens-details` vLLM flag and the per-response `metrics` JSON stay in the response payload — only the *reading* of it is retired (Task 3).

---

### Task 1: Wire fields + generalised engine-metrics parser

**Context:**
ADR-0014 makes the tamad's scrape the sole source of inference stats. This task is the **additive** foundation: three new optional wire fields on `ProcessInfo`, a parser that recognises both `vllm:` and `llamacpp:` counter prefixes (one process = one engine binary, so the body's prefix identifies the engine), and the windowed-rate computation (`Δcounters / Δt`). Nothing downstream of the wire is wired up yet — the per-response path and the merge are untouched, so this commit changes no visible behavior. Key semantics (from the approved spec): rates are `Δ / measured-elapsed` (NOT the nominal 10s, NOT llama.cpp's seconds counters); a window with no traffic yields no observation (the last one is kept until it goes stale); `STALE_MS` tightens 60s → 30s (the single staleness clock); any counter decreasing = engine restart = whole window discarded.

**Files:**
- Modify: `crates/tama-core/proto/tamad.proto` — `message ProcessInfo` (lines 154-170): append fields 12-14
- Rename: `crates/tamad/src/vllm_metrics.rs` → `crates/tamad/src/engine_metrics.rs` (`git mv`); update the declaration `mod vllm_metrics;` → `mod engine_metrics;` in `crates/tamad/src/main.rs:27` and `use crate::vllm_metrics;` → `use crate::engine_metrics;` in `crates/tamad/src/stats.rs:16`
- Modify: `crates/tamad/src/stats.rs` — `SpecState` (lines 52-67) → `EngineState`; `scrape_spec` (lines 201-307): the parse/stamp blocks (lines 245-274) and the final stamping block (lines 289-307)
- Modify: `crates/tama-core/src/proxy/state/rows.rs` — `ModelRow` (lines 88-104) + `row_from` (lines 60-75); extend its tests (lines ~419-463)
- Modify (mechanical — add `tps: None, prompt_tps: None, cache_hit_pct: None` to every `ProcessInfo { ... }` struct literal; do this explicitly in every literal rather than papering over with `..ProcessInfo::default()`, so the diff stays honest): `crates/tamad/src/stats.rs`, `crates/tamad/src/lifecycle.rs` (~lines 109-110), `crates/tama/src/admin.rs`, `crates/tama-core/src/tamad/mod.rs` (2 sites), `crates/tama-core/src/proxy/status.rs`, `crates/tama-core/src/proxy/mod.rs`, `crates/tama-core/src/proxy/state/rows.rs`, `crates/tama-core/src/proxy/lifecycle/tests.rs`, `crates/tama-core/src/proxy/server/tests.rs`, `crates/tama-core/src/proxy/server/metrics.rs` (incl. the `process` test helper at line ~592), `crates/tama-core/src/proxy/forward/tests/request.rs`, `crates/tama-core/src/proxy/lifecycle/spec/tests.rs`, `crates/tama-core/src/proxy/tama_handlers/backend_logs.rs`, `crates/tama-core/src/proxy/tama_handlers/system_tests.rs`, `crates/tama-core/src/proxy/tama_handlers/models/tests/model_handlers.rs`, `crates/tama-core/src/proxy/tama_handlers/models/tests/cancel.rs`, `crates/tama-core/src/proxy/handlers/tests.rs`, `crates/tama-core/src/proxy/handlers/get_model_tests.rs`

**What to implement:**

1. **Proto** — append to `ProcessInfo` after field 11 (keep the ADR-0012 comment on fields 10-11 as-is):

```proto
  // Inference stats observed by the tamad's /metrics scrape (ADR-0014):
  // windowed Δ/Δt over the measured scrape interval. None = no traffic in
  // the window, observation stale (>30s), or the engine doesn't expose
  // the counter.
  optional double tps = 12;
  optional double prompt_tps = 13;
  optional double cache_hit_pct = 14;
```

2. **`engine_metrics.rs`** (the renamed module). Keep `SCRAPE_INTERVAL` / `PER_SCRAPE_TIMEOUT` / `TICK_SCRAPE_BUDGET` unchanged; change `STALE_MS` from `60_000` to `30_000` (update its doc comment: the single 30s staleness clock, ADR-0014).

   Replace `SpecCounters` with:

```rust
/// Cumulative counters summed across all label sets, per engine prefix.
/// A slot is `None` when the engine's body doesn't expose that counter
/// (e.g. pre-2026 vLLM has no `prompt_tokens_by_source`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineCounters {
    /// vllm:generation_tokens_total / llamacpp:tokens_predicted_total
    pub decode_tokens: Option<f64>,
    /// vllm:prompt_tokens_by_source_total{source="local_compute"} (compute-only;
    /// None on pre-2026 vLLM) / llamacpp:prompt_tokens_total (already excludes
    /// cache hits upstream)
    pub prompt_computed: Option<f64>,
    /// vllm:prompt_tokens_by_source_total{source="local_cache_hit"}
    /// / llamacpp:prompt_tokens_cached_total
    pub prompt_cached: Option<f64>,
    /// vllm:spec_decode_num_drafts_total / llamacpp:spec_decode_num_drafts_total
    pub spec_drafts: Option<f64>,
    /// vllm:spec_decode_num_draft_tokens_total / llamacpp:spec_decode_num_draft_tokens_total
    pub spec_draft_tokens: Option<f64>,
    /// vllm:spec_decode_num_accepted_tokens_total / llamacpp:spec_decode_num_accepted_tokens_total
    pub spec_accepted_tokens: Option<f64>,
}

/// Which engine family the scraped body belongs to (body-driven detection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Vllm,
    Llamacpp,
}
```

   Replace `parse_spec_metrics` with `pub fn parse_engine_metrics(body: &str) -> Option<(EngineKind, EngineCounters)>`:
   - Reuse the existing `split_metric_line` / `is_plain_int` machinery for value extraction, but extend `split_metric_line` to also return the raw label-set text so the caller can read the `source` label. **Rule for the label-set span:** from the first `{` to the first `}` that is NOT inside a quoted value (track quote state, toggling on `"` — a legal Prometheus label value may contain `}` inside quotes). If no closing `}` exists, the label set is unparseable — treat the line as having no labels. Keep every existing guarantee (spaced label values, trailing-timestamp handling, non-finite rejection, exact-name match — a longer-suffixed name never matches).
   - Counter table (exact names; sum across all matching label sets):

     | slot | vLLM name | llama.cpp name |
     |---|---|---|
     | `decode_tokens` | `vllm:generation_tokens_total` | `llamacpp:tokens_predicted_total` |
     | `prompt_computed` | `vllm:prompt_tokens_by_source_total` + label `source="local_compute"` | `llamacpp:prompt_tokens_total` |
     | `prompt_cached` | `vllm:prompt_tokens_by_source_total` + label `source="local_cache_hit"` | `llamacpp:prompt_tokens_cached_total` |
     | `spec_drafts` | `vllm:spec_decode_num_drafts_total` | `llamacpp:spec_decode_num_drafts_total` |
     | `spec_draft_tokens` | `vllm:spec_decode_num_draft_tokens_total` | `llamacpp:spec_decode_num_draft_tokens_total` |
     | `spec_accepted_tokens` | `vllm:spec_decode_num_accepted_tokens_total` | `llamacpp:spec_decode_num_accepted_tokens_total` |

   - Kind detection: if any of the SIX vLLM table names matched (NOT "any line with a `vllm:` prefix" — a body containing `vllm:prompt_tokens_total`, a real vLLM metric that is NOT a table name, must still return `None` when no table name is present) → `Vllm`; else if any of the six llama.cpp table names matched → `Llamacpp`; else return `None` (unknown engine — the no-op path). Both prefixes present: `Vllm` wins (documented non-case: one process is one engine binary).
   - For the two `by_source` slots: a line only counts when it carries a label whose key is EXACTLY `source` (anchored — the key appears immediately after `{` or `,`, so e.g. `other_source="local_compute"` never matches) and whose value is `local_compute` / `local_cache_hit` respectively (other `source` values — `local_cache_hit`'s siblings like `external_kv_transfer` — are ignored for both slots).
   - `metrics_url_for` stays unchanged.

   Replace `observe` with:

```rust
/// One scrape window's observation. All `None` except `had_traffic` means
/// "no traffic in this window" (the caller keeps the last observation).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindowObs {
    pub tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub cache_hit_pct: Option<f64>,
    pub spec_accept_pct: Option<f64>,
    /// Window had spec-decoding traffic (Δdraft_tokens > 0).
    pub spec_active: bool,
    pub had_traffic: bool,
}

/// Windowed rates from two successive cumulative counter sets.
///
/// - `prev` is `None` → `None` (first scrape — no delta to compute)
/// - any counter present in both `prev` and `cur` that DECREASED → `None`
///   (engine restart / counter reset — never emit a bogus rate)
/// - no counter advanced (all Δ ≤ 0) → `None` (idle window — the caller
///   keeps the last observation until it goes stale)
/// - otherwise `Some(WindowObs)`:
///   - `tps` = Δdecode_tokens / dt_secs when the counter is present in both
///     (Δ may be 0.0 — a prefill-only window), else `None`
///   - `prompt_tps` = Δprompt_computed / dt_secs (same availability rule)
///   - `cache_hit_pct` = 100 × Δprompt_cached / (Δprompt_cached + Δprompt_computed)
///     when the denominator > 0, else `None` (disjoint-set semantics, matching
///     the retired per-response cache_n/prompt_n)
///   - `spec_accept_pct` = 100 × Δaccepted / Δdraft_tokens when Δdraft_tokens > 0
///   - `spec_active` = Δdraft_tokens > 0
///   - `had_traffic` = true
/// - `dt_secs <= 0.0` → `None` (defensive; the 10s throttle makes this unreachable
///   in production)
pub fn observe(
    prev: Option<EngineCounters>,
    cur: &EngineCounters,
    dt_secs: f64,
) -> Option<WindowObs> { ... }
```

3. **`stats.rs`** — `SpecState` → `EngineState`:

```rust
#[derive(Debug, Default)]
struct EngineState {
    /// Last cumulative counter set (None until the first successful parse).
    prev: Option<engine_metrics::EngineCounters>,
    /// Engine family of the last parsed body (body-driven detection).
    kind: Option<engine_metrics::EngineKind>,
    /// Last scrape attempt, for the per-endpoint scrape throttle.
    last_scrape: Option<Instant>,
    /// Last SUCCESSFUL parse, for the window length (`dt`). A failed attempt
    /// or an unknown body must NOT shorten the next window — the counters
    /// advanced over the full interval since the last successful parse.
    last_parse: Option<Instant>,
    /// The most recent window observation (None until a window had traffic).
    last_obs: Option<engine_metrics::WindowObs>,
    /// Unix millis of the last traffic-bearing observation (poison-pill for freshness).
    last_obs_ms: i64,
}
```

   In `scrape_spec` (lines 201-307), replacing the current parse/stamp logic:
   - Before the fetch: `let prev_parse = self.spec.get(&model_name).and_then(|s| s.last_parse);` — the entry may not exist yet (the `entry().or_default()` binding is created AFTER the fetch, at line ~245), so read via `.get()`; and it is the last *successful parse* (not the last attempt) that anchors the window length
   - After the fetch: `s.last_scrape = Some(Instant::now());` (unchanged — stamped on every attempt, success or failure; it is the throttle anchor only)
   - On a successful 2xx body: `let (kind, cur) = match engine_metrics::parse_engine_metrics(&text) { Some(k) => k, None => { s.kind = None; continue; } };` — note the `None` arm leaves `prev`/`last_parse` untouched (unknown body: the next window's `dt` spans the whole interval, which is correct — the counters advanced over it) — then:
     ```rust
     let dt_secs = prev_parse
         .map(|t| (Instant::now() - t).as_secs_f64())
         .unwrap_or(1.0); // first successful parse: prev is None so observe() returns None anyway
     let obs = engine_metrics::observe(s.prev, &cur, dt_secs);
     s.kind = Some(kind);
     if let Some(o) = obs {
         // ONLY a traffic-bearing window updates the observation and its
         // timestamp — an idle window (observe() → None) must leave the
         // last observation intact until it goes stale (30s, below).
         s.last_obs_ms = now_ms;
         s.last_obs = Some(o);
     }
     s.last_parse = Some(Instant::now()); // dt anchor — every successful parse
     s.prev = Some(cur);
     ```
     (Note the behavioral change from today's `is_vllm`: a body without the counters now sets `s.kind = None` rather than `is_vllm = false` — same effect, and a later body that regains the counters re-stamps `kind`. `s.prev` updates on EVERY successful parse, including idle windows — the reset detection needs the latest cumulative values even when no window observation is emitted.)
   - The final stamping block (replacing lines 285-307) — every field is now 30s-gated:
     ```rust
     let fresh = now_ms - s.last_obs_ms <= engine_metrics::STALE_MS;
     let o = s.last_obs;
     p.tps = if fresh { o.and_then(|o| o.tps) } else { None };
     p.prompt_tps = if fresh { o.and_then(|o| o.prompt_tps) } else { None };
     p.cache_hit_pct = if fresh { o.and_then(|o| o.cache_hit_pct) } else { None };
     p.spec_accept_pct = if fresh { o.and_then(|o| o.spec_accept_pct) } else { None };
     p.spec_decoding_active = fresh && o.is_some_and(|o| o.spec_active);
     ```
   - `lifecycle.rs` (~lines 109-110): add `tps: None, prompt_tps: None, cache_hit_pct: None` to the default `ProcessInfo` literal (next to the existing `spec_accept_pct: None, spec_decoding_active: false`).

4. **`rows.rs`** — add to `ModelRow`:

```rust
    /// Windowed decode tokens/s from the tamad's scrape (ADR-0014); `None` = no
    /// traffic observed, stale, or engine doesn't expose the counter.
    pub tps: Option<f32>,
    pub prompt_tps: Option<f32>,
    pub cache_hit_pct: Option<f32>,
```

   and to `row_from` (mirror the existing `spec_accept_pct` mapping at line 70):

```rust
    tps: p.tps.map(|v| v as f32),
    prompt_tps: p.prompt_tps.map(|v| v as f32),
    cache_hit_pct: p.cache_hit_pct.map(|v| v as f32),
```

**Steps:**
- [ ] In `crates/tamad/src/engine_metrics.rs` (after the `git mv` + import updates), write the failing unit tests first (in the module's existing `mod tests`):
      - `test_parse_llamacpp_body` — an unlabelled `llamacpp:` body (all six counters) → `(Llamacpp, EngineCounters)` with the right values
      - `test_parse_mixed_prefixes_prefers_vllm` — a body containing both prefixes → `Vllm` kind
      - `test_parse_by_source_labels` — `vllm:prompt_tokens_by_source_total` with `source="local_compute"`, `source="local_cache_hit"`, and `source="external_kv_transfer"` label sets: only the first two count, into the right slots, summed across sets
      - `test_parse_pre_2026_vllm_no_by_source` — a vLLM body with `vllm:prompt_tokens_total` but no `by_source` metric → `prompt_computed`/`prompt_cached` are `None`, the other four slots populated, kind `Vllm`
      - `test_observe_window_rates` — explicit `dt_secs = 10.0`: prev all-zero, cur decode=500/computed=200/cached=50/draft_tokens=100/accepted=45 → `tps=50.0`, `prompt_tps=20.0`, `cache_hit_pct=20.0` (50/250), `spec_accept_pct=45.0`, `spec_active=true`, `had_traffic=true`
      - `test_observe_idle_window_none` — cur == prev → `None`
      - `test_observe_reset_none` — any counter in cur below prev → `None`
      - `test_observe_partial_availability` — prev/cur both have `decode_tokens` but `prompt_computed` is `None` in cur → `tps` Some, `prompt_tps`/`cache_hit_pct` None
      - `test_observe_zero_dt_none` — `dt_secs = 0.0` → `None`
      - Carry over the existing `parse_spec_metrics` tests, retargeted at `parse_engine_metrics` (the vLLM spec counters must still parse; a body without the three spec counters but WITH the token counters must still return `Some`)
- [ ] Run `cargo nextest run --package tamad -- engine_metrics`
  - Did it fail with the expected errors (missing fn / wrong values)? If it passed unexpectedly, stop and investigate why.
- [ ] Implement `parse_engine_metrics` + `observe` (+ the `split_metric_line` label-set extension) in `engine_metrics.rs`; set `STALE_MS = 30_000`
- [ ] Run `cargo nextest run --package tamad -- engine_metrics` — all pass?
- [ ] Write the `stats.rs` tick tests (extend the existing mock-engine tests at the bottom of `stats.rs`): the existing `vllm_body(drafts, draft_tokens, accepted)` helper emits ONLY the three spec counters — **extend it** (or add a sibling helper) to also emit `vllm:generation_tokens_total` and the two `vllm:prompt_tokens_by_source_total` label sets (`source="local_compute"` / `source="local_cache_hit"`), and add a `llamacpp_body(...)` helper mirroring it (unlabelled `llamacpp:` counters); extend the existing vLLM tick test to assert `processes[0].tps` / `.prompt_tps` / `.cache_hit_pct` are stamped (use `with_scrape_interval(Duration::from_millis(50))` + a real sleep so `dt_secs` > 0, and assert `Some` with `> 0.0` rather than an exact value — the exact math is covered by the `observe` unit tests); add a llama.cpp tick test mirroring it; add a staleness test that seeds **BOTH** `EngineState.last_obs = Some(WindowObs { all rate fields Some, spec_active: true, had_traffic: true })` **AND** `last_obs_ms = now_ms - 31_000` (direct field access — same-crate test) and asserts all five wire fields are `None`/`false`, PLUS a positive control: the same seeded `last_obs` with a fresh `last_obs_ms` asserts the fields ARE stamped (without the positive control the staleness test is vacuous — it would pass even if `fresh` were computed backwards)
- [ ] Run `cargo nextest run --package tamad -- stats` — all pass?
- [ ] Add the proto fields (step 1) and update every `ProcessInfo { ... }` literal (the 18-file list above) — the compiler will list the broken sites for you; add the three `None` fields to each
- [ ] Add the `ModelRow` mirror + `row_from` mapping, and extend the existing `row_from` tests in `rows.rs` (lines ~419-463) to assert the three new fields
- [ ] Add a proto round-trip test in `crates/tama-core/src/tamad/mod.rs` (follow the existing `ProcessInfo` tests there): `tps: Some(0.0)` round-trips as `Some(0.0)` (explicit presence — `0.0` is NOT the same as absent) and an unset field round-trips as `None`
- [ ] Run `cargo nextest run --package tama-core` and `cargo nextest run --package tamad` — all pass?
- [ ] Run `cargo fmt --all` — clean?
- [ ] Run `cargo clippy --workspace --all-targets -- -D warnings` — clean?
- [ ] Run `cargo check --package tama --no-default-features --features csr` (the `tama` crate literal in `admin.rs` changed) — clean?
- [ ] Commit with message: `feat: add tps/prompt_tps/cache_hit_pct wire fields + generalised engine metrics parser (ADR-0014, task 1)`

**Acceptance criteria:**
- [ ] `parse_engine_metrics` handles vLLM (incl. `by_source` label filtering and pre-2026 absence), llama.cpp (unlabelled), and mixed bodies; unknown bodies → `None`
- [ ] `observe` implements the window semantics (reset → `None`, idle → `None`, Δ/Δt rates, disjoint-set cache ratio)
- [ ] The three new wire fields round-trip with explicit presence; `ModelRow` mirrors them
- [ ] A mock llama.cpp engine's tick stamps `tps`/`prompt_tps`/`cache_hit_pct` on the `ProcessInfo`
- [ ] ALL pre-existing tests pass unchanged (the per-response path and the merge are untouched — this commit is behavior-neutral)

---

### Task 2: Proxy merge switch — the row becomes the sole writer

**Context:**
With the wire fields flowing (Task 1), this task makes the proxy's merge the **sole writer of the inference-stats values**: it overwrites all five fields unconditionally from each live row (including `None` — safe because the row is itself 30s-gated by the tamad; a `None` means "no traffic for 30s" and must blank the entry), and stamps `last_updated_ms` only when the row's `tps` is `Some` (an unconditional stamp would make it ~equal across all entries and `aggregate_inference`'s "latest entry wins" would degenerate to arbitrary). Other code paths still touch the map for lifecycle reasons — `lifecycle/mod.rs` removes an entry on unload, `rename.rs` migrates an entry on rename, `clear_inference_stats` clears on reset, and the conn-error cleanup in `forward/request.rs` removes an unreachable backend's entry — those stay (harmless backstops under the new semantics; the merge re-creates an entry within ~2s whenever a live row still resolves). `status.rs::collect_model_state_snapshots` is the reader surface where the "blank 30s after traffic" behavior surfaces to users (per-model dashboard cards). The per-response path still exists in this commit (Task 3 deletes it) — the merge and the per-response writes coexist for one commit, which is fine: the merge runs every 2s and simply overwrites whatever the per-response path wrote. `aggregate_inference` is **unchanged in code** — its 30s gate becomes a redundant backstop (the entry is already 30s-gated upstream); only its doc comment changes.

**Files:**
- Modify: `crates/tama-core/src/proxy/server/metrics.rs` — `merge_tamad_spec_stats` (lines 28-58) → `merge_tamad_inference_stats`; the call site (line 359); the `aggregate_inference` doc comment (lines 59-65)
- Test: the `mod tests` in the same file (merge tests at lines ~659-760 — see the Steps bullet for the exact helpers)

**What to implement:**

1. Replace `merge_tamad_spec_stats` with:

```rust
/// Fold the tamad-observed inference stats (ADR-0014) off the live rows into
/// the per-server `inference_stats` map.
///
/// Must run in the metrics loop BEFORE the step-2 snapshot — no await sits
/// between the merge and that snapshot, so the merged value lands in this
/// tick's `MetricCurrent` (surfacing within the ~2 s iteration cycle);
/// reordered, it would only surface next tick.
///
/// Semantics (ADR-0014): the row is the SOLE WRITER OF THE VALUES — overwrite
/// all five fields unconditionally, INCLUDING `None`. (Other paths — lifecycle
/// unload, rename, reset, conn-error cleanup — still remove or migrate entries;
/// see the task context.) The row is itself 30s-gated by the tamad (a `None`
/// means "no traffic for 30s" and must blank the entry); the pre-ADR-0014
/// "skip stale defaults" or-merge is retired. `last_updated_ms` is stamped
/// (proxy clock) ONLY when the row's `tps` is `Some` — stamping unconditionally
/// would make it ~equal across all entries (the merge touches every entry every
/// tick) and `aggregate_inference`'s "latest entry wins" would degenerate to
/// arbitrary.
pub(crate) async fn merge_tamad_inference_stats(
    state: &crate::proxy::ProxyState,
    live: &crate::proxy::Rows,
) {
    let cfg = state.config.read().await;
    let model_configs = state.registry.model_configs.read().await;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    for row in live.all() {
        let tps = row.tps;
        let prompt_tps = row.prompt_tps;
        let cache_hit_pct = row.cache_hit_pct;
        let spec_accept_pct = row.spec_accept_pct;
        let active = row.spec_decoding_active;
        let stamp = tps.is_some();
        let servers = cfg.resolve_backends_for_model(&model_configs, &row.key);
        for (server_name, _, _) in &servers {
            let sn = server_name.clone();
            state.metrics.modify_inference_stats(|m| {
                let entry = m.entry(sn.clone()).or_default();
                entry.tps = tps;
                entry.prompt_tps = prompt_tps;
                entry.cache_hit_pct = cache_hit_pct;
                entry.spec_accept_pct = spec_accept_pct;
                entry.spec_decoding_active = active;
                if stamp {
                    entry.last_updated_ms = now_ms;
                }
            });
        }
    }
}
```

   (Compute `now_ms` ONCE per merge call, before the loop — one clock read for all entries. The `Semantics` paragraph above already encodes the "sole writer of the VALUES" wording — keep it as written.)

2. Update the call site (line 359): `merge_tamad_spec_stats(&metrics_state, &live).await;` → `merge_tamad_inference_stats(&metrics_state, &live).await;` (and the surrounding "2a" comment: "Fold the tamad-observed inference stats into the per-server map BEFORE the snapshot (ADR-0014)").

3. Update the `aggregate_inference` doc comment (lines 59-65) to match what the UNCHANGED code actually does: "Live-value aggregation for the broadcast snapshot. tps/prompt_tps AND spec_accept_pct are None when the newest entry is older than the 30 s bucket window — a BACKSTOP since ADR-0014 (entries are already 30s-gated by the tamad; the gate now only fires on edge cases like a wedged row stream). `cache_hit_pct` keeps the newest entry's value when stale — a no-op backstop under ADR-0014 (the entry's `cache_hit_pct` is already `None` after 30s idle, so the ungated read returns `None` anyway). `spec_decoding_active` is OR'd across entries (cluster level) but is no longer sticky per entry: the merge overwrites it, so an entry un-sticks when its backend stops spec-decoding." Do NOT claim `cache_hit_pct` is gated in the stale branch — the code returns it ungated, and the existing test `test_aggregate_inference_stale_gates_tps_and_spec_but_not_sticky_fields` (~line 798) asserts exactly that.

**Steps:**
- [ ] Update the existing merge tests in `server/metrics.rs` (they live at lines ~659-760, built on the `state_with_live_model` / `seed_live_row` / `process` helpers at lines ~592-644 — the `process` helper at line 592 builds a `ProcessInfo` and gains the three new fields in Task 1). The old assertions encode the RETIRED or-merge semantics (e.g. `test_merge_sets_spec_fields_preserving_forwarder_write` asserts the merge must NOT touch `tps`/`last_updated_ms` of a forwarder-written entry) — rework them to assert the NEW overwrite semantics: a row with `spec_accept_pct: Some(44.5)` AND `tps: None` now sets the entry's `tps` to `None` (the row is the sole source of values); a row with `tps: Some(42.0)` stamps `last_updated_ms`. Then add:
      - `test_merge_overwrites_including_none` — seed the map with a full entry (all five fields set, `last_updated_ms` recent); merge a row with all-`None`/`false` fields → the entry's five fields are `None`/`false`
      - `test_merge_stamps_last_updated_only_when_some` — merge a `tps: None` row → `last_updated_ms` unchanged; merge a `tps: Some(1.0)` row → `last_updated_ms` updated
      - `test_merge_replaces_previous_value` — `Some(50.0)` then a row with `Some(30.0)` → `30.0`
      - `test_merge_fanout_aliases` — a model key with two config entries (insert two `ModelConfig` rows under different config keys that both resolve to the same model, mirroring how `resolve_backends_for_model` is exercised in the existing tests) → both entries receive the row's values
- [ ] Run `cargo nextest run --package tama-core -- proxy::server::metrics`
  - Did the updated/added tests fail first (old behavior)? If the new tests passed before the implementation change, stop and investigate.
- [ ] Implement `merge_tamad_inference_stats` (step 1) + the call-site rename (step 2) + the doc-comment updates (step 3)
- [ ] Run `cargo nextest run --package tama-core -- proxy::server::metrics` — all pass?
- [ ] Add a NEW end-to-end test `test_merge_to_aggregate_end_to_end` in the same `mod tests` (there is no runnable full-loop test — the production loop at lines ~351-465 is an untestable infinite loop; the e2e path is the two testable functions). FIRST add the seeding machinery the test needs — the existing `seed_live_row` (line 614) / `state_with_live_model` (line 636) helpers build their `ProcessInfo` via the old `process` helper (line 592) and carry ONLY the spec fields, and `Rows` is not constructible from tests (private `ordered` field) — so: (a) add a SIBLING helper to `process` that also takes `tps`/`prompt_tps`/`cache_hit_pct` (keep the existing helper's signature stable so the existing call sites don't churn), and (b) add a SIBLING `seed_live_row_with_rates(state, model_id, tps, prompt_tps, cache_hit_pct, spec_accept_pct, spec_decoding_active)` that builds the `ProcessInfo` via the sibling `process` helper and inserts via the SAME `stats_full` / `handle_with_latest` / `insert_raw_handle` plumbing `seed_live_row` already uses (`crate::tamad::pool::test_support`). The e2e test, `test_merge_stamps_last_updated_only_when_some` (its `Some(1.0)` half), and `test_merge_replaces_previous_value` all seed their tps-bearing rows through `seed_live_row_with_rates` (a `tps: None` row for the non-stamping half comes from the existing `seed_live_row`); the e2e test seeds a row with `tps: Some(42.0)`, `prompt_tps: Some(120.0)`, `cache_hit_pct: Some(25.0)`, `spec_accept_pct: Some(44.5)`, `spec_decoding_active: true`; then call `merge_tamad_inference_stats(&state, &live).await` then `aggregate_inference(&state.metrics.inference_stats_snapshot(), now_ms, BUCKET_MS)`; assert the aggregated tuple carries `42.0` / `120.0` / `25.0` / `44.5` / `true` — proving wire → row → merge → aggregate
- [ ] Rewrite the STALE `LatestInferenceStats` doc comments in `crates/tama-core/src/proxy/types.rs` (struct doc at lines 12-17 + the per-field docs at lines 21-37) — they currently say "extracted from llama_cpp response `timings` object ... Updated on each non-streaming response" and describe `spec_decoding_active` as "True if draft_n > 0 **has ever been observed**" (the sticky semantics retired by this task). New wording: values come from the tamad's windowed `/metrics` scrape via `merge_tamad_inference_stats` (ADR-0014); `None` = no traffic in the last 30s (or the engine doesn't expose the counter); `spec_decoding_active` is the last window's spec-traffic flag, NOT sticky
- [ ] Run `cargo nextest run --package tama-core` — all pass (including the still-present per-response tests, which now just lose the race to the merge — that's expected and fine)?
- [ ] Run `cargo fmt --all` — clean?
- [ ] Run `cargo clippy --workspace --all-targets -- -D warnings` — clean?
- [ ] Commit with message: `feat: merge overwrites inference stats from tamad rows (sole source, ADR-0014 task 2)`

**Acceptance criteria:**
- [ ] The merge overwrites all five fields including `None`, and stamps `last_updated_ms` only on `Some` tps
- [ ] Alias fan-out delivers one row's values to every matching config entry
- [ ] The end-to-end test proves a live row's values land in the `aggregate_inference` output
- [ ] `aggregate_inference` code is unchanged (only its doc comment changed, and the new comment matches the code — `cache_hit_pct` is NOT claimed to be gated); all pre-existing aggregate tests pass
- [ ] The `LatestInferenceStats` doc comments in `types.rs` describe the ADR-0014 semantics (no stale "per-response" wording remains)

---

### Task 3: Retire the per-response extraction + inject `--metrics` for llama.cpp

**Context:**
The merge is now the sole writer of inference-stats values (Task 2), so the per-response extraction is dead weight — this commit deletes it. Two parts: (a) delete `extract_inference_stats` and everything that exists only to serve it; (b) inject `--metrics` for llama.cpp backends in `build_full_args` (llama.cpp serves its Prometheus endpoint only with the flag — 501 without it — so without injection, llama.cpp inference stats would all be `None`). vLLM needs no flag (its `/metrics` is on by default). The response payload still *contains* the `metrics`/`timings` JSON for API compatibility — we only stop reading it.

**Files:**
- Delete: `crates/tama-core/src/proxy/forward/stats.rs` (whole file — `extract_inference_stats` + `extract_llama_cpp_stats` + `extract_vllm_stats` + their tests)
- Delete: `crates/tama-core/src/proxy/forward/tests/extract_stats.rs` (whole file — dedicated tests for the deleted extractors)
- Modify: `crates/tama-core/src/proxy/forward/mod.rs` — remove `pub(super) mod stats;` (line 6)
- Modify: `crates/tama-core/src/proxy/forward/request.rs` — remove `use super::stats::extract_inference_stats;` (line 8) and the call `let _stats = extract_inference_stats(backend_name, &parsed, &state.metrics);` (line 358); if `parsed` (or its binding) becomes unused as a result, remove that too — the compiler will tell you. Keep the rest of the response handling (status checks, body passthrough) intact
- Modify: `crates/tama-core/src/proxy/forward/sse.rs` — remove `use super::stats::extract_inference_stats;` (line 3) and the `extract_inference_stats(...)` call (line 25); then remove the now-unused `inference_stats: Option<&MetricsState>` **parameter** from `process_sse_line`'s signature (line ~11) — a dead parameter is a `clippy -D warnings` failure — and update its call site(s) in `request.rs` (drop the `Some(&metrics_state)` argument at ~lines 317-322 and the `metrics_state` clone plumbing that exists only to feed it, at ~lines 228-229/298); prune the now-unused `use crate::proxy::state::MetricsState;` imports in `forward/tests.rs` and `forward/tests/sse.rs`
- Modify: `crates/tama-core/src/proxy/forward/tests.rs` — remove `mod extract_stats;`, the `use super::stats::extract_inference_stats;` import (line ~2), the now-dead `make_metrics_state()` helper (lines ~9-11 — its only consumers are the deleted `extract_stats` tests; leaving it is a `clippy -D warnings` dead-code failure), and the now-unused `use crate::proxy::state::MetricsState;` import
- Modify: `crates/tama-core/src/proxy/forward/tests/request.rs` and `crates/tama-core/src/proxy/forward/tests/sse.rs` — in `tests/sse.rs` the stats-specific tests are EXACTLY `test_process_sse_line_extracts_inference_stats` (~line 111) and `test_process_sse_line_extracts_vllm_stats` (~line 135) — delete those two, KEEP everything else (rewrite/passthrough tests) — and update the KEPT tests' `process_sse_line(...)` call sites to drop the trailing `None` argument (e.g. line 64: `process_sse_line("", Some("any-model"), "test-server", &mut out, None)` → drop the `None`). In `tests/request.rs`: delete the tests that assert stats extraction from forwarded responses (search for `inference_stats` assertions), BUT **KEEP `test_forward_request_conn_error_returns_502_and_cleans_up`** (~line 74) — it is a forwarding test (502 + `BadGatewayError` + inference-map cleanup) that seeds via `record_inference_stats` at line 84, which this task deletes: rework its seeding to `state.metrics.modify_inference_stats(|m| { m.insert(backend, LatestInferenceStats { ... }) })` (the assertions stay identical)
- Modify: `crates/tama-core/src/proxy/forward/request.rs` — the conn-error cleanup (`map.remove(backend_name)` at ~lines 497-502): **KEEP it** — it is a harmless backstop (under the new semantics the merge re-creates the entry within ~2s whenever a live row still resolves; the cleanup only matters when the model has left the live rows)
- Modify: `crates/tama-core/src/proxy/state/metrics.rs` — remove `record_inference_stats` (lines 69-72) and ONLY its own test (`test_record_inference_stats_and_snapshot`, ~line 136); the OTHER tests in this file that use `record_inference_stats` purely as seeding — `test_clear_inference_stats` (seed sites ~171/~178) and `test_modify_inference_stats` (seed site ~213) — test `clear_inference_stats`/`modify_inference_stats`, which SURVIVE: rework their seeding to `modify_inference_stats(|m| { m.insert(backend, LatestInferenceStats { ... }) })` (assertions stay identical). The tests in `crates/tama-core/src/proxy/server/metrics.rs` that seed the map via `record_inference_stats` (lines ~661, ~710) get the same rework
- Create: `crates/tama-core/src/config/llama_cpp_args.rs`
- Modify: `crates/tama-core/src/config/mod.rs` — add `mod llama_cpp_args;` next to `mod vllm_args;` (line 7)
- Modify: `crates/tama-core/src/config/resolve/mod.rs` — `build_full_args` (line 227+): after the `--alias` injection block (ends ~line 588), before the sampling merge, add the managed-flag injection

**What to implement:**

1. **Deletions** — exactly the file/line list above. The `LatestInferenceStats` struct itself STAYS (the merge writes it); only `record_inference_stats` (the whole-entry insert used solely by the deleted path) goes. After the deletions, `cargo clippy` will surface any dangling references — fix them by removing the reference, not by re-adding the code.

2. **`crates/tama-core/src/config/llama_cpp_args.rs`**:

```rust
//! Managed llama.cpp flags (ADR-0014).
//!
//! `--metrics` enables llama.cpp's Prometheus `/metrics` endpoint (off by
//! default upstream — the endpoint answers 501 without the flag). The
//! tamad scrapes that endpoint for inference stats (tps, prompt_tps,
//! cache_hit_pct, spec fields), so the proxy injects the flag for
//! llama.cpp backends. Presence-checked: a user-supplied `--metrics` is
//! never duplicated or overridden.

use crate::config::flag_name;

/// Append `--metrics` to grouped args if not already present.
///
/// Returns `true` if the flag was added.
pub fn ensure_metrics(grouped: &mut Vec<String>) -> bool {
    let present = grouped.iter().any(|e| matches!(flag_name(e), Some("--metrics")));
    if present {
        false
    } else {
        grouped.push("--metrics".to_string());
        true
    }
}
```

3. **`build_full_args` injection** — after the `--alias` block (~line 588), before the sampling merge:

```rust
// Managed flag (ADR-0014): the tamad scrapes the engine's /metrics endpoint
// for inference stats; llama.cpp serves it only with --metrics (501 without).
// Inject for llama.cpp backends, presence-checked (user args never overridden).
if is_llama_cpp_backend {
    crate::config::llama_cpp_args::ensure_metrics(&mut grouped);
}
```

   (`is_llama_cpp_backend` is already computed at line 305 via `backend_is_llama_cpp(&server.backend)`. `flag_name` is the existing helper in `config/args_helpers.rs`, re-exported at `crate::config::flag_name` — it handles grouped `--flag value`, inline `--flag=value`, and short forms, so a user's `--metrics` in ANY form suppresses the injection.)

**Steps:**
- [ ] Write the `llama_cpp_args` unit tests first (in the new file's `mod tests`):
      - `test_ensure_metrics_adds_when_absent` — `["--port", "8080"]` → flag appended, returns `true`
      - `test_ensure_metrics_idempotent_when_present_grouped` — `["--metrics"]` → unchanged, returns `false`
      - `test_ensure_metrics_idempotent_when_present_inline` — `["--metrics=true"]` → unchanged, returns `false`
      - `test_ensure_metrics_does_not_touch_other_args` — order of pre-existing entries preserved
- [ ] Add a `build_full_args` integration test in `crates/tama-core/src/config/resolve/` (follow the existing `build_full_args` tests there): a GGUF model config (non-transformers `hf_format`, `backend: "llama.cpp"`) with no `--metrics` in `server.args` → the flat output contains `"--metrics"` exactly once; with `--metrics` already in `server.args` → still exactly once
- [ ] Run `cargo nextest run --package tama-core -- config`
  - Did the new tests fail first? (The integration test fails until step 3 is implemented; the unit tests fail until the file exists.)
- [ ] Implement `llama_cpp_args.rs` + the `mod` declaration + the `build_full_args` injection
- [ ] Run `cargo nextest run --package tama-core -- config` — all pass?
- [ ] Perform the deletions (part 1) — let the compiler guide you: after deleting `stats.rs`, fix the broken references in `forward/mod.rs`, `forward/request.rs` (extract call + the `process_sse_line` call-site cleanup per the file list above), `forward/sse.rs` (extract call + the `inference_stats` parameter removal + import prune), `forward/tests.rs` (`mod extract_stats` + `MetricsState` import), `forward/tests/request.rs` (stats-extraction test deletions + the conn-error test rework), `forward/tests/sse.rs` (the two named stats tests + the `MetricsState` import), `state/metrics.rs` (`record_inference_stats` + its tests), `server/metrics.rs` (the two `record_inference_stats` seed sites at ~661/~710 → rework to `modify_inference_stats`)
- [ ] Run `cargo nextest run --package tama-core` — all remaining tests pass? (Any test that asserted "a forwarded response with `timings`/`metrics` updates the inference stats map" is DELETED by design — the stats now come from the tamad; do not rework such a test to assert the opposite.)
- [ ] Run `cargo fmt --all` — clean?
- [ ] Run `cargo clippy --workspace --all-targets -- -D warnings` — clean?
- [ ] Run `cargo clippy --package tama --features ssr --all-targets -- -D warnings` and `cargo check --package tama --no-default-features --features csr` (the `tama` crate's `admin.rs` still constructs `ProcessInfo` — must still compile)
- [ ] **Full validation gate** (this is the final task — run the complete CI gate): `cargo fmt --all --check` / `cargo clippy --workspace --all-targets -- -D warnings` / `cargo clippy --package tama --features ssr --all-targets -- -D warnings` / `cargo check --package tama --no-default-features --features csr` / `cargo nextest run --workspace`
- [ ] Commit with message: `refactor: retire per-response inference stats extraction; inject --metrics for llama.cpp (ADR-0014 task 3)`

**Acceptance criteria:**
- [ ] `extract_inference_stats` / `extract_llama_cpp_stats` / `extract_vllm_stats` / `record_inference_stats` are gone from the codebase (grep finds no references)
- [ ] `LatestInferenceStats` and the `inference_stats` watch map remain, written (for values) solely by `merge_tamad_inference_stats`; the lifecycle/rename/clear/conn-error paths still remove or migrate entries as before
- [ ] `process_sse_line` no longer takes a `MetricsState` parameter; no dead imports/parameters remain (clippy clean)
- [ ] `test_forward_request_conn_error_returns_502_and_cleans_up` still exists and passes (reworked seeding)
- [ ] `build_full_args` output for a llama.cpp backend contains `--metrics` (injected when absent, never duplicated)
- [ ] The full validation gate passes
