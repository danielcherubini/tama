# LiteLLM-Compatible Model Info Plan

**Goal:** Serve a LiteLLM-shaped `GET /v1/model/info` (and `GET /model/info`) on the proxy so that any plugin written against LiteLLM's discovery API — first and foremost `pi-provider-litellm` — works against tama unchanged: point `LITELLM_URL` at tama and the plugin's `discoverModels` (single `GET /v1/model/info` call) returns every model with the full `model_info` metadata it reads, including the reasoning-effort fields.

**Architecture:** Pure addition, no migration. A new handler in `tama_handlers/models/` reuses the opencode handler's existing per-model data collection (config snapshot, backend `/props` capabilities, backend `/v1/models` context, alias inheritance) — extracted into a shared `collect_model_entries` helper — and maps each `ModelEntry` to the LiteLLM wire shape (`{ data: [ { model_name, litellm_params, model_info } ] }`). All values are derived from data tama already has: `ModelConfig.reasoning_levels` (pi vocabulary, ADR-0008/0009) converted to the wire vocabulary (`off` → `none`), `modalities`, the `/props`-derived capability flags, and the existing context-length resolution chain. Auth and scoping come for free from the existing `auth_middleware` (tama_ Bearer keys) and `scope_middleware` (`/v1/*` → `Inference` scope).

**Tech Stack:** Rust (Axum, serde, `tama-core` proxy crate), JSON wire contract pinned by drift-guard tests.

**References:** `pi-provider-litellm` `src/litellm-api.ts` `discoverModels`/`mergeModelInfo`/`deriveThinkingSupport` (the consumer contract) · LiteLLM `proxy_server.py` `model_info_v1` + `model_prices_and_context_window.json` (the wire contract) · ADR-0008 (derived boolean) · ADR-0009 (`off` stored, `none` on the wire) · `docs/research/reasoning-effort-model-info.md` (Q1/Q5) · plan-189 (the reasoning-effort feature this builds on)

**Conventions (from AGENTS.md):** TDD (failing test → implement → pass); targeted `cargo nextest run --package <crate> -- <filter>` while coding; full gate (fmt + clippy all-targets + SSR clippy + nextest workspace) before the final commit. Commit prefixes: `feat:`, `refactor:`, `docs:`.

**What NOT to change (guardrails):**
- The opencode wire shape (`ModelEntry`) — drift-guarded by `tests/opencode.rs`; Task 1 is a pure refactor and the opencode response must stay byte-identical.
- `GET /health` — it is a liveness probe (`{ "status": "ok", "service": "tama-proxy" }`) used by the control plane; it is NOT LiteLLM's `{ healthy_endpoints: [...] }` and must not be reshaped. Consequence: the plugin's `/health` fallback path (only reached when `/v1/model/info` is unreachable or empty) yields zero models against tama — acceptable, the primary path is what the plugin uses.
- `GET /v1/models` shape (OpenAI-compatible list) — untouched.
- The stored `reasoning_levels` vocabulary (pi words, ADR-0009) — the `off` → `none` conversion happens at serialization only.
- No new DB columns, no new config fields — everything is derived.

---

### Task 1: Extract shared `collect_model_entries` from the opencode handler (pure refactor)

**Context:**
The new LiteLLM handler needs exactly the per-model data the opencode handler already collects: a lock-scoped snapshot of live rows + enabled configs, concurrent `/props` capability fetches (deduplicated by backend URL), concurrent backend `/v1/models` fetches, entry construction via `build_model_entry`, and alias entries that inherit their target's metadata. Duplicating that ~100-line collection in a second handler would create two sources of truth that can drift (e.g., a future change to alias inheritance or the context-resolution chain). This task extracts it into one shared helper as a pure refactor — the opencode endpoint's wire output must be byte-identical before and after.

**Files:**
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/opencode.rs`
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/utils.rs`
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/tests/capabilities.rs` (import path — see step 5)
- Test: `crates/tama-core/src/proxy/tama_handlers/models/tests/opencode.rs` (existing suite is the guard — no new tests required, but it must pass unchanged)

**What to implement:**

1. In `utils.rs`, add:
   ```rust
   /// Collect one `ModelEntry` per enabled config plus alias entries, using
   /// the shared per-backend data sources (live rows, /props capabilities,
   /// backend /v1/models context). Shared by the opencode and LiteLLM
   /// model-info handlers — the single source of truth for this collection.
   pub(super) async fn collect_model_entries(
       state: &Arc<crate::proxy::ProxyState>,
   ) -> Vec<ModelEntry>
   ```
   Body = the ENTIRE body of `handle_opencode_list_models` (from the step-1 comment at line 20 through `drop(aliases)` at line ~147): the `(loaded_models, all_configs)` lock-scoped snapshot, unique-URL `/props` fetch (`fetch_capabilities_from_backend`) into a `url_cap_map`/`cap_map`, unique-URL `/v1/models` fetch (`fetch_models_from_backend`) into `url_model_map`, the `all_configs` loop building entries with `build_model_entry(&state, id, cfg, caps, backend_ctx)`, and the alias loop (skip aliases whose name is already seen; inherit target caps/context; `entry.id = Some(alias_name)`; alias display name). Move the code verbatim — do not change its logic.
2. Move the three helper functions that the body calls out of `opencode.rs` into `utils.rs` (they are collection plumbing, not opencode-specific):
   - `fetch_capabilities_from_backend` (`opencode.rs:218`, `pub(super)`) → `utils.rs`, keep `pub(super)` (still used by `tests/capabilities.rs`).
   - `extract_capabilities` (`opencode.rs:154`, `pub(super)`) → `utils.rs`, keep `pub(super)` (still used by `tests/capabilities.rs` and by `fetch_capabilities_from_backend`).
   - `extract_context_length_from_backend_entry` (`opencode.rs:198`, private) → `utils.rs` as a private `fn` (its only caller is `collect_model_entries`).
   `ModelCapabilities` stays where it is (`mod.rs:7`) — `utils.rs` already does `use super::ModelCapabilities`, so no visibility change is needed.
3. `handle_opencode_list_models` becomes (the current handler has no attributes — only a doc comment; keep that doc comment and add no attributes, this is a pure refactor):
   ```rust
   pub async fn handle_opencode_list_models(
       state: State<Arc<crate::proxy::ProxyState>>,
   ) -> Json<OpencodeModelsResponse> {
       let models = collect_model_entries(&state.0).await;
       Json(OpencodeModelsResponse { models })
   }
   ```
4. `tests/capabilities.rs` line 7: change the import from
   `use super::super::opencode::{extract_capabilities, fetch_capabilities_from_backend};`
   to
   `use super::super::utils::{extract_capabilities, fetch_capabilities_from_backend};`
   (the tests themselves are unchanged). `mod.rs` is untouched in this task (the `litellm` module + `mod litellm;` tests-block entry land in Task 2).
5. Do not touch any response field, ordering, or dedup behavior.

**Steps:**
- [ ] Baseline: `cargo nextest run --package tama-core -- opencode` and `-- capabilities` — record that all tests pass before the change.
- [ ] Implement the extraction in `utils.rs` (helper + 3 moved fns) + slim `opencode.rs` + the `tests/capabilities.rs` import fix.
- [ ] `cargo nextest run --package tama-core -- opencode` and `-- capabilities` — all tests still pass, unchanged.
- [ ] `cargo nextest run --package tama-core -- models` — the whole models test dir passes.
- [ ] `cargo fmt --all`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- [ ] Commit with message: `refactor: extract shared model-entry collection from the opencode handler`

**Acceptance criteria:**
- [ ] `collect_model_entries` exists in `utils.rs` and is the only place that performs the /props + backend /v1/models collection.
- [ ] `fetch_capabilities_from_backend`, `extract_capabilities`, `extract_context_length_from_backend_entry` live in `utils.rs`; `opencode.rs` no longer defines them.
- [ ] `handle_opencode_list_models` is ≤ ~15 lines and calls the helper, with its pre-existing attributes unchanged.
- [ ] Every pre-existing opencode + capabilities test passes without modification (byte-identical wire output).
- [ ] Workspace clippy clean.

---

### Task 2: LiteLLM entry mapping — pure function with unit tests

**Context:**
The heart of the feature: map a collected `ModelEntry` to the LiteLLM wire shape. The contract is defined by what `pi-provider-litellm` reads (`src/litellm-api.ts` `discoverModels` → `mergeModelInfo` → `mapToProviderModel`/`deriveThinkingSupport`) cross-checked against LiteLLM's own `model_info_v1` docstring example and `model_prices_and_context_window.json` schema. Key semantics: the per-level `supports_*_reasoning_effort` flags are **absent** when the model has no configured levels (LiteLLM treats absent as "unknown — advertise nothing"), and present with `true`/`false` when `reasoning_levels` is non-empty (the explicit list is authoritative — the same semantics the plugin's `deriveThinkingSupport` implements). The `off` → `none` conversion is ADR-0009 and reuses the existing `reasoning_options_from_levels` conversion logic.

**Files:**
- Create: `crates/tama-core/src/proxy/tama_handlers/models/litellm.rs`
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/utils.rs` (factor the off→none list conversion out of `reasoning_options_from_levels`)
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/mod.rs` (declare `mod litellm;` in the module list AND add `mod litellm;` to the `#[cfg(test)] mod tests` block at lines 27–33 — without the tests-block entry the test file silently never compiles)
- Test: `crates/tama-core/src/proxy/tama_handlers/models/tests/litellm.rs`

**What to implement:**

1. In `utils.rs`, factor the conversion out so both consumers share it:
   ```rust
   /// Stored levels (pi vocabulary) → wire values (ADR-0009: `off` → `none`,
   /// all other levels pass through). None for absent/empty levels.
   pub(super) fn levels_to_wire_values(levels: &Option<Vec<String>>) -> Option<Vec<String>>
   ```
   `reasoning_options_from_levels` becomes a wrapper: `levels_to_wire_values(levels).map(|values| serde_json::json!([{ "type": "effort", "values": values }]))`. Existing `reasoning_options` behavior (and its tests) must not change.

2. In `litellm.rs`, typed structs (serde; field names are the LiteLLM wire names; `PartialEq` on every struct so the drift-guard test can assert equality — `mode` is a `String`, NOT `&'static str`, because `serde_json::from_value` requires `Deserialize<'de>` for arbitrary lifetimes and `&'static str` does not satisfy it):
   ```rust
   /// One entry of the LiteLLM `/v1/model/info` response.
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub struct LiteLLMModelInfoEntry {
       pub model_name: String,
       pub litellm_params: LiteLLMParams,
       pub model_info: LiteLLMModelInfo,
   }

   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub struct LiteLLMParams {
       /// The name the client calls the model with (same as `model_name`).
       pub model: String,
       /// The backend serving the model (e.g. "llama.cpp", "vllm").
       pub provider: String,
   }

   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub struct LiteLLMModelInfo {
       pub id: String,
       pub mode: String, // always "chat"
       pub litellm_provider: String,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub max_input_tokens: Option<u32>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub max_output_tokens: Option<u32>,
       /// LiteLLM's legacy field: max_output_tokens when known, else
       /// max_input_tokens (see model_prices_and_context_window.json
       /// `sample_spec`).
       #[serde(skip_serializing_if = "Option::is_none")]
       pub max_tokens: Option<u32>,
       pub supports_function_calling: bool,
       pub supports_reasoning: bool,
       pub supports_vision: bool,
       pub supports_audio_input: bool,
       pub supports_pdf_input: bool,
       /// Local inference: no per-token pricing.
       pub input_cost_per_token: f64,
       pub output_cost_per_token: f64,
       pub cache_read_input_token_cost: f64,
       pub cache_creation_input_token_cost: f64,
       // Reasoning-effort fields — present only when the model has configured
       // levels (ADR-0008); absent = "unknown, advertise nothing".
       #[serde(skip_serializing_if = "Option::is_none")]
       pub supports_minimal_reasoning_effort: Option<bool>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub supports_low_reasoning_effort: Option<bool>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub supports_none_reasoning_effort: Option<bool>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub supports_xhigh_reasoning_effort: Option<bool>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub supports_max_reasoning_effort: Option<bool>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub reasoning_effort_levels: Option<Vec<String>>,
   }
   ```

3. The mapping function:
   ```rust
   /// Map a collected model entry to the LiteLLM `/v1/model/info` wire shape.
   pub(super) fn litellm_entry_from_model_entry(entry: &ModelEntry) -> LiteLLMModelInfoEntry
   ```
   Rules (exact):
   - `model_name` = `entry.id.clone().unwrap_or_default()` (entries without an id are not produced by `collect_model_entries` in practice, but the mapping must not panic).
   - `litellm_params` = `{ model: <model_name>, provider: entry.backend }`.
   - `model_info.id` = `model_name`; `mode` = `"chat"`; `litellm_provider` = `entry.backend`.
   - `max_input_tokens` = `entry.limit.context`; `max_output_tokens` = `entry.limit.output`; `max_tokens` = `entry.limit.output.or(entry.limit.context)`.
   - `supports_function_calling` = `entry.tool_call` (defaults to `true` when no /props data exists — `build_model_entry`'s `unwrap_or((true, false))` default, same as the opencode wire).
   - `supports_reasoning` = `entry.reasoning || entry.supports_reasoning_effort` — the ADR-0008 effective flag (/props "backend preserves reasoning" OR user-configured levels).
   - `supports_vision` = `entry.modalities` input list contains `"image"`; `supports_audio_input` = contains `"audio"`; `supports_pdf_input` = contains `"pdf"`; all `false` when modalities is None.
   - All four cost fields = `0.0`.
   - Effort block: let `wire = levels_to_wire_values(&entry.reasoning_levels)`.
     - `wire` is `Some` (levels configured, non-empty): `reasoning_effort_levels = wire`; `supports_minimal_reasoning_effort = Some(wire contains "minimal")`; `supports_low_reasoning_effort = Some(contains "low")`; `supports_none_reasoning_effort = Some(contains "none")`; `supports_xhigh_reasoning_effort = Some(contains "xhigh")`; `supports_max_reasoning_effort = Some(contains "max")`.
     - `wire` is `None` (absent or empty levels): all six fields `None` → omitted from the wire.
     - Note: `medium` and `high` have **no** per-level flag in LiteLLM's schema (they are unconditional for reasoning models; the map only carries flags for minimal/low/none/xhigh/max) — stored `medium`/`high` appear only inside `reasoning_effort_levels`. Do not invent `supports_medium_reasoning_effort`.

4. Unit tests in `tests/litellm.rs` (construct `ModelEntry` values directly — no state, no network):
   - **Qwen3.8 case** (the research doc's first target): entry with `reasoning: false` (/props), `supports_reasoning_effort: true`, `reasoning_levels: Some(vec!["off","low","medium","xhigh"])`, `modalities: Some({ input: ["text","image"], output: ["text"] })`, `limit: { context: Some(262144), output: Some(32768) }`, `tool_call: true`, `backend: "llama.cpp"`, `id: Some("qwen3-8b")`. Assert: `model_name == "qwen3-8b"`; `litellm_params == { model: "qwen3-8b", provider: "llama.cpp" }`; `model_info.max_input_tokens == Some(262144)`, `max_output_tokens == Some(32768)`, `max_tokens == Some(32768)`; `supports_reasoning == true` (OR of false + true); `supports_vision == true`, `supports_audio_input == false`, `supports_pdf_input == false`; `supports_function_calling == true`; all costs `0.0`; `reasoning_effort_levels == Some(["none","low","medium","xhigh"])` (off→none, order preserved); `supports_none_reasoning_effort == Some(true)`; `supports_minimal_reasoning_effort == Some(false)`; `supports_low_reasoning_effort == Some(true)`; `supports_xhigh_reasoning_effort == Some(true)`; `supports_max_reasoning_effort == Some(false)`.
   - **No levels → effort fields absent on the wire**: entry with `reasoning: true` (/props), `supports_reasoning_effort: false`, `reasoning_levels: None`. Serialize to `serde_json::Value` and assert the keys `supports_minimal_reasoning_effort`, `supports_low_reasoning_effort`, `supports_none_reasoning_effort`, `supports_xhigh_reasoning_effort`, `supports_max_reasoning_effort`, `reasoning_effort_levels` are all **absent**; assert `supports_reasoning == true` (props-only).
   - **Empty levels vec → same absence**: `reasoning_levels: Some(vec![])` → all six absent, `supports_reasoning == false`.
   - **off-only**: `reasoning_levels: Some(vec!["off"])` → `reasoning_effort_levels == Some(["none"])`, `supports_none_reasoning_effort == Some(true)`, the other four flags `Some(false)`, `supports_reasoning == true`.
   - **max-only**: `reasoning_levels: Some(vec!["max"])` → `supports_max_reasoning_effort == Some(true)`, `reasoning_effort_levels == Some(["max"])`.
   - **Drift guard**: round-trip the Qwen3.8 case through `serde_json::to_value` → `serde_json::from_value::<LiteLLMModelInfoEntry>` and assert `PartialEq` equality (pins the wire shape, same pattern as `tests/opencode.rs::test_opencode_response_deserializes_into_typed`).

**Steps:**
- [ ] Write the six failing unit tests in `tests/litellm.rs` **and add `mod litellm;` to the `#[cfg(test)] mod tests` block in `mod.rs`** (the module-list `mod litellm;` entry can come with the implementation — but without the tests-block entry the test file is never compiled and the "confirm failure" step below would vacuously match zero tests).
- [ ] `cargo nextest run --package tama-core -- litellm` — confirm the tests fail (compile error or assertion failure, and at least 6 tests were *matched* — "0 tests ran" is a failure of this step, not of the tests; if they somehow pass, stop and investigate).
- [ ] Implement `levels_to_wire_values` in `utils.rs` (refactor `reasoning_options_from_levels` onto it) and `litellm.rs` (structs + mapping).
- [ ] `cargo nextest run --package tama-core -- litellm` — all pass.
- [ ] `cargo nextest run --package tama-core -- opencode` — the `reasoning_options` tests still pass (refactor guard).
- [ ] `cargo fmt --all`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- [ ] Commit with message: `feat: map model entries to the LiteLLM model-info wire shape`

**Acceptance criteria:**
- [ ] All six unit tests pass, including the wire-absence and drift-guard cases.
- [ ] `reasoning_options` serialization is byte-identical to pre-change (existing tests pass).
- [ ] No `supports_medium_reasoning_effort` / `supports_high_reasoning_effort` fields exist anywhere.

---

### Task 3: Handler, routes, auth/scope verification

**Context:**
Expose the mapping at the two LiteLLM paths. `GET /v1/model/info` is the plugin's primary discovery call (single request, all models); `GET /model/info` is the same handler — LiteLLM serves both and other clients may use either. Auth (`tama_` Bearer key via `auth_middleware`) and scoping (`/v1/*` → `Inference` scope via `scope_middleware`) apply automatically because every proxy route passes through `apply_shared_layers` — no new auth code. `GET /model/info` (no `/v1` prefix) matches `scope_middleware::required_scope`'s `None` branch, i.e. no scope requirement — same as `/health`/`/status` today; the auth middleware still gates it when auth is configured.

**Files:**
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/litellm.rs` (add the handler)
- Modify: `crates/tama-core/src/proxy/tama_handlers/models/mod.rs` (re-export)
- Modify: `crates/tama-core/src/proxy/tama_handlers/mod.rs` (re-export `handle_litellm_model_info` next to `handle_opencode_list_models`)
- Modify: `crates/tama-core/src/proxy/server/router.rs` (import + two route entries)
- Modify: `crates/tama-core/src/proxy/scope_middleware.rs` (`required_scope` unit test in the existing `mod tests`)
- Modify: `crates/tama/tests/router_ownership_test.rs` (`EXPECTED_PROXY_PATH_COUNT` 31 → 33)
- Test: `crates/tama-core/src/proxy/tama_handlers/models/tests/litellm.rs` (handler tests, extending Task 2's file)
- Test: `crates/tama-core/src/proxy/tama_handlers/models/tests/helpers.rs` (add the `call_litellm_model_info` helper)

**What to implement:**

1. Handler in `litellm.rs` (note the `litellm_model_id` query filter — LiteLLM serves a single entry for `?litellm_model_id=<id>`; on a match return that one entry, on no match return an empty `data` list, so the route is honest for LiteLLM-shaped clients that use the legacy path):
   ```rust
   /// LiteLLM-compatible model info: `GET /v1/model/info` and `GET /model/info`.
   /// Response: `{ "data": [ { model_name, litellm_params, model_info } ] }` —
   /// one entry per enabled model plus alias entries (same set and inheritance
   /// rules as `/v1/opencode/models`). `?litellm_model_id=<name>` filters to the
   /// single entry whose `model_name` matches (LiteLLM's per-model lookup);
   /// an unknown id yields an empty `data` array, and an empty id behaves like
   /// no param (full list).
   #[axum::debug_handler]
   pub async fn handle_litellm_model_info(
       state: State<Arc<crate::proxy::ProxyState>>,
       params: Query<LiteLLMModelIdQuery>,
   ) -> Json<serde_json::Value> {
       let entries = collect_model_entries(&state.0).await;
       // A present non-empty `litellm_model_id` filters to the matching entry;
       // absent or empty → full list.
       let wanted = params.id.as_deref().filter(|s| !s.is_empty());
       let data: Vec<serde_json::Value> = entries
           .iter()
           .filter(|e| wanted.is_none() || e.id.as_deref() == wanted)
           .map(|e| serde_json::to_value(litellm_entry_from_model_entry(e)).expect("LiteLLM entry serializes"))
           .collect();
       Json(serde_json::json!({ "data": data }))
   }
   ```
   With the named-field extractor (a tuple newtype would NOT work — serde derive for newtypes ignores field renames and never reads the `litellm_model_id` key out of the query map, so axum `Query` would reject every request):
   ```rust
   #[derive(Debug, Deserialize)]
   struct LiteLLMModelIdQuery {
       #[serde(default)]
       #[serde(rename = "litellm_model_id")]
       id: Option<String>,
   }
   ```
   Behavior contract (the tests below pin it): no param → full list; `?litellm_model_id=` (empty value) → full list; `?litellm_model_id=<name>` → the single matching entry, or an empty `data` array when no entry's `model_name` matches.
2. Routes in `router.rs` `proxy_routes()`, in the "OpenCode plugin discovery" section:
   ```rust
   // LiteLLM-compatible discovery (plan-196)
   ("GET", "/v1/model/info", get(handle_litellm_model_info)),
   ("GET", "/model/info", get(handle_litellm_model_info)),
   ```
   (`proxy_route_paths()` derives from `proxy_routes()` — no separate table to update. The cross-crate ownership test `crates/tama/tests/router_ownership_test.rs` pins the proxy route count: `test_proxy_and_management_tables_are_disjoint` asserts `EXPECTED_PROXY_PATH_COUNT = 31` (line ~110) — bump it to **33** (the test's own comment sanctions updating it when intentional). NOTE: that file is `#![cfg(feature = "ssr")]`, so the check must run with the feature: `cargo nextest run --package tama --features ssr -- router_ownership`.)
3. Handler tests (extend `tests/litellm.rs` with the `tests/helpers.rs` pattern — add a `call_litellm_model_info(state)` helper mirroring `call_list_models`, routing `GET /v1/model/info` through a `Router::new().route(...).with_state(state)` oneshot):
   - **Envelope + one model**: fixture built with the existing `create_state_with_model` helper, but the config MUST set `api_name: Some("test-model")` AND `model: Some("test/model")` AND `enabled: true` (without `model`/`api_name`, `build_model_entry` returns `None` and the response is an empty `data` — every existing fixture in `tests/opencode.rs` sets both). Use `reasoning_levels: Some(["off","medium"])`, `context_length: Some(32768)`, no backend loaded. Assert HTTP 200; body has a `data` array of length 1; `data[0].model_name == "test-model"` (api_name wins); `data[0].model_info.max_input_tokens == 32768`; `data[0].model_info.reasoning_effort_levels == ["none","medium"]`; `data[0].model_info.supports_function_calling == true` (no backend → `build_model_entry`'s `(true, false)` default, NOT false).
   - **Alias entry**: fixture with a model + an alias (mirror the alias setup in `tests/opencode.rs` alias tests). Assert `data` has 2 entries; the alias entry's `model_name` is the alias name and its `model_info` mirrors the target's (same context, same effort fields).
   - **`/model/info` parity**: same fixture routed through `GET /model/info` → byte-identical body to `/v1/model/info`.
   - **`litellm_model_id` filter**: the envelope fixture with `?litellm_model_id=test-model` → `data` length 1, `data[0].model_name == "test-model"`; with `?litellm_model_id=does-not-exist` → `data` length 0; with `?litellm_model_id=` (empty value) → `data` length 1 (full list — the contract above).
   - **No-levels model**: config without `reasoning_levels` → `data[0].model_info` has no `reasoning_effort_levels` key (assert absence on the `serde_json::Value`).
   - **Scope**: the scope tests live in `scope_middleware.rs`'s own `mod tests` (line 134) — add a `required_scope` unit test mirroring `test_required_scope_v1_audio_returns_inference` (line 415): `assert_eq!(required_scope("/v1/model/info", &Method::GET), Some(Scope::Inference))` and `assert_eq!(required_scope("/model/info", &Method::GET), None)`. Do not invent a new harness; do not add a full 403 integration test (the existing `test_scope_middleware_key_without_inference_rejected` already covers the middleware mechanics for `/v1/*`).

**Steps:**
- [ ] Write the failing handler tests (fixture + envelope + query-filter assertions) in `tests/litellm.rs` and the `call_litellm_model_info` helper in `tests/helpers.rs`.
- [ ] `cargo nextest run --package tama-core -- litellm` — confirm failure (handler/routes missing).
- [ ] Implement the handler + the two routes + re-exports, and bump `EXPECTED_PROXY_PATH_COUNT` to 33 in `crates/tama/tests/router_ownership_test.rs`.
- [ ] `cargo nextest run --package tama-core -- litellm` — all pass.
- [ ] `cargo nextest run --package tama-core -- scope` — the new `required_scope` unit test passes alongside the existing scope tests.
- [ ] `cargo nextest run --package tama-core -- opencode` and `-- models` — unchanged suites still pass.
- [ ] `cargo nextest run --package tama --features ssr -- router_ownership` — boundary test passes (zero matched tests would mean the ssr feature was dropped — verify a test actually ran).
- [ ] `cargo fmt --all`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- [ ] Commit with message: `feat: serve LiteLLM-shaped model info at /v1/model/info and /model/info`

**Acceptance criteria:**
- [ ] `GET /v1/model/info` returns `{ "data": [...] }` with one entry per enabled model plus aliases, 200 for an `Inference`-scoped key.
- [ ] `GET /model/info` returns a byte-identical body; `?litellm_model_id=` filters to the matching entry (empty `data` when unmatched).
- [ ] A key without the `Inference` scope gets 403 on `/v1/model/info` (consistent with the other `/v1/*` routes).
- [ ] `EXPECTED_PROXY_PATH_COUNT` is 33 and the ssr-gated ownership test passes.
- [ ] Opencode and `/v1/models` suites pass unchanged.

---

### Task 4: Docs — API reference, OpenAPI, plan bookkeeping

**Context:**
The repo documents every client-facing contract in `docs/api/` + `docs/openapi/` (plan-189 precedent). The new endpoint must be documented with its field-by-field mapping so the `off`/`none` duality (ADR-0009) and the absent-when-unknown effort semantics are discoverable, plus plan bookkeeping in `docs/plans/BACKLOG.md` and `docs/plans/README.md` quick stats.

**Files:**
- Modify: `docs/api/models.md` (new section)
- Modify: `docs/openapi/openai-compat.yaml` (two new paths)
- Modify: `docs/plans/BACKLOG.md` (row in the Plans table)
- Modify: `docs/plans/README.md` (quick stats: Total Plans 102, Backlog 1)

**What to implement:**

1. `docs/api/models.md` — new section "LiteLLM-compatible discovery" after the existing endpoint sections:
   - `GET /v1/model/info` and `GET /model/info` (same handler), auth = `tama_` Bearer key with `Inference` scope.
   - Envelope: `{ "data": [ { "model_name", "litellm_params", "model_info" } ] }` with a concrete example entry (use the Qwen3.8 shape from Task 2's first test: `model_name: "qwen3-8b"`, `litellm_params: { model, provider: "llama.cpp" }`, `model_info` with `max_input_tokens: 262144`, `max_output_tokens: 32768`, `supports_reasoning: true`, `supports_vision: true`, `reasoning_effort_levels: ["none","low","medium","xhigh"]`, the five effort flags, costs 0).
   - A mapping table: LiteLLM field → tama source. Rows: `model_name` ← api_name/model (or alias name) · `litellm_params.model`/`model_info.id` ← same · `litellm_params.provider`/`model_info.litellm_provider` ← backend name · `max_input_tokens` ← context-length resolution chain (config override → vLLM `max_model_len` → live backend → HF metadata → model TOML) · `max_output_tokens` ← 1/8 of context, clamped 16K–32K (same heuristic as the opencode `limit.output`) · `max_tokens` ← `max_output_tokens` else `max_input_tokens` · `supports_function_calling` ← backend `/props` · `supports_reasoning` ← `/props` reasoning OR non-empty `reasoning_levels` (ADR-0008) · `supports_vision`/`supports_audio_input`/`supports_pdf_input` ← `modalities.input` · costs ← 0 (local inference) · effort fields ← `reasoning_levels` with `off` → `none` (ADR-0009); absent when no levels configured.
   - Note the known limitation: `GET /health` is a liveness probe, not LiteLLM's `healthy_endpoints` — plugins whose discovery falls back to `/health` get zero models from tama; the `/v1/model/info` primary path is fully supported.
2. `docs/openapi/openai-compat.yaml` — add `/v1/model/info` and `/model/info` paths (GET, 200 with the `data` array schema). The file's `components.responses` contains only `NotFound`, `BadRequest`, `InternalError` — so **create** the missing `Unauthorized` (401) and `Forbidden` (403) response components, each `$ref`-ing the existing `ErrorResponse` schema (follow the shape of the existing three components), and reference them from the new paths. Also add the entry schemas (`LiteLLMModelInfoEntry`, `LiteLLMParams`, `LiteLLMModelInfo` — matching the Task 2 struct fields, with the six effort fields optional) to `components.schemas`. While touching the file, correct the stale info text: the base-URL paragraph claims "Tama does not currently enforce authentication" — API keys (`tama_` prefix, ADR-0001/0002) are supported; reword to "set `OPENAI_API_KEY` to a `tama_` API key (or any non-empty string when API keys are disabled)". Keep the file valid YAML — verify with `python3 -c "import yaml; yaml.safe_load(open('docs/openapi/openai-compat.yaml'))"`.
3. `docs/plans/BACKLOG.md` — under `## Plans` (line 21), the existing subsections are `### Benchmarks Track` and `### Audit Backlog`; there is no generic feature table. Add a new `### Feature Plans (2026-08)` subsection (2-column: Plan | Description, matching the Benchmarks Track table shape) containing the row: `[LiteLLM-Compatible Model Info](plan-196-litellm-model-info.md)` | `GET /v1/model/info` + `/model/info` serving LiteLLM's discovery shape from existing model data (plan-196).
4. `docs/plans/README.md` — quick stats: Total Plans 102, Backlog 1, Completed 99.

**Steps:**
- [ ] Write the `docs/api/models.md` section + example.
- [ ] Add the two OpenAPI paths; validate the YAML parses.
- [ ] Update `BACKLOG.md` + `README.md` stats.
- [ ] Commit with message: `docs: LiteLLM-compatible model info endpoint (plan-196)`

**Acceptance criteria:**
- [ ] `docs/api/models.md` documents both routes, the envelope, the `litellm_model_id` filter, the full mapping table, and the `/health` limitation.
- [ ] `openai-compat.yaml` parses, contains both paths + the new schemas + `Unauthorized`/`Forbidden` components, and the stale no-auth info text is corrected.
- [ ] BACKLOG `### Feature Plans (2026-08)` section exists with the row; README stats are consistent with the file on disk.

---

## Final verification (before PR)

- [ ] `make check` (fmt-check + clippy all-targets + SSR clippy + full test suite incl. `--features ssr`) — green.
- [ ] Manual smoke: start tama with a model that has `reasoning_levels` configured; `curl -s -H "Authorization: Bearer <tama_ key>" http://localhost:<port>/v1/model/info | python3 -m json.tool` — inspect the entry against the Task 2 example.
- [ ] End-to-end with the consumer: `LITELLM_URL=http://localhost:<port> LITELLM_KEY=<tama_ key> pi` with `pi-provider-litellm` installed — the model appears with the correct thinking levels in pi's thinking selector (e.g. Qwen3.8 with `off,low,medium,xhigh` → selector shows exactly those; `max`/`minimal`/`high` hidden).
- [ ] `git log --oneline` shows the 4 commits (refactor, feat mapping, feat handler, docs) in order.
