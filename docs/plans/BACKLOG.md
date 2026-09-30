# Backlog Plans

From the [2026-07-18 codebase audit](../reviews/2026-07-18-codebase-improvement.md) — 40 findings across 8 lenses.

## Execution Order

See [execution-order.md](execution-order.md) for the full dependency-first cascade with phases, coordination flags, and rationale.

### Quick Reference

| Phase | Plans | Description |
|-------|-------|-------------|
| 1 | plan-161, 162, 163 | Error Contract, Model Identity, Backup Restore |
| 3 (parallel) | plan-164, 165, 166 | OAuth tests, forward_request tests, pull handler tests |
| 4 (parallel) | plan-167, 168, 169 | Cleanup, API boilerplate, router consolidation |
| 5 (parallel) | plan-170, 171 | Newtypes, DB query from_row pattern |
| 6 (parallel) | plan-172, 173, 174 | File splits, naming domain terms, typed responses |
| 7 (parallel) | plan-175, 176, 177 | Server SSE, Leptos UI consolidation, ProxyState substructs |
| 8 | plan-178 | Remaining test coverage wave two |

## Plans

### Benchmarks Track (2026-07 — independent of audit backlog)

| Plan | Description |
|------|-------------|
| [Benchmark Bug Fixes](plan-180-benchmark-bug-fixes.md) | history.rs spec/MTP conversion arms, gpu_variant dropdown + DTO (ADR-0005), history refresh, draft_ngl, derived run status |
| [Model Batch/µ-batch Fields](plan-181-model-batch-ubatch-fields.md) | Typed `n_batch`/`n_ubatch` model fields replacing free-text args flags (migration `_0041`) |
| [Benchmarks Frontend Refactor](plan-182-benchmarks-frontend-refactor.md) | ~500-line dedup: shared selectors/submit, LlamaBenchForm extraction, hoisted tab state |
| [Benchmark Suite](plan-183-benchmark-suite.md) | One-button capability-aware suite (ADR-0004), `suite_id` grouping (migration `_0042`) |

Execute in order 180 → 181 → 182 → 183 (180 and 181 are independent of each other).

### Audit Backlog (2026-07-18)

| Plan | Description | Findings |
|------|-------------|----------|
| [Error Contract Unification](plan-161-error-contract.md) | Migrate ~55 flat error sites to structured shape, shared tama-core error helper, fix OpenAPI ErrorResponse schema, shape-assertion tests | F4 |
| [Model Identity ConfigKey](plan-162-model-identity-configkey.md) | ConfigKey newtype as single derivation site, replace 15 open-coded sites, rename dual resolve_model_id fns | F5 |
| [Backup Restore](plan-163-backup-restore.md) | Implement restore for real (extract→validate→merge with job events), route create_backup, fix docs/api/backup.md | F6 |
| [OAuth2 Flow Tests](plan-164-oauth2-flow-tests.md) | ~16 tests for login/callback/logout handlers — state CSRF, wiremock token/userinfo, session cookie | F8 |
| [forward_request Tests](plan-165-forward-request-tests.md) | Dead-PID 502 + cleanup, circuit-breaker 503/trip+unload, metrics, wiremock success path | F9 |
| [Pull Handler Tests](plan-166-pull-handler-tests.md) | Validation, enqueue, job GET/SSE, download+verify orchestration (hash mismatch → failed) via wiremock HF | F10 |
| [File Splits Wave 2](plan-172-file-splits-wave2.md) | Split pull_queue.rs (1756), api/updates.rs (1022), auth.rs (1526) into module dirs, API preserved | F11 |
| [Server SSE Consolidation](plan-175-server-sse-consolidation.md) | serde-tagged PullEvent/UpdateEvent + to_sse_event, shared job_event_stream/broadcast_to_sse, uniform KeepAlive | F12 |
| [API Handler Boilerplate](plan-168-api-handler-boilerplate.md) | Wire submit_benchmark_job, resolve_model_record, resolve_config_dir/open_repository, spawn_blocking discipline, small dedup batch | F13–F15, F20, F37 |
| [Newtypes](plan-170-newtypes.md) | GpuType FromStr/serde adoption, CompactionDevice 422, HfEndpoints + hf_auth_headers, is_valid_repo_id | F16–F18 |
| [Test Coverage Wave 2](plan-178-test-coverage-wave2.md) | Compaction/TTS via lifecycle traits, update-check orchestration, bench/pull engines, un-ignore backends_api tests, tama-mock integration | F22–F24, F36 |
| [Cleanup](plan-167-cleanup.md) | Delete ~1900 lines dead code (config/migrate, Leptos components, jobs.rs), unused deps, blanket allows, rename_legacy, println→tracing, style batch | F26, F34, F38, F39 |
| [Naming Domain Terms](plan-173-naming-domain-terms.md) | GpuType→GpuVariant, server→backend leftovers, download→pull public surface (breaking, migrations), ModelCard→ModelToml, Loading→Starting, [supervisor]→[lifecycle] | F27, F28, F40 |
| [Typed API Responses](plan-174-typed-api-responses.md) | StatusResponse/ModelEntry/OkResponse structs with golden shape tests, drift guards, scoped OpenAPI updates | F19 |
| [DB Query from_row](plan-171-db-query-from-row.md) | Per-record from_row + COLUMNS const for 6 record types, column-drift guard test | F30 |
| [Leptos UI Consolidation](plan-176-leptos-ui-consolidation.md) | Shared wasm-safe types via #[path] inclusion, collapse mirror types, DOM/request helpers + patch_request, benchmark form-state hook | F29, F31 |
| [ProxyState Sub-structs](plan-177-proxystate-substructs.md) | RegistryState/MetricsState/PullState composition, domain methods over lock-guard accessors, shim migration | F32 |
| [Router Consolidation](plan-169-router-consolidation.md) | Single-source route table (31 routes), process helpers to crate::process, cross-crate ownership test, fix shadowed /system/health | F33 |
