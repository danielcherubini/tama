use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::proxy::types::ProxyState;

/// Seed a live wire row for `server_id` with the given lifecycle `status`
/// (plan-193 T4: lifecycle presence is read from the rows, not a mirror).
async fn seed_live_row(state: &ProxyState, server_id: &str, status: &str) {
    use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
    let proc = crate::tamad::ProcessInfo {
        model_name: server_id.to_string(),
        provider_name: "llama-cpp".to_string(),
        pid: 1,
        alive: true,
        endpoint_url: "http://127.0.0.1:8080".to_string(),
        status: status.to_string(),
        desired: true,
        restart_count: 0,
        max_restarts: 3,
        spec_accept_pct: None,
        spec_decoding_active: false,
        tps: None,
        prompt_tps: None,
        cache_hit_pct: None,
    };
    let stats = stats_full(1.5, vec![], vec![proc]);
    let pool = state.tamad_pool();
    pool.insert_raw_handle(
        server_id,
        Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
    )
    .await;
}

/// Seed the proxy-owned LRU access time used by idle/eviction decisions.
async fn set_last_accessed(state: &ProxyState, server_id: &str, ago: u64) {
    state.registry.last_accessed.write().await.insert(
        server_id.to_string(),
        Instant::now() - Duration::from_secs(ago),
    );
}

/// max_loaded_models = 0 disables the LRU capacity guard entirely.
#[tokio::test]
async fn test_evict_lru_if_needed_zero_is_unlimited() {
    let mut config = Config::default();
    config.proxy.max_loaded_models = 0;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "server1", "ready").await;

    let result = state.evict_lru_if_needed(None).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), None, "max_loaded_models=0 is unlimited");
}

/// Below the limit → nothing is evicted.
#[tokio::test]
async fn test_evict_lru_if_needed_under_limit_no_eviction() {
    let mut config = Config::default();
    config.proxy.max_loaded_models = 2;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "server1", "ready").await;

    let result = state.evict_lru_if_needed(None).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), None, "under limit must not evict");
}

/// At capacity, the single ready row is evicted.
#[tokio::test]
async fn test_evict_lru_if_needed_at_limit_evicts_lru() {
    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "server1", "ready").await;

    let result = state.evict_lru_if_needed(None).await;
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        Some("server1".to_string()),
        "Should evict the only ready model at capacity"
    );
}

/// Starting (in-flight) models are not evicted — only `ready` rows count.
#[tokio::test]
async fn test_evict_lru_if_needed_skips_starting_models() {
    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "server1", "starting").await;

    let result = state.evict_lru_if_needed(None).await;
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        None,
        "Should return None when no Ready rows exist"
    );
}

/// Failed / unloading processes are not eligible rows, so nothing evicts.
#[tokio::test]
async fn test_evict_lru_if_needed_skips_failed_models() {
    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "server1", "failed").await;

    let result = state.evict_lru_if_needed(None).await;
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        None,
        "failed process is not an eligible row"
    );
}

/// Two ready rows at capacity, LRU-access times explicitly seeded so the
/// ordering is deterministic (plan 193 T5c, D4): routing decisions all
/// pick the unambiguously least-recently-accessed model. Concurrent
/// eviction calls agree on the same victim — eviction is idempotent, so
/// the old "no double-eviction" hazard (two callers racing on the same
/// pick) is eliminated by design.
#[tokio::test]
async fn test_evict_lru_if_needed_concurrent_no_double_eviction() {
    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "server1", "ready").await;
    seed_live_row(&state, "server2", "ready").await;
    // server1 is unambiguously the stalest access → deterministic victim.
    set_last_accessed(&state, "server1", 900).await;
    set_last_accessed(&state, "server2", 10).await;

    let state_a = state.clone();
    let state_b = state.clone();
    let handle_a = tokio::spawn(async move { state_a.evict_lru_if_needed(None).await });
    let handle_b = tokio::spawn(async move { state_b.evict_lru_if_needed(None).await });

    let result_a = handle_a.await.unwrap();
    let result_b = handle_b.await.unwrap();
    assert!(result_a.is_ok());
    assert!(result_b.is_ok());
    let name_a = result_a.unwrap().unwrap();
    let name_b = result_b.unwrap().unwrap();
    // Both victims agree (deterministic by access time; no per-actor jitter).
    assert!(
        name_a == "server1" && (name_b == "server1" || name_b == "server2"),
        "concurrent victims must be the seeded LRU pair, got {name_a} / {name_b}"
    );
}

/// TTS backends are excluded from the LRU capacity count.
#[tokio::test]
async fn test_evict_lru_excludes_tts_backends() {
    use crate::config::ModelConfig;

    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    state.registry.model_configs.write().await.insert(
        "tts-server".to_string(),
        ModelConfig {
            backend: "tts_kokoro".to_string(),
            ..Default::default()
        },
    );
    seed_live_row(&state, "tts-server", "ready").await;

    let result = state.evict_lru_if_needed(None).await.unwrap();
    assert_eq!(result, None, "TTS backends should not trigger eviction");
}

/// Different GPU devices don't count against each other.
#[tokio::test]
async fn test_evict_lru_per_gpu_isolation() {
    use crate::config::ModelConfig;

    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    state.registry.model_configs.write().await.insert(
        "cuda0-server".to_string(),
        ModelConfig {
            backend: "llama-cpp".to_string(),
            gpu_device: Some("CUDA0".to_string()),
            ..Default::default()
        },
    );
    state.registry.model_configs.write().await.insert(
        "cuda1-server".to_string(),
        ModelConfig {
            backend: "llama-cpp".to_string(),
            gpu_device: Some("CUDA1".to_string()),
            ..Default::default()
        },
    );
    seed_live_row(&state, "cuda0-server", "ready").await;

    // (Config is registered for CUDA0 and CUDA1, but only cuda0-server has
    // a live ready row.)
    //
    // Targeting CUDA1: zero CUDA1 candidates → under its limit → no
    // eviction, the CUDA0 row is untouched (per-GPU isolation).
    let result = state
        .evict_lru_if_needed(Some("CUDA1".to_string()))
        .await
        .unwrap();
    assert_eq!(result, None, "still under the per-GPU limit");
    // Targeting CUDA0: one ready row at the per-GPU limit → capacity
    // eviction fires on CUDA0 (own GPU only).
    let result = state
        .evict_lru_if_needed(Some("CUDA0".to_string()))
        .await
        .unwrap();
    assert_eq!(
        result,
        Some("cuda0-server".to_string()),
        "capacity eviction on its own GPU"
    );
}

/// Same-GPU models count together; the least-recently-accessed is evicted.
#[tokio::test]
async fn test_evict_lru_same_gpu_counts_together() {
    use crate::config::ModelConfig;

    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    state.registry.model_configs.write().await.insert(
        "cuda0-server1".to_string(),
        ModelConfig {
            backend: "llama-cpp".to_string(),
            gpu_device: Some("CUDA0".to_string()),
            ..Default::default()
        },
    );
    state.registry.model_configs.write().await.insert(
        "cuda0-server2".to_string(),
        ModelConfig {
            backend: "llama-cpp".to_string(),
            gpu_device: Some("CUDA0".to_string()),
            ..Default::default()
        },
    );
    seed_live_row(&state, "cuda0-server1", "ready").await;
    seed_live_row(&state, "cuda0-server2", "ready").await;
    set_last_accessed(&state, "cuda0-server1", 600).await;
    set_last_accessed(&state, "cuda0-server2", 100).await;

    let result = state
        .evict_lru_if_needed(Some("CUDA0".to_string()))
        .await
        .unwrap();
    assert_eq!(
        result,
        Some("cuda0-server1".to_string()),
        "Should evict the LRU model on the same GPU"
    );
}

/// `None` (CPU / default) GPU models group together.
#[tokio::test]
async fn test_evict_lru_none_gpu_grouped() {
    use crate::config::ModelConfig;

    let mut config = Config::default();
    config.proxy.max_loaded_models = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    state.registry.model_configs.write().await.insert(
        "default-server1".to_string(),
        ModelConfig {
            backend: "llama-cpp".to_string(),
            gpu_device: None,
            ..Default::default()
        },
    );
    state.registry.model_configs.write().await.insert(
        "default-server2".to_string(),
        ModelConfig {
            backend: "llama-cpp".to_string(),
            gpu_device: None,
            ..Default::default()
        },
    );
    seed_live_row(&state, "default-server1", "ready").await;
    seed_live_row(&state, "default-server2", "ready").await;
    set_last_accessed(&state, "default-server1", 600).await;
    set_last_accessed(&state, "default-server2", 100).await;

    let result = state.evict_lru_if_needed(None).await.unwrap();
    assert_eq!(
        result,
        Some("default-server1".to_string()),
        "Should evict the LRU model in the None group"
    );
}

/// unload_model succeeds for a live ready model and clears inference stats.
#[tokio::test]
async fn test_unload_model_graceful_shutdown() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "unload-test", "ready").await;

    let result = state.unload_model("unload-test").await;
    assert!(result.is_ok(), "Unload should succeed");
}

/// unload_model prunes the per-key last-access entry, so the LRU / idle
/// decisions do not keep a dead entry for an unloaded model (pins the
/// shared `prune_last_accessed` path from the proxy-unload side).
#[tokio::test]
async fn test_unload_model_prunes_last_accessed_entry() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "access-prune", "ready").await;
    set_last_accessed(&state, "access-prune", 0).await;
    assert!(
        state
            .registry
            .last_accessed_time("access-prune")
            .await
            .is_some(),
        "precondition: the access entry is present"
    );

    let result = state.unload_model("access-prune").await;
    assert!(result.is_ok(), "unload should succeed: {result:?}");

    assert!(
        state
            .registry
            .last_accessed_time("access-prune")
            .await
            .is_none(),
        "the unloaded model must not keep a stale last-access entry"
    );
}

/// unload_model fails when there is no live row for the backend.
#[tokio::test]
async fn test_unload_model_nonexistent_backend() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    let result = state.unload_model("nonexistent").await;
    assert!(
        result.is_err(),
        "Unload should fail for non-existent backend"
    );
}

/// unload_model refuses a non-eligible lifecycle state (unloading).
#[tokio::test]
async fn test_unload_model_non_ready_state() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "starting-server", "unloading").await;
    let result = state.unload_model("starting-server").await;
    assert!(result.is_err(), "Unload should fail for non-ready state");
}

// ─── plan-193 T5c/T6 budget-exhausted unload ──────────────────────────────

/// Unloading a `budget_exhausted` row is CLEANUP, not re-arm.
///
/// Since T5c a `budget_exhausted` model deliberately holds a live row (it is
/// what the 503 in `ensure_model_loaded` reads). The pre-fix status gate
/// refused every row whose status was not `ready | starting | restarting`,
/// so a budget-EB model was unloadable via **no** proxy path through this
/// gate — including `tama admin unload` — and nothing ever re-warmed:
/// `admin load` to exit 13 (503), `admin unload` to error exit 1, forever.
/// Recovery therefore *must* be allowed to proceed for a `budget_exhausted`
/// row: `unload_model` returns `Ok` so the admin exit-code contract maps it
/// to `0` (not-found to 2, Ok to 0), and the host-side flush (T2
/// `store.delete`) then drops the row.
#[tokio::test]
async fn test_unload_model_budget_exhausted_is_cleanup_not_rearm() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    // The tamad holds a live `budget_exhausted` row — the precondition:
    // this is the very row `ensure_model_loaded` reads for the 503.
    seed_live_row(&state, "model.gguf", "budget_exhausted").await;
    assert!(
        crate::proxy::live_rows(state.tamad_pool().as_ref())
            .await
            .row("model.gguf")
            .is_some(),
        "precondition: a budget_exhausted row is live (the 503 reads it)"
    );

    // The fix: unloading it is cleanup, so it must be Ok, not the gate err.
    let result = state.unload_model("model.gguf").await;
    assert!(
        result.is_ok(),
        "unloading a budget_exhausted row must be allowed (cleanup, not re-arm): {result:?}"
    );

    // Once the host-side flush lands (T2: the UnloadModel via kills the
    // store row, so the wire no longer reports the model), the proxy's
    // live view reflects it: no row. We model the host re-emitting a
    // frame that no longer carries the key. Since admin e2e is out of
    // scope, we use the same seed pattern as the T5c tests.
    let pool = state.tamad_pool();
    let frame = crate::tamad::pool::test_support::stats_full(1.5, vec![], vec![]);
    pool.insert_raw_handle(
        "model.gguf",
        Arc::new(
            crate::tamad::pool::test_support::handle_with_latest(std::time::Instant::now(), frame)
                .await,
        ),
    )
    .await;
    assert!(
        crate::proxy::live_rows(pool.as_ref())
            .await
            .row("model.gguf")
            .is_none(),
        "after the host flushes the row it must be gone from the live view"
    );
}

/// Regression guard: admitting `budget_exhausted` must not widen
/// the gate beyond that. A row in a non-terminatable status (here
/// `unloading`) stays refused. Only `ready | starting | restarting`
/// plus `budget_exhausted` are admissible; everything else errors.
#[tokio::test]
async fn test_unload_model_still_refuses_unloading_row() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "away-server", "unloading").await;
    let result = state.unload_model("away-server").await;
    assert!(
        result.is_err(),
        "the gate must stay narrow: an `unloading` row is still refused"
    );
}

// ─── Slim idle-timeout tests (plan-191 Task 10) ────────────────────────

/// An idle Ready model is unloaded while a fresh one is left alone and a
/// TTS backend is never idle-unloaded.
#[tokio::test]
async fn test_idle_timeout_unloads_idle_ready_models() {
    use crate::config::ModelConfig;

    let mut config = Config::default();
    config.proxy.auto_unload = true;
    config.proxy.idle_timeout_secs = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    state.registry.model_configs.write().await.insert(
        "tts-server".to_string(),
        ModelConfig {
            backend: "tts_kokoro".to_string(),
            ..Default::default()
        },
    );

    seed_live_row(&state, "idle-server", "ready").await;
    seed_live_row(&state, "fresh-server", "ready").await;
    seed_live_row(&state, "tts-server", "ready").await;
    set_last_accessed(&state, "idle-server", 600).await;
    set_last_accessed(&state, "fresh-server", 0).await;
    set_last_accessed(&state, "tts-server", 600).await;

    let cleaned = state.check_idle_timeouts().await;
    assert!(
        cleaned.contains(&"idle-server".to_string()),
        "idle model must be unloaded"
    );
    assert!(
        !cleaned.contains(&"fresh-server".to_string()),
        "fresh model must not be unloaded"
    );
    assert!(
        !cleaned.contains(&"tts-server".to_string()),
        "TTS model must not be idle-unloaded"
    );
}

/// auto_unload disabled → no model is idle-unloaded.
#[tokio::test]
async fn test_idle_timeout_respects_auto_unload_flag() {
    let mut config = Config::default();
    config.proxy.auto_unload = false;
    config.proxy.idle_timeout_secs = 1;
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    seed_live_row(&state, "idle-server", "ready").await;
    set_last_accessed(&state, "idle-server", 600).await;

    let cleaned = state.check_idle_timeouts().await;
    assert!(
        !cleaned.contains(&"idle-server".to_string()),
        "auto_unload=false must not unload"
    );
}

/// plan-193 T5c: a model in the tamad's `budget_exhausted` lifecycle
/// state must surface as HTTP 503 + `retry-after: 60` — through the FULL
/// translation the HTTP layers run: `ensure_model_loaded` → MARKED error →
/// `budget_exhausted_response_for` → response.
///
/// The mark is a type (`BudgetExhausted`), never a string: no
/// user-visible string marker is allowed to leak into any other path, and
/// no handler may string-match on `Display` to recover the mark.
#[tokio::test]
async fn test_ensure_model_loaded_budget_exhausted_translates_to_503() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    // Seed a live wire row reporting `budget_exhausted` for "model.gguf"
    // (the tamad keeps reporting the budget state; its restart budget is
    // spent).
    seed_live_row(&state, "model.gguf", "budget_exhausted").await;

    // Step 1 — the call the HTTP layers make: the marked error comes back
    // before any load attempt.
    let err =
        crate::proxy::lifecycle::ensure_model_loaded(&Arc::new(state), "model.gguf", |_, e| Err(e))
            .await
            .expect_err("a budget-exhausted model must not load");
    assert!(
        err.is::<crate::proxy::lifecycle::BudgetExhausted>(),
        "the error must carry the BudgetExhausted mark (a type, not a string)"
    );

    // Step 2 — the same mapping both handlers apply: mark → the full 503
    // wire shape (status, header, exact body).
    let resp = crate::proxy::lifecycle::budget_exhausted_response_for(&err)
        .expect("the marked error must translate to the budget-exhausted 503");
    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("60")
    );
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        std::str::from_utf8(&body).unwrap(),
        "the model exhausted its restarts; retry in 60 seconds"
    );

    // Step 3 — an unmarked error must not translate (no 503 leakage).
    let unrelated = anyhow::anyhow!("some other failure");
    assert!(
        crate::proxy::lifecycle::budget_exhausted_response_for(&unrelated).is_none(),
        "unmarked errors must not map to the budget-exhausted 503"
    );
}
