use crate::config::{Config, ModelConfig};
use crate::proxy::ProxyState;
use axum::body::Body;
use axum::extract::Request;
use std::sync::Arc;

// ── Shared helper functions ──────────────────────────────────────────────

/// Create a default ProxyState for testing.
pub fn create_test_state() -> ProxyState {
    let config = Config::default();
    ProxyState::new(config, None, crate::db::pool::test_dummy_pool())
}

/// Seed a live `ready` wire row for `model_id` on the state's tamad pool
/// (plan-193 T4: handlers read endpoints/state from the live ProcessInfo
/// rows, so tests must seed them the way the tamad stream would).
pub async fn seed_live_row(state: &ProxyState, model_id: &str, endpoint: &str) {
    use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
    let proc = crate::tamad::ProcessInfo {
        model_name: model_id.to_string(),
        provider_name: "llama_cpp".to_string(),
        pid: 1,
        alive: true,
        endpoint_url: endpoint.to_string(),
        status: "ready".to_string(),
        desired: true,
        restart_count: 0,
        max_restarts: 3,
        spec_accept_pct: None,
        spec_decoding_active: false,
        tps: None,
        prompt_tps: None,
        cache_hit_pct: None,
        last_obs_ms: None,
    };
    let stats = stats_full(1.5, vec![], vec![proc]);
    let pool = state.tamad_pool();
    pool.insert_raw_handle(
        model_id,
        Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
    )
    .await;
}

/// Create a POST request with the given body for testing forward handlers.
pub fn create_forward_post_request(body: &[u8]) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .body(Body::from(body.to_vec()))
        .unwrap()
}

/// Create a GET request for testing forward handlers.
pub fn create_forward_get_request() -> Request<Body> {
    Request::get("/v1/models").body(Body::empty()).unwrap()
}

/// Helper: set up a ProxyState with two Ready backends and model configs.
pub async fn create_state_with_two_backends(
    backend1_url: &str,
    backend2_url: &str,
) -> Arc<ProxyState> {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    // Add model configs
    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "model-a".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("api-model-a".to_string()),
                model: Some("test/model-a".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
        mc.insert(
            "model-b".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("api-model-b".to_string()),
                model: Some("test/model-b".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
        // Unloaded model (enabled but no backend loaded)
        mc.insert(
            "model-c".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("api-model-c".to_string()),
                model: Some("test/model-c".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    // (plan-193 T5c: the mirror block is gone. Two `ready` wire rows
    // below are the loaded state for the forward / list-model handlers.)

    seed_live_row(&state, "model-a", backend1_url).await;
    seed_live_row(&state, "model-b", backend2_url).await;

    Arc::new(state)
}
