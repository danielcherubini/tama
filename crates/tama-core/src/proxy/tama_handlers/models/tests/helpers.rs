use std::sync::Arc;

use crate::config::{Config, ModelConfig};
use crate::proxy::tama_handlers::models::{handle_litellm_model_info, handle_opencode_list_models};
use crate::proxy::ProxyState;
use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::Router;
use tower::ServiceExt;

/// Helper: create a ProxyState with a single model config.
pub async fn create_state_with_model(model_cfg: ModelConfig) -> Arc<ProxyState> {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
    let mut mc = state.registry.model_configs.write().await;
    mc.insert("test-model".to_string(), model_cfg);
    drop(mc);
    Arc::new(state)
}

/// Helper: build the router and call handle_opencode_list_models.
pub async fn call_list_models(state: Arc<ProxyState>) -> serde_json::Value {
    let app = Router::new()
        .route(
            "/v1/opencode/models",
            axum::routing::get(handle_opencode_list_models),
        )
        .with_state(state);

    let request = Request::get("/v1/opencode/models")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// Helper: build the router serving the LiteLLM handler at both discovery
/// paths and call it with the given URI (e.g. `"/v1/model/info"` or
/// `"/model/info?litellm_model_id=test-model"`). Returns the status and the
/// raw body bytes (so parity tests can compare byte-identical bodies).
pub async fn call_litellm_model_info(state: Arc<ProxyState>, uri: &str) -> (StatusCode, Vec<u8>) {
    let app = Router::new()
        .route(
            "/v1/model/info",
            axum::routing::get(handle_litellm_model_info),
        )
        .route("/model/info", axum::routing::get(handle_litellm_model_info))
        .with_state(state);

    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, body.to_vec())
}
