use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::Router;
use tower::ServiceExt;

use super::helpers::create_state_with_model;
use crate::config::ModelConfig;
use crate::proxy::tama_handlers::models::handle_tama_cancel_load;

/// Seed a live wire row for `model_id` (plan-193 T4: `handle_cancel_load`
/// checks availability via rows, not the mirror).
async fn seed_live(state: &Arc<crate::proxy::ProxyState>, model_id: &str, status: &str) {
    use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
    let proc = crate::tamad::ProcessInfo {
        model_name: model_id.to_string(),
        provider_name: "llama_cpp".to_string(),
        pid: 1,
        alive: true,
        endpoint_url: "http://127.0.0.1:1".to_string(),
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
        model_id,
        Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
    )
    .await;
}

/// Cancel returns 200 for a Starting model with a PID.
/// The fake PID (99999) has no real process group, so kill_process_group
/// returns Ok(()) on ESRCH and is_process_group_alive returns false.
#[tokio::test]
async fn test_cancel_returns_200_for_starting_model() {
    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        ..Default::default()
    })
    .await;

    // Insert a Starting entry with a fake PID
    seed_live(&state, "test-model", "starting").await;

    let app = Router::new()
        .route(
            "/tama/v1/models/:id/cancel",
            axum::routing::post(handle_tama_cancel_load),
        )
        .with_state(state);

    let request = Request::post("/tama/v1/models/test-model/cancel")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(status, StatusCode::OK, "Expected 200 OK");
    assert_eq!(json["loaded"], false, "Expected loaded: false");
    assert_eq!(json["id"], "test-model", "Expected id: test-model");

    // plan-193 T5: the handler no longer removes a local mirror entry;
    // lifecycle truth is the tamad rows, so there is no mirror to purge.
}

/// Cancel for a Ready (already loaded) model → 200: cancel operates on
/// desired state (plan-191 Task 5), so a loaded model is cleared +
/// unloaded rather than rejected with 409.
#[tokio::test]
async fn test_cancel_ready_model_unloads() {
    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        ..Default::default()
    })
    .await;

    // Insert a Ready entry
    seed_live(&state, "test-model", "ready").await;

    let app = Router::new()
        .route(
            "/tama/v1/models/:id/cancel",
            axum::routing::post(handle_tama_cancel_load),
        )
        .with_state(state.clone());

    let request = Request::post("/tama/v1/models/test-model/cancel")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(status, StatusCode::OK, "Expected 200 OK");
    assert_eq!(json["loaded"], false, "Expected loaded: false");

    // plan-193 T5: cancel no longer purges a local mirror entry — the
    // lifecycle rows (live from the tamad) are the source of truth.
}

/// Cancel returns 404 for a non-existing model.
#[tokio::test]
async fn test_cancel_returns_404_for_non_existing_model() {
    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        ..Default::default()
    })
    .await;

    // No model entries in the models map

    let app = Router::new()
        .route(
            "/tama/v1/models/:id/cancel",
            axum::routing::post(handle_tama_cancel_load),
        )
        .with_state(state);

    let request = Request::post("/tama/v1/models/nonexistent/cancel")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "Expected 404 Not Found for non-existing model"
    );
    assert_eq!(
        json["error"]["type"], "ModelNotLoadingError",
        "Expected ModelNotLoadingError type"
    );
}

/// Cancel returns 404 for a Failed model (not in a loading state).
#[tokio::test]
async fn test_cancel_returns_404_for_failed_model() {
    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        ..Default::default()
    })
    .await;

    // Insert a Failed entry

    let app = Router::new()
        .route(
            "/tama/v1/models/:id/cancel",
            axum::routing::post(handle_tama_cancel_load),
        )
        .with_state(state);

    let request = Request::post("/tama/v1/models/test-model/cancel")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "Expected 404 Not Found for Failed model"
    );
    assert_eq!(
        json["error"]["type"], "ModelNotLoadingError",
        "Expected ModelNotLoadingError type for Failed model"
    );
}
