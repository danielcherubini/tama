use super::*;

use crate::proxy::types::LatestInferenceStats;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::http::request::Parts;
use axum::http::StatusCode;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::proxy::ProxyState;

// ── Helpers ─────────────────────────────────────────────────────────────

/// POST variant of the parent's GET-only `make_parts` helper.
fn make_post_parts(path: &str) -> Parts {
    let req = axum::http::Request::post(path).body(()).unwrap();
    let (parts, _) = req.into_parts();
    parts
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn test_state() -> Arc<ProxyState> {
    Arc::new(ProxyState::new(
        crate::config::Config::default(),
        None,
        crate::db::pool::test_dummy_pool(),
    ))
}

/// Register a live `ready` wire row for `model_id` on the state's tamad pool
/// (plan-193 T4: `forward_request` reads the endpoint from rows, so tests
/// must seed the live ProcessInfo the same way the tamad stream would). The
/// mirror entry (circuit breaker, failure bookkeeping) is inserted separately.
async fn seed_live_row(state: &Arc<ProxyState>, model_id: &str, endpoint: &str) {
    use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
    let proc = crate::tamad::ProcessInfo {
        model_name: model_id.to_string(),
        provider_name: "llama.cpp".to_string(),
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
    };
    let stats = stats_full(1.5, vec![], vec![proc]);
    let pool = state.tamad_pool();
    pool.insert_raw_handle(
        "t1",
        Arc::new(handle_with_latest(Instant::now(), stats).await),
    )
    .await;
}

// ── Tests ───────────────────────────────────────────────────────────────

/// Crashed backend (nothing listening) surfaces as a connection error at
/// forward time → 502 + `BadGatewayError`, model removal, inference_stats
/// cleanup. (Plan-191 Task 10: the proxy never pings local pids — liveness
/// converged via connection errors + the reconciler mirror.)
#[tokio::test]
async fn test_forward_request_conn_error_returns_502_and_cleans_up() {
    let state = test_state();

    // Insert a model whose backend URL is not listening (simulates a crash;
    // the recorded pid is irrelevant — the proxy no longer inspects pids).
    seed_live_row(&state, "test-model", "http://127.0.0.1:1").await;

    // Pre-seed inference_stats so we can assert it's cleared.
    state
        .metrics
        .record_inference_stats("test-model", LatestInferenceStats::default());

    let resp = forward_request(
        &state,
        "test-model",
        &make_post_parts("/v1/chat/completions"),
        b"{}",
        Some("test-model"),
    )
    .await;

    let status = resp.status();
    let body = body_json(resp).await;

    // Status + error type + message
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["type"], "BadGatewayError");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Backend error"),);

    // (plan-193 T5c: there is no local model map left to clear; the
    // wire row is the tamad's fact, and the tamad's own lifecycle
    // reaps its dead process.)

    // Inference stats entry cleared
    assert!(!state
        .metrics
        .inference_stats_snapshot()
        .contains_key("test-model"));
}

/// Connection-error path counts exactly one `failed_request` and one
/// `total_request` (no cooldown counter is involved — the model state is
/// removed immediately instead of accumulating failures).
#[tokio::test]
async fn test_forward_request_conn_error_counts_failed_and_removes_model() {
    let state = test_state();

    // Seed the wire row with a backend that is not listening.
    seed_live_row(&state, "test-model", "http://127.0.0.1:1").await;

    let _resp = forward_request(
        &state,
        "test-model",
        &make_post_parts("/v1/chat/completions"),
        b"{}",
        Some("test-model"),
    )
    .await;

    // Connection-error path: total=1, successful=0, failed=1.
    assert_eq!(
        state
            .metrics
            .counters
            .total_requests
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        state
            .metrics
            .counters
            .successful_requests
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        state
            .metrics
            .counters
            .failed_requests
            .load(Ordering::Relaxed),
        1
    );
}

/// Model not loaded (no entry in `state.models()`) → 502 + `BackendUrlError`.
#[tokio::test]
async fn test_forward_request_model_not_loaded_returns_502() {
    let state = test_state();
    // Empty state — no models inserted.

    let resp = forward_request(
        &state,
        "nonexistent-model",
        &make_post_parts("/v1/chat/completions"),
        b"{}",
        Some("nonexistent-model"),
    )
    .await;

    let status = resp.status();
    let body = body_json(resp).await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["type"], "BackendUrlError");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("is not loaded"));
}

// ── Metrics and Proxied-Response Tests (wiremock backend) ───────────────

/// Success path: mock returns 200 → successful/total request metrics
/// updated, model name rewritten in the response body.
#[tokio::test]
async fn test_forward_request_success_increments_metrics_and_rewrites_model() {
    let server = MockServer::start().await;

    // Mock the backend to return a valid chat completion response.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-1",
            "model": "backend-model",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        })))
        .mount(&server)
        .await;

    let state = test_state();

    // Insert a model with pre-existing failure count of 2 — success path should reset to 0.
    seed_live_row(&state, "test-model", server.uri().as_str()).await;

    let resp = forward_request(
        &state,
        "test-model",
        &make_post_parts("/v1/chat/completions"),
        br#"{"model":"test-model","messages":[]}"#,
        Some("public-name"),
    )
    .await;

    let status = resp.status();
    let json = body_json(resp).await;

    // Status 200 — success proxied through.
    assert_eq!(status, StatusCode::OK);

    // Model name rewritten: backend-model → public-name.
    assert_eq!(json["model"].as_str().unwrap(), "public-name");

    // Response content preserved.
    assert_eq!(
        json["choices"][0]["message"]["content"].as_str().unwrap(),
        "hi"
    );

    // Metrics: total_requests == 1 (fresh state), successful_requests == 1,
    // failed_requests == 0.
    assert_eq!(
        state
            .metrics
            .counters
            .total_requests
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        state
            .metrics
            .counters
            .successful_requests
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        state
            .metrics
            .counters
            .failed_requests
            .load(Ordering::Relaxed),
        0
    );
}

/// Alias rewrite: the request body's `model` field is replaced with the resolved
/// name before forwarding, and the backend receives the complete, valid JSON body.
///
/// Regression test for the content-length truncation bug: the rewritten body is a
/// different size than the original, so a stale forwarded content-length header
/// would truncate the body mid-string ("Unterminated string" errors on the backend).
#[tokio::test]
async fn test_forward_request_rewrites_model_in_request_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-1",
            "model": "resolved-model",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        })))
        .mount(&server)
        .await;

    let state = test_state();

    seed_live_row(&state, "resolved-model", server.uri().as_str()).await;

    // Body padded large enough that a stale (smaller) content-length header
    // would visibly truncate it. "org/alias" is shorter than "resolved-model"
    // so the rewritten body grows.
    let long_content = "x".repeat(10_000);
    let original_body = serde_json::json!({
        "model": "org/alias",
        "messages": [{"role": "user", "content": long_content}]
    });
    let body_bytes = serde_json::to_vec(&original_body).unwrap();

    let resp = forward_request(
        &state,
        "resolved-model",
        &make_post_parts("/v1/chat/completions"),
        &body_bytes,
        Some("resolved-model"),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::OK);

    // The backend must have received exactly one request whose body is complete,
    // valid JSON with the model field rewritten to the resolved name.
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let sent_body: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("forwarded body must be valid JSON");
    assert_eq!(sent_body["model"].as_str().unwrap(), "resolved-model");
    assert_eq!(
        sent_body["messages"][0]["content"].as_str().unwrap(),
        long_content
    );
}
