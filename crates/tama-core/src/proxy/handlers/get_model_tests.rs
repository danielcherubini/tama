use super::models::handle_get_model;
use crate::config::{Config, ModelConfig};
use crate::proxy::ProxyState;
use axum::{
    body::to_bytes,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde_json::Value as JsonValue;
use std::sync::Arc;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use super::tests::*;

/// Seed a live `ready` wire row for `model_id` on the state's tamad pool so
/// `handle_get_model` (plan-193 T4: rows, not the mirror) sees the backend
/// as loaded and pulls its `/v1/models` from the live endpoint.
async fn seed_live_proxy(state: &ProxyState, model_id: &str, endpoint: &str) {
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
        "t1",
        Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
    )
    .await;
}

// ── handle_get_model: basic config lookup tests ──────────────────────────

#[tokio::test]
async fn test_handle_get_model_by_config_key_returns_api_name() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    // Populate model_configs
    {
        let mut mc = state_arc.registry.model_configs.write().await;
        mc.insert(
            "config-key-1".to_string(),
            ModelConfig {
                backend: "llama.cpp".to_string(),
                api_name: Some("api-name-1".to_string()),
                model: Some("test/model-1".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    let state = State(state_arc);

    let response = handle_get_model(state, Path("config-key-1".to_string())).await;
    let status = response.status();
    assert_eq!(status, StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json.get("id").unwrap().as_str(), Some("api-name-1"));
}

#[tokio::test]
async fn test_handle_get_model_by_api_name_returns_api_name() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    // Populate model_configs
    {
        let mut mc = state_arc.registry.model_configs.write().await;
        mc.insert(
            "config-key-1".to_string(),
            ModelConfig {
                backend: "llama.cpp".to_string(),
                api_name: Some("api-name-1".to_string()),
                model: Some("test/model-1".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    let state = State(state_arc);

    let response = handle_get_model(state, Path("api-name-1".to_string())).await;
    let status = response.status();
    assert_eq!(status, StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json.get("id").unwrap().as_str(), Some("api-name-1"));
}

#[tokio::test]
async fn test_handle_get_model_without_api_name_falls_back_to_config_key() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    // Populate model_configs
    {
        let mut mc = state_arc.registry.model_configs.write().await;
        mc.insert(
            "config-key-2".to_string(),
            ModelConfig {
                backend: "llama.cpp".to_string(),
                api_name: None,
                model: Some("test/model-2".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    let state = State(state_arc);

    let response = handle_get_model(state, Path("config-key-2".to_string())).await;
    let status = response.status();
    assert_eq!(status, StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json.get("id").unwrap().as_str(), Some("config-key-2"));
}

// ── handle_get_model: backend fetch tests ──────────────────────────────

/// Test that handle_get_model fetches from backend when model is loaded,
/// preserves `meta` data, and injects `ready: true`.
#[tokio::test]
async fn test_handle_get_model_fetches_from_backend_with_meta() {
    let mock_server = MockServer::start().await;

    // Mock backend returns model with meta
    let backend_response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "llama3.gguf",
                "object": "model",
                "created": 1700000000,
                "owned_by": "backend1",
                "meta": {
                    "general_name": "Llama 3",
                    "architecture": "llama"
                }
            }
        ]
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    // Add model config
    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "test-model".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-api-model".to_string()),
                model: Some("llama3.gguf".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    seed_live_proxy(&state, "test-model", mock_server.uri().as_str()).await;

    let state_arc = Arc::new(state);
    let state = State(state_arc.clone());

    // Query by config key
    let response = handle_get_model(state.clone(), Path("test-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    // Should have meta from backend
    assert!(
        json.get("meta").is_some(),
        "meta should be preserved from backend response"
    );
    assert_eq!(
        json["meta"]["general_name"], "Llama 3",
        "meta.general_name should match backend response"
    );
    // ready should be injected as true
    assert_eq!(json["ready"], true, "Loaded model should have ready: true");
}

/// Test that handle_get_model falls back to config when model is not loaded.
/// Response should have no `meta` and `ready: false`.
#[tokio::test]
async fn test_handle_get_model_fallback_to_config_when_not_loaded() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    // Add model config but do NOT add it to loaded models
    {
        let mut mc = state_arc.registry.model_configs.write().await;
        mc.insert(
            "unloaded-model".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-unloaded-model".to_string()),
                model: Some("test/unloaded".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    let state = State(state_arc.clone());

    // Query by config key
    let response = handle_get_model(state.clone(), Path("unloaded-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    // Should use api_name as id
    assert_eq!(json["id"], "my-unloaded-model");
    // Should NOT have meta
    assert!(
        json.get("meta").is_none(),
        "Unloaded model should not have meta"
    );
    // ready should be false
    assert_eq!(
        json["ready"], false,
        "Unloaded model should have ready: false"
    );
}

/// Test that handle_get_model returns 404 for unknown model IDs.
#[tokio::test]
async fn test_handle_get_model_404_for_unknown_model() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    let state = State(state_arc.clone());

    // Query with a model_id that doesn't exist in config
    let response = handle_get_model(state.clone(), Path("totally-unknown-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["error"]["type"], "NotFoundError");
}

/// Test that handle_get_model works when backend returns multiple models
/// and matches by config's model field (file path).
#[tokio::test]
async fn test_handle_get_model_matches_by_model_field_when_multiple() {
    let mock_server = MockServer::start().await;

    // Backend returns multiple models
    let backend_response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "/path/to/model-a.gguf",
                "object": "model",
                "created": 1700000000,
                "owned_by": "backend1"
            },
            {
                "id": "/path/to/model-b.gguf",
                "object": "model",
                "created": 1700000001,
                "owned_by": "backend1"
            }
        ]
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    // Config's model field matches model-b
    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "my-model".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-api-name".to_string()),
                model: Some("/path/to/model-b.gguf".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    seed_live_proxy(&state, "my-model", mock_server.uri().as_str()).await;

    let state_arc = Arc::new(state);
    let state = State(state_arc.clone());

    let response = handle_get_model(state.clone(), Path("my-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    // Should match model-b by config's model field, and normalize id to api_name
    assert_eq!(json["id"], "my-api-name", "Should normalize id to api_name");
    assert_eq!(json["ready"], true);
}

/// Test that handle_get_model falls back to config when backend query fails.
#[tokio::test]
async fn test_handle_get_model_backend_failure_fallback() {
    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    // Add model config
    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "fail-model".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("fail-api".to_string()),
                model: Some("test/fail".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    seed_live_proxy(&state, "fail-model", "http://localhost:59999").await;

    let state_arc = Arc::new(state);
    let state = State(state_arc.clone());

    let response = handle_get_model(state.clone(), Path("fail-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    // Should fall back to config-based response
    assert_eq!(json["id"], "fail-api");
    assert!(json.get("meta").is_none());
    assert_eq!(json["ready"], false);
}

/// Test handle_get_model normalizes id when backend entry is found via alias.
#[tokio::test]
async fn test_handle_get_model_normalizes_id_from_alias() {
    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "gemma-4-E2B-it-UD-IQ3_XXS.gguf",
                "object": "model",
                "created": 1779728594,
                "owned_by": "llamacpp",
                "aliases": ["unsloth/gemma-4-E2B-it-GGUF"],
                "meta": {"n_ctx": 32768}
            }
        ]
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .mount(&mock_server)
        .await;

    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "gemma-e2b".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("unsloth/gemma-4-E2B-it-GGUF".to_string()),
                model: Some("unsloth/gemma-4-E2B-it-GGUF".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    seed_live_proxy(&state, "gemma-e2b", mock_server.uri().as_str()).await;

    let state_arc = Arc::new(state);
    let state = State(state_arc.clone());

    // Look up by config key
    let response = handle_get_model(state.clone(), Path("gemma-e2b".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    // ID should be normalized to api_name
    assert_eq!(
        json["id"], "unsloth/gemma-4-E2B-it-GGUF",
        "ID should be normalized to api_name"
    );
    // Meta should be preserved
    assert!(json.get("meta").is_some(), "meta should be preserved");
    assert_eq!(json["ready"], true);

    // Also look up by api_name
    let response = handle_get_model(
        state.clone(),
        Path("unsloth/gemma-4-E2B-it-GGUF".to_string()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["id"], "unsloth/gemma-4-E2B-it-GGUF");
    assert_eq!(json["ready"], true);
}

// ── handle_get_model: reasoning-effort fields (plan 189, review fix) ──────

/// Loaded (Ready backend) model with reasoning levels: the response carries
/// supportsReasoningEffort, reasoningLevels and derived reasoning_options
/// (off → none), matching the list handler's shape.
#[tokio::test]
async fn test_handle_get_model_loaded_with_reasoning_levels() {
    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "test/leveled.gguf",
                "object": "model",
                "created": 1700000000,
                "owned_by": "backend1",
                "meta": {"n_ctx": 8192}
            }
        ]
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "leveled-model".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-leveled-model".to_string()),
                model: Some("test/leveled.gguf".to_string()),
                enabled: true,
                reasoning_levels: Some(vec![
                    "off".to_string(),
                    "low".to_string(),
                    "medium".to_string(),
                    "xhigh".to_string(),
                ]),
                ..Default::default()
            },
        );
    }

    seed_live_proxy(&state, "leveled-model", mock_server.uri().as_str()).await;

    let state_arc = Arc::new(state);
    let state = State(state_arc.clone());

    let response = handle_get_model(state.clone(), Path("leveled-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["id"], "my-leveled-model");
    assert_eq!(json["ready"], true);
    assert_eq!(json["supportsReasoningEffort"], true);
    assert_eq!(
        json["reasoningLevels"],
        serde_json::json!(["off", "low", "medium", "xhigh"])
    );
    assert_eq!(
        json["reasoning_options"],
        serde_json::json!([
            { "type": "effort", "values": ["none", "low", "medium", "xhigh"] }
        ])
    );
}

/// Loaded model without reasoning levels: response stays byte-identical to
/// the pre-change shape (no reasoning-effort keys).
#[tokio::test]
async fn test_handle_get_model_loaded_without_reasoning_levels_unchanged() {
    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "plain.gguf",
                "object": "model",
                "created": 1700000000,
                "owned_by": "backend1"
            }
        ]
    });
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = Config::default();
    let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "plain-model".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-plain-model".to_string()),
                model: Some("plain.gguf".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    seed_live_proxy(&state, "plain-model", mock_server.uri().as_str()).await;

    let state_arc = Arc::new(state);
    let state = State(state_arc.clone());

    let response = handle_get_model(state.clone(), Path("plain-model".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(
        json,
        serde_json::json!({
            "id": "my-plain-model",
            "object": "model",
            "created": 1700000000,
            "owned_by": "backend1",
            "ready": true
        }),
        "loaded entry without levels must be structurally identical to the pre-change shape"
    );
}

/// Fallback (unloaded) model with reasoning levels: the config-based entry
/// carries supportsReasoningEffort, reasoningLevels and reasoning_options.
#[tokio::test]
async fn test_handle_get_model_fallback_with_reasoning_levels() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    {
        let mut mc = state_arc.registry.model_configs.write().await;
        mc.insert(
            "leveled-unloaded".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-leveled-unloaded".to_string()),
                model: Some("test/leveled-unloaded".to_string()),
                enabled: true,
                reasoning_levels: Some(vec!["off".to_string(), "high".to_string()]),
                ..Default::default()
            },
        );
    }

    let state = State(state_arc.clone());

    let response = handle_get_model(state.clone(), Path("leveled-unloaded".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["id"], "my-leveled-unloaded");
    assert_eq!(json["ready"], false);
    assert_eq!(json["supportsReasoningEffort"], true);
    assert_eq!(json["reasoningLevels"], serde_json::json!(["off", "high"]));
    assert_eq!(
        json["reasoning_options"],
        serde_json::json!([
            { "type": "effort", "values": ["none", "high"] }
        ])
    );
}

/// Fallback (unloaded) model without reasoning levels: response stays
/// byte-identical to the pre-change shape (no reasoning-effort keys).
#[tokio::test]
async fn test_handle_get_model_fallback_without_reasoning_levels_unchanged() {
    let state_inner = create_test_state();
    let state_arc = Arc::new(state_inner);

    {
        let mut mc = state_arc.registry.model_configs.write().await;
        mc.insert(
            "plain-unloaded".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                api_name: Some("my-plain-unloaded".to_string()),
                model: Some("test/plain-unloaded".to_string()),
                enabled: true,
                ..Default::default()
            },
        );
    }

    let state = State(state_arc.clone());

    let response = handle_get_model(state.clone(), Path("plain-unloaded".to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_parts, body) = response.into_response().into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    let json: JsonValue = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(
        json,
        serde_json::json!({
            "id": "my-plain-unloaded",
            "object": "model",
            "created": 0,
            "owned_by": "llama_cpp",
            "ready": false
        }),
        "fallback entry without levels must be structurally identical to the pre-change shape"
    );
}
