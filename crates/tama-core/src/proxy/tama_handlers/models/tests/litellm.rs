//! Unit tests for the LiteLLM model-info mapping (`litellm.rs`) and the
//! `handle_litellm_model_info` handler (plan-196 Task 3).
//!
//! The mapping tests construct `ModelEntry` values directly — no state, no
//! network. The handler tests use the `create_state_with_model` fixture.

use crate::config::ModelModalities;
use crate::proxy::tama_handlers::models::litellm::{
    handle_litellm_model_info, litellm_entry_from_model_entry, LiteLLMModelInfoEntry,
};
use crate::proxy::tama_handlers::models::{ModelEntry, ModelLimit};
use axum::body::Body;
use axum::extract::Request;
use axum::Router;
use tower::ServiceExt;

/// Baseline entry with every optional field empty; tests override what they need.
fn base_entry() -> ModelEntry {
    ModelEntry {
        id: None,
        name: "Test: Model".to_string(),
        model: None,
        backend: "llama.cpp".to_string(),
        context_length: None,
        limit: ModelLimit {
            context: None,
            output: None,
        },
        quant: None,
        gpu_layers: None,
        modalities: None,
        tool_call: true,
        reasoning: false,
        attachment: false,
        temperature: true,
        supports_reasoning_effort: false,
        reasoning_levels: None,
        reasoning_options: None,
    }
}

/// Qwen3.8 (the research doc's first target): props says no reasoning, but
/// user-configured levels make effective reasoning true.
#[test]
fn test_qwen38_entry_maps_to_litellm_wire_shape() {
    let entry = base_entry();
    let entry = ModelEntry {
        id: Some("qwen3-8b".to_string()),
        backend: "llama.cpp".to_string(),
        limit: ModelLimit {
            context: Some(262144),
            output: Some(32768),
        },
        modalities: Some(ModelModalities {
            input: vec!["text".to_string(), "image".to_string()],
            output: vec!["text".to_string()],
        }),
        tool_call: true,
        reasoning: false,
        supports_reasoning_effort: true,
        reasoning_levels: Some(vec![
            "off".to_string(),
            "low".to_string(),
            "medium".to_string(),
            "xhigh".to_string(),
        ]),
        ..entry
    };

    let mapped = litellm_entry_from_model_entry(&entry);

    assert_eq!(mapped.model_name, "qwen3-8b");
    assert_eq!(mapped.litellm_params.model, "qwen3-8b");
    assert_eq!(mapped.litellm_params.provider, "llama.cpp");

    let info = &mapped.model_info;
    assert_eq!(info.id, "qwen3-8b");
    assert_eq!(info.mode, "chat");
    assert_eq!(info.litellm_provider, "llama.cpp");
    assert_eq!(info.max_input_tokens, Some(262144));
    assert_eq!(info.max_output_tokens, Some(32768));
    assert_eq!(info.max_tokens, Some(32768));

    // ADR-0008 effective flag: OR of props (false) and configured levels (true).
    assert!(
        info.supports_reasoning,
        "props false OR levels true must be true"
    );
    assert!(info.supports_function_calling);
    assert!(info.supports_vision, "image in input modalities");
    assert!(!info.supports_audio_input);
    assert!(!info.supports_pdf_input);

    // Local inference: no per-token pricing.
    assert_eq!(info.input_cost_per_token, 0.0);
    assert_eq!(info.output_cost_per_token, 0.0);
    assert_eq!(info.cache_read_input_token_cost, 0.0);
    assert_eq!(info.cache_creation_input_token_cost, 0.0);

    // off→none (ADR-0009), order preserved.
    assert_eq!(
        info.reasoning_effort_levels,
        Some(vec![
            "none".to_string(),
            "low".to_string(),
            "medium".to_string(),
            "xhigh".to_string()
        ])
    );
    assert_eq!(info.supports_none_reasoning_effort, Some(true));
    assert_eq!(info.supports_minimal_reasoning_effort, Some(false));
    assert_eq!(info.supports_low_reasoning_effort, Some(true));
    assert_eq!(info.supports_xhigh_reasoning_effort, Some(true));
    assert_eq!(info.supports_max_reasoning_effort, Some(false));
}

/// No configured levels: all six effort fields are absent from the wire
/// (LiteLLM treats absent as "unknown — advertise nothing").
#[test]
fn test_no_reasoning_levels_omits_effort_fields_on_wire() {
    let entry = base_entry();
    let entry = ModelEntry {
        reasoning: true,
        supports_reasoning_effort: false,
        reasoning_levels: None,
        ..entry
    };

    let mapped = litellm_entry_from_model_entry(&entry);
    let value = serde_json::to_value(&mapped).expect("entry must serialize");

    for key in [
        "supports_minimal_reasoning_effort",
        "supports_low_reasoning_effort",
        "supports_none_reasoning_effort",
        "supports_xhigh_reasoning_effort",
        "supports_max_reasoning_effort",
        "reasoning_effort_levels",
    ] {
        assert!(
            value.get("model_info").unwrap().get(key).is_none(),
            "{key} must be absent when no levels are configured"
        );
    }
    // Props-only reasoning still surfaces.
    assert!(
        value
            .get("model_info")
            .unwrap()
            .get("supports_reasoning")
            .unwrap()
            .as_bool()
            .unwrap(),
        "supports_reasoning must be true from props alone"
    );
}

/// Empty levels vec behaves like absent levels: all six fields omitted,
/// and effective reasoning stays false.
#[test]
fn test_empty_reasoning_levels_omits_effort_fields_on_wire() {
    let entry = base_entry();
    let entry = ModelEntry {
        reasoning: false,
        supports_reasoning_effort: false,
        reasoning_levels: Some(vec![]),
        ..entry
    };

    let mapped = litellm_entry_from_model_entry(&entry);
    let value = serde_json::to_value(&mapped).expect("entry must serialize");

    for key in [
        "supports_minimal_reasoning_effort",
        "supports_low_reasoning_effort",
        "supports_none_reasoning_effort",
        "supports_xhigh_reasoning_effort",
        "supports_max_reasoning_effort",
        "reasoning_effort_levels",
    ] {
        assert!(
            value.get("model_info").unwrap().get(key).is_none(),
            "{key} must be absent for an empty levels list"
        );
    }
    assert_eq!(
        value
            .get("model_info")
            .unwrap()
            .get("supports_reasoning")
            .unwrap()
            .as_bool(),
        Some(false),
        "supports_reasoning must be false when props is false and no levels"
    );
}

/// Only `off` configured: converts to `none` and flips just the none flag.
#[test]
fn test_off_only_levels_map_to_none() {
    let entry = base_entry();
    let entry = ModelEntry {
        reasoning: true,
        supports_reasoning_effort: true,
        reasoning_levels: Some(vec!["off".to_string()]),
        ..entry
    };

    let mapped = litellm_entry_from_model_entry(&entry);
    let info = &mapped.model_info;

    assert_eq!(info.reasoning_effort_levels, Some(vec!["none".to_string()]));
    assert_eq!(info.supports_none_reasoning_effort, Some(true));
    assert_eq!(info.supports_minimal_reasoning_effort, Some(false));
    assert_eq!(info.supports_low_reasoning_effort, Some(false));
    assert_eq!(info.supports_xhigh_reasoning_effort, Some(false));
    assert_eq!(info.supports_max_reasoning_effort, Some(false));
    assert!(
        info.supports_reasoning,
        "off-only levels still imply reasoning support"
    );
}

/// Only `max` configured: flips just the max flag.
#[test]
fn test_max_only_levels_map_to_max() {
    let entry = base_entry();
    let entry = ModelEntry {
        reasoning: true,
        supports_reasoning_effort: true,
        reasoning_levels: Some(vec!["max".to_string()]),
        ..entry
    };

    let mapped = litellm_entry_from_model_entry(&entry);
    let info = &mapped.model_info;

    assert_eq!(info.supports_max_reasoning_effort, Some(true));
    assert_eq!(info.reasoning_effort_levels, Some(vec!["max".to_string()]));
    assert_eq!(info.supports_minimal_reasoning_effort, Some(false));
    assert_eq!(info.supports_low_reasoning_effort, Some(false));
    assert_eq!(info.supports_none_reasoning_effort, Some(false));
    assert_eq!(info.supports_xhigh_reasoning_effort, Some(false));
}

/// Drift guard: the serialized wire shape round-trips losslessly through the
/// typed struct (same pattern as `tests/opencode.rs::
/// test_opencode_response_deserializes_into_typed`).
#[test]
fn test_litellm_entry_round_trips_through_serde_json() {
    let entry = base_entry();
    let entry = ModelEntry {
        id: Some("qwen3-8b".to_string()),
        backend: "llama.cpp".to_string(),
        limit: ModelLimit {
            context: Some(262144),
            output: Some(32768),
        },
        modalities: Some(ModelModalities {
            input: vec!["text".to_string(), "image".to_string()],
            output: vec!["text".to_string()],
        }),
        tool_call: true,
        reasoning: false,
        supports_reasoning_effort: true,
        reasoning_levels: Some(vec![
            "off".to_string(),
            "low".to_string(),
            "medium".to_string(),
            "xhigh".to_string(),
        ]),
        ..entry
    };

    let mapped = litellm_entry_from_model_entry(&entry);
    let value = serde_json::to_value(&mapped).expect("entry must serialize");
    let parsed: LiteLLMModelInfoEntry = serde_json::from_value(value.clone())
        .expect("serialized wire shape must deserialize into LiteLLMModelInfoEntry");

    assert_eq!(
        parsed, mapped,
        "round-trip must be lossless — struct fields must match the wire shape exactly"
    );
    assert_eq!(
        serde_json::to_value(&parsed).expect("parsed must serialize"),
        value,
        "re-serialization must be byte-identical (same Value)"
    );
}

// ── Handler tests (plan-196 Task 3) ──────────────────────────────────────

/// Envelope + one model: config with api_name, model, enabled, reasoning
/// levels, and a config-level context_length; no backend loaded.
#[tokio::test]
async fn test_handler_envelope_with_one_model() {
    use super::super::tests::helpers::{call_litellm_model_info, create_state_with_model};
    use crate::config::ModelConfig;

    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        reasoning_levels: Some(vec!["off".to_string(), "medium".to_string()]),
        context_length: Some(32768),
        ..Default::default()
    })
    .await;

    let (status, body) = call_litellm_model_info(state, "/v1/model/info").await;
    assert_eq!(status, 200, "GET /v1/model/info must return 200");
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let data = value
        .get("data")
        .and_then(|d| d.as_array())
        .expect("envelope must be { data: [...] }");
    assert_eq!(data.len(), 1, "one enabled model → one entry");

    let entry = &data[0];
    assert_eq!(
        entry.get("model_name").unwrap().as_str(),
        Some("test-model"),
        "api_name wins for model_name"
    );
    let info = entry.get("model_info").unwrap();
    assert_eq!(
        info.get("max_input_tokens").unwrap().as_u64(),
        Some(32768),
        "config context_length must surface as max_input_tokens"
    );
    assert_eq!(
        info.get("reasoning_effort_levels"),
        Some(&serde_json::json!(["none", "medium"])),
        "off → none conversion must survive the handler (order preserved)"
    );
    // No backend loaded → build_model_entry's (true, false) default, NOT false.
    assert_eq!(
        info.get("supports_function_calling").unwrap().as_bool(),
        Some(true),
        "tool_call must default to true when no backend /props data exists"
    );
}

/// Alias entry: model + alias → two entries; the alias entry's model_name is
/// the alias name and its model_info mirrors the target's.
#[tokio::test]
async fn test_handler_alias_entry_mirrors_target() {
    use super::super::tests::helpers::{call_litellm_model_info, create_state_with_model};
    use crate::config::ModelConfig;

    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        reasoning_levels: Some(vec!["off".to_string(), "medium".to_string()]),
        context_length: Some(32768),
        ..Default::default()
    })
    .await;

    {
        let mut aliases = state.registry.aliases.write().await;
        aliases.insert("my-alias".to_string(), "test-model".to_string());
    }

    let (status, body) = call_litellm_model_info(state, "/v1/model/info").await;
    assert_eq!(status, 200);
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let data = value.get("data").unwrap().as_array().unwrap().clone();
    assert_eq!(data.len(), 2, "model + alias → two entries");

    let alias_entry = data
        .iter()
        .find(|e| e.get("model_name").unwrap().as_str() == Some("my-alias"))
        .expect("alias entry must exist with the alias name");
    let target_entry = data
        .iter()
        .find(|e| e.get("model_name").unwrap().as_str() == Some("test-model"))
        .expect("target entry must exist");

    let alias_info = alias_entry.get("model_info").unwrap();
    let target_info = target_entry.get("model_info").unwrap();
    // Same context and same effort fields as the target.
    assert_eq!(
        alias_info.get("max_input_tokens"),
        target_info.get("max_input_tokens"),
        "alias must mirror the target's context"
    );
    assert_eq!(
        alias_info.get("reasoning_effort_levels"),
        target_info.get("reasoning_effort_levels"),
        "alias must mirror the target's effort fields"
    );
    assert_eq!(
        alias_info.get("supports_function_calling"),
        target_info.get("supports_function_calling")
    );
    // The alias entry's own identity fields.
    assert_eq!(alias_info.get("id").unwrap().as_str(), Some("my-alias"));
    assert_eq!(
        alias_entry
            .get("litellm_params")
            .unwrap()
            .get("model")
            .unwrap()
            .as_str(),
        Some("my-alias")
    );
}

/// `/model/info` parity: the same fixture routed through `GET /model/info`
/// returns a byte-identical body to `GET /v1/model/info`.
#[tokio::test]
async fn test_handler_model_info_parity_with_v1() {
    use super::super::tests::helpers::{call_litellm_model_info, create_state_with_model};
    use crate::config::ModelConfig;

    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        reasoning_levels: Some(vec!["off".to_string(), "medium".to_string()]),
        context_length: Some(32768),
        ..Default::default()
    })
    .await;

    let (status_v1, body_v1) = call_litellm_model_info(state.clone(), "/v1/model/info").await;
    let (status_plain, body_plain) = call_litellm_model_info(state, "/model/info").await;
    assert_eq!(status_v1, 200);
    assert_eq!(status_plain, 200);
    assert_eq!(
        body_v1, body_plain,
        "/model/info must return a byte-identical body to /v1/model/info"
    );
}

/// `litellm_model_id` filter: a matching name returns the single entry, an
/// unknown name returns an empty `data` array, and an empty value behaves
/// like no param (full list).
#[tokio::test]
async fn test_handler_litellm_model_id_filter() {
    use super::super::tests::helpers::{call_litellm_model_info, create_state_with_model};
    use crate::config::ModelConfig;

    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        model: Some("test/model".to_string()),
        enabled: true,
        reasoning_levels: Some(vec!["off".to_string(), "medium".to_string()]),
        context_length: Some(32768),
        ..Default::default()
    })
    .await;

    // Matching name → the single matching entry.
    let (status, body) =
        call_litellm_model_info(state.clone(), "/v1/model/info?litellm_model_id=test-model").await;
    assert_eq!(status, 200);
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let data = value.get("data").unwrap().as_array().unwrap().clone();
    assert_eq!(data.len(), 1);
    assert_eq!(
        data[0].get("model_name").unwrap().as_str(),
        Some("test-model")
    );

    // Unknown name → empty data array.
    let (status, body) = call_litellm_model_info(
        state.clone(),
        "/v1/model/info?litellm_model_id=does-not-exist",
    )
    .await;
    assert_eq!(status, 200);
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let data = value.get("data").unwrap().as_array().unwrap().clone();
    assert_eq!(data.len(), 0, "unknown id must yield an empty data array");

    // Empty value → full list (behaves like no param).
    let (status, body) = call_litellm_model_info(state, "/v1/model/info?litellm_model_id=").await;
    assert_eq!(status, 200);
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let data = value.get("data").unwrap().as_array().unwrap().clone();
    assert_eq!(data.len(), 1, "empty id must behave like no param");
}

/// No-levels model: the entry's model_info has no `reasoning_effort_levels`
/// key (and no per-level effort flags) — absent when unknown.
#[tokio::test]
async fn test_handler_no_levels_omits_effort_fields() {
    use super::super::tests::helpers::{call_litellm_model_info, create_state_with_model};
    use crate::config::ModelConfig;

    let state = create_state_with_model(ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-plain".to_string()),
        model: Some("test/plain".to_string()),
        enabled: true,
        ..Default::default()
    })
    .await;

    let (status, body) = call_litellm_model_info(state, "/v1/model/info").await;
    assert_eq!(status, 200);
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let data = value.get("data").unwrap().as_array().unwrap().clone();
    assert_eq!(data.len(), 1);

    let info = &data[0].get("model_info").unwrap();
    for key in [
        "reasoning_effort_levels",
        "supports_minimal_reasoning_effort",
        "supports_low_reasoning_effort",
        "supports_none_reasoning_effort",
        "supports_xhigh_reasoning_effort",
        "supports_max_reasoning_effort",
    ] {
        assert!(
            info.get(key).is_none(),
            "{key} must be absent when no levels are configured"
        );
    }
}

/// Sanity: the handler is reachable at both paths with no models configured
/// (empty data array, 200) — the route is honest for an empty catalog.
#[tokio::test]
async fn test_handler_empty_catalog_returns_empty_data() {
    use crate::proxy::ProxyState;
    use std::sync::Arc;

    let config = crate::config::Config::default();
    let state = Arc::new(ProxyState::new(
        config,
        None,
        crate::db::pool::test_dummy_pool(),
    ));

    let app = Router::new()
        .route(
            "/v1/model/info",
            axum::routing::get(handle_litellm_model_info),
        )
        .route("/model/info", axum::routing::get(handle_litellm_model_info))
        .with_state(state);

    for uri in ["/v1/model/info", "/model/info"] {
        let request = Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), 200);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value.get("data").unwrap().as_array().unwrap().len(),
            0,
            "no enabled models → empty data array at {uri}"
        );
    }
}
