//! Mapping of a collected [`ModelEntry`] to the LiteLLM `/v1/model/info`
//! wire shape (the `model_info_v1` docstring / `model_prices_and_context_window.json`
//! schema that `pi-provider-litellm` consumes).
//!
//! Reasoning-effort semantics (ADR-0008): the per-level `supports_*_reasoning_effort`
//! flags are **absent** when the model has no configured levels (LiteLLM treats
//! absent as "unknown — advertise nothing") and present with `true`/`false` when
//! `reasoning_levels` is non-empty — the explicit list is authoritative. The
//! `off` → `none` conversion is ADR-0009 and reuses `levels_to_wire_values`.

use axum::extract::{Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::utils::{collect_model_entries, levels_to_wire_values, ModelEntry};

/// Query extractor for `?litellm_model_id=<name>`. Named-field on purpose:
/// a tuple newtype would NOT work — serde derive for newtypes ignores field
/// renames and never reads the `litellm_model_id` key out of the query map,
/// so axum `Query` would reject every request.
#[derive(Debug, Deserialize)]
pub struct LiteLLMModelIdQuery {
    #[serde(default)]
    #[serde(rename = "litellm_model_id")]
    id: Option<String>,
}

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
) -> Json<LiteLLMModelInfoResponse> {
    let entries = collect_model_entries(&state.0).await;
    // A present non-empty `litellm_model_id` filters to the matching entry;
    // absent or empty → full list.
    let wanted = params.id.as_deref().filter(|s| !s.is_empty());
    let data = entries
        .iter()
        .filter(|e| wanted.is_none() || e.id.as_deref() == wanted)
        .map(litellm_entry_from_model_entry)
        .collect();
    Json(LiteLLMModelInfoResponse { data })
}

/// The LiteLLM `/model/info` response envelope: `{ "data": [...] }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiteLLMModelInfoResponse {
    pub data: Vec<LiteLLMModelInfoEntry>,
}

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

/// Map a collected model entry to the LiteLLM `/v1/model/info` wire shape.
pub(super) fn litellm_entry_from_model_entry(entry: &ModelEntry) -> LiteLLMModelInfoEntry {
    let model_name = entry.id.clone().unwrap_or_default();

    // Reasoning-effort block: absent (all None) when no levels are configured;
    // the explicit list is authoritative otherwise. Note: `medium` and `high`
    // have no per-level flag in LiteLLM's schema — they appear only inside
    // `reasoning_effort_levels`.
    let wire = levels_to_wire_values(&entry.reasoning_levels);
    let (supports_minimal, supports_low, supports_none, supports_xhigh, supports_max) = match &wire
    {
        Some(values) => {
            let has = |level: &str| values.iter().any(|v| v == level);
            (
                Some(has("minimal")),
                Some(has("low")),
                Some(has("none")),
                Some(has("xhigh")),
                Some(has("max")),
            )
        }
        None => (None, None, None, None, None),
    };

    // Modality flags: all false when modalities is None.
    let (supports_vision, supports_audio_input, supports_pdf_input) = entry
        .modalities
        .as_ref()
        .map_or((false, false, false), |m| {
            (
                m.input.iter().any(|s| s == "image"),
                m.input.iter().any(|s| s == "audio"),
                m.input.iter().any(|s| s == "pdf"),
            )
        });

    LiteLLMModelInfoEntry {
        model_name: model_name.clone(),
        litellm_params: LiteLLMParams {
            model: model_name.clone(),
            provider: entry.backend.clone(),
        },
        model_info: LiteLLMModelInfo {
            id: model_name,
            mode: "chat".to_string(),
            litellm_provider: entry.backend.clone(),
            max_input_tokens: entry.limit.context,
            max_output_tokens: entry.limit.output,
            max_tokens: entry.limit.output.or(entry.limit.context),
            supports_function_calling: entry.tool_call,
            // ADR-0008 effective flag: /props "backend preserves reasoning"
            // OR user-configured levels.
            supports_reasoning: entry.reasoning || entry.supports_reasoning_effort,
            supports_vision,
            supports_audio_input,
            supports_pdf_input,
            input_cost_per_token: 0.0,
            output_cost_per_token: 0.0,
            cache_read_input_token_cost: 0.0,
            cache_creation_input_token_cost: 0.0,
            supports_minimal_reasoning_effort: supports_minimal,
            supports_low_reasoning_effort: supports_low,
            supports_none_reasoning_effort: supports_none,
            supports_xhigh_reasoning_effort: supports_xhigh,
            supports_max_reasoning_effort: supports_max,
            reasoning_effort_levels: wire,
        },
    }
}
