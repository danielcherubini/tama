use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::ModelModalities;
use crate::models::ConfigKey;
use crate::proxy::handlers::models::{
    fetch_models_from_backend, find_model_in_entries, BackendModelEntry,
};
use crate::proxy::ProxyState;

use super::ModelCapabilities;

/// Capitalize the first character of a string, preserve the rest unchanged.
pub fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}

/// Generate a pretty display name from an HF repo name.
/// e.g., "unsloth/Qwen3.5-35B-A3B-GGUF" -> "Unsloth: Qwen3.5 35B A3B"
/// Strips common file suffixes like "GGUF".
pub fn generate_display_name(hf_repo: &str) -> String {
    let parts: Vec<&str> = hf_repo.split('/').collect();
    let (org, model_name) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        (hf_repo, hf_repo)
    };

    let model_name_processed = model_name
        .replace(['-', '_'], " ")
        .split_whitespace()
        .filter(|word| !word.eq_ignore_ascii_case("GGUF"))
        .map(capitalize_first)
        .collect::<Vec<_>>()
        .join(" ");

    format!("{}: {}", capitalize_first(org), model_name_processed)
}

/// Resolve a raw model identifier (db id, repo id, or config key) to a config_key string.
///
/// Accepts three forms (in priority order):
/// 1. Integer db_id — looked up against `config.db_id` in the in-memory map.
/// 2. Repo id with a slash (e.g. `Unsloth/Foo-GGUF`) — normalized to the
///    lowercased double-dash config_key (e.g. `unsloth--foo-gguf`).
/// 3. Anything else — returned unchanged, on the assumption it is already a
///    config_key, api_name, or model field that downstream lookups will handle.
///
/// Steps 1 and 2 both honour the case-insensitive repo_id contract established
/// by the `COLLATE NOCASE` migration on `model_configs.repo_id`: the in-memory
/// HashMap is keyed by the lowercased repo_id, so a repo id in any case
/// resolves to the same bucket.
pub(super) async fn resolve_config_key(state: &ProxyState, raw: &str) -> String {
    if let Ok(id) = raw.parse::<i64>() {
        let configs = state.registry.model_configs.read().await;
        if let Some((key, _)) = configs.iter().find(|(_, c)| c.db_id == Some(id)) {
            return key.clone();
        }
    }
    if raw.contains('/') {
        return ConfigKey::from_repo_id(raw).to_string();
    }
    raw.to_string()
}

/// Context/output limits sub-object of an opencode model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLimit {
    pub context: Option<u32>,
    pub output: Option<u32>,
}

/// One model entry in the `/v1/opencode/models` discovery response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: Option<String>,
    pub name: String,
    pub model: Option<String>,
    pub backend: String,
    pub context_length: Option<u32>,
    pub limit: ModelLimit,
    pub quant: Option<String>,
    pub gpu_layers: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<ModelModalities>,
    pub tool_call: bool,
    pub reasoning: bool,
    pub attachment: bool,
    pub temperature: bool,
    /// Derived at serialization: true when the model has configured
    /// reasoning levels (see ADR-0008). Wire name is camelCase.
    #[serde(rename = "supportsReasoningEffort")]
    pub supports_reasoning_effort: bool,
    /// Raw stored levels (pi vocabulary). Absent when None.
    #[serde(rename = "reasoningLevels", skip_serializing_if = "Option::is_none")]
    pub reasoning_levels: Option<Vec<String>>,
    /// Opencode-canonical derived field (snake_case on purpose —
    /// byte-compatible with the models.dev catalog). Absent when None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_options: Option<serde_json::Value>,
}

/// Stored levels (pi vocabulary) → wire values (ADR-0009: `off` → `none`,
/// all other levels pass through). None for absent/empty levels.
pub(super) fn levels_to_wire_values(levels: &Option<Vec<String>>) -> Option<Vec<String>> {
    let levels = levels.as_ref()?;
    if levels.is_empty() {
        return None;
    }
    Some(
        levels
            .iter()
            .map(|l| {
                if l == "off" {
                    "none".to_string()
                } else {
                    l.clone()
                }
            })
            .collect(),
    )
}

/// Build the opencode-canonical `reasoning_options` value from stored
/// levels: [{ "type": "effort", "values": [...] }] with `off` mapped to
/// `none` (ADR-0009). Returns None for empty/absent levels.
pub(crate) fn reasoning_options_from_levels(
    levels: &Option<Vec<String>>,
) -> Option<serde_json::Value> {
    levels_to_wire_values(levels)
        .map(|values| serde_json::json!([{ "type": "effort", "values": values }]))
}

/// Extract capability flags from a /props response body.
/// Returns (tool_call, reasoning) tuple. Defaults to (true, false) on any error.
pub(super) fn extract_capabilities(body: &[u8]) -> (bool, bool) {
    let value = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(v) => v,
        Err(_) => return (true, false),
    };

    let mut tool_call = true; // default
    let mut reasoning = false; // default

    // Check chat_template_caps.supports_tool_calls
    if let Some(supports_tool_calls) = value
        .get("chat_template_caps")
        .and_then(|c| c.get("supports_tool_calls"))
        .and_then(|v| v.as_bool())
    {
        tool_call = supports_tool_calls;
    }

    // Check chat_template_caps.supports_preserve_reasoning
    if let Some(supports_reasoning) = value
        .get("chat_template_caps")
        .and_then(|c| c.get("supports_preserve_reasoning"))
        .and_then(|v| v.as_bool())
    {
        reasoning = supports_reasoning;
    }

    // Also check default_generation_settings.params.reasoning_format != "none"
    if let Some(reasoning_format) = value
        .get("default_generation_settings")
        .and_then(|d| d.get("params"))
        .and_then(|p| p.get("reasoning_format"))
        .and_then(|v| v.as_str())
    {
        if !reasoning_format.eq_ignore_ascii_case("none") {
            reasoning = true;
        }
    }

    (tool_call, reasoning)
}

/// Query a single backend's /props endpoint and extract capability flags.
/// Returns (tool_call, reasoning) tuple. Defaults to (true, false) on any error.
pub(super) async fn fetch_capabilities_from_backend(
    client: &reqwest::Client,
    backend_url: &str,
) -> (bool, bool) {
    let url = format!("{}/props", backend_url);
    match client
        .get(&url)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) => match resp.bytes().await {
            Ok(bytes) => extract_capabilities(&bytes),
            Err(_) => (true, false),
        },
        Err(_) => (true, false),
    }
}

/// Extract context length from a backend /v1/models entry.
/// Checks `max_model_len` (vLLM) first, then falls back to `meta.n_ctx` (llama.cpp).
fn extract_context_length_from_backend_entry(entry: &BackendModelEntry) -> Option<u32> {
    // vLLM: max_model_len
    entry
        .extra
        .get("max_model_len")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        // llama.cpp: meta.n_ctx
        .or_else(|| {
            entry
                .extra
                .get("meta")
                .and_then(|m| m.get("n_ctx"))
                .and_then(|v| v.as_u64())
                .and_then(|v| u32::try_from(v).ok())
        })
}

/// Collect one `ModelEntry` per enabled config plus alias entries, using
/// the shared per-backend data sources (live rows, /props capabilities,
/// backend /v1/models context). Shared by the opencode and LiteLLM
/// model-info handlers — the single source of truth for this collection.
pub(super) async fn collect_model_entries(
    state: &Arc<crate::proxy::ProxyState>,
) -> Vec<ModelEntry> {
    // 1. Snapshot data under locks — clone out so locks are dropped before any .await below.
    // `all_configs` is a clone of the HashMap contents, not the guard, so no explicit drop needed.
    let (loaded_models, all_configs): (
        HashMap<String, String>,
        HashMap<String, crate::config::ModelConfig>,
    ) = {
        // Live wire rows (plan-193 T4 flip) — config_name → endpoint for ready rows.
        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        let configs = state.registry.model_configs.read().await;
        let loaded: HashMap<String, String> = live
            .all()
            .iter()
            .filter(|r| r.status == "ready" && !r.endpoint.is_empty())
            .map(|r| (r.key.clone(), r.endpoint.clone()))
            .collect();
        (loaded, configs.clone())
    }; // locks dropped

    // 2. Fetch capabilities (/props) and models (/v1/models) for all loaded backends concurrently.
    // Deduplicate backend URLs — multiple configs can share the same backend.
    let unique_urls: Vec<_> = loaded_models
        .values()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    // Fetch capabilities from /props for each unique URL
    let cap_futures: Vec<_> = unique_urls
        .iter()
        .map(|url| fetch_capabilities_from_backend(&state.client, url))
        .collect();
    let cap_results: Vec<(bool, bool)> = futures::future::join_all(cap_futures).await;

    // Build url -> ModelCapabilities map
    let url_cap_map: HashMap<_, _> = unique_urls
        .iter()
        .zip(cap_results)
        .map(|(url, (tc, r))| {
            (
                url.clone(),
                ModelCapabilities {
                    tool_call: tc,
                    reasoning: r,
                },
            )
        })
        .collect();

    // Fetch /v1/models from each unique backend URL
    let model_futures: Vec<_> = unique_urls
        .iter()
        .map(|url| fetch_models_from_backend(state, url))
        .collect();
    let model_results: Vec<Vec<BackendModelEntry>> = futures::future::join_all(model_futures).await;

    // Build url -> Vec<BackendModelEntry> map
    let url_model_map: HashMap<_, _> = unique_urls.into_iter().zip(model_results).collect();

    // Build config_name -> ModelCapabilities map (for backward compat with cap_map lookups)
    let cap_map: HashMap<_, _> = loaded_models
        .iter()
        .filter_map(|(name, url)| url_cap_map.get(url).map(|caps| (name.clone(), *caps)))
        .collect();

    // 3. Build model entries with capabilities and context lengths
    let mut models: Vec<ModelEntry> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    for (id, cfg) in all_configs.iter().filter(|(_, cfg)| cfg.enabled) {
        let caps = cap_map.get(id);

        // Look up backend context length from /v1/models response
        let backend_ctx = loaded_models
            .get(id)
            .and_then(|url| url_model_map.get(url))
            .and_then(|entries| find_model_in_entries(entries, cfg.model.as_deref()))
            .as_ref()
            .and_then(extract_context_length_from_backend_entry);

        if let Some(entry) = build_model_entry(state, id, cfg, caps, backend_ctx).await {
            if let Some(api_id) = entry.id.as_deref() {
                seen_ids.insert(api_id.to_string());
            }
            models.push(entry);
        }
    }

    // 4. Add alias entries — inherit capabilities and context_length from target model
    let aliases = state.registry.aliases.read().await;
    for (alias_name, resolved_model) in aliases.iter() {
        if seen_ids.contains(alias_name.as_str()) {
            continue;
        }

        let resolved_lower = resolved_model.to_lowercase();
        let target_cfg = all_configs.iter().find(|(_, cfg)| {
            cfg.enabled
                && (cfg.api_name.as_ref().map(|s| s.to_lowercase()) == Some(resolved_lower.clone())
                    || cfg.model.as_ref().map(|s| s.to_lowercase()) == Some(resolved_lower.clone()))
        });

        if let Some((key, cfg)) = target_cfg {
            let caps = cap_map.get(key);
            // Look up backend context length for the target config
            let backend_ctx = loaded_models
                .get(key)
                .and_then(|url| url_model_map.get(url))
                .and_then(|entries| find_model_in_entries(entries, cfg.model.as_deref()))
                .as_ref()
                .and_then(extract_context_length_from_backend_entry);

            if let Some(mut entry) = build_model_entry(state, key, cfg, caps, backend_ctx).await {
                entry.id = Some(alias_name.clone());
                // Derive a display name from the alias slug (not from the target model).
                let alias_display = alias_name
                    .replace(['-', '_'], " ")
                    .split_whitespace()
                    .map(capitalize_first)
                    .collect::<Vec<_>>()
                    .join(" ");
                entry.name = alias_display;
                models.push(entry);
                seen_ids.insert(alias_name.clone());
            }
        }
    }
    drop(aliases);

    models
}

/// Response wrapper for `/v1/opencode/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpencodeModelsResponse {
    pub models: Vec<ModelEntry>,
}

/// Build a model entry from a config entry.
pub(super) async fn build_model_entry(
    state: &ProxyState,
    id: &str,
    cfg: &crate::config::ModelConfig,
    capabilities: Option<&ModelCapabilities>,
    backend_context_length: Option<u32>,
) -> Option<ModelEntry> {
    // Use model field first, fall back to api_name.
    let hf_repo = cfg.model.as_deref().or(cfg.api_name.as_deref())?;

    // Resolve unified metadata from whichever source is populated.
    let meta = crate::models::ResolvedModelMetadata::resolve(cfg);
    // Note: meta.context_length is intentionally NOT used here — the chain
    // below interleaves the live backend-reported value between vLLM and HF
    // tiers, which resolve() cannot do (see metadata.rs resolve() docs).
    // Only meta.quant is consumed from the resolved metadata.

    // Context length resolution order (highest to lowest priority):
    // 1. cfg.context_length — explicit user override (GGUF column)
    // 2. cfg.vllm.max_model_len — vLLM config
    // 3. backend_context_length — live backend-reported value
    // 4. cfg.hf_context_length — pull-time HF parse
    // 5. model_toml — lowest priority fallback
    let context_length = cfg
        .context_length
        .or(cfg.vllm.max_model_len)
        .or(backend_context_length)
        .or(cfg.hf_context_length);
    // If no context_length found yet, fall back to model_toml (async)
    let context_length = if context_length.is_some() {
        context_length
    } else {
        let model_toml = state.get_model_toml(id).await;
        model_toml.and_then(|m| {
            let quant_key = meta.quant.as_deref().unwrap_or_default();
            m.quants
                .get(quant_key)
                .and_then(|q| q.context_length)
                .or(m.model.default_context_length)
        })
    };
    let modalities = cfg.modalities.clone();

    // Output limit: 1/8 of context window, floored at 16K and capped at 32K.
    let output_limit = context_length.map(|ctx| (ctx / 8).clamp(16384, 32768));

    // API id: prefer api_name, fall back to model — preserve original casing.
    let api_id = cfg.api_name.clone().or_else(|| cfg.model.clone());

    // Generate a pretty display name with org prefix.
    let parts: Vec<&str> = hf_repo.split('/').collect();
    let (org, model_name) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        (hf_repo, hf_repo)
    };

    let model_name_processed = model_name
        .replace(['-', '_'], " ")
        .split_whitespace()
        .filter(|word| !word.eq_ignore_ascii_case("GGUF"))
        .map(capitalize_first)
        .collect::<Vec<_>>()
        .join(" ");

    let pretty_name = format!("{}: {}", capitalize_first(org), model_name_processed);

    // Derive attachment from modalities
    let attachment = cfg
        .modalities
        .as_ref()
        .is_some_and(|m| m.input.iter().any(|s| s == "image"));

    // Use provided capabilities or config-derived defaults
    let (tool_call, props_reasoning) = capabilities
        .map(|c| (c.tool_call, c.reasoning))
        .unwrap_or((true, false));

    // Effective reasoning: props-derived OR derived from configured levels
    // (fixes vLLM-served thinking models, which have no /props).
    let reasoning = props_reasoning || cfg.supports_reasoning_effort();

    Some(ModelEntry {
        id: api_id,
        name: pretty_name,
        model: cfg.model.clone(),
        backend: cfg.backend.clone(),
        context_length,
        limit: ModelLimit {
            context: context_length,
            output: output_limit,
        },
        quant: meta.quant,
        gpu_layers: cfg.gpu_layers.map(|n| n.to_string()),
        modalities,
        tool_call,
        reasoning,
        attachment,
        temperature: true,
        supports_reasoning_effort: cfg.supports_reasoning_effort(),
        reasoning_levels: cfg.reasoning_levels.clone(),
        reasoning_options: reasoning_options_from_levels(&cfg.reasoning_levels),
    })
}
