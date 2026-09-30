use std::sync::Arc;

use crate::proxy::ProxyState;
use axum::extract::State;
use axum::Json;

use super::utils::{collect_model_entries, OpencodeModelsResponse};

/// Handle listing all enabled models for OpenCode plugin discovery.
/// Returns rich metadata including context limits, modalities, and capabilities.
/// Aliases are included with the same metadata as their target model, using the alias name as `id`.
pub async fn handle_opencode_list_models(
    state: State<Arc<ProxyState>>,
) -> Json<OpencodeModelsResponse> {
    let models = collect_model_entries(&state.0).await;
    Json(OpencodeModelsResponse { models })
}
