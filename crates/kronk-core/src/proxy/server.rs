use crate::proxy::ProxyState;
use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{MatchedPath, OriginalUri},
    http::{header, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use serde_json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use toml;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};

pub struct ProxyServer {
    state: Arc<ProxyState>,
}

impl ProxyServer {
    pub fn new(state: Arc<ProxyState>) -> Self {
        Self { state }
    }

    pub async fn run(self, addr: SocketAddr) -> Result<()> {
        info!("Starting proxy server on {}", addr);

        let app = Router::new()
            .route("/chat/completions", post(handle_chat_completions))
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/models", get(handle_list_models))
            .route("/models/:model_id", get(handle_get_model))
            .route("/health", get(handle_health))
            .fallback(handle_fallback)
            .layer(TraceLayer::new_for_http())
            .with_state(self.state);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn handle_chat_completions(
    state: Arc<ProxyState>,
    body: Bytes,
    matched_path: MatchedPath,
    uri: OriginalUri,
) -> Result<impl IntoResponse> {
    debug!(
        "Received chat/completions request to {}",
        matched_path.as_str()
    );

    // Parse the request body to extract model name
    let request: serde_json::Value =
        serde_json::from_slice(&body).context("Failed to parse request body")?;

    let model_name = request
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'model' field in request"))?;

    info!("Routing request for model: {}", model_name);

    // Check if model is already loaded
    let is_loaded = state.is_model_loaded(model_name).await;

    if !is_loaded {
        // Try to load the model
        let model_card = state
            .get_model_card(model_name)
            .await
            .ok();

        if let Err(e) = state.load_model(model_name, &model_card).await {
            warn!("Failed to load model '{}': {}", model_name, e);
        }
    }

    // Update last accessed time
    state.update_last_accessed(model_name).await;

    // Forward the request to the backend
    let response = forward_request_to_backend(state, body, model_name).await?;

    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        response,
    ))
}

async fn handle_list_models(state: Arc<ProxyState>) -> impl IntoResponse {
    let models: Vec<String> = state
        .models
        .read()
        .await
        .keys()
        .cloned()
        .collect();

    let response = serde_json::json!({
        "object": "list",
        "data": models
    });

    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        response,
    ))
}

async fn handle_get_model(state: Arc<ProxyState>, model_id: String) -> impl IntoResponse {
    let model_state = state.get_model_state(&model_id).await;

    let response = if let Some(state) = model_state {
        serde_json::json!({
            "id": model_id,
            "object": "model",
            "created": state.load_time.elapsed().as_secs(),
            "owned_by": state.backend,
            "ready": true
        })
    } else {
        serde_json::json!({
            "error": {
                "message": "Model not found",
                "type": "NotFoundError"
            }
        })
    };

    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        response,
    ))
}

async fn handle_health() -> impl IntoResponse {
    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::json!({
            "status": "ok",
            "service": "kronk-proxy"
        }),
    ))
}

async fn handle_fallback() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        "Not Found",
    )
}

/// Forward a request to the backend and stream the response back.
async fn forward_request_to_backend(
    state: Arc<ProxyState>,
    body: Bytes,
    model_name: String,
) -> Result<Bytes> {
    // Get the backend URL for this model
    let backend_url = state
        .get_backend_url(&model_name)
        .await
        .with_context(|| format!("No backend URL for model '{}'", model_name))?;

    info!("Forwarding request to backend at {}", backend_url);

    // For now, return a placeholder response
    // In a full implementation, we would:
    // 1. Make a request to the backend
    // 2. Stream the SSE response back to the client
    // 3. Handle errors and timeouts

    let response = serde_json::json!({
        "error": {
            "message": "Backend forwarding not fully implemented yet",
            "type": "InternalServerError"
        }
    });

    Ok(response.to_string().into_bytes())
}

/// Get the model card for a model name.
async fn get_model_card(state: &ProxyState, model_name: &str) -> Option<crate::models::card::ModelCard> {
    let configs_dir = state.config_data.read().await.configs_dir().ok()?;
    
    // Try to find the model card file
    // Format: configs.d/<company>--<model>.toml
    let card_path = configs_dir.join(format!("{}--{}.toml", model_name.split('/').next().unwrap_or(""), model_name));
    
    if card_path.exists() {
        let content = std::fs::read_to_string(&card_path).ok()?;
        let card: crate::models::card::ModelCard = toml::from_str(&content).ok()?;
        Some(card)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proxy_server_creation() {
        let state = Arc::new(ProxyState {
            config: ProxyConfig::default(),
            models: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            registry: Arc::new(tokio::sync::RwLock::new(crate::backends::registry::BackendRegistry::default())),
            config_data: Arc::new(tokio::sync::RwLock::new(crate::config::Config::default())),
            process_map: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        });

        let server = ProxyServer::new(state);
        assert!(true);
    }
}