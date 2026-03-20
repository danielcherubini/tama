use crate::proxy::ProxyState;
use anyhow::Context;
use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tracing::info;

pub struct ProxyServer {
    state: Arc<ProxyState>,
}

impl ProxyServer {
    pub fn new(state: Arc<ProxyState>) -> Self {
        Self { state }
    }

    pub async fn run(self, addr: std::net::SocketAddr) -> anyhow::Result<()> {
        info!("Starting proxy server on {}", addr);

        let app = Router::new()
            .route("/chat/completions", post(handle_chat_completions))
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/models", get(handle_list_models))
            .route("/models/:model_id", get(handle_get_model))
            .route("/health", get(handle_health))
            .fallback(handle_fallback)
            .with_state(self.state.clone());

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

#[axum::debug_handler]
async fn handle_chat_completions(
    state: State<Arc<ProxyState>>,
    req: Request<Body>,
) -> Json<serde_json::Value> {
    let body_bytes = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return Json(serde_json::json!({
                "error": {
                    "message": "Bad Request",
                    "type": "BadRequestError"
                }
            }))
        }
    };

    let request: serde_json::Value =
        match serde_json::from_slice(&body_bytes).context("Failed to parse request body") {
            Ok(r) => r,
            Err(_) => serde_json::json!({}),
        };

    let model_name = request
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    info!("Routing request for model: {}", model_name);

    let is_loaded = state.is_model_loaded(model_name).await;

    if !is_loaded {
        let model_card = state.get_model_card(model_name).await;
        if let Some(card) = model_card {
            let _ = state.load_model(model_name, Some(&card)).await;
        }
    }

    state.update_last_accessed(model_name).await;

    Json(serde_json::json!({
        "error": {
            "message": "Backend forwarding not fully implemented yet",
            "type": "InternalServerError"
        }
    }))
}

#[axum::debug_handler]
async fn handle_list_models(state: State<Arc<ProxyState>>) -> Json<serde_json::Value> {
    let models: Vec<String> = state.models.read().await.keys().cloned().collect();

    Json(serde_json::json!({
        "object": "list",
        "data": models
    }))
}

#[axum::debug_handler]
async fn handle_get_model(
    state: State<Arc<ProxyState>>,
    model_id: String,
) -> Json<serde_json::Value> {
    let model_state = state.get_model_state(&model_id).await;

    if let Some(state) = model_state {
        Json(serde_json::json!({
            "id": model_id,
            "object": "model",
            "created": state.load_time.elapsed().as_secs(),
            "owned_by": state.backend,
            "ready": true
        }))
    } else {
        Json(serde_json::json!({
            "error": {
                "message": "Model not found",
                "type": "NotFoundError"
            }
        }))
    }
}

#[axum::debug_handler]
async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "kronk-proxy"
    }))
}

#[axum::debug_handler]
async fn handle_fallback() -> StatusCode {
    StatusCode::NOT_FOUND
}
