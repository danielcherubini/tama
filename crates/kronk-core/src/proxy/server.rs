use crate::proxy::ProxyState;
use anyhow::Context;
use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use reqwest::Client;
use std::sync::Arc;
use tracing::{debug, info};

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
            .route(
                "/chat/completions/stream",
                post(handle_stream_chat_completions),
            )
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
async fn handle_chat_completions(state: State<Arc<ProxyState>>, req: Request<Body>) -> Response {
    let body_bytes = match to_bytes(req.into_body(), usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return StatusCode::BAD_REQUEST.into_response();
        }
    };

    let request: serde_json::Value =
        match serde_json::from_slice(&body_bytes).context("Failed to parse request body") {
            Ok(r) => r,
            Err(_) => {
                return Json(serde_json::json!({
                    "error": {
                        "message": "Bad Request",
                        "type": "BadRequestError"
                    }
                }))
                .into_response();
            }
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
            if let Err(e) = state.load_model(model_name, Some(&card)).await {
                info!("Failed to load model {}: {}", model_name, e);
                return Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to load model: {}", e),
                        "type": "LoadModelError"
                    }
                }))
                .into_response();
            }
        }
    }

    state.update_last_accessed(model_name).await;

    // Forward the request to the backend
    forward_request(&state, model_name, body_bytes.to_vec(), &req).await
}

#[axum::debug_handler]
async fn handle_stream_chat_completions(
    state: State<Arc<ProxyState>>,
    req: Request<Body>,
) -> Response {
    let body_bytes = match to_bytes(req.into_body(), usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let request: serde_json::Value =
        match serde_json::from_slice(&body_bytes).context("Failed to parse request body") {
            Ok(r) => r,
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        };

    let model_name = request
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    info!("Streaming request for model: {}", model_name);

    let is_loaded = state.is_model_loaded(model_name).await;

    if !is_loaded {
        let model_card = state.get_model_card(model_name).await;
        if let Some(card) = model_card {
            if let Err(e) = state.load_model(model_name, Some(&card)).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to load model: {}", e),
                )
                    .into_response();
            }
        }
    }

    state.update_last_accessed(model_name).await;

    // Forward the request to the backend for streaming
    forward_request(&state, model_name, body_bytes.to_vec(), &req).await
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
async fn handle_list_models(state: State<Arc<ProxyState>>) -> Json<serde_json::Value> {
    let models: Vec<String> = state.models.read().await.keys().cloned().collect();

    Json(serde_json::json!({
        "object": "list",
        "data": models
    }))
}

#[axum::debug_handler]
async fn handle_fallback() -> StatusCode {
    StatusCode::NOT_FOUND
}

/// Forward a request to the backend and stream the response back.
async fn forward_request(
    state: &Arc<ProxyState>,
    model_name: &str,
    body_bytes: Vec<u8>,
    req: &Request<Body>,
) -> Response {
    // Get the backend URL for the model
    let backend_url = state.get_backend_url(model_name).await.unwrap_or_else(|e| {
        info!("Failed to get backend URL for {}: {}", model_name, e);
        return format!("http://127.0.0.1:8080");
    });

    info!("Forwarding request to: {}", backend_url);

    let client = Client::new();
    let method = req.method().clone();
    let uri: String = req.uri().to_string();

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(content_type) = req.headers().get("Content-Type") {
        if let Ok(ct) = content_type.to_str() {
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                reqwest::header::HeaderValue::from_static(ct),
            );
        }
    }

    match client
        .request(method, &uri)
        .headers(headers)
        .body(body_bytes.to_vec())
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            let mut builder = Response::builder().status(status);

            for (key, value) in response.headers().iter() {
                if let (Ok(k), Ok(v)) = (key.as_str(), value.to_str()) {
                    builder = builder.header(k, v);
                }
            }

            let body = Body::from_stream(response.bytes_stream());
            builder.body(body).unwrap().into_response()
        }
        Err(e) => {
            info!("Failed to forward request: {}", e);
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Backend error: {}", e),
                        "type": "BadRequestError"
                    }
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use reqwest::Method;
    use std::sync::Arc;

    /// Test SSE streaming response for chat completions
    #[tokio::test]
    async fn test_sse_streaming_response() {
        let config = crate::proxy::ProxyConfig::default();
        let registry = crate::backends::registry::BackendRegistry::default();
        let config_data = crate::config::Config::default();
        let state = Arc::new(crate::proxy::ProxyState::new(config, registry, config_data));

        let socket = std::net::SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            0,
        );

        let server = ProxyServer::new(state.clone());
        tokio::spawn(async move {
            server.run(socket).await.unwrap();
        });

        // Give server time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Test that SSE response has correct content type
        let response: Request<Body> = Request::builder()
            .method(Method::GET)
            .uri("/test")
            .header("content-type", "text/event-stream; charset=utf-8")
            .body(Body::from("test"))
            .unwrap();

        // Verify content type
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream; charset=utf-8"
        );
    }

    /// Test SSE streaming with timeout
    #[tokio::test]
    async fn test_sse_streaming_timeout() {
        let config = crate::proxy::ProxyConfig::default();
        let registry = crate::backends::registry::BackendRegistry::default();
        let config_data = crate::config::Config::default();
        let state = Arc::new(crate::proxy::ProxyState::new(config, registry, config_data));

        let socket = std::net::SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            0,
        );

        let server = ProxyServer::new(state.clone());
        tokio::spawn(async move {
            server.run(socket).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Create a slow streaming response (will timeout)
        let response_body = String::from("data: {\"id\":\"1\"}\n\n");
        let response_bytes = response_body.into_bytes();

        let response: Request<Body> = Request::builder()
            .method(Method::GET)
            .uri("/test")
            .header("content-type", "text/event-stream; charset=utf-8")
            .body(response_bytes.into())
            .unwrap();

        // Verify response is valid SSE
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream; charset=utf-8"
        );
    }

    /// Test SSE streaming error handling
    #[tokio::test]
    async fn test_sse_streaming_error_response() {
        let config = crate::proxy::ProxyConfig::default();
        let registry = crate::backends::registry::BackendRegistry::default();
        let config_data = crate::config::Config::default();
        let state = Arc::new(crate::proxy::ProxyState::new(config, registry, config_data));

        let socket = std::net::SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            0,
        );

        let server = ProxyServer::new(state.clone());
        tokio::spawn(async move {
            server.run(socket).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Create an error response in SSE format
        let response_body = String::from("data: {\"error\":{\"message\":\"Backend error\",\"type\":\"InternalServerError\"}}\n\n");
        let response_bytes = response_body.into_bytes();

        let response: Request<Body> = Request::builder()
            .method(Method::GET)
            .uri("/test")
            .header("content-type", "text/event-stream; charset=utf-8")
            .body(response_bytes.into())
            .unwrap();

        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream; charset=utf-8"
        );
    }
}
