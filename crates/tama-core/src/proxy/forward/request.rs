use super::headers::{filter_request_headers, strip_response_headers};
use super::json::rewrite_json_model_name;
use super::langfuse::{
    compute_energy_cost, extract_langfuse_headers, extract_request_fields, extract_timings,
    extract_usage, get_gpu_power_watts, parse_sse_accumulated, LangfuseTelemetry,
};
use super::sse::process_sse_line;
use crate::proxy::{api_keys::AuthSubject, ProxyState};
use axum::{body::Body, http::request::Parts, response::IntoResponse};
use bytes::{Bytes, BytesMut};
use futures_util::stream::StreamExt;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

pub async fn forward_request(
    state: &Arc<ProxyState>,
    backend_name: &str,
    parts: &Parts,
    body_bytes: &[u8],
    model_name: Option<&str>,
) -> axum::response::Response {
    state
        .metrics
        .counters
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // There used to be a per-backend failure counter map here; with it
    // gone (plan-193 T5c) there is no failure memory, so a crashed or
    // repeatedly-failing backend surfaces as a connection error below and
    // the tamad's own restart budget (`budget_exhausted` → HTTP 503)
    // bounds it.

    // Flip: the routing target comes from the live model rows (plan-193 T4),
    // not a local state cache. Offline host → no row → "not loaded".
    let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
    let backend_url = match live.row(backend_name) {
        Some(r) if !r.endpoint.is_empty() => r.endpoint.clone(),
        _ => {
            info!("No backend URL for model '{}' (not loaded?)", backend_name);
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                axum::response::Json(serde_json::json!({
                    "error": {
                        "message": format!("Model '{}' is not loaded", backend_name),
                        "type": "BackendUrlError"
                    }
                })),
            )
                .into_response();
        }
    };

    // Combine backend_url with the request path and query
    let path_and_query = match parts.uri.path_and_query() {
        Some(pq) => pq,
        None => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::response::Json(serde_json::json!({
                    "error": {
                        "message": "Invalid request URI",
                        "type": "BadRequestError"
                    }
                })),
            )
                .into_response();
        }
    };

    let (path, query) = path_and_query
        .as_str()
        .split_once('?')
        .unwrap_or((path_and_query.as_str(), ""));

    let target_uri = format!("{}{}", backend_url, path);

    // Resolve GPU device from model config using backend_name (the correct HashMap key).
    // Clone to own the value — can't borrow from temporary RwLockReadGuard.
    let gpu_info: String = state
        .registry
        .model_configs
        .read()
        .await
        .get(backend_name)
        .and_then(|mc| mc.gpu_device.clone())
        .unwrap_or_else(|| "default".to_string());

    // plan-195 emission convention: per-request forward noise is DEBUG —
    // the 50k-row retention window must not be burned by one line per
    // proxied request.
    debug!(gpu = %gpu_info, "Forwarding request to: {}", target_uri);

    let method = parts.method.clone();

    let headers = filter_request_headers(&parts.headers);

    // Capture Langfuse telemetry context before sending the request.
    // extract_request_fields must be called here — body_bytes is a reference to the
    // original request body and remains valid after .send(). It is shadowed later by
    // response.body() inside the non-streaming else branch.
    let langfuse_headers = extract_langfuse_headers(&parts.headers);

    // Extract user_id from auth subject (AuthSubject in request extensions).
    // Used as fallback for Langfuse when no langfuse_trace_user_id header is present.
    let auth_subject: Option<AuthSubject> = parts.extensions.get::<AuthSubject>().cloned();
    let auth_user_id: Option<String> = match auth_subject {
        Some(AuthSubject::User { username }) => Some(username),
        Some(AuthSubject::Key { key_id, .. }) => {
            // DB lookup for key name (async Postgres pool).
            // Only done when langfuse is enabled (checked via langfuse_cfg below,
            // but we always resolve here to keep the logic simple).
            crate::proxy::api_keys::ApiKeyStore::new(state.db_pool())
                .get_key_name(key_id)
                .await
                .ok()
                .flatten()
        }
        None => None,
    };

    let telemetry_start = Instant::now();

    // Read langfuse config once — reused for body injection (streaming) and
    // telemetry collection (both streaming and non-streaming paths).
    let langfuse_cfg = state.config.read().await.langfuse.clone();

    // Extract request fields from body (before body_bytes is shadowed by response).
    let langfuse_req_fields = extract_request_fields(body_bytes).unwrap_or_default();

    let mut query_string = query.to_string();
    if !query_string.is_empty() {
        query_string = format!("?{}", query_string);
    }

    // Rewrite model name (alias → resolved) and optionally inject langfuse
    // stream_options — single parse-and-serialize to avoid corrupting large bodies.
    let body_to_send = if let Ok(mut body) = serde_json::from_slice::<serde_json::Value>(body_bytes)
    {
        // ADR-0009: no backend accepts "off" as reasoning_effort — the
        // ecosystem off-word is "none" (vLLM 400s on "off"; llama.cpp would
        // silently leave thinking on). Normalize for all clients.
        // The modified flag is intentionally discarded — this branch
        // re-serializes the body unconditionally, so zero-copy pass-through
        // is not a concern here (unlike the remote path in chat.rs).
        let _ = normalize_reasoning_effort_body(&mut body);
        if let Some(obj) = body.as_object_mut() {
            // Rewrite model field to resolved name (for alias support)
            if let Some(name) = model_name {
                if !name.is_empty() {
                    obj.insert("model".to_string(), serde_json::json!(name.to_string()));
                }
            }
            // Inject stream_options for langfuse (streaming chat only)
            if langfuse_cfg.enabled {
                let is_streaming_chat = parts.uri.path().ends_with("/chat/completions")
                    && obj.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
                if is_streaming_chat {
                    let stream_opts = obj
                        .entry("stream_options")
                        .or_insert_with(|| serde_json::json!({}));
                    if let Some(opts) = stream_opts.as_object_mut() {
                        opts.insert("include_usage".to_string(), serde_json::json!(true));
                    }
                }
            }
        }
        serde_json::to_vec(&body).unwrap_or_else(|_| body_bytes.to_vec())
    } else {
        body_bytes.to_vec()
    };

    match state
        .client
        .request(method, format!("{}{}", target_uri, query_string))
        .headers(headers)
        .body(body_to_send)
        .send()
        .await
    {
        Ok(response) => {
            let status = response.status();
            if status.is_success() {
                state
                    .metrics
                    .counters
                    .successful_requests
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // plan-193 T5c: no consecutive-failure counter to reset (mirror gone).
            } else {
                state
                    .metrics
                    .counters
                    .failed_requests
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // (Plan-193 T5c: the per-backend failure bookkeeping that used to
                // live on the deleted state cache is gone with it.)
            }

            let mut builder = axum::response::Response::builder().status(status);

            for (key, value) in strip_response_headers(response.headers()) {
                builder = builder.header(&key, value);
            }

            // Check if this is a streaming response
            let is_streaming = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(|ct| ct.contains("text/event-stream"))
                .unwrap_or(false);

            // Langfuse config already read once above — reuse it here.
            let capture_streaming = langfuse_cfg.enabled && langfuse_cfg.capture_streaming;
            let langfuse_client = state.langfuse_client.read().await.clone();

            let body = if is_streaming {
                // Streaming response — rewrite the model name in each SSE chunk.
                // Uses unfold to own the partial-line buffer across chunks (Send-safe).
                // When Langfuse streaming capture is enabled, tee raw bytes via mpsc
                // for background accumulation and telemetry reporting.
                let model_name: Option<String> = model_name.map(|s| s.to_string());
                let byte_stream = response.bytes_stream();

                // Channel for tee'd bytes — None when capture disabled.
                let (tx, rx) = if capture_streaming {
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
                    (Some(tx), Some(rx))
                } else {
                    (None, None)
                };

                // Spawn background accumulation + reporting (only if capture enabled).
                if let Some(mut rx) = rx {
                    let max_bytes = langfuse_cfg.telemetry_max_bytes;
                    let (trace_id, user_id, session_id, metadata, tags) = langfuse_headers.clone();
                    let auth_user_id = auth_user_id.clone();
                    let (req_model, input, model_params) = langfuse_req_fields.clone();
                    let start_time = telemetry_start;

                    tokio::spawn(async move {
                        let mut buf = BytesMut::new();
                        let mut total_bytes = 0usize;
                        while let Some(chunk) = rx.recv().await {
                            if total_bytes + chunk.len() <= max_bytes {
                                buf.extend_from_slice(&chunk);
                                total_bytes += chunk.len();
                            }
                            // Keep draining the channel even if over limit (must consume all)
                        }
                        let raw = String::from_utf8_lossy(&buf).into_owned();
                        let (content, usage, timings) = parse_sse_accumulated(&raw);

                        if let Some(client) = langfuse_client {
                            let telemetry = LangfuseTelemetry {
                                model: req_model,
                                input: if langfuse_cfg.capture_input {
                                    input
                                } else {
                                    None
                                },
                                model_params,
                                output: if langfuse_cfg.capture_output {
                                    content
                                } else {
                                    None
                                },
                                usage,
                                timings,
                                start_time,
                                end_time: Some(std::time::Instant::now()),
                                trace_id,
                                user_id: user_id.or(auth_user_id),
                                session_id,
                                metadata,
                                tags,
                                energy_cost: None,
                                energy_wh: None,
                                gpu_watts: None,
                            };
                            client.report_generation(telemetry).await;
                        }
                    });
                }

                let transformed_stream = futures_util::stream::unfold(
                    (byte_stream, String::new()),
                    move |(mut stream, mut line_buf)| {
                        let model_name = model_name.clone();
                        let tx = tx.clone(); // Option<UnboundedSender<Bytes>> clone for closure
                        async move {
                            let chunk_result = stream.next().await?;
                            let result: Result<Bytes, reqwest::Error> = match chunk_result {
                                Ok(chunk) => {
                                    // Tee: send clone to background accumulator (if channel active).
                                    if let Some(ref sender) = tx {
                                        let _ = sender.send(chunk.clone());
                                    }

                                    let chunk_str = String::from_utf8_lossy(&chunk);
                                    let mut out = String::new();

                                    for ch in chunk_str.chars() {
                                        line_buf.push(ch);
                                        if ch == '\n' {
                                            let line = line_buf.clone();
                                            line_buf.clear();
                                            process_sse_line(
                                                &line,
                                                model_name.as_deref(),
                                                &mut out,
                                            );
                                        }
                                    }

                                    Ok(Bytes::from(out.into_bytes()))
                                }
                                Err(e) => Err(e),
                            };
                            Some((result, (stream, line_buf)))
                        }
                    },
                );
                Body::from_stream(transformed_stream)
            } else {
                // Non-streaming response - parse, rewrite, and re-serialize
                let body_bytes = match response.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("Failed to read backend response body: {}", e);
                        return (
                            axum::http::StatusCode::BAD_GATEWAY,
                            axum::response::Json(serde_json::json!({
                                "error": {
                                    "message": "Failed to read backend response".to_string(),
                                    "type": "BadGatewayError"
                                }
                            })),
                        )
                            .into_response();
                    }
                };
                // Only attempt JSON rewrite if content is valid JSON
                let new_body = if let Ok(parsed) = serde_json::from_slice::<JsonValue>(&body_bytes)
                {
                    // Collect Langfuse telemetry (non-streaming path) — fire-and-forget.
                    // MUST be before rewrite_json_model_name which consumes `parsed`.
                    {
                        if langfuse_cfg.enabled {
                            let langfuse_client = state.langfuse_client.read().await.clone();

                            // Use langfuse_req_fields captured before the send (body_bytes is shadowed here by response body)
                            let (req_model, input, model_params) = langfuse_req_fields.clone();

                            // Extract response fields from &parsed (borrow — parsed still owned)
                            let usage = extract_usage(&parsed);
                            let timings = extract_timings(&parsed);

                            // Extract output (completion text)
                            let output = if langfuse_cfg.capture_output {
                                // Chat completions: choices[0].message.content
                                parsed
                                    .get("choices")
                                    .and_then(|c| c.as_array())
                                    .and_then(|c| c.first())
                                    .and_then(|c| c.get("message"))
                                    .and_then(|m| m.get("content"))
                                    .and_then(|c| c.as_str())
                                    .map(|s| s.to_string())
                                    // Completions (non-chat): choices[0].text
                                    .or_else(|| {
                                        parsed
                                            .get("choices")
                                            .and_then(|c| c.as_array())
                                            .and_then(|c| c.first())
                                            .and_then(|c| c.get("text"))
                                            .and_then(|t| t.as_str())
                                            .map(|s| s.to_string())
                                    })
                            } else {
                                None
                            };

                            // Compute energy cost (best-effort — uses first GPU's power_w)
                            let (energy_cost, energy_wh, gpu_watts) = {
                                let metrics = state.metrics.system_metrics_snapshot().await;
                                let power_w = get_gpu_power_watts(&metrics);
                                if let (Some(pw), Some(t)) = (power_w, &timings) {
                                    match compute_energy_cost(
                                        pw,
                                        t.prompt_ms,
                                        t.predicted_ms,
                                        langfuse_cfg.electricity_price_per_kwh,
                                    ) {
                                        Some((wh, cost)) => (Some(cost), Some(wh), Some(pw)),
                                        None => (None, None, None),
                                    }
                                } else {
                                    (None, None, None)
                                }
                            };

                            // Build LangfuseTelemetry
                            let (trace_id, user_id, session_id, metadata, tags) =
                                langfuse_headers.clone();
                            let auth_user_id = auth_user_id.clone();
                            let telemetry = LangfuseTelemetry {
                                model: req_model,
                                input: if langfuse_cfg.capture_input {
                                    input
                                } else {
                                    None
                                },
                                model_params,
                                output,
                                usage,
                                timings,
                                start_time: telemetry_start,
                                end_time: Some(Instant::now()),
                                trace_id,
                                user_id: user_id.or(auth_user_id),
                                session_id,
                                metadata,
                                tags,
                                energy_cost,
                                energy_wh,
                                gpu_watts,
                            };

                            // Spawn background reporting task
                            if let Some(client) = langfuse_client {
                                tokio::spawn(async move {
                                    client.report_generation(telemetry).await;
                                });
                            }
                        }
                    }

                    let rewritten = rewrite_json_model_name(parsed, model_name);
                    serde_json::to_vec(&rewritten).unwrap_or(body_bytes.to_vec())
                } else {
                    // Not JSON, pass through unchanged
                    body_bytes.to_vec()
                };
                Body::from(new_body)
            };

            match builder.body(body) {
                Ok(resp) => resp.into_response(),
                Err(e) => {
                    tracing::error!("Failed to build response body: {}", e);
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        axum::response::Json(serde_json::json!({
                            "error": {
                                "message": "Internal error building response",
                                "type": "InternalError"
                            }
                        })),
                    )
                        .into_response()
                }
            }
        }
        Err(e) => {
            state
                .metrics
                .counters
                .failed_requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // The proxy cannot inspect the (remote) backend process
            // (ADR-0010). A connection-level failure (refused/reset) or a
            // timeout means the backend is not accepting traffic — clean up
            // immediately instead of letting the circuit breaker accumulate
            // failures; auto-load re-establishes it. Any other
            // error (the backend answered, e.g. 5xx) is a transient failure
            // counted as a transient failure.
            let backend_unreachable = e.is_connect() || e.is_timeout();

            if backend_unreachable {
                info!(
                    "Backend for backend '{}' is unreachable (request error = {}), clearing stale inference stats",
                    backend_name, e
                );
                state.metrics.modify_inference_stats(|map| {
                    map.remove(backend_name);
                });
            } else {
                // Process is alive — this is a transient error (timeout, busy,
                // etc.). There is no per-backend circuit-breaker counter to
                // bump now that the mirror is gone (plan-193 T5c).
            }

            info!("Failed to forward request: {}", e);
            (
                axum::http::StatusCode::BAD_GATEWAY,
                axum::response::Json(serde_json::json!({
                    "error": {
                        "message": format!("Backend error: {}", e),
                        "type": "BadGatewayError"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Rewrite `reasoning_effort: "off"` → `"none"` (ADR-0009).
///
/// No backend accepts `"off"` as `reasoning_effort` — the ecosystem off-word
/// is `"none"` (vLLM 400s on `"off"`; llama.cpp would silently leave thinking
/// on). All other values pass through verbatim.
///
/// Returns true if the body was modified.
pub(crate) fn normalize_reasoning_effort_body(body: &mut serde_json::Value) -> bool {
    if body
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        == Some("off")
    {
        body["reasoning_effort"] = serde_json::json!("none");
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `reasoning_effort: "off"` is rewritten to `"none"` (ADR-0009).
    #[test]
    fn test_normalize_reasoning_effort_off_to_none() {
        let mut body = serde_json::json!({"model": "x", "reasoning_effort": "off"});
        assert!(normalize_reasoning_effort_body(&mut body));
        assert_eq!(body["reasoning_effort"], "none");
    }

    /// Other string values pass through verbatim.
    #[test]
    fn test_normalize_reasoning_effort_low_unchanged() {
        let mut body = serde_json::json!({"model": "x", "reasoning_effort": "low"});
        assert!(!normalize_reasoning_effort_body(&mut body));
        assert_eq!(body["reasoning_effort"], "low");
    }

    /// Absent field: body is not modified.
    #[test]
    fn test_normalize_reasoning_effort_absent_unchanged() {
        let mut body = serde_json::json!({"model": "x", "stream": true});
        assert!(!normalize_reasoning_effort_body(&mut body));
        assert!(body.get("reasoning_effort").is_none());
    }

    /// Non-string value: body is not modified.
    #[test]
    fn test_normalize_reasoning_effort_non_string_unchanged() {
        let mut body = serde_json::json!({"model": "x", "reasoning_effort": 42});
        assert!(!normalize_reasoning_effort_body(&mut body));
        assert_eq!(body["reasoning_effort"], 42);
    }
}
