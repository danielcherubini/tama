//! Backend-specific log endpoint: GET /tama/v1/logs/:backend

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::Arc;

use crate::server::AppState;

/// Maximum number of lines to return (clamp for the `lines` query parameter).
pub const MAX_LINES: usize = 10_000;

/// Query parameters for GET /tama/v1/logs/:backend
#[derive(serde::Deserialize)]
pub struct BackendLogsQuery {
    /// Number of lines to return (default: 200)
    #[serde(default = "default_lines")]
    pub lines: usize,
}

fn default_lines() -> usize {
    200
}

/// Validate a backend name for use in log file paths.
///
/// Returns `true` if the name is non-empty, ≤64 characters, and contains only
/// alphanumeric characters, underscores, or hyphens.
pub fn is_valid_backend_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// GET /tama/v1/logs/:backend — return the last N lines of a backend's log file.
pub async fn get_backend_logs(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(backend): axum::extract::Path<String>,
    axum::extract::Query(query): axum::extract::Query<BackendLogsQuery>,
) -> impl IntoResponse {
    // 1. Check logs_dir is configured
    let dir = match &state.logs_dir {
        Some(d) => d.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "logs_dir not configured"})),
            )
                .into_response()
        }
    };

    // 2. Validate backend name
    if !is_valid_backend_name(&backend) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid backend name"})),
        )
            .into_response();
    }

    // 3. Build log path
    let path = dir.join(format!("{}.log", backend));

    // 4. Check file existence
    if !path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("No logs found for '{}'", backend)})),
        )
            .into_response();
    }

    // 5. Read log lines using spawn_blocking
    let n = query.lines.min(MAX_LINES);
    let path_clone = path.clone();
    let lines =
        tokio::task::spawn_blocking(move || tama_core::logging::tail_lines(&path_clone, n)).await;

    match lines {
        Ok(Ok(result)) => Json(serde_json::json!({ "lines": result })).into_response(),
        Ok(Err(e)) => {
            tracing::warn!("Failed to read backend log {}: {}", path.display(), e);
            Json(serde_json::json!({ "lines": Vec::<String>::new() })).into_response()
        }
        Err(join_err) => {
            tracing::warn!(
                "Failed to read backend log {} (spawn_blocking): {}",
                path.display(),
                join_err
            );
            Json(serde_json::json!({ "lines": Vec::<String>::new() })).into_response()
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_backend_names() {
        // Valid names from spec
        assert!(is_valid_backend_name("llama_cpp"));
        assert!(is_valid_backend_name("ik_llama"));
        assert!(is_valid_backend_name("tts_kokoro"));
        assert!(is_valid_backend_name("custom-backend"));
        assert!(is_valid_backend_name("abc123"));
    }

    #[test]
    fn test_invalid_backend_names() {
        // Invalid names from spec
        assert!(!is_valid_backend_name(""));
        assert!(!is_valid_backend_name("../etc/passwd"));
        assert!(!is_valid_backend_name("../../logs"));
        assert!(!is_valid_backend_name("name with spaces"));
        assert!(!is_valid_backend_name("name/with/slashes"));
        // Note: "name..double" contains only valid characters (alphanumeric + dots)
        // but dots are NOT allowed, so this should be invalid
        assert!(!is_valid_backend_name("name..double"));
        assert!(!is_valid_backend_name(&"a".repeat(65))); // 65+ chars
        assert!(!is_valid_backend_name("name.with.dots"));
        assert!(!is_valid_backend_name("name\0null"));
        assert!(!is_valid_backend_name("UPPER CASE"));
    }

    #[test]
    fn test_max_lines_constant() {
        assert_eq!(MAX_LINES, 10_000);
    }
}
