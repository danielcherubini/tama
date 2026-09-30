//! Scope-based authorization middleware for API keys.
//!
//! After authentication, this middleware checks that API key subjects
//! have the required scope for the requested path and HTTP method.
//! OAuth2-authenticated users bypass all scope checks.

use axum::{
    body::Body,
    extract::Request,
    http::{header, Method, StatusCode},
    middleware::Next,
    response::Response,
};

use crate::proxy::api_keys::{AuthSubject, Scope};

/// Determine the required scope for a given path and HTTP method.
///
/// Returns `None` for paths that don't require scope checking
/// (e.g., `/health`, `/metrics`, forwarded llama.cpp routes).
///
/// Route mapping:
/// - `/v1/*` → `Inference` scope
/// - `/model/info` (GET) → `Inference` scope (LiteLLM-compatible twin of `/v1/model/info`)
/// - `GET/HEAD/DELETE /tama/v1/*` → `ManagementRead` scope
/// - `POST/PUT/PATCH /tama/v1/*` → `ManagementWrite` scope
pub fn required_scope(path: &str, method: &Method) -> Option<Scope> {
    if path.starts_with("/v1/") || path == "/v1" {
        return Some(Scope::Inference);
    }
    if path == "/model/info" {
        // LiteLLM-compatible twin of `/v1/model/info` — same handler, same data,
        // so it must carry the same authorization contract.
        return Some(Scope::Inference);
    }
    if path.starts_with("/tama/v1/") {
        return if matches!(
            *method,
            Method::POST | Method::PUT | Method::PATCH | Method::DELETE
        ) {
            Some(Scope::ManagementWrite)
        } else {
            Some(Scope::ManagementRead)
        };
    }
    None
}

/// Scope-based authorization middleware.
///
/// 1. Extracts `AuthSubject` from request extensions (set by `auth_middleware`).
/// 2. If `AuthSubject::User` → bypasses (full access).
/// 3. If `AuthSubject::Key` → checks scopes against path + method.
/// 4. If no `AuthSubject` (skip path) → passes through.
///
/// Missing scope returns 403 with a JSON body containing the required scope.
pub async fn scope_middleware(req: Request, next: Next) -> Response {
    // Extract AuthSubject from request extensions
    let subject = req.extensions().get::<AuthSubject>().cloned();
    #[cfg(test)]
    eprintln!(
        "scope_middleware: path={} method={} subject={:?}",
        req.uri().path(),
        req.method(),
        subject.as_ref().map(|s| match s {
            AuthSubject::User { username } => format!("User({username})"),
            AuthSubject::Key { key_id, scopes } => format!("Key({key_id}, {:?})", scopes),
        })
    );

    // No auth subject (skip path) → pass through
    let Some(subject) = subject else {
        return next.run(req).await;
    };

    // OAuth2 users bypass all scope checks
    if matches!(subject, AuthSubject::User { .. }) {
        return next.run(req).await;
    }

    // API key — check scopes
    let AuthSubject::Key { key_id: _, scopes } = subject else {
        unreachable!("User case already handled above")
    };

    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let required = match required_scope(&path, &method) {
        Some(s) => s,
        None => return next.run(req).await, // No scope required for this path
    };

    // Check if the key has the required scope
    // management:write implies management:read
    let has_scope = match required {
        Scope::Inference => scopes.contains(&Scope::Inference),
        Scope::ManagementRead => {
            scopes.contains(&Scope::ManagementRead) || scopes.contains(&Scope::ManagementWrite)
        }
        Scope::ManagementWrite => scopes.contains(&Scope::ManagementWrite),
    };

    if !has_scope {
        return forbidden_response(&required);
    }

    next.run(req).await
}

/// Build a 403 JSON response with the required scope information.
fn forbidden_response(required: &Scope) -> Response {
    let body = serde_json::json!({
        "error": "forbidden",
        "message": format!("missing required scope: {}", scope_to_string(required)),
        "required_scope": scope_to_string(required),
    })
    .to_string();
    Response::builder()
        .status(StatusCode::FORBIDDEN)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("build forbidden response")
}

/// Convert a Scope enum to its string representation for error messages.
fn scope_to_string(scope: &Scope) -> String {
    match scope {
        Scope::Inference => "inference".to_string(),
        Scope::ManagementRead => "management:read".to_string(),
        Scope::ManagementWrite => "management:write".to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::post;
    use axum::{routing::get, Router};
    use std::sync::Arc;
    use tower::util::ServiceExt;

    async fn test_handler() -> &'static str {
        "ok"
    }

    fn make_app_with_subject(subject: AuthSubject) -> Router {
        Router::new()
            .route("/v1/models", get(test_handler))
            .route("/v1/chat/completions", post(post_handler).get(test_handler))
            .route("/tama/v1/models", get(test_handler))
            .route("/tama/v1/models/test/load", post(post_handler))
            .route("/tama/v1/models/test", get(test_handler))
            .route("/health", get(test_handler))
            .layer(axum::middleware::from_fn(scope_middleware))
            .with_state(Arc::new(()))
            .layer(axum::middleware::from_fn(
                move |mut req: Request<Body>, next: Next| {
                    let subject = subject.clone();
                    async move {
                        req.extensions_mut().insert(subject.clone());
                        next.run(req).await
                    }
                },
            ))
    }

    async fn post_handler() -> &'static str {
        "created"
    }

    /// Test that a key with only `management:read` scope cannot access `/v1/*` routes.
    #[tokio::test]
    async fn test_scope_middleware_key_without_inference_rejected() {
        let subject = AuthSubject::Key {
            key_id: 1,
            scopes: vec![Scope::ManagementRead],
        };
        let app = make_app_with_subject(subject);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("forbidden"));
        assert!(body_str.contains("inference"));
    }

    /// Test that `AuthSubject::User` bypasses all scope checks.
    #[tokio::test]
    async fn test_scope_middleware_user_bypasses() {
        let subject = AuthSubject::User {
            username: "admin".to_string(),
        };
        let app = make_app_with_subject(subject);

        // User should be able to access any route
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Management write route
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tama/v1/models/test/load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Test that a key with `inference` scope can access `/v1/*` routes.
    #[tokio::test]
    async fn test_scope_middleware_key_with_inference_passes() {
        let subject = AuthSubject::Key {
            key_id: 1,
            scopes: vec![Scope::Inference],
        };
        let app = make_app_with_subject(subject);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // GET /v1/models
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Test that a key with `management:read` scope can GET `/tama/v1/*` routes.
    #[tokio::test]
    async fn test_scope_middleware_key_management_read_get_passes() {
        let subject = AuthSubject::Key {
            key_id: 1,
            scopes: vec![Scope::ManagementRead],
        };
        let app = make_app_with_subject(subject);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/tama/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // GET /tama/v1/models/:id
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/tama/v1/models/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Test that a key with only `management:read` scope cannot POST to `/tama/v1/*`.
    #[tokio::test]
    async fn test_scope_middleware_key_management_read_post_rejected() {
        let subject = AuthSubject::Key {
            key_id: 1,
            scopes: vec![Scope::ManagementRead],
        };
        let app = make_app_with_subject(subject);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tama/v1/models/test/load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("forbidden"));
        assert!(body_str.contains("management:write"));
    }

    /// Test that a key with `management:write` scope can GET `/tama/v1/*` (write implies read).
    #[tokio::test]
    async fn test_scope_middleware_key_management_write_get_passes() {
        let subject = AuthSubject::Key {
            key_id: 1,
            scopes: vec![Scope::ManagementWrite],
        };
        let app = make_app_with_subject(subject);

        // GET should pass with management:write
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/tama/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // POST should also pass with management:write
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tama/v1/models/test/load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Test that requests without an AuthSubject pass through (skip paths).
    #[tokio::test]
    async fn test_scope_middleware_no_subject_passes() {
        let app = Router::new()
            .route("/v1/models", get(test_handler))
            .route("/health", get(test_handler))
            .layer(axum::middleware::from_fn(scope_middleware))
            .with_state(Arc::new(()));

        // No AuthSubject inserted — should pass through
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── required_scope helper tests ───────────────────────────────────────

    #[test]
    fn test_required_scope_v1_audio_returns_inference() {
        assert_eq!(
            required_scope("/v1/audio/speech", &Method::POST),
            Some(Scope::Inference)
        );
    }

    #[test]
    fn test_required_scope_v1_compaction_returns_inference() {
        assert_eq!(
            required_scope("/v1/compaction", &Method::POST),
            Some(Scope::Inference)
        );
    }

    #[test]
    fn test_required_scope_v1_opencode_returns_inference() {
        assert_eq!(
            required_scope("/v1/opencode/models", &Method::GET),
            Some(Scope::Inference)
        );
    }

    #[test]
    fn test_required_scope_v1_model_info_returns_inference() {
        assert_eq!(
            required_scope("/v1/model/info", &Method::GET),
            Some(Scope::Inference)
        );
    }

    #[test]
    fn test_required_scope_model_info_returns_inference() {
        // The non-prefixed /model/info is the LiteLLM-compatible twin of
        // /v1/model/info (same handler, same data) and must require the
        // same Inference scope; the auth middleware still gates it.
        assert_eq!(
            required_scope("/model/info", &Method::GET),
            Some(Scope::Inference)
        );
    }

    #[test]
    fn test_required_scope_tama_management_read_get() {
        assert_eq!(
            required_scope("/tama/v1/models", &Method::GET),
            Some(Scope::ManagementRead)
        );
    }

    #[test]
    fn test_required_scope_tama_management_read_head() {
        assert_eq!(
            required_scope("/tama/v1/system/health", &Method::HEAD),
            Some(Scope::ManagementRead)
        );
    }

    #[test]
    fn test_required_scope_tama_management_write_post() {
        assert_eq!(
            required_scope("/tama/v1/models/test/load", &Method::POST),
            Some(Scope::ManagementWrite)
        );
    }

    #[test]
    fn test_required_scope_tama_management_write_put() {
        assert_eq!(
            required_scope("/tama/v1/models/test", &Method::PUT),
            Some(Scope::ManagementWrite)
        );
    }

    #[test]
    fn test_required_scope_tama_management_write_delete() {
        assert_eq!(
            required_scope("/tama/v1/models/test", &Method::DELETE),
            Some(Scope::ManagementWrite)
        );
    }

    #[test]
    fn test_required_scope_tama_management_write_patch() {
        assert_eq!(
            required_scope("/tama/v1/models/test", &Method::PATCH),
            Some(Scope::ManagementWrite)
        );
    }

    #[test]
    fn test_required_scope_health_returns_none() {
        assert_eq!(required_scope("/health", &Method::GET), None);
    }

    #[test]
    fn test_required_scope_metrics_returns_none() {
        assert_eq!(required_scope("/metrics", &Method::GET), None);
    }

    #[test]
    fn test_required_scope_forwarded_route_returns_none() {
        assert_eq!(required_scope("/completion", &Method::POST), None);
    }

    #[test]
    fn test_required_scope_tokenize_returns_none() {
        assert_eq!(required_scope("/tokenize", &Method::POST), None);
    }

    #[test]
    fn test_required_scope_v1_exact_match() {
        assert_eq!(required_scope("/v1", &Method::POST), Some(Scope::Inference));
    }

    // ── Integration tests: full auth → scope flow ─────────────────────────

    /// Helper: create a Postgres schema with a seeded API key.
    /// Returns the proxy state, the schema guard (kept alive), and the raw key.
    async fn make_test_db(
        scopes: &[Scope],
    ) -> (
        std::sync::Arc<crate::proxy::ProxyState>,
        crate::testing::postgres::SchemaGuard,
        String,
    ) {
        let guard = crate::testing::postgres::with_schema().await;

        let key = crate::proxy::api_keys::generate_key();
        crate::proxy::api_keys::ApiKeyStore::new(Arc::new(guard.pool.clone()))
            .create_key("test-key", &key, scopes, "admin", None)
            .await
            .unwrap();

        let config = crate::config::Config {
            proxy: crate::config::ProxyConfig {
                authenticator_skip_paths: vec![
                    "/health".to_string(),
                    "/metrics".to_string(),
                    "/login".to_string(),
                    "/login/callback".to_string(),
                    "/login/error".to_string(),
                ],
                api_keys_enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let proxy_state = Arc::new(crate::proxy::ProxyState::new(
            config,
            None,
            Arc::new(guard.pool.clone()),
        ));

        (proxy_state, guard, key)
    }

    /// Integration test: auth with valid key → scope check → handler passes.
    /// Key with `inference` scope → POST /v1/chat/completions → 200
    #[tokio::test]
    async fn test_full_auth_then_scope_flow_inference_key() {
        let (state, _guard, key) = make_test_db(&[Scope::Inference]).await;

        let app = Router::new()
            .route("/v1/chat/completions", post(post_handler).get(test_handler))
            .route("/health", get(test_handler))
            .route("/tama/v1/models", get(test_handler))
            .route("/tama/v1/models/test/load", post(post_handler))
            .layer(axum::middleware::from_fn(scope_middleware))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                crate::proxy::auth::auth_middleware,
            ))
            .with_state(state);

        // Valid inference key → POST /v1/chat/completions should pass
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", key))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Same key → POST /tama/v1/models/test/load should be 403 (no management scope)
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tama/v1/models/test/load")
                    .header("Authorization", format!("Bearer {}", key))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("forbidden"));
        assert!(body_str.contains("management:write"));
    }

    /// Integration test: auth with management:write key → scope check → handler passes.
    /// Key with `management:write` scope → POST /tama/v1/models/test/load → 200
    #[tokio::test]
    async fn test_full_auth_then_scope_flow_management_write_key() {
        let (state, _guard, key) = make_test_db(&[Scope::ManagementWrite]).await;

        let app = Router::new()
            .route("/v1/chat/completions", post(post_handler).get(test_handler))
            .route("/health", get(test_handler))
            .route("/tama/v1/models", get(test_handler))
            .route("/tama/v1/models/test/load", post(post_handler))
            .layer(axum::middleware::from_fn(scope_middleware))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                crate::proxy::auth::auth_middleware,
            ))
            .with_state(state);

        // Management write key → GET /tama/v1/models should pass (write implies read)
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/tama/v1/models")
                    .header("Authorization", format!("Bearer {}", key))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Management write key → POST /tama/v1/models/test/load should pass
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tama/v1/models/test/load")
                    .header("Authorization", format!("Bearer {}", key))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Same key → POST /v1/chat/completions should be 403 (no inference scope)
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", key))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    /// Integration test: skip paths bypass both auth and scope middlewares.
    #[tokio::test]
    async fn test_full_auth_then_scope_flow_skip_paths() {
        let (state, _guard, _key) = make_test_db(&[Scope::Inference]).await;

        let app = Router::new()
            .route("/v1/chat/completions", post(post_handler).get(test_handler))
            .route("/health", get(test_handler))
            .route("/metrics", get(test_handler))
            .layer(axum::middleware::from_fn(scope_middleware))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                crate::proxy::auth::auth_middleware,
            ))
            .with_state(state);

        // /health should pass without any auth (skip path)
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // /metrics should pass without any auth (skip path)
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
