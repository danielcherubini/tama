#![cfg(feature = "ssr")]

use std::collections::HashMap;
use std::sync::Arc;

/// Paths owned by the tama (web) route table under /tama/v1.
/// MAINTENANCE: when you add a /tama/v1 route to `crates/tama/src/router.rs`,
/// add its path here. This list is the tripwire that keeps core's proxy
/// router and this crate's management router disjoint (audit F33).
const TAMA_MANAGED_PATHS: &[&str] = &[
    "/tama/v1/system/capabilities",
    "/tama/v1/installations",
    "/tama/v1/installations/install",
    "/tama/v1/installations/:name/update",
    "/tama/v1/installations/:name",
    "/tama/v1/installations/:name/default-args",
    "/tama/v1/installations/:name/default-env",
    "/tama/v1/installations/:name/versions/:version",
    "/tama/v1/installations/check-updates",
    "/tama/v1/installations/:name/versions",
    "/tama/v1/installations/:name/activate",
    "/tama/v1/installations/:name/source",
    "/tama/v1/installations/:name/rename",
    "/tama/v1/installations/jobs/:id",
    "/tama/v1/installations/jobs/:id/events",
    "/tama/v1/installations/compaction",
    // Deprecated aliases for old /tama/v1/backends/* paths
    "/tama/v1/backends",
    "/tama/v1/backends/install",
    "/tama/v1/backends/:name/update",
    "/tama/v1/backends/:name",
    "/tama/v1/backends/:name/default-args",
    "/tama/v1/backends/:name/default-env",
    "/tama/v1/backends/:name/versions/:version",
    "/tama/v1/backends/check-updates",
    "/tama/v1/backends/:name/versions",
    "/tama/v1/backends/:name/activate",
    "/tama/v1/backends/:name/source",
    "/tama/v1/backends/:name/rename",
    "/tama/v1/backends/jobs/:id",
    "/tama/v1/backends/jobs/:id/events",
    "/tama/v1/backends/compaction",
    "/tama/v1/backup",
    "/tama/v1/restore/preview",
    "/tama/v1/restore",
    "/tama/v1/self-update/update",
    "/tama/v1/self-update/check",
    "/tama/v1/self-update/events",
    "/tama/v1/updates/check",
    "/tama/v1/updates/check/:item_type/:item_id",
    "/tama/v1/updates/events",
    "/tama/v1/updates/apply/backend/:name",
    "/tama/v1/updates/apply/model/:id",
    "/tama/v1/updates",
    "/tama/v1/config",
    "/tama/v1/config/structured",
    "/tama/v1/models",
    "/tama/v1/models/:id",
    "/tama/v1/models/:id/rename",
    "/tama/v1/models/:id/refresh",
    "/tama/v1/models/:id/verify",
    "/tama/v1/models/:id/quants/:quant_key",
    "/tama/v1/benchmarks/run",
    "/tama/v1/benchmarks/spec-run",
    "/tama/v1/benchmarks/mtp-run",
    "/tama/v1/benchmarks/jobs/:id",
    "/tama/v1/benchmarks/jobs/:id/events",
    "/tama/v1/benchmarks/history",
    "/tama/v1/benchmarks/history/:id",
    "/tama/v1/pulls/:job_id/cancel",
    "/tama/v1/pulls/active",
    "/tama/v1/pulls/history",
    "/tama/v1/pulls/events",
    "/tama/v1/aliases",
    "/tama/v1/aliases/:id",
    "/tama/v1/hf/*repo_id",
    "/tama/v1/docs",
    "/tama/v1/logs",
    "/tama/v1/logs/sources",
    "/tama/v1/logs/summary",
    "/tama/v1/logs/status",
    "/tama/v1/logs/stream",
    "/tama/v1/logs/events",
    "/tama/v1/logs/export",
    "/tama/v1/system/health",
    "/tama/v1/providers",
    "/tama/v1/providers/:name",
    "/tama/v1/tamads",
    "/tama/v1/tamads/:id",
    "/tama/v1/tamads/:id/health",
];

/// Core's proxy table and tama's management table must be disjoint —
/// a path in both is a shadow-route bug (audit F33).
#[test]
fn test_proxy_and_management_tables_are_disjoint() {
    const EXPECTED_TAMA_PATH_COUNT: usize = 79;
    assert_eq!(
        TAMA_MANAGED_PATHS.len(),
        EXPECTED_TAMA_PATH_COUNT,
        "tama-managed route count changed — update test if intentional"
    );

    let proxy_paths: std::collections::HashSet<&str> =
        tama_core::proxy::server::router::proxy_route_paths()
            .into_iter()
            .map(|(_, p)| p)
            .collect();

    const EXPECTED_PROXY_PATH_COUNT: usize = 33;
    assert_eq!(
        proxy_paths.len(),
        EXPECTED_PROXY_PATH_COUNT,
        "proxy route count changed unexpectedly — update test if intentional"
    );
    for path in TAMA_MANAGED_PATHS {
        assert!(
            !proxy_paths.contains(path),
            "route {path} exists in BOTH routers — pick exactly one owner"
        );
    }
}

/// The unified (production) app must serve API routes as API responses,
/// never as the SPA's index.html (the system/health shadow bug).
#[tokio::test]
async fn test_unified_app_serves_api_not_spa_html() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let config = tama_core::config::Config::default();
        let state = Arc::new(tama_core::proxy::ProxyState::new(
            config,
            None,
            tama_test_support::test_dummy_pool(),
        ));
        let web_state = Arc::new(tama_web::web_types::WebState {
            jobs: Some(Arc::new(tama_web::web_types::JobManager::new())),
            capabilities: None,
            update_checker: Arc::new(tama_core::updates::UpdateChecker::default()),
            binary_version: "test".to_string(),
            update_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upload_lock: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            db_pool: tama_test_support::test_dummy_pool(),
            log_filter: None,
            log_status: None,
            log_read: None,
            log_tail: None,
            log_events_tx: Arc::new(tokio::sync::Mutex::new(None)),
        });
        let web_routes = tama_web::router::build_web_routes(web_state);
        let server = tama_core::proxy::ProxyServer::new(state).await;
        let app = server.into_unified_router(web_routes).await;
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let client = reqwest::Client::new();

    for path in ["/tama/v1/system/health", "/tama/v1/logs", "/tama/v1/models"] {
        let resp = client
            .get(format!("http://{addr}{path}"))
            .send()
            .await
            .unwrap();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            !content_type.contains("text/html"),
            "GET {path} returned SPA HTML (shadowed by the web UI fallback); content-type: {content_type}"
        );
    }
}
