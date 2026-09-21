use super::*;
use futures_util::StreamExt;
use std::sync::Arc;

/// Seed a live `ready` wire row so `handle_metrics` (plan-193 T4: rows)
/// sees the mock backend as a Ready, routeable endpoint.
async fn seed_live_row(state: &Arc<crate::proxy::ProxyState>, model_id: &str, endpoint: &str) {
    use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
    let proc = crate::tamad::ProcessInfo {
        model_name: model_id.to_string(),
        provider_name: "llama_cpp".to_string(),
        pid: 1,
        alive: true,
        endpoint_url: endpoint.to_string(),
        status: "ready".to_string(),
        desired: true,
        restart_count: 0,
        max_restarts: 3,
        spec_accept_pct: None,
        spec_decoding_active: false,
        tps: None,
        prompt_tps: None,
        cache_hit_pct: None,
        last_obs_ms: None,
    };
    let stats = stats_full(1.5, vec![], vec![proc]);
    let pool = state.tamad_pool();
    pool.insert_raw_handle(
        model_id,
        Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
    )
    .await;
}

#[tokio::test]
async fn test_proxy_routes_exist() {
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        None,
        crate::db::pool::test_dummy_pool(),
    ));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    let server = ProxyServer::new(state.clone()).await;
    let app = server.into_router().await;
    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Test health endpoint
    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://{}/health", bound_addr))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Test models endpoint
    let response = client
        .get(format!("http://{}/v1/models", bound_addr))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Test status endpoint
    let response = client
        .get(format!("http://{}/status", bound_addr))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// Verify that /metrics returns Prometheus content type and tama: prefixed metrics.
#[tokio::test]
async fn test_metrics_returns_prometheus_format() {
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        None,
        crate::db::pool::test_dummy_pool(),
    ));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    let server = ProxyServer::new(state.clone()).await;
    let app = server.into_router().await;
    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://{}/metrics", bound_addr))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    // Check content type
    let content_type = response.headers().get("content-type").unwrap();
    assert!(
        content_type.to_str().unwrap().contains("text/plain"),
        "content type should be text/plain, got: {}",
        content_type.to_str().unwrap()
    );

    // Check body contains tama: prefixed metrics
    let body = response.text().await.unwrap();
    assert!(
        body.contains("tama:total_requests"),
        "missing tama:total_requests"
    );
    assert!(
        body.contains("tama:successful_requests"),
        "missing tama:successful_requests"
    );
    assert!(
        body.contains("tama:failed_requests"),
        "missing tama:failed_requests"
    );
    assert!(
        body.contains("tama:models_loaded"),
        "missing tama:models_loaded"
    );
    assert!(
        body.contains("tama:models_unloaded"),
        "missing tama:models_unloaded"
    );
    assert!(
        body.contains("tama:active_models"),
        "missing tama:active_models"
    );
    // Check Prometheus format markers
    assert!(body.contains("# HELP"), "missing # HELP lines");
    assert!(body.contains("# TYPE"), "missing # TYPE lines");
}

/// Verify that /metrics gracefully handles no backends (returns just Tama metrics).
#[tokio::test]
async fn test_metrics_no_backends_returns_tama_only() {
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        None,
        crate::db::pool::test_dummy_pool(),
    ));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    let server = ProxyServer::new(state.clone()).await;
    let app = server.into_router().await;
    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let body = client
        .get(format!("http://{}/metrics", bound_addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    // Should have Tama metrics with 0 active models
    assert!(
        body.contains("tama:active_models 0"),
        "should have 0 active models"
    );
    // Should NOT have any backend metrics (no llamacpp: lines)
    assert!(
        !body.contains("llamacpp:"),
        "should have no backend metrics"
    );
}

/// Verify that /metrics merges backend metrics correctly with {server} labels.
#[tokio::test]
async fn test_metrics_merges_backend_metrics() {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Start a mock backend server
    let mock_server = MockServer::start().await;

    // Mock the /metrics endpoint with a simple Prometheus metric
    let mock_body =
        "# HELP test:counter A test counter.\n# TYPE test:counter counter\ntest:counter 42\n";
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(mock_body))
        .mount(&mock_server)
        .await;

    let backend_url = mock_server.uri().to_string();

    // Create state and register the mock as a Ready backend
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        None,
        crate::db::pool::test_dummy_pool(),
    ));

    // (ready row seeded below; the mirror insert is gone - plan-193 T5c)
    seed_live_row(&state, "test-model", backend_url.as_str()).await;

    // Start the proxy server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    let server = ProxyServer::new(state.clone()).await;
    let app = server.into_router().await;
    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Fetch /metrics and verify the merged output
    let client = reqwest::Client::new();
    let body = client
        .get(format!("http://{}/metrics", bound_addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    // Should contain Tama metrics
    assert!(body.contains("tama:total_requests"), "missing Tama metrics");
    assert!(
        body.contains("tama:active_models 1"),
        "should have 1 active model, got: {}",
        body
    );

    // Should contain backend metric with server label
    assert!(
        body.contains("test:counter{server=\"test-model\"} 42"),
        "backend metric should have server label, got: {}",
        body
    );
}

/// `test_metrics_task_persists_to_db` now lives in
/// `crates/tama-core/tests/metrics_collector.rs` on the Postgres harness
/// (plan-190 Task 4 — system metrics persist to Postgres).

#[tokio::test]
async fn test_metrics_task_broadcasts_samples() {
    let tmp = tempfile::tempdir().unwrap();
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        Some(tmp.path().to_path_buf()),
        crate::db::pool::test_dummy_pool(),
    ));

    let mut rx = state.metrics.metrics_tx.subscribe();

    let _server = ProxyServer::new(state.clone()).await;

    let result = tokio::time::timeout(std::time::Duration::from_secs(4), rx.recv()).await;
    assert!(
        result.is_ok(),
        "Expected to receive a MetricsSnapshot within 4s, but timeout occurred"
    );
    let snapshot = result.unwrap().unwrap();
    assert!(
        !snapshot.buckets.is_empty(),
        "Expected at least one bucket in the broadcast"
    );
    let sample = &snapshot.buckets[0];
    assert!(sample.ts_unix_ms > 0, "ts_unix_ms should be positive");
    assert!(
        sample.cpu_usage_pct >= 0.0,
        "cpu_usage_pct should be non-negative"
    );
    assert!(sample.ram_total_mib > 0, "ram_total_mib should be positive");
}

#[tokio::test]
async fn test_metric_sample_broadcast_populates_models_field() {
    use crate::config::ModelConfig;
    use std::collections::BTreeMap;

    let tmp = tempfile::tempdir().unwrap();

    // Build a Config with exactly one known model so the assertions are
    // deterministic. We clear the default fixtures shipped by
    // `Config::default()` first.
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        Some(tmp.path().to_path_buf()),
        crate::db::pool::test_dummy_pool(),
    ));

    // Manually insert a model into model_configs since it's no longer in Config
    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "alpha".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                args: vec![],
                sampling: None,
                model: None,
                quant: None,

                mmproj: None,
                port: None,
                health_check: None,
                enabled: true,
                context_length: None,
                num_parallel: Some(1),
                kv_unified: false,
                profile: None,
                api_name: None,
                gpu_layers: None,
                cache_type_k: None,
                cache_type_v: None,
                quants: BTreeMap::new(),
                modalities: None,
                display_name: None,
                db_id: None,
                ..Default::default()
            },
        );
    }

    // Subscribe BEFORE starting the server so we don't miss the first tick.
    let mut rx = state.metrics.metrics_tx.subscribe();

    let _server = ProxyServer::new(state.clone()).await;

    let snapshot = tokio::time::timeout(std::time::Duration::from_secs(4), rx.recv())
        .await
        .expect("Expected to receive a MetricsSnapshot within 4s, but timeout occurred")
        .expect("metrics_tx channel closed before any sample was broadcast");

    // The metrics loop must populate `MetricCurrent.models` from
    // `ProxyState::collect_model_statuses`, which reflects the current
    // configuration.
    assert!(
        !snapshot.buckets.is_empty(),
        "Expected at least one bucket in the broadcast"
    );
    let sample = &snapshot.current;
    assert_eq!(
        sample.models.len(),
        1,
        "Expected exactly one model in current.models, got: {:?}",
        sample.models
    );
    assert_eq!(sample.models[0].id, "alpha");
    assert_eq!(sample.models[0].backend, "llama_cpp");
    assert!(
        !matches!(
            sample.models[0].state,
            crate::gpu::ModelState::Ready
        ),
        "Expected the configured model to be reported as not ready since no backend was started, got: {:?}",
        sample.models[0]
    );
    assert_eq!(
        sample.models_loaded, 0,
        "Expected models_loaded counter to be 0 when no model is loaded"
    );
}

#[tokio::test]
async fn test_system_metrics_stream_emits_samples() {
    use bytes::Bytes;

    let tmp = tempfile::tempdir().unwrap();
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        Some(tmp.path().to_path_buf()),
        crate::db::pool::test_dummy_pool(),
    ));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    let server = ProxyServer::new(state.clone()).await;
    let app = server.into_router().await;
    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .get(format!(
            "http://{}/tama/v1/system/metrics/stream",
            bound_addr
        ))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("text/event-stream"));

    let mut stream = response.bytes_stream();
    let mut found_snapshot = false;
    while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(4), stream.next())
        .await
        .unwrap()
    {
        let chunk: Bytes = chunk.unwrap();
        let data = String::from_utf8_lossy(&chunk);
        if data.contains("event: snapshot") {
            // Parse the data: line to extract data: line
            for line in data.lines() {
                if let Some(data_line) = line.strip_prefix("data: ") {
                    let snapshot: crate::gpu::MetricsSnapshot =
                        serde_json::from_str(data_line).unwrap();
                    assert!(!snapshot.buckets.is_empty());
                    assert!(snapshot.buckets[0].ts_unix_ms > 0);
                    assert!(snapshot.buckets[0].ram_total_mib > 0);
                    found_snapshot = true;
                    break;
                }
            }
            if found_snapshot {
                break;
            }
        }
    }

    assert!(
        found_snapshot,
        "Expected to receive a snapshot event within 4s, but none was found"
    );
}

/// Round-trip test: the SSE `sample` events emitted by
/// `/tama/v1/system/metrics/stream` must serialize the new
/// `MetricSample.models` field in a wire format that the client-side
/// `crate::gpu::MetricSample` Deserialize impl can read back without
/// error.
///
/// We configure the proxy with exactly one known model so the assertions
/// over the deserialized `Vec<ModelStateSnapshot>` are deterministic, then
/// connect to the SSE endpoint, wait for an `event: sample`, parse the
/// `data:` payload as a `MetricSample`, and assert that
/// `sample.models` is a `Vec<crate::models::ModelStateSnapshot>` carrying the
/// configured model.
#[tokio::test]
async fn test_system_metrics_stream_sample_models_round_trip() {
    use crate::config::ModelConfig;
    use bytes::Bytes;
    use std::collections::BTreeMap;

    let tmp = tempfile::tempdir().unwrap();

    // Build a Config with exactly one known model so the deserialized
    // `sample.models` Vec has a deterministic shape we can assert on.
    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        Some(tmp.path().to_path_buf()),
        crate::db::pool::test_dummy_pool(),
    ));

    // Manually insert a model into model_configs since it's no longer in Config
    {
        let mut mc = state.registry.model_configs.write().await;
        mc.insert(
            "alpha".to_string(),
            ModelConfig {
                backend: "llama_cpp".to_string(),
                args: vec![],
                sampling: None,
                model: None,
                quant: None,

                mmproj: None,
                port: None,
                health_check: None,
                enabled: true,
                context_length: None,
                num_parallel: Some(1),
                kv_unified: false,
                profile: None,
                api_name: None,
                gpu_layers: None,
                cache_type_k: None,
                cache_type_v: None,
                quants: BTreeMap::new(),
                modalities: None,
                display_name: None,
                db_id: None,
                ..Default::default()
            },
        );
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound_addr = listener.local_addr().unwrap();

    let server = ProxyServer::new(state.clone()).await;
    let app = server.into_router().await;
    let _handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .get(format!(
            "http://{}/tama/v1/system/metrics/stream",
            bound_addr
        ))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("text/event-stream"));

    let mut stream = response.bytes_stream();
    let mut parsed_snapshot: Option<crate::gpu::MetricsSnapshot> = None;
    let mut buf = String::new();
    while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(4), stream.next())
        .await
        .unwrap()
    {
        let chunk: Bytes = chunk.unwrap();
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // SSE events are delimited by a blank line. Iterate over each
        // complete event currently in the buffer.
        while let Some(idx) = buf.find("\n\n") {
            let event_block = buf[..idx].to_string();
            buf = buf[idx + 2..].to_string();

            let mut event_name: Option<&str> = None;
            let mut data_line: Option<&str> = None;
            for line in event_block.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event_name = Some(rest);
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data_line = Some(rest);
                }
            }

            if event_name == Some("snapshot") {
                let data_line = data_line
                    .expect("snapshot event must include a data: line carrying the JSON payload");
                // The critical assertion: the JSON produced by the
                // server must deserialize cleanly into MetricsSnapshot,
                // including the `models` field in `current`.
                let snapshot: crate::gpu::MetricsSnapshot = serde_json::from_str(data_line)
                    .expect("MetricsSnapshot JSON from SSE stream must deserialize without error");
                assert!(!snapshot.buckets.is_empty());
                parsed_snapshot = Some(snapshot.clone());
                break;
            }
        }

        if parsed_snapshot.is_some() {
            break;
        }
    }

    let snapshot = parsed_snapshot
        .expect("Expected to receive a snapshot event within 4s, but none was found");
    let sample = &snapshot.current;

    // Statically prove `sample.models` is a `Vec<crate::models::ModelStateSnapshot>`.
    // If the field's type ever changes, this binding will fail to
    // type-check, which is exactly the regression we want to catch.
    let models: &Vec<crate::models::ModelStateSnapshot> = &sample.models;

    // The configured model must round-trip through JSON serialization
    // unchanged. We picked a deterministic single-model config above so
    // we can assert on the exact contents.
    assert_eq!(
        models.len(),
        1,
        "Expected exactly one model in sample.models after JSON round-trip, got: {:?}",
        models
    );
    assert_eq!(models[0].id, "alpha");
    assert_eq!(models[0].backend, "llama_cpp");
    assert!(
        !matches!(
            models[0].state,
            crate::gpu::ModelState::Ready
        ),
        "Expected the configured model to be reported as not ready since no backend was started, got: {:?}",
        models[0]
    );
    assert_eq!(
        sample.models_loaded, 0,
        "Expected models_loaded counter to be 0 when no model is loaded"
    );

    // The network field should deserialize correctly from the SSE stream.
    // In the test environment, a network interface is available so network stats
    // are populated and round-trip through JSON serialization. Network now lives
    // in 30s buckets (for bar charts).
    assert!(
        snapshot
            .buckets
            .last()
            .and_then(|h| h.network.as_ref())
            .is_some(),
        "network should be Some when a network interface is available"
    );
}

#[tokio::test]
async fn test_proxy_loads_models_from_db_on_startup() {
    use crate::config::ModelConfig;
    let guard = crate::testing::postgres::with_schema().await;

    // Pre-populate Postgres with a model config (plan-190 Task 5)
    let mc = ModelConfig {
        backend: "llama_cpp".to_string(),
        display_name: Some("DB Model".to_string()),
        ..Default::default()
    };
    crate::db::save_model_config(&guard.pool, "db-model-key", &mc)
        .await
        .unwrap();

    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        None,
        Arc::new(guard.pool.clone()),
    ));

    // Start the server (which should load models from DB)
    let _server = ProxyServer::new(state.clone()).await;

    // Verify that the model from DB is now in the proxy state
    let model_configs = state.registry.model_configs.read().await;
    assert!(
        model_configs.contains_key("db-model-key"),
        "Expected model 'db-model-key' to be loaded from DB"
    );
    let model = model_configs.get("db-model-key").unwrap();
    assert_eq!(model.display_name.as_deref(), Some("DB Model"));

    guard.finish().await;
}

/// Test that aliases are loaded from the DB into the in-memory cache at startup.
/// Without this, /v1/models and /v1/opencode/models return zero aliases because
/// the alias cache is never populated.
#[tokio::test]
async fn test_proxy_loads_aliases_from_db_on_startup() {
    use crate::config::ModelConfig;

    let guard = crate::testing::postgres::with_schema().await;

    // Pre-populate Postgres with a model config and an alias (plan-190 Task 5)
    let mc = ModelConfig {
        backend: "llama_cpp".to_string(),
        api_name: Some("test-model".to_string()),
        ..Default::default()
    };
    let model_id = crate::db::save_model_config(&guard.pool, "owner--test-repo", &mc)
        .await
        .unwrap();
    crate::db::queries::insert_alias(&guard.pool, "my-alias", model_id, None)
        .await
        .unwrap();

    let config = crate::config::Config::default();
    let state = Arc::new(crate::proxy::ProxyState::new(
        config,
        None,
        Arc::new(guard.pool.clone()),
    ));

    // Start the server (which should load aliases from DB)
    let _server = ProxyServer::new(state.clone()).await;

    // Verify that the alias from DB is now in the proxy state's alias cache
    let aliases = state.registry.aliases.read().await;
    assert!(
        aliases.contains_key("my-alias"),
        "Expected alias 'my-alias' to be loaded from DB into cache"
    );
    // The alias should resolve to the model's api_name (or repo_id as fallback)
    assert_eq!(
        aliases.get("my-alias"),
        Some(&"test-model".to_string()),
        "alias should resolve to the model's api_name"
    );

    guard.finish().await;
}
