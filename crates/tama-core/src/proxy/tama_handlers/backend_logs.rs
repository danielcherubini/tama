//! Tamad engine-log tail core.
//!
//! `tail_one_tamad_source` is the per-(host, model) gRPC `Logs` RPC tail
//! reused by the structured read API: `logs_api::TamadTailProvider` calls
//! it for each `tamad:<host>:model:<name>` source to surface the engine's
//! raw output as a legacy tail adapter.

/// Per-tamad budget for the engine-log tail (open + drain).
/// A wedged RPC must not stall the whole endpoint.
const TAMAD_LOGS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Idempotent per-(host, model) RPC tail: opens the `Logs` stream (bounded
/// single attempt — a wedge must not stop a fanout… or a single
/// source) and drains it to `max_lines`.
pub(crate) async fn tail_one_tamad_source(
    pool: &crate::tamad::pool::TamadPool,
    host: &str,
    model: Option<&str>,
    max_lines: usize,
) -> Vec<String> {
    for handle in pool.list_handles().await {
        if handle.connection.name != host {
            continue;
        }
        if !handle.is_online().await {
            continue;
        }
        let Some(stats) = handle.latest().await else {
            // Online but no snapshot yet — nothing to enumerate.
            continue;
        };
        for p in &stats.processes {
            if !matches!(p.status.as_str(), "starting" | "ready") {
                continue;
            }
            if let Some(m) = model {
                if m != p.model_name {
                    continue;
                }
            }
            let req = crate::tamad::LogsRequest {
                provider_name: p.provider_name.clone(),
                model_name: p.model_name.clone(),
            };

            let open = tokio::time::timeout(TAMAD_LOGS_TIMEOUT, handle.logs(&req)).await;
            let mut stream = match open {
                Ok(Ok(stream)) => stream,
                Ok(Err(e)) => {
                    tracing::warn!(host, model = %p.model_name, error = %e, "tamad logs RPC failed; skipping source");
                    continue;
                }
                Err(_) => {
                    tracing::warn!(host, model = %p.model_name, "tamad logs RPC timed out; skipping source");
                    continue;
                }
            };

            let drained = tokio::time::timeout(TAMAD_LOGS_TIMEOUT, async {
                let mut lines: Vec<String> = Vec::new();
                while lines.len() < max_lines {
                    match stream.message().await {
                        Ok(Some(entry)) => {
                            // Skip blank lines (container log noise).
                            if !entry.message.trim().is_empty() {
                                lines.push(entry.message);
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            tracing::warn!(host, model = %p.model_name, error = %e, "tamad log stream error; keeping partial tail");
                            break;
                        }
                    }
                }
                lines
            })
            .await;

            return match drained {
                Ok(lines) => lines,
                Err(_) => {
                    tracing::warn!(host, model = %p.model_name, "tamad log stream stalled; skipping source");
                    Vec::new()
                }
            };
        }
    }
    Vec::new()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::tail_one_tamad_source;
    use crate::tamad::pool::test_support::{grpc_conn, start_stub, wait_for, StubTamad};
    use crate::tamad::pool::TamadPool;

    /// A `StubTamad` wired for scripted stats processes + log tails.
    /// `fail_first_n = usize::MAX` simulates a never-online tamad.
    /// (Full literal — the shared harness has no builder.)
    fn logs_stub(
        fail_first_n: usize,
        stats_processes: Vec<crate::tamad::ProcessInfo>,
        log_messages: Vec<String>,
    ) -> StubTamad {
        let (down_tx, _) = tokio::sync::watch::channel(false);
        StubTamad {
            fail_first_n,
            succeed_until: usize::MAX,
            down: Arc::new(down_tx),
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            successes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            pull_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            pull_job_id: "job-stub".to_string(),
            pull_model_fail: Arc::new(tokio::sync::Mutex::new(false)),
            install_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            install_job_id: "job-install".to_string(),
            install_dispatch_fail: Arc::new(tokio::sync::Mutex::new(false)),
            update_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            update_job_id: "job-update".to_string(),
            update_dispatch_fail: Arc::new(tokio::sync::Mutex::new(false)),
            remove_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            remove_dispatch_fail: Arc::new(tokio::sync::Mutex::new(false)),
            stream_job_events: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            stream_job_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            stream_job_events_by_id: Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            bench_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            bench_job_id: "job-bench".to_string(),
            bench_dispatch_fail: Arc::new(tokio::sync::Mutex::new(false)),
            stats_gpus: vec![],
            stats_processes,
            logs_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            log_messages,
            load_requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            load_delays: std::collections::HashMap::new(),
            load_model_fail: Arc::new(tokio::sync::Mutex::new(false)),
            stream_log_frames: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            stream_log_calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            stream_log_refuse: false,
        }
    }

    fn process(model: &str, status: &str) -> crate::tamad::ProcessInfo {
        crate::tamad::ProcessInfo {
            model_name: model.to_string(),
            provider_name: "llama.cpp".to_string(),
            pid: 4242,
            alive: true,
            endpoint_url: "http://127.0.0.1:18099".to_string(),
            status: status.to_string(),
            desired: false,
            restart_count: 0,
            max_restarts: 0,
            spec_accept_pct: None,
            spec_decoding_active: false,
            tps: None,
            prompt_tps: None,
            cache_hit_pct: None,
            last_obs_ms: None,
        }
    }

    /// No registered handles → no tail lines (and no failure).
    #[tokio::test]
    async fn test_tail_one_tamad_source_empty_pool() {
        let guard = crate::testing::postgres::with_schema().await;
        let pool = TamadPool::new(Arc::new(guard.pool.clone()));
        let lines = tail_one_tamad_source(&pool, "hostA", Some("model-x"), 200).await;
        assert!(lines.is_empty(), "empty pool must yield no lines");
        guard.finish().await;
    }

    /// A never-online tamad is skipped entirely: no `Logs` RPC is even
    /// attempted, and the tail stays empty (the endpoint stays healthy).
    #[tokio::test]
    async fn test_tail_one_tamad_source_offline_tamad_skipped() {
        let guard = crate::testing::postgres::with_schema().await;
        let db_pool = Arc::new(guard.pool.clone());
        let stub = logs_stub(
            usize::MAX, // stream_stats fails forever → never online
            vec![process("model-x", "ready")],
            vec!["line".into()],
        );
        let addr = start_stub(stub.clone()).await;
        let pool = TamadPool::new(db_pool).with_backoff_base(std::time::Duration::from_millis(20));
        let conn = grpc_conn("uuid-logs-off", "hostA", &format!("grpc://{addr}"));
        pool.upsert_connection(&conn).await.unwrap();

        // Give the stream task a couple of (failing) reconnect attempts.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let handle = pool.get("uuid-logs-off").await.expect("handle");
        assert!(!handle.is_online().await, "stub must stay offline");

        let lines = tail_one_tamad_source(&pool, "hostA", Some("model-x"), 200).await;
        assert!(lines.is_empty(), "offline tamad must yield no lines");
        assert!(
            stub.logs_requests.lock().await.is_empty(),
            "no Logs RPC may be attempted for an offline tamad"
        );

        guard.finish().await;
    }

    /// Happy path: one online tamad reporting one `ready` process → the
    /// tail returns its lines; the RPC carries both the provider name and
    /// the model name (precise routing).
    #[tokio::test]
    async fn test_tail_one_tamad_source_happy_path() {
        let guard = crate::testing::postgres::with_schema().await;
        let db_pool = Arc::new(guard.pool.clone());
        let stub = logs_stub(
            0,
            vec![process("model-x", "ready")],
            vec!["engine line one".into(), "engine line two".into()],
        );
        let addr = start_stub(stub.clone()).await;
        let pool = TamadPool::new(db_pool).with_backoff_base(std::time::Duration::from_millis(20));
        let conn = grpc_conn("uuid-logs-hp", "hostA", &format!("grpc://{addr}"));
        pool.upsert_connection(&conn).await.unwrap();
        let handle = pool.get("uuid-logs-hp").await.expect("handle");

        assert!(
            wait_for(|| async {
                handle.is_online().await
                    && handle
                        .latest()
                        .await
                        .map(|s| s.processes.len() == 1)
                        .unwrap_or(false)
            })
            .await,
            "handle should come online with the process snapshot"
        );

        let lines = tail_one_tamad_source(&pool, "hostA", Some("model-x"), 200).await;
        assert_eq!(
            lines,
            vec!["engine line one".to_string(), "engine line two".to_string()]
        );

        // The RPC carried both fields (model_name precise + provider_name).
        let reqs = stub.logs_requests.lock().await;
        assert_eq!(reqs.len(), 1, "exactly one Logs RPC");
        assert_eq!(reqs[0].model_name, "model-x");
        assert_eq!(reqs[0].provider_name, "llama.cpp");

        guard.finish().await;
    }

    /// Non-`starting`/`ready` processes are never tailed, and blank lines
    /// from the tail are filtered out of the result.
    #[tokio::test]
    async fn test_tail_one_tamad_source_skips_inactive_and_blanks() {
        let guard = crate::testing::postgres::with_schema().await;
        let db_pool = Arc::new(guard.pool.clone());
        let stub = logs_stub(
            0,
            vec![process("model-a", "ready"), process("model-b", "failed")],
            vec![
                "keep me".into(),
                String::new(),
                "   ".into(),
                "keep too".into(),
            ],
        );
        let addr = start_stub(stub.clone()).await;
        let pool = TamadPool::new(db_pool).with_backoff_base(std::time::Duration::from_millis(20));
        let conn = grpc_conn("uuid-logs-skip", "hostB", &format!("grpc://{addr}"));
        pool.upsert_connection(&conn).await.unwrap();
        let handle = pool.get("uuid-logs-skip").await.expect("handle");

        assert!(
            wait_for(|| async {
                handle
                    .latest()
                    .await
                    .map(|s| s.processes.len() == 2)
                    .unwrap_or(false)
            })
            .await,
            "snapshot with both processes should arrive"
        );

        let lines = tail_one_tamad_source(&pool, "hostB", Some("model-a"), 200).await;
        assert_eq!(
            lines,
            vec!["keep me".to_string(), "keep too".to_string()],
            "blank lines must be filtered"
        );
        let reqs = stub.logs_requests.lock().await;
        assert_eq!(reqs.len(), 1, "only the ready process is tailed");
        assert_eq!(reqs[0].model_name, "model-a");

        guard.finish().await;
    }
}
