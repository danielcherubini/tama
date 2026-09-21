use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::pull_queue::PullQueueService;
use super::state::repo_pull::{RepoPullError, RepoPullStart, RepoPullStatusDto};
use super::state::{MetricsState, PullState, RegistryState};

/// Metrics for the proxy server.
#[derive(Debug, Default)]
pub struct ProxyMetrics {
    pub total_requests: std::sync::atomic::AtomicU64,
    pub successful_requests: std::sync::atomic::AtomicU64,
    pub failed_requests: std::sync::atomic::AtomicU64,
}

/// Latest inference stats for a backend server.
///
/// Sole source (ADR-0014): the tamad's windowed `/metrics` scrape, folded
/// into this map by `merge_tamad_inference_stats`. Fields are `Option<f32>`
/// — `None` when the value was not observed in the last 30 s (or the
/// engine doesn't expose the counter).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LatestInferenceStats {
    /// Windowed decode tokens/s (Δ generation tokens / Δt over the scrape window)
    pub tps: Option<f32>,
    /// Windowed prompt processing speed in tokens per second (compute-only tokens)
    pub prompt_tps: Option<f32>,
    /// Windowed cache hit rate percentage, None if no prompt tokens in the window
    pub cache_hit_pct: Option<f32>,
    /// Windowed speculative decoding acceptance rate, None if no draft tokens in the window
    pub spec_accept_pct: Option<f32>,
    /// The last window's spec-traffic flag (Δ draft tokens > 0) — NOT sticky:
    /// the merge overwrites it, so it un-sticks 30 s after the last spec traffic
    pub spec_decoding_active: bool,
    /// Unix ms timestamp of the last merge that observed `tps` (proxy clock)
    pub last_updated_ms: i64,
}

/// Manages proxy state and model lifecycle.
///
/// Composed from three domain sub-structs: `registry` (models/configs/aliases),
/// `metrics` (counters/channels), `pull` (jobs/downloads). Remaining fields
/// are standalone configuration or service handles.
impl Clone for ProxyState {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            metrics: self.metrics.clone(),
            pull: self.pull.clone(),
            config: Arc::clone(&self.config),
            client: self.client.clone(),
            db_dir: self.db_dir.clone(),
            config_write_semaphore: Arc::clone(&self.config_write_semaphore),
            cookie_key: self.cookie_key.clone(),
            langfuse_client: Arc::clone(&self.langfuse_client),
            remote_forwarder: self.remote_forwarder.clone(),
            tamad_pool: Arc::clone(&self.tamad_pool),
            started_at: self.started_at,
            db_pool: self.db_pool.clone(),
        }
    }
}

pub struct ProxyState {
    /// Model registry: loaded backends, model configs, and alias caches.
    pub(crate) registry: RegistryState,
    /// Metrics and channel handles: counters, system metrics, inference stats.
    pub(crate) metrics: MetricsState,
    /// Pull job state: active jobs, in-flight downloads, queue service.
    pub(crate) pull: PullState,
    pub(crate) config: Arc<tokio::sync::RwLock<crate::config::Config>>,
    pub(crate) client: reqwest::Client,
    pub(crate) db_dir: Option<std::path::PathBuf>,
    /// Semaphore controlling concurrent post-pull config writes.
    /// Replaces the old global CONFIG_WRITE_LOCK to allow controlled
    /// parallelism (default capacity=4) instead of full serialization.
    pub(crate) config_write_semaphore: Arc<tokio::sync::Semaphore>,
    /// Signing key for session cookies (OAuth2 OIDC login).
    pub(crate) cookie_key: cookie::Key,
    /// Langfuse observability client, initialized from config at startup.
    /// Wrapped in RwLock so it can be refreshed when config is updated via PATCH.
    pub(crate) langfuse_client:
        Arc<tokio::sync::RwLock<Option<Arc<crate::proxy::forward::langfuse::LangfuseClient>>>>,
    /// HTTP forwarder for remote OpenAI-compatible providers.
    pub(crate) remote_forwarder: crate::proxy::remote::RemoteForwarder,
    /// Pool of live per-tamad stats streams, keyed by tamad ID (plan-191
    /// Task 4). Replaces the old lazy `tamad_clients` cache — the pool owns
    /// one reconnecting `StreamStats` connection per registered tamad.
    pub(crate) tamad_pool: Arc<crate::tamad::pool::TamadPool>,
    /// When this proxy process started (uptime for the health endpoint,
    /// plan-191 Task 9).
    pub(crate) started_at: std::time::Instant,
    /// Postgres pool (plan-190 Task 9: always present). `main.rs` is the
    /// single owner of the pool and hands the same `Arc<PgPool>` to both
    /// `ProxyState` and `WebState`.
    pub(crate) db_pool: Arc<sqlx::PgPool>,
}

impl ProxyState {
    /// The Postgres pool (always present; plan-190 Task 9).
    pub fn db_pool(&self) -> Arc<sqlx::PgPool> {
        self.db_pool.clone()
    }

    /// The tamad stats-stream pool (plan-191 Task 4). Used by the proxy
    /// startup sequence (`load_all`), the management API (register/update/
    /// delete refresh), and the dashboard fan-out in `tama_handlers`.
    pub fn tamad_pool(&self) -> Arc<crate::tamad::pool::TamadPool> {
        Arc::clone(&self.tamad_pool)
    }
    /// Process status on the TAMAD wire for `name` (plan 193 T5c:
    /// process-state queries read from the live rows, not a mirror).
    ///
    /// `Some(endpoint)` IFF that key has a live, addressable wire row
    /// for the process — a row exists AND its `alive` bit is set
    /// (`starting` / `restarting` rows are included; this is not
    /// `ready`-only) — returning the row's endpoint URL. `None`
    /// otherwise: no row for the key at all, or a row whose `alive`
    /// is false (a dead `budget_exhausted` row).
    pub async fn process_status(&self, name: &str) -> Option<String> {
        crate::proxy::live_rows(self.tamad_pool().as_ref())
            .await
            .row(name)
            .filter(|r| r.alive)
            .map(|r| r.endpoint)
    }

    /// Start a whole-repo `hf` CLI pull.
    ///
    /// `model_id` is the pre-created stub row (None = no DB update on
    /// completion). Takes `&Arc<Self>` so the spawned wait-loop can clone the
    /// state and outlive the caller.
    pub async fn start_repo_pull(
        self: &Arc<Self>,
        repo_id: &str,
        model_id: Option<i64>,
    ) -> Result<RepoPullStart, RepoPullError> {
        super::state::repo_pull::start_repo_pull(self, repo_id, model_id).await
    }

    /// Live status snapshot of a whole-repo pull job, or `None` if the job id
    /// is unknown.
    ///
    /// `bytes_done` prefers the relay-mirrored counter (the pull runs on the
    /// pull host, ADR-0010 — the files may not be local to the proxy); a
    /// local directory scan (`scan_dir_bytes`, wrapped in `spawn_blocking` so
    /// the recursive walk never blocks a web worker) covers jobs without a
    /// relay counter yet and single-host setups where both views agree.
    pub async fn get_repo_pull_status(&self, job_id: &str) -> Option<RepoPullStatusDto> {
        let job = self.pull.get_repo_pull(job_id).await?;
        let dest = job.dest.clone();
        let scanned =
            tokio::task::spawn_blocking(move || super::state::repo_pull::scan_dir_bytes(&dest))
                .await
                .unwrap_or(0);
        let bytes_done = job.bytes_done.max(scanned);
        Some(RepoPullStatusDto {
            job_id: job.job_id.clone(),
            status: job.status.to_string(),
            bytes_done,
            total_bytes: job.total_bytes,
            error: job.error.clone(),
            context_length: job.context_length,
        })
    }

    /// The host-side job id for an in-memory pull job (plan-191 follow-up
    /// B); the cancel endpoint uses it to route `CancelJob` to the pull
    /// host (`None` when unknown or when no host id has been recorded yet).
    pub async fn pull_job_tamad_job_id(&self, job_id: &str) -> Option<String> {
        self.pull
            .pull_jobs
            .read()
            .await
            .get(job_id)
            .and_then(|j| j.tamad_job_id.clone())
    }

    /// Best-effort dispatch of `CancelJob` to the pull host (plan-191
    /// follow-up B). The remote call is idempotent; every failure path is
    /// logged and swallowed — best-effort by design (the relay converges to
    /// the terminal state regardless).
    pub async fn cancel_pull_host_job(&self, tamad_job_id: &str) {
        let Some(backend) = self.config.read().await.proxy.pull_backend.clone() else {
            return;
        };
        let Some(handle) = self.tamad_pool.get(&backend).await else {
            return;
        };
        match handle.cancel_job(tamad_job_id).await {
            Ok(true) => {
                tracing::info!(
                    tamad = %backend,
                    tamad_job_id,
                    "host job cancel dispatched"
                );
            }
            Ok(false) => {
                tracing::debug!(
                    tamad = %backend,
                    tamad_job_id,
                    "host job already terminal (no-op cancel)"
                );
            }
            Err(e) => {
                tracing::warn!(
                    tamad = %backend,
                    tamad_job_id,
                    error = %e,
                    "host job cancel failed (best-effort)"
                );
            }
        }
    }

    /// Cancel + kill a running whole-repo pull job.
    ///
    /// Err message is user-facing: "not found" / "already finished".
    pub async fn cancel_repo_pull(&self, job_id: &str) -> Result<(), String> {
        // Hand the cancel to the pull host before we mark the local job
        // cancelled (the relay will converge the in-memory state to
        // `cancelled` when the tamad sends its terminal event).
        if let Some(job) = self.pull.get_repo_pull(job_id).await {
            if let Some(tamad_job_id) = job.tamad_job_id.clone() {
                if job.status == crate::proxy::state::RepoPullStatus::Running {
                    self.cancel_pull_host_job(&tamad_job_id).await;
                }
            }
        }
        self.pull.cancel_repo_pull(job_id).await
    }

    /// Gracefully shut down the proxy state.
    ///
    /// This method is called during a hard restart to clean up resources:
    /// - Closes the metrics broadcast channel to stop metrics streaming
    /// - Clears active pull jobs
    /// - Clears in-flight pulls
    pub async fn shutdown(&self) {
        // Close the metrics broadcast channel to stop the metrics stream
        let _ = self
            .metrics
            .metrics_tx
            .send(crate::gpu::MetricsSnapshot::default());

        // Clear inference stats (there is no model mirror to clear, plan-193
        // T5c) and the per-key access map is cleared with it, so
        // nothing stale survives a restart.
        self.registry.last_accessed.write().await.clear();
        self.metrics.clear_inference_stats();

        // Clear pull jobs and in-flight pulls
        self.pull.clear().await;
    }

    /// Returns a reference to the HTTP client.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Returns a reference to the database directory.
    pub fn db_dir(&self) -> &Option<std::path::PathBuf> {
        &self.db_dir
    }

    /// Returns a reference to the pull queue service.
    pub fn pull_queue(&self) -> &Option<Arc<PullQueueService>> {
        &self.pull.pull_queue
    }

    /// Sets the pull queue service. Used by tests in other workspace crates.
    #[allow(dead_code)]
    pub fn set_pull_queue(&mut self, queue: Option<Arc<PullQueueService>>) {
        self.pull.pull_queue = queue;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Test that the get_repo_pull_status delegate builds a DTO with a
    /// computed bytes_done (recursive file sizes under dest) and that an
    /// unknown job id yields None.
    #[tokio::test]
    async fn test_get_repo_pull_status_dto() {
        let state = Arc::new(ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        ));
        let dest = tempfile::tempdir().unwrap();
        std::fs::write(dest.path().join("a.bin"), vec![0u8; 100]).unwrap();
        let nested = dest.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("b.bin"), vec![1u8; 50]).unwrap();

        state
            .pull
            .upsert_repo_pull(crate::proxy::state::RepoPullJob {
                job_id: "job-dto".to_string(),
                repo_id: "owner/repo".to_string(),
                model_id: Some(7),
                dest: dest.path().to_path_buf(),
                total_bytes: Some(300),
                status: crate::proxy::state::RepoPullStatus::Running,
                error: None,
                cancel_requested: false,
                context_length: None,
                stderr_tail: Arc::new(tokio::sync::Mutex::new(Vec::new())),
                tamad_job_id: None,
                bytes_done: 0,
            })
            .await;

        let dto = state
            .get_repo_pull_status("job-dto")
            .await
            .expect("running job should have a DTO");
        assert_eq!(dto.job_id, "job-dto");
        assert_eq!(dto.status, "running");
        assert_eq!(dto.bytes_done, 150);
        assert_eq!(dto.total_bytes, Some(300));
        assert!(dto.error.is_none());
        assert!(dto.context_length.is_none());

        assert!(state.get_repo_pull_status("missing").await.is_none());
    }

    /// Test that the cancel_repo_pull delegate surfaces user-facing errors
    /// ("not found" / "already finished") and flags a running job.
    #[tokio::test]
    async fn test_cancel_repo_pull_delegate() {
        let state = Arc::new(ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        ));

        assert_eq!(
            state.cancel_repo_pull("missing").await,
            Err("not found".to_string())
        );

        state
            .pull
            .upsert_repo_pull(crate::proxy::state::RepoPullJob {
                job_id: "job-cancel-dto".to_string(),
                repo_id: "owner/repo".to_string(),
                model_id: None,
                dest: std::path::PathBuf::from("/tmp/models/owner/repo"),
                total_bytes: None,
                status: crate::proxy::state::RepoPullStatus::Running,
                error: None,
                cancel_requested: false,
                context_length: None,
                stderr_tail: Arc::new(tokio::sync::Mutex::new(Vec::new())),
                tamad_job_id: None,
                bytes_done: 0,
            })
            .await;

        state
            .cancel_repo_pull("job-cancel-dto")
            .await
            .expect("first cancel should succeed");
        assert_eq!(
            state.cancel_repo_pull("job-cancel-dto").await,
            Err("already finished".to_string())
        );

        // The DTO reflects the cancellation (status is terminal, no error).
        let dto = state
            .get_repo_pull_status("job-cancel-dto")
            .await
            .expect("job should still be queryable");
        assert_eq!(dto.status, "cancelled");
        assert!(dto.error.is_none());
    }

    /// Test that the start_repo_pull delegate (Arc receiver, public boundary)
    /// validates the repo id before any other work.
    #[tokio::test]
    async fn test_start_repo_pull_delegate_invalid_id() {
        let state = Arc::new(ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        ));
        let err = state
            .start_repo_pull("a/b\\c", None)
            .await
            .expect_err("invalid repo id must be rejected");
        assert!(
            matches!(err, crate::proxy::RepoPullError::InvalidRepoId(_)),
            "expected InvalidRepoId, got: {err:?}"
        );
    }

    /// Verify the public surface exposes service handles and sub-struct
    /// composition — not lock guards.
    #[tokio::test]
    async fn test_proxy_state_public_surface() {
        let state = ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        );
        let _: &reqwest::Client = state.client();
        let _: &Option<std::path::PathBuf> = state.db_dir();
        let _: &Option<Arc<PullQueueService>> = state.pull_queue();
        // Sub-structs are composed and independently cloneable.
        let _registry = state.registry.clone();
        let _metrics = state.metrics.clone();
        let _pull = state.pull.clone();
    }

    #[test]
    fn test_latest_inference_stats_default() {
        let stats = LatestInferenceStats::default();
        assert!(stats.tps.is_none());
        assert!(stats.prompt_tps.is_none());
        assert!(stats.cache_hit_pct.is_none());
        assert!(stats.spec_accept_pct.is_none());
        assert!(!stats.spec_decoding_active);
        assert_eq!(stats.last_updated_ms, 0);
    }

    #[test]
    fn test_latest_inference_stats_clone_copy() {
        let stats = LatestInferenceStats {
            tps: Some(50.0),
            prompt_tps: Some(200.0),
            cache_hit_pct: Some(85.5),
            spec_accept_pct: Some(90.0),
            spec_decoding_active: true,
            last_updated_ms: 1234567890,
        };
        // Test Copy
        let stats2: LatestInferenceStats = stats;
        assert_eq!(stats2.tps, Some(50.0));
        assert!(stats2.spec_decoding_active);
        // Original is still usable after copy
        assert_eq!(stats.tps, Some(50.0));
        // Test Clone
        let stats3 = stats;
        assert_eq!(stats3.prompt_tps, Some(200.0));
    }

    #[test]
    fn test_latest_inference_stats_serialization() {
        let stats = LatestInferenceStats {
            tps: Some(50.0),
            prompt_tps: Some(200.0),
            cache_hit_pct: Some(85.5),
            spec_accept_pct: Some(90.0),
            spec_decoding_active: true,
            last_updated_ms: 1700000000000,
        };

        let json = serde_json::to_string(&stats).expect("serialization failed");
        let value: serde_json::Value = serde_json::from_str(&json).expect("deserialization failed");

        // All 6 fields must be present
        assert!(value.get("tps").is_some(), "missing field: tps");
        assert!(
            value.get("prompt_tps").is_some(),
            "missing field: prompt_tps"
        );
        assert!(
            value.get("cache_hit_pct").is_some(),
            "missing field: cache_hit_pct"
        );
        assert!(
            value.get("spec_accept_pct").is_some(),
            "missing field: spec_accept_pct"
        );
        assert!(
            value.get("spec_decoding_active").is_some(),
            "missing field: spec_decoding_active"
        );
        assert!(
            value.get("last_updated_ms").is_some(),
            "missing field: last_updated_ms"
        );

        // Correct types: f32 -> number, bool -> bool, i64 -> number
        assert_eq!(value["tps"], serde_json::json!(50.0));
        assert_eq!(value["prompt_tps"], serde_json::json!(200.0));
        assert_eq!(value["cache_hit_pct"], serde_json::json!(85.5));
        assert_eq!(value["spec_accept_pct"], serde_json::json!(90.0));
        assert_eq!(value["spec_decoding_active"], serde_json::json!(true));
        assert_eq!(
            value["last_updated_ms"],
            serde_json::json!(1700000000000_i64)
        );

        // Test with None values (not yet observed)
        let empty = LatestInferenceStats::default();
        let json_empty = serde_json::to_string(&empty).expect("serialization failed");
        let value_empty: serde_json::Value =
            serde_json::from_str(&json_empty).expect("deserialization failed");
        assert!(value_empty["tps"].is_null());
        assert!(value_empty["prompt_tps"].is_null());
        assert!(value_empty["cache_hit_pct"].is_null());
        assert!(value_empty["spec_accept_pct"].is_null());
        assert_eq!(
            value_empty["spec_decoding_active"],
            serde_json::json!(false)
        );
        assert_eq!(value_empty["last_updated_ms"], serde_json::json!(0_i64));
    }

    #[test]
    fn test_inference_stats_watch_round_trip() {
        let (tx, mut rx) =
            tokio::sync::watch::channel::<HashMap<String, LatestInferenceStats>>(HashMap::new());
        // Initial value is empty
        assert!(rx.borrow_and_update().is_empty());
        // Send stats for a backend
        let mut map = HashMap::new();
        map.insert(
            "backend-a".to_string(),
            LatestInferenceStats {
                tps: Some(42.0),
                prompt_tps: Some(100.0),
                cache_hit_pct: Some(75.0),
                spec_accept_pct: Some(80.0),
                spec_decoding_active: true,
                last_updated_ms: 999,
            },
        );
        tx.send_replace(map);
        // Verify
        let received = rx.borrow_and_update();
        assert_eq!(received.len(), 1);
        let stats = received.get("backend-a").unwrap();
        assert_eq!(stats.tps, Some(42.0));
        assert_eq!(stats.cache_hit_pct, Some(75.0));
        assert!(stats.spec_decoding_active);
        assert_eq!(stats.last_updated_ms, 999);
    }

    #[test]
    fn test_inference_stats_per_backend_isolation() {
        let (tx, mut rx) =
            tokio::sync::watch::channel::<HashMap<String, LatestInferenceStats>>(HashMap::new());

        // Insert stats for backend-a
        let mut map = HashMap::new();
        map.insert(
            "backend-a".to_string(),
            LatestInferenceStats {
                tps: Some(50.0),
                prompt_tps: Some(200.0),
                cache_hit_pct: Some(85.0),
                spec_accept_pct: Some(90.0),
                spec_decoding_active: true,
                last_updated_ms: 1000,
            },
        );
        tx.send_replace(map);

        // Insert stats for backend-b
        let mut map2 = rx.borrow_and_update().clone();
        map2.insert(
            "backend-b".to_string(),
            LatestInferenceStats {
                tps: Some(30.0),
                prompt_tps: Some(100.0),
                cache_hit_pct: Some(50.0),
                spec_accept_pct: None,
                spec_decoding_active: false,
                last_updated_ms: 2000,
            },
        );
        tx.send_replace(map2);

        // Verify both backends have independent stats
        let received = rx.borrow_and_update();
        assert_eq!(received.len(), 2);

        let a = received.get("backend-a").unwrap();
        assert_eq!(a.tps, Some(50.0));
        assert!(a.spec_decoding_active);

        let b = received.get("backend-b").unwrap();
        assert_eq!(b.tps, Some(30.0));
        assert!(!b.spec_decoding_active);
        assert!(b.spec_accept_pct.is_none());
    }
}
