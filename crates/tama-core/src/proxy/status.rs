use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::types::ProxyState;

/// Lifecycle state of a model as reported by the `/status` endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusModelState {
    Ready,
    #[serde(rename = "starting")]
    Loading,
    Unloading,
    Failed,
    Idle,
}

/// Per-model entry in the `/status` response `models` map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusModelEntry {
    pub id: Option<i64>,
    pub display_name: Option<String>,
    pub backend: String,
    pub backend_path: Option<String>,
    pub model: Option<String>,
    pub quant: Option<String>,
    pub context_length: Option<u32>,
    pub enabled: bool,
    pub api_name: Option<String>,
    pub state: StatusModelState,
    pub backend_pid: Option<u32>,
    pub load_time_secs: Option<u64>,
    pub last_accessed_secs_ago: Option<u64>,
    pub idle_timeout_remaining_secs: Option<u64>,
    pub consecutive_failures: Option<u32>,
}

/// VRAM usage snapshot for the `/status` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramStatus {
    pub used_mib: u64,
    pub total_mib: u64,
}

/// Atomic counter snapshot for the `/status` endpoint (distinct from the
/// live `proxy::types::ProxyMetrics` counters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyMetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub models_loaded: u64,
    pub models_unloaded: u64,
}

/// Typed response for the `/status` endpoint.
///
/// `vram` and `gpu_utilization_pct` use `skip_serializing_if` so that `None`
/// values are omitted from the wire JSON — matching the behaviour clients
/// received before the endpoint was fully typed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub cpu_usage_pct: f32,
    pub ram_used_mib: u64,
    pub ram_total_mib: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_utilization_pct: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram: Option<VramStatus>,
    pub auto_unload: bool,
    pub idle_timeout_secs: u64,
    pub metrics: ProxyMetricsSnapshot,
    pub models: std::collections::BTreeMap<String, StatusModelEntry>,
}

/// Resolve the dashboard host-grouping name for a model.
///
/// The frontend groups active models into host fleet cards by
/// `host_name == hosts[].name`, and the SSE `hosts[]` array is populated
/// from `TamadHandle.connection.name` (see `tama_handlers::system`). So the
/// attribution value must be the tamad **connection display name**, not the
/// provider's name (a user nickname that matches no host card).
///
/// Derivation: resolve the model's owning provider, then look up the
/// provider's `tamad_id` in the live tamad pool and return the handle's
/// connection name. Pool-first is sufficient: a model can only be `ready`
/// when its tamad was online to load it, so the live grouping case always
/// has a handle. Returns `None` (dashboard "Unassigned") when the provider
/// is unresolvable, has no tamad assigned, or the tamad has no pool handle.
pub(crate) async fn resolve_host_name(state: &ProxyState, model_id: &str) -> Option<String> {
    let provider = crate::proxy::lifecycle::spec::resolve_provider_for_model(state, model_id)
        .await
        .ok()?;
    let tamad_id = provider.tamad_id?;
    let handle = state.tamad_pool().get(&tamad_id).await?;
    Some(handle.connection.name.clone())
}

/// Map a wire process status onto the dashboard `ModelState` enum.
///
/// Only the eligible row statuses are ever seen (rows.rs filters the rest),
/// so `starting`/`restarting` map to `Starting` and everything else falls
/// back to `Idle` defensively.
fn row_model_state(status: &str) -> crate::gpu::ModelState {
    match status {
        "ready" => crate::gpu::ModelState::Ready,
        "starting" | "restarting" => crate::gpu::ModelState::Starting,
        _ => crate::gpu::ModelState::Idle,
    }
}

impl ProxyState {
    /// Build the per-model status snapshot embedded in `MetricSample.models`.
    ///
    /// Iterates over every configured model, resolves its backends, and reports
    /// the lifecycle state (`idle`, `loading`, `ready`, `unloading`, `failed`).
    /// The returned vector is sorted by `id` so dashboard rows do not shuffle
    /// between SSE samples.
    pub async fn collect_model_state_snapshots(&self) -> Vec<crate::models::ModelStateSnapshot> {
        let config = self.config.read().await;
        let model_configs = self.registry.model_configs.read().await;

        // Borrow inference_stats before acquiring runtime to avoid lock-order issues.
        let inference_stats = self.metrics.inference_stats_snapshot();

        let rows = crate::proxy::live_rows(self.tamad_pool().as_ref()).await;
        let mut out: Vec<crate::models::ModelStateSnapshot> =
            Vec::with_capacity(model_configs.len());
        for (model_id, model_cfg) in model_configs.iter() {
            // Determine the model's lifecycle state from its live wire row
            // (plan-193 Task 4 read-side source flip: rows, not the mirror).
            // A row exists only for a loaded, eligible process; an idle
            // model — or an offline host — has no row and reads as Idle.
            let servers = config.resolve_backends_for_model(&model_configs, model_id);
            let row = rows.row(model_id);
            let (model_state, error_message, is_docker) = match row {
                Some(r) => (row_model_state(&r.status), None, false),
                None => (crate::gpu::ModelState::Idle, None, false),
            };

            // Look up the first matching backend's inference stats.
            // first-server-wins: for the current usage (one server per model) this is sufficient.
            let server_stats = servers
                .iter()
                .find_map(|(sn, _, _)| inference_stats.get(sn));
            // Resolve unified metadata from whichever source is populated.
            let meta = crate::models::ResolvedModelMetadata::resolve(model_cfg);
            let status = crate::models::ModelStateSnapshot {
                id: model_id.clone(),
                db_id: model_cfg.db_id,
                api_name: model_cfg.api_name.clone(),
                display_name: model_cfg.display_name.clone(),
                backend: model_cfg.backend.clone(),
                state: model_state,
                quant: meta.quant,
                context_length: meta.context_length,
                hf_architecture_type: model_cfg.hf_architecture_type.clone(),
                hf_base_model: model_cfg.hf_base_model.clone(),
                hf_format: model_cfg.hf_format.clone(),
                gpu_variant: model_cfg
                    .gpu_variant
                    .as_ref()
                    .map(|v| v.variant_folder().to_string()),
                cache_type_k: meta.kv_cache_k,
                cache_type_v: meta.kv_cache_v,
                spec_types: model_cfg.spec_decoding.spec_types.clone(),
                gpu_device: model_cfg.gpu_device.clone(),
                error_message,
                tps: server_stats.and_then(|s| s.tps),
                prompt_tps: server_stats.and_then(|s| s.prompt_tps),
                is_docker,
                host_name: None,
            };
            out.push(status);
        }
        // Stable order so dashboard rows don't shuffle between samples.
        out.sort_by(|a, b| a.id.cmp(&b.id));

        // Attribute each model to its host for the dashboard's host-centric
        // grouping. Display-only: any resolution error (missing provider,
        // ambiguous local providers, remote provider, provider without a
        // tamad, tamad not in the pool, DB error) leaves `host_name` as None
        // so the model lands in the dashboard's "Unassigned" group. The
        // registry/config guards are dropped first — provider resolution
        // re-acquires them and hits the DB, which must not happen while
        // locks are held. One await per model is acceptable: the snapshot
        // cadence is ~2s and the model count is small.
        drop(config);
        drop(model_configs);
        for snapshot in out.iter_mut() {
            snapshot.host_name = resolve_host_name(self, &snapshot.id).await;
        }
        out
    }

    /// Build a comprehensive status response for the proxy.
    ///
    /// Returns a typed `StatusResponse` suitable for JSON serialization.
    /// Models are an object keyed by name, fields are flat per model
    /// (not nested in a `runtime` sub-object), and `idle_timeout_secs`
    /// is at the top level.
    pub async fn build_status_response(&self) -> StatusResponse {
        use std::sync::atomic::Ordering::Relaxed;

        let sys_metrics = self.metrics.system_metrics_snapshot().await;

        let config = self.config.read().await;
        let model_configs = self.registry.model_configs.read().await;
        let auto_unload = config.proxy.auto_unload;
        let idle_timeout_secs = config.proxy.idle_timeout_secs;
        let mut models_obj = std::collections::BTreeMap::new();
        let rows = crate::proxy::live_rows(self.tamad_pool().as_ref()).await;
        // The proxy-owned per-key access snapshot (plan-193 T5c LRU source,
        // + the countdown fields below).
        let access = self.registry.last_accessed.read().await;

        for (model_name, model_config) in model_configs.iter() {
            let backend_path = config
                .backends
                .get(&model_config.backend)
                .and_then(|b| b.path.clone());

            // Lifecycle state from the live wire rows (plan-193 T4/T5c):
            // `ready` → Ready, `starting`/`restarting` → Loading, any
            // other state (incl. `budget_exhausted`) or no row (idle
            // model / offline host) → Idle. The rich fields read from
            // the row (+ the per-key access map) instead of the mirror:
            // `backend_pid` is the wire process id; `last_accessed_secs_ago` /
            // `idle_timeout_remaining_secs` are arithmetic on the proxy's
            // per-key access entry (no entry = the model was never touched
            // by the proxy → both are reported as `None`); `load_time_secs`
            // and `consecutive_failures` never had a row source, so they
            // are permanently `None`.
            let state = match rows.row(model_name) {
                Some(r) if r.status == "ready" => StatusModelState::Ready,
                Some(r) if r.status == "starting" || r.status == "restarting" => {
                    StatusModelState::Loading
                }
                _ => StatusModelState::Idle,
            };

            // Rich fields (plan-193 T5c): ready rows carry the process
            // id; the per-key access entry carries the idle countdown.
            // No access entry = never touched = both read None, same as
            // the mirror never touched one.
            let (backend_pid, last_accessed_secs_ago, idle_timeout_remaining_secs) =
                if let Some(row) = rows.row(model_name) {
                    if row.status == "ready" {
                        let pid = Some(row.pid.max(0) as u32);
                        if let Some(last) = access.get(&row.key) {
                            let elapsed = Instant::now().duration_since(*last);
                            let remaining = if auto_unload {
                                let timeout = Duration::from_secs(idle_timeout_secs);
                                Some(if elapsed < timeout {
                                    (timeout - elapsed).as_secs()
                                } else {
                                    0
                                })
                            } else {
                                None
                            };
                            (pid, Some(elapsed.as_secs()), remaining)
                        } else {
                            (pid, None, None)
                        }
                    } else {
                        (None, None, None)
                    }
                } else {
                    (None, None, None)
                };

            let entry = StatusModelEntry {
                id: model_config.db_id,
                display_name: model_config.display_name.clone(),
                backend: model_config.backend.clone(),
                backend_path: backend_path.clone(),
                model: model_config.model.clone(),
                quant: model_config.quant.clone(),
                context_length: model_config.context_length,
                enabled: model_config.enabled,
                api_name: model_config.api_name.clone(),
                state,
                backend_pid,
                load_time_secs: None,
                last_accessed_secs_ago,
                idle_timeout_remaining_secs,
                consecutive_failures: None,
            };

            models_obj.insert(model_name.clone(), entry);
        }

        let metrics = &self.metrics.counters;

        StatusResponse {
            cpu_usage_pct: sys_metrics.cpu_usage_pct,
            ram_used_mib: sys_metrics.ram_used_mib,
            ram_total_mib: sys_metrics.ram_total_mib,
            gpu_utilization_pct: sys_metrics.gpu_utilization_pct,
            vram: sys_metrics.vram.map(|v| VramStatus {
                used_mib: v.used_mib,
                total_mib: v.total_mib,
            }),
            auto_unload,
            idle_timeout_secs,
            metrics: ProxyMetricsSnapshot {
                total_requests: metrics.total_requests.load(Relaxed),
                successful_requests: metrics.successful_requests.load(Relaxed),
                failed_requests: metrics.failed_requests.load(Relaxed),
                // plan-193 T4/T5c: the wire `models_loaded`/`models_unloaded`
                // names survive, sourced from the live row ready count (the
                // in-memory AtomicU64 counters are gone).
                models_loaded: rows.ready_count(),
                models_unloaded: 0,
            },
            models: models_obj,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BackendConfig, Config, ModelConfig};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn make_model_config(backend: &str) -> ModelConfig {
        ModelConfig {
            backend: backend.to_string(),
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
        }
    }

    /// Seed a live wire row for `model_id` with the given status (plan-193
    /// T4: `collect_model_state_snapshots` reads rows, not the mirror).
    async fn seed_live_row(state: &ProxyState, model_id: &str, status: &str) {
        use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
        let proc = crate::tamad::ProcessInfo {
            model_name: model_id.to_string(),
            provider_name: "llama_cpp".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://127.0.0.1:8000".to_string(),
            status: status.to_string(),
            desired: true,
            restart_count: 0,
            max_restarts: 3,
            spec_accept_pct: None,
            spec_decoding_active: false,
            tps: None,
            prompt_tps: None,
            cache_hit_pct: None,
        };
        let stats = stats_full(1.5, vec![], vec![proc]);
        let pool = state.tamad_pool();
        pool.insert_raw_handle(
            model_id,
            Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
        )
        .await;
    }

    #[allow(clippy::too_many_arguments)]
    fn make_model_config_with_vllm(
        backend: &str,
        quant: Option<String>,
        context_length: Option<u32>,
        cache_type_k: Option<String>,
        cache_type_v: Option<String>,
        vllm_quantization: Option<String>,
        vllm_max_model_len: Option<u32>,
        vllm_kv_cache_dtype: Option<String>,
    ) -> ModelConfig {
        ModelConfig {
            backend: backend.to_string(),
            args: vec![],
            sampling: None,
            model: None,
            quant,
            mmproj: None,
            port: None,
            health_check: None,
            enabled: true,
            context_length,
            num_parallel: Some(1),
            kv_unified: false,
            profile: None,
            api_name: None,
            gpu_layers: None,
            cache_type_k,
            cache_type_v,
            quants: BTreeMap::new(),
            modalities: None,
            display_name: None,
            db_id: None,
            vllm: crate::config::VllmConfig {
                quantization: vllm_quantization,
                kv_cache_dtype: vllm_kv_cache_dtype,
                max_model_len: vllm_max_model_len,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// When `state.models()` has no runtime entries, every configured model
    /// should be reported as `loaded == false`, with the returned vector
    /// sorted by id ascending and the `backend` field matching the
    /// corresponding `ModelConfig.backend` value.
    #[tokio::test]
    async fn test_collect_model_state_snapshots_reports_idle_when_no_runtime_entry() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Populate model_configs
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("zephyr".to_string(), make_model_config("llama_cpp"));
            mc.insert("alpha".to_string(), make_model_config("vllm"));
        }

        // Sanity check: no per-key access entries yet.
        assert!(state.registry.last_accessed.read().await.is_empty());

        let statuses = state.collect_model_state_snapshots().await;

        // Length matches the number of configured models.
        assert_eq!(statuses.len(), 2);

        // Every entry is reported as not loaded.
        assert!(
            !statuses
                .iter()
                .any(|s| matches!(s.state, crate::gpu::ModelState::Ready)),
            "expected every status to not be ready, got: {:?}",
            statuses
        );

        // Entries are sorted by id ascending.
        let ids: Vec<&str> = statuses.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "zephyr"]);

        // Backend field matches the configured backend name for each model.
        assert_eq!(statuses[0].id, "alpha");
        assert_eq!(statuses[0].backend, "vllm");
        assert_eq!(statuses[1].id, "zephyr");
        assert_eq!(statuses[1].backend, "llama_cpp");
    }

    /// When a live wire row reports `ready` under the
    /// server name that resolves for one of the configured models, that
    /// model should be reported as `loaded == true` while all other
    /// configured models remain `loaded == false`. The returned vector
    /// must still be sorted by id ascending and carry the configured
    /// `backend` value.
    #[tokio::test]
    async fn test_collect_model_state_snapshots_reports_loaded_when_server_is_ready() {
        let mut config = Config::default();
        // Add backends so resolve_backends_for_model can match models.
        config.backends.insert(
            "vllm".to_string(),
            BackendConfig {
                path: None,
                version: None,
                gpu_variant: None,
            },
        );
        config.backends.insert(
            "llama_cpp".to_string(),
            BackendConfig {
                path: None,
                version: None,
                gpu_variant: None,
            },
        );
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Populate model_configs
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("zephyr".to_string(), make_model_config("llama_cpp"));
            mc.insert("alpha".to_string(), make_model_config("vllm"));
        }

        // Seed a live `ready` wire row for "alpha" (plan-193 T4: snapshots
        // read rows, not the mirror).
        seed_live_row(&state, "alpha", "ready").await;

        let statuses = state.collect_model_state_snapshots().await;

        // Length matches the number of configured models.
        assert_eq!(statuses.len(), 2);

        // Entries are sorted by id ascending.
        let ids: Vec<&str> = statuses.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "zephyr"]);

        // Exactly one model is reported as loaded.
        let loaded_count = statuses
            .iter()
            .filter(|s| matches!(s.state, crate::gpu::ModelState::Ready))
            .count();
        assert_eq!(
            loaded_count, 1,
            "expected exactly one loaded model, got: {:?}",
            statuses
        );

        // alpha is loaded with the configured backend.
        assert_eq!(statuses[0].id, "alpha");
        assert_eq!(
            statuses[0].state,
            crate::gpu::ModelState::Ready,
            "expected alpha to be ready"
        );
        assert_eq!(statuses[0].backend, "vllm");

        // zephyr is not loaded but still carries its configured backend.
        assert_eq!(statuses[1].id, "zephyr");
        assert_eq!(
            statuses[1].state,
            crate::gpu::ModelState::Idle,
            "expected zephyr to not be ready"
        );
        assert_eq!(statuses[1].backend, "llama_cpp");
    }

    /// `collect_model_state_snapshots` should only treat a `ready` wire row as
    /// "loaded". Other variants like `Starting` and `Failed` must be
    /// reported as `loaded == false` so the dashboard does not falsely
    /// claim a model is serving traffic while it is still booting or has
    /// crashed.
    #[tokio::test]
    async fn test_collect_model_state_snapshots_ignores_non_ready_states() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Populate model_configs
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("alpha".to_string(), make_model_config("llama_cpp"));
        }

        // The server name `resolve_backends_for_model("alpha")` returns is
        // the config key itself, since `make_model_config` leaves
        // `model` as `None`.
        let backend_name = "alpha".to_string();

        // --- Case 1: Starting must NOT count as loaded ---------------------
        seed_live_row(&state, &backend_name, "starting").await;

        let statuses = state.collect_model_state_snapshots().await;
        assert_eq!(
            statuses.len(),
            1,
            "expected one status entry per configured model, got: {:?}",
            statuses
        );
        let alpha = statuses
            .iter()
            .find(|s| s.id == "alpha")
            .expect("alpha entry missing from collect_model_state_snapshots output");
        assert_eq!(
            alpha.state,
            crate::gpu::ModelState::Starting,
            "a starting wire row must be reported as starting, not ready, got: {:?}",
            alpha
        );

        // --- Case 2: a dead/failed process is NOT a live row ------
        // Flip (plan-193 T4): a failed backend reports no eligible process
        // on the wire, so the model reads Idle — "no host = no models",
        // never a phantom "failed" entry.
        seed_live_row(&state, &backend_name, "failed").await;

        let statuses = state.collect_model_state_snapshots().await;
        assert_eq!(
            statuses.len(),
            1,
            "expected one status entry per configured model, got: {:?}",
            statuses
        );
        let alpha = statuses
            .iter()
            .find(|s| s.id == "alpha")
            .expect("alpha entry missing from collect_model_state_snapshots output");
        assert_eq!(
            alpha.state,
            crate::gpu::ModelState::Idle,
            "a failed/dead process is not a live row and must read Idle, got: {:?}",
            alpha
        );
    }

    /// Regression guard for #192: the `Starting` placeholder mirror that
    /// `load_spec_on_tamad` inserts before the blocking `LoadModel` RPC must
    /// be surfaced as `starting` (the Active Models filter keeps it visible
    /// while the model loads), not as `idle` (no runtime entry).
    #[tokio::test]
    async fn test_collect_model_state_snapshots_maps_load_window_starting_mirror() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("start-model".to_string(), make_model_config("llama_cpp"));
        }

        // The load-window is now a live `starting` wire row (the tamad emits
        // it while the in-flight `LoadModel` RPC blocks), not a local mirror
        // placeholder — and the dashboard reads rows (plan-193 T4).
        seed_live_row(&state, "start-model", "starting").await;

        let statuses = state.collect_model_state_snapshots().await;
        assert_eq!(
            statuses.len(),
            1,
            "expected one status entry per configured model, got: {:?}",
            statuses
        );
        let entry = statuses
            .iter()
            .find(|s| s.id == "start-model")
            .expect("start-model entry missing from collect_model_state_snapshots output");
        assert_eq!(
            entry.state,
            crate::gpu::ModelState::Starting,
            "the load-window Starting mirror must be reported as Starting (not Idle): {:?}",
            entry
        );
    }

    /// `host_name` is display-only: when the owning provider cannot be
    /// resolved (here: the dummy pool has no providers table at all, so the
    /// DB lookup fails), the snapshot must still be produced with
    /// `host_name == None` rather than erroring or panicking.
    #[tokio::test]
    async fn test_collect_model_state_snapshots_host_name_none_when_provider_unresolvable() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("zephyr".to_string(), make_model_config("llama_cpp"));
        }

        let statuses = state.collect_model_state_snapshots().await;
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].id, "zephyr");
        assert!(
            statuses[0].host_name.is_none(),
            "unresolvable provider must yield host_name None, got: {:?}",
            statuses[0].host_name
        );
    }

    /// `host_name` feeds the dashboard's host-fleet grouping
    /// (`host_name == hosts[].name`), and `hosts[].name` is the tamad
    /// **connection display name** (`TamadHandle.connection.name`). When the
    /// provider's nickname differs from the tamad's display name (here:
    /// provider "local-radiance" on tamad "tama"), the snapshot must carry
    /// the connection name — the provider name matches no host card and the
    /// model would land in "Unassigned".
    #[tokio::test]
    async fn test_collect_model_state_snapshots_host_name_uses_tamad_connection_name() {
        use crate::tamad::pool::test_support::grpc_conn;
        use crate::testing::postgres::with_schema;

        let guard = with_schema().await;
        let pool = Arc::new(guard.pool.clone());
        let config = Config::default();
        let state = ProxyState::new(config, None, pool.clone());

        // Provider whose nickname differs from the tamad's display name.
        crate::db::queries::insert_provider(
            pool.as_ref(),
            "local-radiance",
            "local",
            "llama_cpp",
            Some("tamad-1"),
            None,
            None,
        )
        .await
        .unwrap();

        let mut mc = make_model_config("llama_cpp");
        mc.provider_name = Some("local-radiance".to_string());
        state
            .registry
            .model_configs
            .write()
            .await
            .insert("zephyr".to_string(), mc);

        // Pool handle for the provider's tamad: connection.name is what the
        // SSE hosts[] array exposes as HostStats.name.
        let conn = grpc_conn("tamad-1", "tama", "grpc://127.0.0.1:1");
        state.tamad_pool().upsert_connection(&conn).await.unwrap();

        let statuses = state.collect_model_state_snapshots().await;
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].id, "zephyr");
        assert_eq!(
            statuses[0].host_name.as_deref(),
            Some("tama"),
            "host_name must be the tamad connection display name (hosts[].name), \
             not the provider nickname"
        );

        let _ = guard.finish().await;
    }

    /// When the provider resolves but its `tamad_id` has no live handle in
    /// the pool (tamad offline/never connected), `host_name` falls back to
    /// None — the model lands in the dashboard's "Unassigned" group rather
    /// than being mis-attributed.
    #[tokio::test]
    async fn test_collect_model_state_snapshots_host_name_none_when_tamad_not_in_pool() {
        use crate::testing::postgres::with_schema;

        let guard = with_schema().await;
        let pool = Arc::new(guard.pool.clone());
        let config = Config::default();
        let state = ProxyState::new(config, None, pool.clone());

        crate::db::queries::insert_provider(
            pool.as_ref(),
            "local-radiance",
            "local",
            "llama_cpp",
            Some("tamad-1"),
            None,
            None,
        )
        .await
        .unwrap();

        let mut mc = make_model_config("llama_cpp");
        mc.provider_name = Some("local-radiance".to_string());
        state
            .registry
            .model_configs
            .write()
            .await
            .insert("zephyr".to_string(), mc);

        // No handle upserted for "tamad-1".
        let statuses = state.collect_model_state_snapshots().await;
        assert_eq!(statuses.len(), 1);
        assert!(
            statuses[0].host_name.is_none(),
            "missing pool handle must yield host_name None, got: {:?}",
            statuses[0].host_name
        );

        let _ = guard.finish().await;
    }

    // ── Drift-guard: /status response round-trip ──────────────────────────────

    /// The StatusResponse struct must faithfully represent the full wire shape.
    /// Deserializing the serialized body back into StatusResponse and comparing
    /// against the raw Value ensures no fields are silently dropped or invented.
    #[tokio::test]
    async fn test_status_response_roundtrip_lossless() {
        use crate::gpu::{SystemMetrics, VramInfo};

        // Build a minimal fixture with one ready model and vram present.
        let mut config = Config::default();
        config.backends.insert(
            "llama_cpp".to_string(),
            BackendConfig {
                path: Some("/opt/llama".into()),
                version: None,
                gpu_variant: None,
            },
        );
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "ready-model".to_string(),
                ModelConfig {
                    backend: "llama_cpp".to_string(),
                    display_name: Some("Ready".to_string()),
                    model: Some("test/model".to_string()),
                    db_id: Some(42),
                    enabled: true,
                    ..Default::default()
                },
            );
        }

        // Seed a ready wire row for the in-profile model (plan-193 T5c:
        // state comes from rows).
        seed_live_row(&state, "ready-model", "ready").await;

        {
            state
                .metrics
                .set_system_metrics(SystemMetrics {
                    cpu_usage_pct: 10.0,
                    ram_used_mib: 512,
                    ram_total_mib: 4096,
                    gpu_utilization_pct: Some(50),
                    vram: Some(VramInfo {
                        used_mib: 100,
                        total_mib: 8192,
                    }),
                    ..Default::default()
                })
                .await;
        }

        let response = state.build_status_response().await;
        let body_bytes = serde_json::to_vec(&response).unwrap();

        // Deserialize into the typed struct.
        let parsed: StatusResponse =
            serde_json::from_slice(&body_bytes).expect("body must deserialize into StatusResponse");

        // Lossless round-trip: re-serialize parsed and compare to original Value.
        let raw_value: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("body must be valid JSON");
        assert_eq!(
            serde_json::to_value(&parsed).expect("parsed must serialize"),
            raw_value,
            "StatusResponse round-trip must be lossless — struct fields must match wire shape exactly"
        );
    }

    // ── vLLM field resolution in ModelStateSnapshot ──────────────────────────

    /// When GGUF `quant` is `Some`, it takes priority over `vllm.quantization`.
    #[tokio::test]
    async fn test_vllm_quant_fallback_gguf_wins() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    Some("Q4_K_M".to_string()), // GGUF quant
                    None,
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM quantization
                    None,
                    None,
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.quant,
            Some("Q4_K_M".to_string()),
            "GGUF quant should take priority over vLLM quantization"
        );
    }

    /// When GGUF `quant` is `None`, falls back to `vllm.quantization`.
    #[tokio::test]
    async fn test_vllm_quant_fallback_uses_vllm() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None, // GGUF quant is None
                    None,
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM quantization
                    None,
                    None,
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.quant,
            Some("fp8".to_string()),
            "Should fall back to vLLM quantization when GGUF quant is None"
        );
    }

    /// When both GGUF `quant` and `vllm.quantization` are `None`, returns `None`.
    #[tokio::test]
    async fn test_vllm_quant_fallback_both_none() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm", None, // GGUF quant
                    None, None, None, None, // vLLM quantization
                    None, None,
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.quant, None,
            "Should be None when both GGUF quant and vLLM quantization are None"
        );
    }

    /// When GGUF `context_length` is `Some`, it takes priority over `vllm.max_model_len`.
    #[tokio::test]
    async fn test_vllm_context_length_fallback_gguf_wins() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    Some(4096), // GGUF context_length
                    None,
                    None,
                    None,
                    Some(8192), // vLLM max_model_len
                    None,
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.context_length,
            Some(4096),
            "GGUF context_length should take priority over vLLM max_model_len"
        );
    }

    /// When GGUF `context_length` is `None`, falls back to `vllm.max_model_len`.
    #[tokio::test]
    async fn test_vllm_context_length_fallback_uses_vllm() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    None, // GGUF context_length is None
                    None,
                    None,
                    None,
                    Some(8192), // vLLM max_model_len
                    None,
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.context_length,
            Some(8192),
            "Should fall back to vLLM max_model_len when GGUF context_length is None"
        );
    }

    /// When both GGUF `context_length` and `vllm.max_model_len` are `None`, returns `None`.
    #[tokio::test]
    async fn test_vllm_context_length_fallback_both_none() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm", None, None, // GGUF context_length
                    None, None, None, None, // vLLM max_model_len
                    None,
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.context_length, None,
            "Should be None when both GGUF context_length and vLLM max_model_len are None"
        );
    }

    /// When GGUF `cache_type_k` is `Some`, it takes priority over `vllm.kv_cache_dtype`.
    #[tokio::test]
    async fn test_vllm_cache_type_k_fallback_gguf_wins() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    None,
                    Some("q4_0".to_string()), // GGUF cache_type_k
                    None,
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM kv_cache_dtype
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.cache_type_k,
            Some("q4_0".to_string()),
            "GGUF cache_type_k should take priority over vLLM kv_cache_dtype"
        );
    }

    /// When GGUF `cache_type_k` is `None`, falls back to `vllm.kv_cache_dtype`.
    #[tokio::test]
    async fn test_vllm_cache_type_k_fallback_uses_vllm() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    None,
                    None, // GGUF cache_type_k is None
                    None,
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM kv_cache_dtype
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.cache_type_k,
            Some("fp8".to_string()),
            "Should fall back to vLLM kv_cache_dtype when GGUF cache_type_k is None"
        );
    }

    /// When GGUF `cache_type_v` is `Some`, it takes priority over `vllm.kv_cache_dtype`.
    #[tokio::test]
    async fn test_vllm_cache_type_v_fallback_gguf_wins() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    None,
                    None,
                    Some("q8_0".to_string()), // GGUF cache_type_v
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM kv_cache_dtype
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.cache_type_v,
            Some("q8_0".to_string()),
            "GGUF cache_type_v should take priority over vLLM kv_cache_dtype"
        );
    }

    /// When GGUF `cache_type_v` is `None`, falls back to `vllm.kv_cache_dtype`.
    #[tokio::test]
    async fn test_vllm_cache_type_v_fallback_uses_vllm() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    None,
                    None,
                    None, // GGUF cache_type_v is None
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM kv_cache_dtype
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.cache_type_v,
            Some("fp8".to_string()),
            "Should fall back to vLLM kv_cache_dtype when GGUF cache_type_v is None"
        );
    }

    /// `cache_type_k` and `cache_type_v` resolve independently — each falls back
    /// to `vllm.kv_cache_dtype` only when its own column is `None`.
    #[tokio::test]
    async fn test_vllm_cache_types_resolve_independently() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // cache_type_k is Some, cache_type_v is None — only V should fall back
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "safetensors-model".to_string(),
                make_model_config_with_vllm(
                    "vllm",
                    None,
                    None,
                    Some("q4_0".to_string()), // GGUF cache_type_k
                    None,                     // GGUF cache_type_v is None
                    None,
                    None,
                    Some("fp8".to_string()), // vLLM kv_cache_dtype
                ),
            );
        }

        let statuses = state.collect_model_state_snapshots().await;
        let snap = statuses
            .iter()
            .find(|s| s.id == "safetensors-model")
            .unwrap();
        assert_eq!(
            snap.cache_type_k,
            Some("q4_0".to_string()),
            "cache_type_k should use GGUF value"
        );
        assert_eq!(
            snap.cache_type_v,
            Some("fp8".to_string()),
            "cache_type_v should fall back to vLLM kv_cache_dtype"
        );
    }
}
