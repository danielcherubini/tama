use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Width of each aggregation bucket in milliseconds (30 seconds).
const BUCKET_MS: i64 = 30_000;

/// Maximum number of frozen (complete) buckets to retain in the ring.
/// Together with the trailing in-progress bucket this yields ~31 bars for
/// a 15-minute window.
const MAX_FROZEN_BUCKETS: usize = 30;

/// Floor a Unix millisecond timestamp to the start of its 30s wall-clock
/// bucket. Stable across ticks — the boundary depends only on the timestamp,
/// not on when the collector started.
fn bucket_start(ts_ms: i64) -> i64 {
    (ts_ms / BUCKET_MS) * BUCKET_MS
}

/// Fold the tamad-observed inference stats (ADR-0014) off the live rows into
/// the per-server `inference_stats` map.
///
/// Must run in the metrics loop BEFORE the step-2 snapshot — no await sits
/// between the merge and that snapshot, so the merged value lands in this
/// tick's `MetricCurrent` (surfacing within the ~2 s iteration cycle);
/// reordered, it would only surface next tick.
///
/// Semantics (ADR-0014): the row is the SOLE WRITER OF THE VALUES — overwrite
/// all five fields unconditionally, INCLUDING `None`. (Other paths — lifecycle
/// unload, rename, reset, conn-error cleanup — still remove or migrate entries;
/// see the task context.) The row is itself 30s-gated by the tamad (a `None`
/// means "no traffic for 30s" and must blank the entry); the pre-ADR-0014
/// "skip stale defaults" or-merge is retired. `last_updated_ms` is stamped
/// (proxy clock) ONLY when the row's `tps` is `Some` — stamping unconditionally
/// would make it ~equal across all entries (the merge touches every entry every
/// tick) and `aggregate_inference`'s "latest entry wins" would degenerate to
/// arbitrary.
pub(crate) async fn merge_tamad_inference_stats(
    state: &crate::proxy::ProxyState,
    live: &crate::proxy::Rows,
) {
    let cfg = state.config.read().await;
    let model_configs = state.registry.model_configs.read().await;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    for row in live.all() {
        let tps = row.tps;
        let prompt_tps = row.prompt_tps;
        let cache_hit_pct = row.cache_hit_pct;
        let spec_accept_pct = row.spec_accept_pct;
        let active = row.spec_decoding_active;
        let stamp = tps.is_some();
        let servers = cfg.resolve_backends_for_model(&model_configs, &row.key);
        for (server_name, _, _) in &servers {
            let sn = server_name.clone();
            state.metrics.modify_inference_stats(|m| {
                let entry = m.entry(sn.clone()).or_default();
                entry.tps = tps;
                entry.prompt_tps = prompt_tps;
                entry.cache_hit_pct = cache_hit_pct;
                entry.spec_accept_pct = spec_accept_pct;
                entry.spec_decoding_active = active;
                if stamp {
                    entry.last_updated_ms = now_ms;
                }
            });
        }
    }
}

/// Live-value aggregation for the broadcast snapshot. tps/prompt_tps AND
/// spec_accept_pct are None when the newest entry is older than the 30 s
/// bucket window — a BACKSTOP since ADR-0014 (entries are already 30s-gated
/// by the tamad; the gate now only fires on edge cases like a wedged row
/// stream). `cache_hit_pct` keeps the newest entry's value when stale — a
/// no-op backstop under ADR-0014 (the entry's `cache_hit_pct` is already
/// `None` after 30s idle, so the ungated read returns `None` anyway).
/// `spec_decoding_active` is OR'd across entries (cluster level) but is no
/// longer sticky per entry: the merge overwrites it, so an entry un-sticks
/// when its backend stops spec-decoding.
// The 6-element return tuple is consumed once, into locals at the metrics
// loop's single call site — a struct would add indirection for zero callers.
#[allow(clippy::type_complexity)]
pub(crate) fn aggregate_inference(
    inference_map: &std::collections::HashMap<String, crate::proxy::types::LatestInferenceStats>,
    now_ms: i64,
    stale_threshold_ms: i64,
) -> (
    Option<f32>,
    Option<f32>,
    Option<f32>,
    Option<f32>,
    bool,
    Option<i64>,
) {
    let latest = inference_map.values().max_by_key(|s| s.last_updated_ms);
    match latest {
        Some(s) if now_ms - s.last_updated_ms <= stale_threshold_ms => (
            s.tps,
            s.prompt_tps,
            s.cache_hit_pct,
            s.spec_accept_pct,
            inference_map.values().any(|s| s.spec_decoding_active),
            Some(s.last_updated_ms),
        ),
        _ => (
            None,
            None,
            latest.and_then(|s| s.cache_hit_pct),
            None,
            inference_map.values().any(|s| s.spec_decoding_active),
            latest.map(|s| s.last_updated_ms),
        ),
    }
}

/// Collect a CPU/RAM-only system snapshot from the persistent
/// [`sysinfo::System`].
///
/// The proxy card reports the host the proxy process itself runs on — the
/// one host fact the proxy may keep sampling (ADR-0010, plan-191 Task 10):
/// no GPU fields, no subprocesses. Per-host GPU facts come from the tamad
/// stats streams (Task 9).
fn collect_system_metrics_cpu_ram(sys: &mut sysinfo::System) -> crate::gpu::SystemMetrics {
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    crate::gpu::SystemMetrics {
        cpu_usage_pct: sys.global_cpu_info().cpu_usage(),
        ram_used_mib: sys.used_memory() / 1024 / 1024,
        ram_total_mib: sys.total_memory() / 1024 / 1024,
        gpu_utilization_pct: None,
        vram: None,
        gpus: Vec::new(),
        network: None,
    }
}

/// Accumulator for the in-progress 30s bucket. Sums each 2s sample's fields
/// and produces a [`MetricBucket`] with averaged values when frozen or polled.
///
/// When a new sample's timestamp falls in a different 30s window than the
/// accumulator's, the caller freezes the old bucket (via [`to_bucket`]) and
/// starts a fresh accumulator for the new window. This guarantees completed
/// buckets are immutable — only the trailing in-progress bucket changes as
/// new samples arrive.
struct BucketAccumulator {
    bucket_start_ms: i64,
    cpu_sum: f64,
    ram_used_sum: f64,
    ram_total_last: u64,
    network_dl_sum: f64,
    network_ul_sum: f64,
    has_network: bool,
    /// Per-GPU running utilization sum. Index aligns with the order GPUs
    /// appear in `MetricSample.gpus`. Resized as new GPUs appear; a missing
    /// index is treated as 0.0 utilization.
    gpu_util_sums: Vec<f64>,
    /// Running sum of generation tok/s from samples that reported values.
    tps_sum: f64,
    /// Count of samples that reported a non-None tps value.
    tps_count: usize,
    /// Running sum of prompt tok/s from samples that reported values.
    prompt_tps_sum: f64,
    /// Count of samples that reported a non-None prompt_tps value.
    prompt_tps_count: usize,
    count: usize,
}

impl BucketAccumulator {
    fn new(bucket_start_ms: i64) -> Self {
        Self {
            bucket_start_ms,
            cpu_sum: 0.0,
            ram_used_sum: 0.0,
            ram_total_last: 0,
            network_dl_sum: 0.0,
            network_ul_sum: 0.0,
            has_network: false,
            gpu_util_sums: Vec::new(),
            tps_sum: 0.0,
            tps_count: 0,
            prompt_tps_sum: 0.0,
            prompt_tps_count: 0,
            count: 0,
        }
    }

    /// Add a 2s sample's graphable fields to this bucket's running sums.
    fn add(&mut self, sample: &crate::gpu::MetricSample) {
        self.cpu_sum += sample.cpu_usage_pct as f64;
        self.ram_used_sum += sample.ram_used_mib as f64;
        self.ram_total_last = sample.ram_total_mib;
        if let Some(ref net) = sample.network {
            self.network_dl_sum += net.download_mibps;
            self.network_ul_sum += net.upload_mibps;
            self.has_network = true;
        }
        // Track per-GPU utilization sums. Resize to fit the highest GPU index
        // seen so far; missing indices stay 0.0 (treated as 0% util).
        for (i, gpu) in sample.gpus.iter().enumerate() {
            if i >= self.gpu_util_sums.len() {
                self.gpu_util_sums.resize(i + 1, 0.0);
            }
            self.gpu_util_sums[i] += gpu.utilization_pct.unwrap_or(0) as f64;
        }
        // Accumulate inference stats (only from samples that reported values)
        if let Some(tps) = sample.tps {
            self.tps_sum += tps as f64;
            self.tps_count += 1;
        }
        if let Some(prompt_tps) = sample.prompt_tps {
            self.prompt_tps_sum += prompt_tps as f64;
            self.prompt_tps_count += 1;
        }
        self.count += 1;
    }

    /// Produce a [`MetricBucket`] from the accumulated sums. `complete`
    /// should be `true` when the 30s window has elapsed (frozen) or `false`
    /// for the trailing in-progress bucket.
    fn to_bucket(&self, complete: bool) -> crate::gpu::MetricBucket {
        let n = self.count.max(1) as f64;
        crate::gpu::MetricBucket {
            ts_unix_ms: self.bucket_start_ms,
            cpu_usage_pct: (self.cpu_sum / n) as f32,
            ram_used_mib: (self.ram_used_sum / n) as u64,
            ram_total_mib: self.ram_total_last,
            network: if self.has_network {
                Some(crate::network::NetworkStats {
                    download_mibps: self.network_dl_sum / n,
                    upload_mibps: self.network_ul_sum / n,
                })
            } else {
                None
            },
            gpu_utils: self.gpu_util_sums.iter().map(|s| (s / n) as f32).collect(),
            tps: if self.tps_count > 0 {
                (self.tps_sum / self.tps_count as f64) as f32
            } else {
                0.0
            },
            prompt_tps: if self.prompt_tps_count > 0 {
                (self.prompt_tps_sum / self.prompt_tps_count as f64) as f32
            } else {
                0.0
            },
            complete,
        }
    }
}

/// Feed a sample into the accumulator, freezing the previous bucket when the
/// sample crosses a 30s boundary. Shared between DB-seed replay and the live
/// collection loop so both paths produce identical bucket structure.
fn feed_sample(
    frozen: &mut VecDeque<crate::gpu::MetricBucket>,
    accum: &mut Option<BucketAccumulator>,
    sample: &crate::gpu::MetricSample,
) {
    let bs = bucket_start(sample.ts_unix_ms);
    if let Some(a) = accum.as_mut() {
        if bs > a.bucket_start_ms {
            // Sample crossed into a new 30s window — freeze the old bucket.
            frozen.push_back(a.to_bucket(true));
            while frozen.len() > MAX_FROZEN_BUCKETS {
                frozen.pop_front();
            }
            *a = BucketAccumulator::new(bs);
        }
    } else {
        *accum = Some(BucketAccumulator::new(bs));
    }
    accum.as_mut().unwrap().add(sample);
}

/// Start the system metrics collection background task.
///
/// Creates an in-memory history buffer seeded from Postgres, then spawns
/// a task that periodically collects system metrics, persists them,
/// updates the buffer, and broadcasts to subscribers.
///
/// Returns a `JoinHandle` that can be stored to prevent task cancellation.
pub fn start_metrics_collector(
    state: Arc<crate::proxy::ProxyState>,
) -> tokio::task::JoinHandle<()> {
    let mut frozen_buckets: VecDeque<crate::gpu::MetricBucket> =
        VecDeque::with_capacity(MAX_FROZEN_BUCKETS + 1);
    let mut accum: Option<BucketAccumulator> = None;

    // Spawn background task to refresh system metrics every 2s.
    // Each tick: collect metrics, build unified sample (system + inference),
    // persist to Postgres, update in-memory buffer, broadcast full buffer.
    let metrics_state = Arc::clone(&state);
    tokio::spawn(async move {
        // Seed in-memory bucket accumulator from Postgres. Replay the most
        // recent raw rows through the same feed_sample() path used by the
        // live loop so the seeded buckets have identical structure to live ones.
        let pool = metrics_state.db_pool();
        if let Ok(rows) = crate::db::queries::get_recent_system_metrics(&pool, 450).await {
            for row in rows {
                let sample = row_into_sample(&row);
                feed_sample(&mut frozen_buckets, &mut accum, &sample);
            }
        }

        let mut sys = sysinfo::System::new();

        // Network detection — done once before the loop
        let primary_interface = crate::network::get_primary_interface();
        if let Some(ref iface) = primary_interface {
            tracing::info!("Using primary network interface: {}", iface);
        }

        // Before the loop: Create Networks instance and establish baseline
        let mut net = sysinfo::Networks::new_with_refreshed_list();
        let mut prev_rx: u64 = 0;
        let mut prev_tx: u64 = 0;

        // First refresh to establish baseline
        net.refresh();

        // Capture baseline cumulative values so the first tick doesn't include
        // all historical bytes since system boot
        if let Some(ref iface) = primary_interface {
            if let Some(iface_data) = net.get(iface) {
                prev_rx = iface_data.total_received();
                prev_tx = iface_data.total_transmitted();
            }
        }

        loop {
            // 1. Collect system metrics (spawn_blocking, unchanged pattern).
            // CPU/RAM only: the proxy samples its own process host for its
            // dashboard card — never local GPUs (ADR-0010; per-host GPU
            // facts come from the tamad stats streams, plan-191 Task 9).
            let (snapshot, returned_sys) = tokio::task::spawn_blocking(move || {
                let snapshot = collect_system_metrics_cpu_ram(&mut sys);
                (snapshot, sys)
            })
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("system metrics collection panicked: {}", e);
                (crate::gpu::SystemMetrics::default(), sysinfo::System::new())
            });
            sys = returned_sys;

            // 1b. Collect network stats
            let (network_stats, cum_rx, cum_tx) = if let Some(ref iface) = primary_interface {
                let (stats, rx, tx) =
                    crate::network::collect_network_stats(iface, &mut net, prev_rx, prev_tx);
                prev_rx = rx;
                prev_tx = tx;
                (stats, rx, tx)
            } else {
                (None, 0, 0)
            };

            // 1c. Attach network stats to the system snapshot
            let mut snapshot = snapshot;
            snapshot.network = network_stats.clone();

            // Update the cached snapshot read by /tama/v1/system/health.
            metrics_state
                .metrics
                .set_system_metrics(snapshot.clone())
                .await;

            // 2. Collect live rows first — they feed both the spec merge
            // below and `models_loaded` further down (plan-193 T4).
            let live = crate::proxy::live_rows(metrics_state.tamad_pool().as_ref()).await;

            // 2a. Fold the tamad-observed inference stats into the
            // per-server map BEFORE the snapshot (ADR-0014)
            merge_tamad_inference_stats(&metrics_state, &live).await;

            // 2b. Read latest inference stats and aggregate across servers
            // (latest-server values, freshness-gated; sticky flag OR'd).
            let inference_map = metrics_state.metrics.inference_stats_snapshot();
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let (
                tps,
                prompt_tps,
                cache_hit_pct,
                spec_accept_pct,
                spec_decoding_active,
                inference_last_updated_ms,
            ) = aggregate_inference(&inference_map, now_ms, BUCKET_MS);

            // 3. Collect model statuses.
            let model_statuses = metrics_state.collect_model_state_snapshots().await;
            // The wire `models_loaded` name is kept, but its source is now the
            // live may-still row ready count (plan-193 T4) — a *current*
            // ready-count semantics switch, driven by rows.ready_count(), not
            // a tally against the staging mirror.
            let models_loaded = live.ready_count() as u64;

            // 4. Build unified MetricSample WITH inference fields
            let sample = crate::gpu::MetricSample {
                ts_unix_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0),
                cpu_usage_pct: snapshot.cpu_usage_pct,
                ram_used_mib: snapshot.ram_used_mib,
                ram_total_mib: snapshot.ram_total_mib,
                gpu_utilization_pct: snapshot.gpu_utilization_pct,
                vram: snapshot.vram.clone(),
                gpus: snapshot.gpus.clone(),
                models_loaded,
                models: model_statuses,
                tps,
                prompt_tps,
                cache_hit_pct,
                spec_accept_pct,
                spec_decoding_active,
                inference_last_updated_ms,
                network: network_stats.clone(),
            };

            // 5. Persist to Postgres (include inference fields in SystemMetricsRow)
            let row = crate::db::queries::SystemMetricsRow {
                ts_unix_ms: sample.ts_unix_ms,
                cpu_usage_pct: sample.cpu_usage_pct,
                ram_used_mib: sample.ram_used_mib as i64,
                ram_total_mib: sample.ram_total_mib as i64,
                gpu_utilization_pct: sample.gpu_utilization_pct.map(|v| v as i64),
                vram_used_mib: sample.vram.as_ref().map(|v| v.used_mib as i64),
                vram_total_mib: sample.vram.as_ref().map(|v| v.total_mib as i64),
                models_loaded: sample.models_loaded as i64,
                tps: sample.tps.map(|v| v as f64),
                prompt_tps: sample.prompt_tps.map(|v| v as f64),
                cache_hit_pct: sample.cache_hit_pct.map(|v| v as f64),
                spec_accept_pct: sample.spec_accept_pct.map(|v| v as f64),
                net_rx_bytes: Some(cum_rx as i64),
                net_tx_bytes: Some(cum_tx as i64),
            };
            // Persist (non-fatal — a DB hiccup must not kill the collector)
            let retention_secs = metrics_state
                .config
                .read()
                .await
                .proxy
                .metrics_retention_secs;
            let cutoff_ms = sample.ts_unix_ms - (retention_secs as i128 * 1000) as i64;
            let pool = metrics_state.db_pool();
            if let Err(e) = crate::db::queries::insert_system_metric(&pool, &row, cutoff_ms).await {
                tracing::warn!("failed to persist system metric: {}", e);
            }

            // 6. Feed this sample into the bucket accumulator. This freezes
            // the previous bucket when the sample crosses a 30s boundary,
            // guaranteeing completed buckets are immutable thereafter.
            feed_sample(&mut frozen_buckets, &mut accum, &sample);

            // Derive the trailing in-progress bucket from the accumulator.
            let in_progress = accum.as_ref().map(|a| a.to_bucket(false));

            // 7. Broadcast: frozen buckets + in-progress last + current
            //    (instantaneous CPU/RAM/Network + GPU/model/inference state).
            let mut all_buckets = frozen_buckets.make_contiguous().to_vec();
            if let Some(b) = in_progress {
                all_buckets.push(b);
            }
            let current = sample.clone().into_current();
            let snapshot = crate::gpu::MetricsSnapshot {
                buckets: all_buckets,
                current,
            };
            metrics_state.metrics.publish_metrics(snapshot);

            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    })
}

/// Convert a `SystemMetricsRow` from Postgres into a `MetricSample`.
/// Used to seed the in-memory history buffer on startup.
fn row_into_sample(row: &crate::db::queries::SystemMetricsRow) -> crate::gpu::MetricSample {
    crate::gpu::MetricSample {
        ts_unix_ms: row.ts_unix_ms,
        cpu_usage_pct: row.cpu_usage_pct,
        ram_used_mib: row.ram_used_mib.max(0) as u64,
        ram_total_mib: row.ram_total_mib.max(0) as u64,
        gpu_utilization_pct: row.gpu_utilization_pct.and_then(|v| {
            if (0..=100).contains(&v) {
                Some(v as u8)
            } else {
                None
            }
        }),
        vram: row.vram_used_mib.and_then(|used| {
            row.vram_total_mib.map(|total| crate::gpu::VramInfo {
                used_mib: used.max(0) as u64,
                total_mib: total.max(0) as u64,
            })
        }),
        models_loaded: row.models_loaded.max(0) as u64,
        models: vec![], // Not stored in DB — seeded samples have no model status
        gpus: vec![],   // historical rows don't store per-GPU; left empty
        tps: row.tps.map(|v| v as f32),
        prompt_tps: row.prompt_tps.map(|v| v as f32),
        cache_hit_pct: row.cache_hit_pct.map(|v| v as f32),
        spec_accept_pct: row.spec_accept_pct.map(|v| v as f32),
        spec_decoding_active: false,     // Transient — not in DB
        inference_last_updated_ms: None, // Transient — not in DB
        network: None,                   // Throughput not reconstructable from single row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `MetricSample` with only the fields needed for
    /// `BucketAccumulator` tests. All non-inference fields use defaults.
    fn sample(tps: Option<f32>, prompt_tps: Option<f32>) -> crate::gpu::MetricSample {
        crate::gpu::MetricSample {
            ts_unix_ms: 0,
            cpu_usage_pct: 0.0,
            ram_used_mib: 0,
            ram_total_mib: 0,
            gpu_utilization_pct: None,
            vram: None,
            gpus: vec![],
            models_loaded: 0,
            models: vec![],
            tps,
            prompt_tps,
            cache_hit_pct: None,
            spec_accept_pct: None,
            spec_decoding_active: false,
            inference_last_updated_ms: None,
            network: None,
        }
    }

    /// Mixed Some/None samples average only over Some values.
    ///
    /// Three samples are added: two with Some(tps) and one with None.
    /// The bucket tps must be the mean of the two Some values, not diluted
    /// by the None sample. The same logic applies to prompt_tps.
    #[test]
    fn test_mixed_some_none_averages_only_some() {
        let mut acc = BucketAccumulator::new(0);

        acc.add(&sample(Some(10.0), Some(20.0)));
        acc.add(&sample(None, None));
        acc.add(&sample(Some(30.0), Some(40.0)));

        let bucket = acc.to_bucket(true);

        // tps: mean of 10.0 and 30.0 = 20.0 (None sample excluded)
        assert_eq!(bucket.tps, 20.0);
        // prompt_tps: mean of 20.0 and 40.0 = 30.0 (None sample excluded)
        assert_eq!(bucket.prompt_tps, 30.0);
    }

    /// All None produces 0.0 for both tps and prompt_tps.
    #[test]
    fn test_all_none_produces_zero() {
        let mut acc = BucketAccumulator::new(0);

        acc.add(&sample(None, None));
        acc.add(&sample(None, None));
        acc.add(&sample(None, None));

        let bucket = acc.to_bucket(true);

        assert_eq!(bucket.tps, 0.0);
        assert_eq!(bucket.prompt_tps, 0.0);
    }

    /// A single sample passes its tps/prompt_tps values through unchanged.
    #[test]
    fn test_single_sample_passes_through() {
        let mut acc = BucketAccumulator::new(0);

        acc.add(&sample(Some(55.5), Some(123.0)));

        let bucket = acc.to_bucket(true);

        assert_eq!(bucket.tps, 55.5);
        assert_eq!(bucket.prompt_tps, 123.0);
    }

    // ── inference-stats merge (ADR-0014) / aggregate_inference ────────

    use crate::config::ModelConfig;
    use std::collections::HashMap;

    fn model_config(backend: &str, model: Option<&str>) -> ModelConfig {
        ModelConfig {
            backend: backend.to_string(),
            model: model.map(|m| m.to_string()),
            // serde's `default = "super::default_enabled"` applies only at
            // deserialization — the derived Default leaves `enabled: false`,
            // which `resolve_backends_for_model` skips entirely.
            enabled: true,
            ..Default::default()
        }
    }

    /// A ready vLLM process carrying one tamad spec-decode observation.
    fn process(
        model_name: &str,
        spec_accept_pct: Option<f64>,
        spec_decoding_active: bool,
    ) -> crate::tamad::ProcessInfo {
        crate::tamad::ProcessInfo {
            model_name: model_name.to_string(),
            provider_name: "vllm".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://127.0.0.1:8000".to_string(),
            status: "ready".to_string(),
            desired: true,
            restart_count: 0,
            max_restarts: 3,
            spec_accept_pct,
            spec_decoding_active,
            tps: None,
            prompt_tps: None,
            cache_hit_pct: None,
        }
    }

    /// Sibling to [`process`]: a ready vLLM process also carrying the
    /// tamad's windowed rates (the ADR-0014 wire fields).
    fn process_with_rates(
        model_name: &str,
        tps: Option<f64>,
        prompt_tps: Option<f64>,
        cache_hit_pct: Option<f64>,
        spec_accept_pct: Option<f64>,
        spec_decoding_active: bool,
    ) -> crate::tamad::ProcessInfo {
        crate::tamad::ProcessInfo {
            model_name: model_name.to_string(),
            provider_name: "vllm".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://127.0.0.1:8000".to_string(),
            status: "ready".to_string(),
            desired: true,
            restart_count: 0,
            max_restarts: 3,
            spec_accept_pct,
            spec_decoding_active,
            tps,
            prompt_tps,
            cache_hit_pct,
        }
    }

    /// Seed a fresh live `ready` row into the state's tamad pool so
    /// [`crate::proxy::live_rows`] surfaces it with the given spec values.
    async fn seed_live_row(
        state: &crate::proxy::ProxyState,
        model_id: &str,
        spec_accept_pct: Option<f64>,
        spec_decoding_active: bool,
    ) {
        use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
        let stats = stats_full(
            1.5,
            vec![],
            vec![process(model_id, spec_accept_pct, spec_decoding_active)],
        );
        let pool = state.tamad_pool();
        pool.insert_raw_handle(
            model_id,
            Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
        )
        .await;
    }

    /// Sibling to [`seed_live_row`]: seed a fresh live `ready` row carrying
    /// the given windowed rates (the ADR-0014 wire fields), via the same
    /// `stats_full` / `handle_with_latest` / `insert_raw_handle` plumbing.
    async fn seed_live_row_with_rates(
        state: &crate::proxy::ProxyState,
        model_id: &str,
        tps: Option<f64>,
        prompt_tps: Option<f64>,
        cache_hit_pct: Option<f64>,
        spec_accept_pct: Option<f64>,
        spec_decoding_active: bool,
    ) {
        use crate::tamad::pool::test_support::{handle_with_latest, stats_full};
        let stats = stats_full(
            1.5,
            vec![],
            vec![process_with_rates(
                model_id,
                tps,
                prompt_tps,
                cache_hit_pct,
                spec_accept_pct,
                spec_decoding_active,
            )],
        );
        let pool = state.tamad_pool();
        pool.insert_raw_handle(
            model_id,
            Arc::new(handle_with_latest(std::time::Instant::now(), stats).await),
        )
        .await;
    }

    /// Fixture: empty `ProxyState` + one model config under `config_key`
    /// + a live ready tamad row reporting the given spec values.
    async fn state_with_live_model(
        config_key: &str,
        spec_accept_pct: Option<f64>,
        spec_decoding_active: bool,
    ) -> crate::proxy::ProxyState {
        let state = crate::proxy::ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        );
        state
            .registry
            .model_configs
            .write()
            .await
            .insert(config_key.to_string(), model_config("vllm", None));
        seed_live_row(&state, config_key, spec_accept_pct, spec_decoding_active).await;
        state
    }

    /// Merging a row into an existing forwarder-written entry OVERWRITES
    /// the entry's values — the row is the SOLE source of values (ADR-0014):
    /// a `tps: None` row blanks `tps`, and a `tps: Some` row stamps
    /// `last_updated_ms`.
    #[tokio::test]
    async fn test_merge_overwrites_forwarder_write() {
        let state = state_with_live_model("alpha", Some(44.5), true).await;
        state.metrics.modify_inference_stats(|m| {
            m.insert(
                "alpha".to_string(),
                crate::proxy::types::LatestInferenceStats {
                    tps: Some(50.0),
                    prompt_tps: Some(200.0),
                    cache_hit_pct: Some(85.0),
                    spec_accept_pct: None,
                    spec_decoding_active: false,
                    last_updated_ms: 123,
                },
            );
        });

        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;

        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("alpha").expect("existing entry updated in place");
        assert_eq!(stats.spec_accept_pct, Some(44.5));
        assert!(stats.spec_decoding_active);
        assert_eq!(stats.tps, None, "a tps: None row blanks the entry's tps");
        assert_eq!(
            stats.prompt_tps, None,
            "the row is the sole source of values"
        );
        assert_eq!(
            stats.cache_hit_pct, None,
            "the row is the sole source of values"
        );
        assert_eq!(stats.last_updated_ms, 123, "a tps: None row does not stamp");

        // A tps-bearing row stamps last_updated_ms.
        seed_live_row_with_rates(&state, "alpha", Some(42.0), None, None, Some(44.5), true).await;
        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;
        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("alpha").unwrap();
        assert_eq!(stats.tps, Some(42.0));
        assert!(
            stats.last_updated_ms > 123,
            "a tps: Some row stamps last_updated_ms"
        );
    }

    /// No existing entry → one is created by `or_default()` and carries
    /// the row's values (the row is the sole source).
    #[tokio::test]
    async fn test_merge_creates_entry_when_absent() {
        let state = state_with_live_model("beta", Some(44.5), true).await;

        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;

        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("beta").expect("merge creates the entry");
        assert_eq!(stats.spec_accept_pct, Some(44.5));
        assert!(stats.spec_decoding_active);
        assert_eq!(stats.tps, None, "or_default() entry + tps: None row");
        assert_eq!(stats.last_updated_ms, 0, "a tps: None row does not stamp");
    }

    /// A row at its stale defaults (all `None` / `false`) BLANKS the entry:
    /// the row is 30s-gated by the tamad, so a `None` means "no traffic for
    /// 30s" and the entry's values must go (ADR-0014).
    #[tokio::test]
    async fn test_merge_overwrites_including_none() {
        let state = state_with_live_model("gamma", None, false).await;
        state.metrics.modify_inference_stats(|m| {
            m.insert(
                "gamma".to_string(),
                crate::proxy::types::LatestInferenceStats {
                    tps: Some(50.0),
                    prompt_tps: Some(200.0),
                    cache_hit_pct: Some(85.0),
                    spec_accept_pct: Some(30.0),
                    spec_decoding_active: true,
                    last_updated_ms: 500,
                },
            );
        });

        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;

        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("gamma").unwrap();
        assert_eq!(stats.tps, None, "a None row blanks tps");
        assert_eq!(stats.prompt_tps, None, "a None row blanks prompt_tps");
        assert_eq!(stats.cache_hit_pct, None, "a None row blanks cache_hit_pct");
        assert_eq!(
            stats.spec_accept_pct, None,
            "a None row blanks spec_accept_pct"
        );
        assert!(
            !stats.spec_decoding_active,
            "a false row un-sticks spec_decoding_active"
        );
        assert_eq!(stats.last_updated_ms, 500, "a tps: None row does not stamp");
    }

    /// `last_updated_ms` is stamped ONLY when the row's `tps` is `Some` —
    /// an unconditional stamp would make it ~equal across all entries (the
    /// merge touches every entry every tick) and `aggregate_inference`'s
    /// "latest entry wins" would degenerate to arbitrary.
    #[tokio::test]
    async fn test_merge_stamps_last_updated_only_when_some() {
        let state = state_with_live_model("delta", None, false).await;
        state.metrics.modify_inference_stats(|m| {
            m.insert(
                "delta".to_string(),
                crate::proxy::types::LatestInferenceStats {
                    tps: Some(50.0),
                    prompt_tps: None,
                    cache_hit_pct: None,
                    spec_accept_pct: None,
                    spec_decoding_active: false,
                    last_updated_ms: 1000,
                },
            );
        });

        // A tps: None row must not touch last_updated_ms.
        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;
        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("delta").unwrap();
        assert_eq!(
            stats.last_updated_ms, 1000,
            "a tps: None row must not stamp"
        );

        // A tps: Some(1.0) row stamps last_updated_ms.
        seed_live_row_with_rates(&state, "delta", Some(1.0), None, None, None, false).await;
        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;
        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("delta").unwrap();
        assert!(
            stats.last_updated_ms > 1000,
            "a tps: Some row stamps last_updated_ms"
        );
        assert_eq!(stats.tps, Some(1.0));
    }

    /// A newer row's values replace the previous merge's values.
    #[tokio::test]
    async fn test_merge_replaces_previous_value() {
        let state = state_with_live_model("epsilon", None, false).await;

        seed_live_row_with_rates(&state, "epsilon", Some(50.0), None, None, None, false).await;
        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;
        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("epsilon").unwrap();
        assert_eq!(stats.tps, Some(50.0));

        seed_live_row_with_rates(&state, "epsilon", Some(30.0), None, None, None, false).await;
        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;
        let snap = state.metrics.inference_stats_snapshot();
        let stats = snap.get("epsilon").unwrap();
        assert_eq!(
            stats.tps,
            Some(30.0),
            "the newer row's value replaces the previous one"
        );
    }

    /// A model key resolving to multiple model configs (aliases) delivers
    /// the row's values to EVERY resolved config entry.
    #[tokio::test]
    async fn test_merge_fanout_aliases() {
        let state = crate::proxy::ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        );
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("a1".to_string(), model_config("vllm", Some("dl-shared")));
            mc.insert("b2".to_string(), model_config("vllm", Some("dl-shared")));
        }
        seed_live_row_with_rates(
            &state,
            "dl-shared",
            Some(42.0),
            Some(120.0),
            Some(25.0),
            Some(44.5),
            true,
        )
        .await;

        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;

        let snap = state.metrics.inference_stats_snapshot();
        for key in ["a1", "b2"] {
            let stats = snap
                .get(key)
                .unwrap_or_else(|| panic!("{key} entry missing"));
            assert_eq!(stats.tps, Some(42.0));
            assert_eq!(stats.prompt_tps, Some(120.0));
            assert_eq!(stats.cache_hit_pct, Some(25.0));
            assert_eq!(stats.spec_accept_pct, Some(44.5));
            assert!(stats.spec_decoding_active);
        }
    }

    /// A model key resolving to multiple model configs updates EVERY
    /// resolved server's inference entry.
    #[tokio::test]
    async fn test_merge_updates_all_resolving_servers() {
        let state = crate::proxy::ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        );
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert("a1".to_string(), model_config("vllm", Some("dl-shared")));
            mc.insert("b2".to_string(), model_config("vllm", Some("dl-shared")));
        }
        seed_live_row(&state, "dl-shared", Some(44.5), true).await;

        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;

        let snap = state.metrics.inference_stats_snapshot();
        assert_eq!(snap.get("a1").unwrap().spec_accept_pct, Some(44.5));
        assert_eq!(snap.get("b2").unwrap().spec_accept_pct, Some(44.5));
        assert!(snap.get("a1").unwrap().spec_decoding_active);
        assert!(snap.get("b2").unwrap().spec_decoding_active);
    }

    /// Wire → row → merge → aggregate: a live tamad row's windowed rates
    /// flow through `merge_tamad_inference_stats` into
    /// `aggregate_inference`'s output (ADR-0014).
    #[tokio::test]
    async fn test_merge_to_aggregate_end_to_end() {
        let state = crate::proxy::ProxyState::new(
            crate::config::Config::default(),
            None,
            crate::db::pool::test_dummy_pool(),
        );
        state
            .registry
            .model_configs
            .write()
            .await
            .insert("omega".to_string(), model_config("vllm", None));
        seed_live_row_with_rates(
            &state,
            "omega",
            Some(42.0),
            Some(120.0),
            Some(25.0),
            Some(44.5),
            true,
        )
        .await;

        let live = crate::proxy::live_rows(state.tamad_pool().as_ref()).await;
        merge_tamad_inference_stats(&state, &live).await;

        let inference_map = state.metrics.inference_stats_snapshot();
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let (tps, prompt_tps, cache_hit_pct, spec_accept_pct, active, last) =
            aggregate_inference(&inference_map, now_ms, BUCKET_MS);
        assert_eq!(tps, Some(42.0), "wire → row → merge → aggregate");
        assert_eq!(prompt_tps, Some(120.0));
        assert_eq!(cache_hit_pct, Some(25.0));
        assert_eq!(spec_accept_pct, Some(44.5));
        assert!(active);
        assert!(
            last.is_some(),
            "the stamped last_updated_ms surfaces in the aggregate"
        );
    }

    fn agg_map(last_updated_ms: i64) -> HashMap<String, crate::proxy::types::LatestInferenceStats> {
        let mut m = HashMap::new();
        m.insert(
            "srv".to_string(),
            crate::proxy::types::LatestInferenceStats {
                tps: Some(50.0),
                prompt_tps: Some(200.0),
                cache_hit_pct: Some(90.0),
                spec_accept_pct: Some(44.5),
                spec_decoding_active: true,
                last_updated_ms,
            },
        );
        m
    }

    /// A fresh (within the 30s window) newest entry surfaces tps,
    /// prompt_tps, cache_hit_pct, the tamad-merged spec_accept_pct, the
    /// OR'd spec_decoding_active flag, and its last_updated_ms.
    #[test]
    fn test_aggregate_inference_fresh_reports_all_fields() {
        let (tps, prompt_tps, cache_hit_pct, spec_accept_pct, active, last) =
            aggregate_inference(&agg_map(90_000), 100_000, BUCKET_MS);
        assert_eq!(tps, Some(50.0));
        assert_eq!(prompt_tps, Some(200.0));
        assert_eq!(cache_hit_pct, Some(90.0));
        assert_eq!(spec_accept_pct, Some(44.5), "fresh merged value surfaces");
        assert!(active);
        assert_eq!(last, Some(90_000));
    }

    /// A stale (now - last > window) newest entry gates tps/prompt_tps
    /// (existing behavior) AND spec_accept_pct (new freshness gate) — so a
    /// merged rate can't linger beside a "—" tok/s. cache_hit_pct and the
    /// OR'd spec_decoding_active flag keep their sticky semantics.
    #[test]
    fn test_aggregate_inference_stale_gates_tps_and_spec_but_not_sticky_fields() {
        let (tps, prompt_tps, cache_hit_pct, spec_accept_pct, active, last) =
            aggregate_inference(&agg_map(40_000), 100_000, BUCKET_MS);
        assert_eq!(tps, None, "60s-old tps gated (existing behavior)");
        assert_eq!(
            prompt_tps, None,
            "60s-old prompt_tps gated (existing behavior)"
        );
        assert_eq!(
            spec_accept_pct, None,
            "60s-old tamad-merged rate must not linger"
        );
        assert_eq!(
            cache_hit_pct,
            Some(90.0),
            "cache_hit_pct is not freshness-gated"
        );
        assert!(
            active,
            "spec_decoding_active is sticky — not freshness-gated"
        );
        assert_eq!(
            last,
            Some(40_000),
            "last_updated_ms reports the newest entry regardless of staleness"
        );
    }

    /// The boundary is inclusive: `now - last == stale_threshold_ms` is still
    /// fresh (matching the pre-existing `<=` tps gate).
    #[test]
    fn test_aggregate_inference_exact_stale_threshold_is_fresh() {
        let (tps, _prompt_tps, _c, spec_accept_pct, _a, _l) =
            aggregate_inference(&agg_map(70_000), 100_000, BUCKET_MS);
        assert_eq!(
            tps,
            Some(50.0),
            "boundary pin: now - last == threshold is fresh"
        );
        assert_eq!(spec_accept_pct, Some(44.5));
    }

    /// An empty map yields all-None fields and a false flag.
    #[test]
    fn test_aggregate_inference_empty_map() {
        let (tps, prompt_tps, cache_hit_pct, spec_accept_pct, active, last) =
            aggregate_inference(&HashMap::new(), 1_000, BUCKET_MS);
        assert_eq!(tps, None);
        assert_eq!(prompt_tps, None);
        assert_eq!(cache_hit_pct, None);
        assert_eq!(spec_accept_pct, None);
        assert!(!active);
        assert_eq!(last, None);
    }
}
