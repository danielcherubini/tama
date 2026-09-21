//! Host stats collector for the tamad daemon.
//!
//! Stateful on purpose: CPU% is a *delta* between two samples taken on the
//! same `sysinfo::System`. Creating a fresh `System` per tick would yield a
//! meaningless (always-0-ish) CPU reading — the same reason
//! `tama-core/src/proxy/server/metrics.rs` holds one `System` across its
//! loop. `tick` is blocking (GPU detection shells out to nvidia-smi /
//! reads sysfs) and must be called via `tokio::task::spawn_blocking`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::engine_metrics;
use crate::state::TamadState;
use tama_core::tamad::GpuInfo;
use tama_core::tamad::ProcessInfo;
use tama_core::tamad::SystemStats;

/// Collects a full host stats snapshot (CPU/RAM/swap/disk + per-GPU info)
/// on a fixed cadence, reusing one `sysinfo::System` across ticks.
pub struct StatsCollector {
    state: Arc<TamadState>,
    /// Refreshed once per tick; persists across ticks so CPU% is a real
    /// inter-sample delta.
    sys: sysinfo::System,
    /// Refreshed per tick.
    disks: sysinfo::Disks,
    /// Engine-metrics scrape state per model_name (all ready+alive
    /// backends, any engine — the scraped body determines the engine).
    engine: HashMap<String, EngineState>,
    /// Overridable in tests (`Duration::ZERO` = scrape every tick).
    scrape_interval: Duration,
    /// Blocking HTTP client for `/metrics` scrapes (per-scrape timeout),
    /// or `None` when the client could not be built. Built lazily on the
    /// first tick — constructing a blocking reqwest client inside an
    /// async context panics, and `new` runs there at service boot — while
    /// a tick always runs via `spawn_blocking`.
    ///
    /// A build failure (e.g. a TLS-provider init on a misconfigured host)
    /// is NOT fatal: it stores `None` and spec scraping is silently
    /// disabled (one-time `debug!`), so a scrape problem's blast radius
    /// never extends past this feature to the rest of the host stats.
    http: OnceLock<Option<reqwest::blocking::Client>>,
}

/// Per-endpoint engine-metrics scrape state. `prev` is the last cumulative
/// counter set (diffed on the next scrape); the `last_*` fields are the
/// most recent observation until it goes stale or is evicted.
#[derive(Debug, Default)]
struct EngineState {
    /// Last cumulative counter set (None until the first successful parse).
    prev: Option<engine_metrics::EngineCounters>,
    /// Engine family of the last parsed body (body-driven detection).
    kind: Option<engine_metrics::EngineKind>,
    /// Last scrape attempt, for the per-endpoint scrape throttle.
    last_scrape: Option<Instant>,
    /// Last SUCCESSFUL parse, for the window length (`dt`). A failed attempt
    /// or an unknown body must NOT shorten the next window — the counters
    /// advanced over the full interval since the last successful parse.
    last_parse: Option<Instant>,
    /// The most recent window observation (None until a window had traffic).
    last_obs: Option<engine_metrics::WindowObs>,
    /// Unix millis of the last traffic-bearing observation (poison-pill for
    /// freshness).
    last_obs_ms: i64,
}

impl StatsCollector {
    /// Build a collector and take one baseline sample so the first tick
    /// already has a meaningful CPU delta.
    pub fn new(state: Arc<TamadState>) -> Self {
        let mut sys = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::new()
                .with_cpu(sysinfo::CpuRefreshKind::everything())
                .with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        Self {
            state,
            sys,
            disks: sysinfo::Disks::new_with_refreshed_list(),
            engine: HashMap::new(),
            scrape_interval: engine_metrics::SCRAPE_INTERVAL,
            http: OnceLock::new(),
        }
    }

    /// Lazily build the blocking scrape client (first-tick, from a
    /// non-async thread) and return an owned handle. Cloning a
    /// `Client` is cheap (clones share the underlying connection pool),
    /// and an owned copy keeps this call from holding a borrow of `self`
    /// across the mutable `self.spec` work in the scrape loop.
    ///
    /// A build failure (e.g. TLS/OpenSSL init on a misconfigured host)
    /// is not fatal: it returns `None` with a one-time `debug!` rather
    /// than panicking — a panic out of this `spawn_blocking` tick would
    /// drop the whole host stats stream and re-panic on every proxy
    /// reconnect until tamad is restarted. The happy path is unchanged.
    fn http(&self) -> Option<reqwest::blocking::Client> {
        self.http
            .get_or_init(|| {
                match reqwest::blocking::Client::builder()
                    .timeout(engine_metrics::PER_SCRAPE_TIMEOUT)
                    .build()
                {
                    Ok(client) => Some(client),
                    Err(e) => {
                        tracing::debug!("spec scrape disabled: {e}");
                        None
                    }
                }
            })
            .clone()
    }

    /// Override the per-endpoint scrape throttle (tests use ~0ms).
    #[cfg(test)]
    pub(crate) fn with_scrape_interval(mut self, interval: Duration) -> Self {
        self.scrape_interval = interval;
        self
    }

    /// Refresh all subsystems and return a full snapshot.
    ///
    /// Blocking — call from `tokio::task::spawn_blocking`.
    pub fn tick(&mut self, mut processes: Vec<ProcessInfo>) -> SystemStats {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        // CPU/RAM/swap straight from the persistent System (NOT from
        // SystemMetrics — swap is not populated by collect_system_metrics_with).
        let cpu_usage_percent = self.sys.global_cpu_info().cpu_usage() as f64;
        let memory_total_bytes = self.sys.total_memory() as i64;
        let memory_used_bytes = self.sys.used_memory() as i64;
        let swap_total_bytes = self.sys.total_swap() as i64;
        let swap_used_bytes = self.sys.used_swap() as i64;

        let (disk_total_bytes, disk_free_bytes) =
            Self::disk_usage_for(&mut self.disks, &self.state.models_dir);

        // Reuse the same System for GPU detection (its internals refresh
        // CPU/memory again — harmless; we only consume `.gpus`).
        let metrics = crate::gpu::system::collect_system_metrics_with(&mut self.sys);
        let gpus = map_gpus(&metrics.gpus);

        // Spec-decode observation: scrape ready+alive engine /metrics
        // endpoints and stamp the diffed acceptance rate onto their
        // ProcessInfo entries. Never blocks more than the scrape budget.
        self.scrape_spec(&mut processes);

        SystemStats {
            cpu_usage_percent,
            memory_total_bytes,
            memory_used_bytes,
            swap_total_bytes,
            swap_used_bytes,
            disk_total_bytes,
            disk_free_bytes,
            gpus,
            processes,
        }
    }

    /// Total/available bytes of the filesystem containing `dir`.
    ///
    /// Longest mount-point prefix of `dir` wins; the `/` mount is always a
    /// valid fallback, so a real host always resolves to a disk.
    fn disk_usage_for(disks: &mut sysinfo::Disks, dir: &Path) -> (i64, i64) {
        disks.refresh();
        let mut best_len: usize = 0;
        let mut best: Option<(u64, u64)> = None;
        for disk in disks.iter() {
            let mount = disk.mount_point();
            if dir.starts_with(mount) {
                let len = mount.components().count();
                if len > best_len {
                    best_len = len;
                    best = Some((disk.total_space(), disk.available_space()));
                }
            }
        }
        match best {
            Some((total, free)) => (total as i64, free as i64),
            // Empty disk list (shouldn't happen on a real host).
            None => (0, 0),
        }
    }

    /// Scrape `/metrics` for every ready+alive process and diff the engine
    /// counters; stamp `tps` / `prompt_tps` / `cache_hit_pct` /
    /// `spec_accept_pct` / `spec_decoding_active` on the matching entries
    /// (every field is 30s-gated, ADR-0014). Blocking (HTTP) — legitimate
    /// only because `tick` already runs via `spawn_blocking`. The tick must
    /// never linger: the proxy's 5s `LIVE_FRAME_MAX_AGE` freshness gate
    /// blanks every model on the host if a tick overshoots, so scrapes are
    /// throttled per endpoint and the cumulative scrape work is capped at
    /// `TICK_SCRAPE_BUDGET`.
    /// The budget is preflighted *before* a scrape is started: a send
    /// can run up to the full `PER_SCRAPE_TIMEOUT` before timing out and
    /// cannot be interrupted, so a post-hoc check would admit one extra
    /// scrape and let a hanging engine push the tick past the budget.
    /// Skipped models simply retry next tick.
    fn scrape_spec(&mut self, processes: &mut [ProcessInfo]) {
        // Scraping is a logged no-op when the client could not be built
        // (the one-time `debug!` fired inside `http()`); the rest of the
        // tick proceeds with host metrics.
        let Some(client) = self.http() else {
            return;
        };
        let now = Instant::now();
        // Intentional skew: captured before the scrape loop, so a
        // `last_obs_ms` stamped during this tick is up to
        // `TICK_SCRAPE_BUDGET` (~3s) older than at emit — values blank
        // slightly early, never linger (the safe direction).
        let now_ms = unix_now_ms();
        let mut scrape_elapsed = Duration::ZERO;

        for p in processes.iter_mut() {
            if p.status != "ready" || !p.alive {
                continue;
            }
            let model_name = p.model_name.clone();
            // The entry may not exist yet (the `entry().or_default()` binding
            // is created AFTER the fetch), so read via `.get()`. It is the
            // last SUCCESSFUL parse (not the last attempt) that anchors the
            // window length — a failed attempt or an unknown body must not
            // shorten the next window.
            let prev_parse = self.engine.get(&model_name).and_then(|s| s.last_parse);
            let throttled = self.engine.get(&model_name).is_some_and(|s| {
                s.last_scrape.is_some_and(|t| {
                    self.scrape_interval > Duration::ZERO
                        && now.duration_since(t) < self.scrape_interval
                })
            });
            // Preflight the budget: admit a new scrape only when the
            // remaining `TICK_SCRAPE_BUDGET` can cover a full
            // `PER_SCRAPE_TIMEOUT` — the send can take up to that long
            // before timing out and cannot be cancelled once started, so
            // refusing it now keeps total scrape work within the budget
            // even against a hanging engine.
            if throttled
                || scrape_elapsed + engine_metrics::PER_SCRAPE_TIMEOUT
                    >= engine_metrics::TICK_SCRAPE_BUDGET
            {
                continue;
            }
            let Some(url) = engine_metrics::metrics_url_for(&p.endpoint_url) else {
                continue;
            };

            let t0 = Instant::now();
            let outcome = client.get(&url).send().and_then(|r| {
                let ok = r.status().is_success();
                r.text().map(move |t| (ok, t))
            });
            scrape_elapsed += t0.elapsed();
            let s = self.engine.entry(model_name).or_default();
            s.last_scrape = Some(Instant::now());

            // ANY failure (send error, non-2xx, text error): debug-log
            // (never warn — down engines would spam the log), keep the
            // last observation, move on.
            let text = match outcome {
                Ok((true, t)) => t,
                Ok((false, _)) => {
                    tracing::debug!("{} engine scrape: non-2xx response", p.model_name);
                    continue;
                }
                Err(e) => {
                    tracing::debug!("{} engine scrape failed: {e}", p.model_name);
                    continue;
                }
            };

            let (kind, cur) = match engine_metrics::parse_engine_metrics(&text) {
                Some(k) => k,
                // Unknown engine body: leave `prev`/`last_parse` untouched
                // (the next window's `dt` spans the whole interval — the
                // counters advanced over it) and memo that the engine is
                // unrecognised.
                None => {
                    s.kind = None;
                    continue;
                }
            };
            let dt_secs = prev_parse
                .map(|t| (Instant::now() - t).as_secs_f64())
                .unwrap_or(1.0); // first successful parse: prev is None so
                                 // observe() returns None anyway
            let obs = engine_metrics::observe(s.prev, &cur, dt_secs);
            s.kind = Some(kind);
            if let Some(o) = obs {
                // ONLY a traffic-bearing window updates the observation and
                // its timestamp — an idle window (observe() → None) must
                // leave the last observation intact until it goes stale
                // (30s, below).
                s.last_obs_ms = now_ms;
                s.last_obs = Some(o);
            }
            s.last_parse = Some(Instant::now()); // dt anchor — every successful parse
            s.prev = Some(cur);
        }

        // Evict state for models no longer in this tick's process list — a
        // restarted engine gets a fresh `prev` (its counters were reset).
        let current: std::collections::HashSet<&str> =
            processes.iter().map(|p| p.model_name.as_str()).collect();
        self.engine
            .retain(|name, _| current.contains(name.as_str()));

        // Emit observations to the tracked entries. Every field is now
        // 30s-gated (ADR-0014): an entry stamps the last traffic-bearing
        // observation while fresh, and blanks (defaults) as soon as it
        // stops being ready or the observation goes stale.
        for p in processes.iter_mut() {
            if p.status != "ready" || !p.alive {
                continue;
            }
            let Some(s) = self.engine.get(&p.model_name) else {
                continue;
            };
            let fresh = now_ms - s.last_obs_ms <= engine_metrics::STALE_MS;
            let o = s.last_obs;
            p.tps = if fresh { o.and_then(|o| o.tps) } else { None };
            p.prompt_tps = if fresh {
                o.and_then(|o| o.prompt_tps)
            } else {
                None
            };
            p.cache_hit_pct = if fresh {
                o.and_then(|o| o.cache_hit_pct)
            } else {
                None
            };
            p.spec_accept_pct = if fresh {
                o.and_then(|o| o.spec_accept_pct)
            } else {
                None
            };
            p.spec_decoding_active = fresh && o.is_some_and(|o| o.spec_active);
        }
    }
}

/// Unix time in millis (falls back to 0 before the epoch — pre-epoch
/// always reads as stale, which is the safe side).
fn unix_now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Map `GpuDeviceStats` list to proto `GpuInfo`.
fn map_gpus(gpus: &[tama_core::gpu::GpuDeviceStats]) -> Vec<GpuInfo> {
    gpus.iter()
        .enumerate()
        .map(|(position, g)| {
            // "GPU0" → 0; if the suffix is not a bare integer, use position.
            let digits: String = g.device_id.chars().filter(|c| c.is_ascii_digit()).collect();
            let index = digits.parse::<i32>().unwrap_or(position as i32);
            let (vram_total_bytes, vram_used_bytes) = match &g.vram {
                Some(v) => (
                    v.total_mib as i64 * 1024 * 1024,
                    v.used_mib as i64 * 1024 * 1024,
                ),
                None => (0, 0),
            };
            GpuInfo {
                index,
                name: g.name.clone(),
                // GpuDeviceStats carries no driver version today; the proto
                // field is reserved for the future.
                driver_version: String::new(),
                vram_total_bytes,
                vram_used_bytes,
                utilization_percent: g.utilization_pct.map(|u| u as f64).unwrap_or(0.0),
                temperature_c: g.temperature_c.map(|t| t as f64).unwrap_or(0.0),
                power_w: g.power_w.map(|p| p as f64).unwrap_or(0.0),
                fan_percent: g.fan_pct.map(|f| f as f64).unwrap_or(0.0),
            }
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> Arc<TamadState> {
        let dir = tempfile::tempdir().unwrap();
        // Keep the tempdir alive for the test's lifetime via a leak-free
        // guard: the state only needs models_dir as a path string.
        let args = crate::CliArgs {
            addr: "127.0.0.1:50051".to_string(),
            protocol: "grpc".to_string(),
            name: Some("stats-test".to_string()),
            public_url: None,
            models_dir: Some(dir.path().join("models")),
            data_dir: Some(dir.keep()),
            no_replay_desired: false,
            container_runtime: crate::host_installs::docker::runtime::ContainerRuntime::default(),
        };
        Arc::new(TamadState::from_cli(&args).unwrap())
    }

    /// A tick yields real memory numbers, a plausible non-NaN CPU across
    /// two ticks (proves the persistent-System delta works), positive disk
    /// figures for the models-dir filesystem, and structurally valid GPU
    /// entries on whatever hardware the test host has (GPU-less hosts
    /// yield an empty list without panicking).
    #[test]
    fn test_tick_host_snapshot() {
        let collector = StatsCollector::new(test_state());
        let mut collector = collector;

        let first = collector.tick(vec![]);
        for g in &first.gpus {
            assert!(
                g.vram_used_bytes <= g.vram_total_bytes,
                "vram_used must not exceed vram_total"
            );
            assert!((0.0..=100.0).contains(&g.utilization_percent));
        }
        assert!(
            first.memory_total_bytes > 0,
            "memory_total_bytes must be non-zero on a real host"
        );
        assert!(first.memory_used_bytes >= 0);
        assert!(
            first.disk_total_bytes > 0,
            "models-dir filesystem must have a positive total size"
        );
        assert!(first.disk_free_bytes >= 0);
        assert!(
            !first.cpu_usage_percent.is_nan(),
            "first tick CPU must not be NaN"
        );

        let second = collector.tick(vec![]);
        assert!(
            !second.cpu_usage_percent.is_nan(),
            "second tick CPU must not be NaN"
        );
        assert!(
            (0.0..=100.0).contains(&second.cpu_usage_percent),
            "CPU usage must be in 0..=100, got {}",
            second.cpu_usage_percent
        );

        // Processes pass through untouched (invalid port → scrape fails
        // silently, spec fields stay at their defaults; no DNS in tests).
        let proc = ProcessInfo {
            model_name: "m".to_string(),
            provider_name: "p".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://127.0.0.1:0".to_string(),
            status: "ready".to_string(),
            desired: false,
            restart_count: 0,
            max_restarts: 0,
            spec_accept_pct: None,
            spec_decoding_active: false,
            tps: None,
            prompt_tps: None,
            cache_hit_pct: None,
        };
        let third = collector.tick(vec![proc.clone()]);
        assert_eq!(third.processes.len(), 1);
        assert_eq!(third.processes[0].model_name, "m");
    }

    /// `map_gpus` parses device indices, multiplies VRAM MiB→bytes, and
    /// defaults None fields to 0.
    #[test]
    fn test_map_gpus() {
        use tama_core::gpu::{GpuDeviceStats, GpuVendor, VramInfo};

        let gpus = vec![
            GpuDeviceStats {
                device_id: "GPU0".to_string(),
                vendor: GpuVendor::Nvidia,
                name: "RTX 4090".to_string(),
                utilization_pct: Some(42),
                vram: Some(VramInfo {
                    used_mib: 1024,
                    total_mib: 24576,
                }),
                temperature_c: Some(71),
                power_w: Some(350),
                fan_pct: Some(40),
                pci_bus: None,
                uuid: None,
            },
            GpuDeviceStats {
                device_id: "unknown".to_string(),
                vendor: GpuVendor::Amd,
                name: "Mystery".to_string(),
                utilization_pct: None,
                vram: None,
                temperature_c: None,
                power_w: None,
                fan_pct: None,
                pci_bus: None,
                uuid: None,
            },
        ];

        let out = map_gpus(&gpus);
        assert_eq!(out.len(), 2);

        assert_eq!(out[0].index, 0);
        assert_eq!(out[0].name, "RTX 4090");
        assert_eq!(out[0].driver_version, "");
        assert_eq!(out[0].vram_total_bytes, 24576 * 1024 * 1024);
        assert_eq!(out[0].vram_used_bytes, 1024 * 1024 * 1024);
        assert_eq!(out[0].utilization_percent, 42.0);
        assert_eq!(out[0].temperature_c, 71.0);
        assert_eq!(out[0].power_w, 350.0);
        assert_eq!(out[0].fan_percent, 40.0);

        // Unparseable device_id → position in the vec; None fields → 0.
        assert_eq!(out[1].index, 1);
        assert_eq!(out[1].vram_total_bytes, 0);
        assert_eq!(out[1].vram_used_bytes, 0);
        assert_eq!(out[1].utilization_percent, 0.0);
        assert_eq!(out[1].temperature_c, 0.0);
        assert_eq!(out[1].power_w, 0.0);
        assert_eq!(out[1].fan_percent, 0.0);
    }

    /// A ready+alive process feeding a mock engine /metrics endpoint.
    fn spec_process(endpoint_url: String) -> ProcessInfo {
        ProcessInfo {
            model_name: "m".to_string(),
            provider_name: "vllm".to_string(),
            pid: 1,
            alive: true,
            endpoint_url,
            status: "ready".to_string(),
            desired: false,
            restart_count: 0,
            max_restarts: 0,
            spec_accept_pct: None,
            spec_decoding_active: false,
            tps: None,
            prompt_tps: None,
            cache_hit_pct: None,
        }
    }

    /// vLLM body: the token counters plus the three spec counters, one
    /// label set each.
    fn vllm_body(
        gen: f64,
        computed: f64,
        cached: f64,
        drafts: f64,
        draft_tokens: f64,
        accepted: f64,
    ) -> String {
        format!(
            "\n# HELP vllm:generation_tokens_total Total generated tokens\n\
             vllm:generation_tokens_total{{model_name=\"m\",engine=\"0\"}} {gen}\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_compute\",model_name=\"m\",engine=\"0\"}} {computed}\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_cache_hit\",model_name=\"m\",engine=\"0\"}} {cached}\n\
             vllm:spec_decode_num_drafts_total{{model_name=\"m\",engine=\"0\"}} {drafts}\n\
             vllm:spec_decode_num_draft_tokens_total{{model_name=\"m\",engine=\"0\"}} {draft_tokens}\n\
             vllm:spec_decode_num_accepted_tokens_total{{model_name=\"m\",engine=\"0\"}} {accepted}\n"
        )
    }

    /// llama.cpp body: the six unlabelled `llamacpp:` counters.
    fn llamacpp_body(
        gen: f64,
        computed: f64,
        cached: f64,
        drafts: f64,
        draft_tokens: f64,
        accepted: f64,
    ) -> String {
        format!(
            "\n# HELP llamacpp:tokens_predicted_total Total predicted tokens\n\
             llamacpp:tokens_predicted_total {gen}\n\
             llamacpp:prompt_tokens_total {computed}\n\
             llamacpp:prompt_tokens_cached_total {cached}\n\
             llamacpp:spec_decode_num_drafts_total {drafts}\n\
             llamacpp:spec_decode_num_draft_tokens_total {draft_tokens}\n\
             llamacpp:spec_decode_num_accepted_tokens_total {accepted}\n"
        )
    }

    /// Two ticks against mock vLLM engines: the first scrape only seeds
    /// the cumulative counters (no delta yet → defaults), and the second,
    /// after the counters advance, reports the windowed acceptance rate
    /// and marks the engine active. Detection is body-driven: only the
    /// metrics *body* decides vLLM-ness. Ticks run on a worker thread —
    /// the blocking reqwest client must not run in an async context.
    #[test]
    fn test_tick_spec_scrape_vllm_positive() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // Tick-1 endpoint seeds the cumulative counters at zero; the tick-2
        // endpoint serves the real-log window (165/371 ≈ 44.5%) plus the
        // token counters (500/200/50).
        let seed = rt.block_on(MockServer::start());
        let next = rt.block_on(MockServer::start());
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(vllm_body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                )
                .mount(&seed),
        );
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(vllm_body(500.0, 200.0, 50.0, 115.0, 371.0, 165.0)),
                )
                .mount(&next),
        );

        let (first, second) = std::thread::spawn(move || {
            let mut collector =
                StatsCollector::new(test_state()).with_scrape_interval(Duration::from_millis(50));
            let first = collector.tick(vec![spec_process(seed.uri())]);
            // Guarantee at least one scrape interval elapses between the
            // two ticks: on fast CI machines tick 2 can land <50ms after
            // tick 1's scrape, the `last_scrape` guard would throttle it
            // and the 2nd tick would observe no fresh window (the
            // presence-assert below would fail) — and a real sleep also
            // keeps `dt_secs` > 0 for the windowed rates.
            std::thread::sleep(Duration::from_millis(60));
            let second = collector.tick(vec![spec_process(next.uri())]);
            (first, second)
        })
        .join()
        .unwrap();

        // Tick 1: first scrape seeds prev — no window yet.
        assert_eq!(first.processes.len(), 1);
        assert!(first.processes[0].spec_accept_pct.is_none());
        assert!(!first.processes[0].spec_decoding_active);
        assert!(first.processes[0].tps.is_none());
        assert!(first.processes[0].prompt_tps.is_none());
        assert!(first.processes[0].cache_hit_pct.is_none());

        // Tick 2: the windowed observation is stamped — the spec
        // acceptance rate is the real-log vector, and the token counters
        // yield the windowed tps / prompt_tps / cache_hit_pct (the exact
        // math is covered by the engine_metrics `observe` unit tests; here
        // we only assert presence + sign).
        let p = &second.processes[0];
        let Some(pct) = p.spec_accept_pct else {
            panic!("expected a spec acceptance rate on tick 2");
        };
        assert!((44.4..=44.55).contains(&pct), "expected ~44.47, got {pct}");
        assert!(p.spec_decoding_active);
        assert!(
            p.tps.is_some_and(|v| v > 0.0),
            "windowed tps stamped, got {:?}",
            p.tps
        );
        assert!(
            p.prompt_tps.is_some_and(|v| v > 0.0),
            "windowed prompt_tps stamped, got {:?}",
            p.prompt_tps
        );
        assert!(
            p.cache_hit_pct.is_some_and(|v| (0.0..=100.0).contains(&v)),
            "windowed cache_hit_pct stamped, got {:?}",
            p.cache_hit_pct
        );
    }

    /// A mock llama.cpp engine (unlabelled `llamacpp:` counters) gets the
    /// windowed rates stamped on tick 2, mirroring the vLLM test — the
    /// engine is detected from the body, not the provider name.
    #[test]
    fn test_tick_spec_scrape_llamacpp_positive() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let seed = rt.block_on(MockServer::start());
        let next = rt.block_on(MockServer::start());
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(llamacpp_body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                )
                .mount(&seed),
        );
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(llamacpp_body(500.0, 200.0, 50.0, 115.0, 371.0, 165.0)),
                )
                .mount(&next),
        );

        let (first, second) = std::thread::spawn(move || {
            let mut collector =
                StatsCollector::new(test_state()).with_scrape_interval(Duration::from_millis(50));
            let first = collector.tick(vec![spec_process(seed.uri())]);
            std::thread::sleep(Duration::from_millis(60));
            let second = collector.tick(vec![spec_process(next.uri())]);
            (first, second)
        })
        .join()
        .unwrap();

        assert_eq!(first.processes.len(), 1);
        assert!(first.processes[0].tps.is_none());
        assert!(first.processes[0].prompt_tps.is_none());
        assert!(first.processes[0].cache_hit_pct.is_none());
        assert!(first.processes[0].spec_accept_pct.is_none());
        assert!(!first.processes[0].spec_decoding_active);

        let p = &second.processes[0];
        assert!(
            p.tps.is_some_and(|v| v > 0.0),
            "llama.cpp windowed tps stamped, got {:?}",
            p.tps
        );
        assert!(
            p.prompt_tps.is_some_and(|v| v > 0.0),
            "llama.cpp windowed prompt_tps stamped, got {:?}",
            p.prompt_tps
        );
        assert!(
            p.cache_hit_pct.is_some_and(|v| (0.0..=100.0).contains(&v)),
            "llama.cpp windowed cache_hit_pct stamped, got {:?}",
            p.cache_hit_pct
        );
        let Some(pct) = p.spec_accept_pct else {
            panic!("expected a spec acceptance rate on tick 2");
        };
        assert!((44.4..=44.55).contains(&pct), "expected ~44.47, got {pct}");
        assert!(p.spec_decoding_active);
    }

    /// Non-2xx response (501) takes the `Ok((false, _))` failure arm:
    /// no observation is stamped and the `last_parse` dt anchor does not
    /// move, so the next windowed rate spans the FULL interval since the
    /// last successful parse — the failed attempt does not shorten the
    /// window. Three ticks: seed (200, zero counters) → 501 → 200 with
    /// the counters advanced; the tick-3 rate is consistent with dt
    /// spanning both 3s sleeps (~6s → ~83 tps), not just the second
    /// one (~3s → ~167 tps).
    #[test]
    fn test_tick_spec_scrape_non_2xx_does_not_shorten_window() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // Tick 1: seed the cumulative counters at zero. Tick 2: 501
        // (the non-2xx arm). Tick 3: the counters advanced (500/200/50
        // plus the 165/371 spec vector).
        let seed = rt.block_on(MockServer::start());
        let fail = rt.block_on(MockServer::start());
        let next = rt.block_on(MockServer::start());
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(vllm_body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                )
                .mount(&seed),
        );
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(ResponseTemplate::new(501))
                .mount(&fail),
        );
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(vllm_body(500.0, 200.0, 50.0, 115.0, 371.0, 165.0)),
                )
                .mount(&next),
        );

        let mut collector =
            StatsCollector::new(test_state()).with_scrape_interval(Duration::from_millis(50));

        // Ticks run on the plain test thread (not in an async context
        // — the `rt.block_on` enter-guard is released before the blocking
        // scrape), so no `std::thread::spawn` wrapper is needed.
        // Tick 1: first successful parse seeds `prev` — no window yet.
        let first = collector.tick(vec![spec_process(seed.uri())]);
        assert!(first.processes[0].spec_accept_pct.is_none());
        assert!(first.processes[0].tps.is_none());
        let parse_after_seed = collector
            .engine
            .get("m")
            .unwrap()
            .last_parse
            .expect("tick 1 seeds last_parse");

        // Tick 2: the 501 takes the non-2xx arm — no observation, and
        // the dt anchor must stay at tick 1's parse.
        std::thread::sleep(Duration::from_millis(3000));
        let second = collector.tick(vec![spec_process(fail.uri())]);
        let p = &second.processes[0];
        assert!(p.spec_accept_pct.is_none());
        assert!(!p.spec_decoding_active);
        assert!(p.tps.is_none());
        assert!(
            collector
                .engine
                .get("m")
                .unwrap()
                .last_parse
                .is_some_and(|t| t == parse_after_seed),
            "a non-2xx response must not move the last_parse dt anchor"
        );

        // Tick 3: the counters advanced — the window spans BOTH 3s
        // sleeps (~6s → ~83 tps). If the failed attempt had shortened
        // it, the rate would be ~2x higher (~500/3s ≈ 167 tps).
        std::thread::sleep(Duration::from_millis(3000));
        let third = collector.tick(vec![spec_process(next.uri())]);
        let p = &third.processes[0];
        let Some(pct) = p.spec_accept_pct else {
            panic!("expected a spec acceptance rate on tick 3");
        };
        assert!((44.4..=44.55).contains(&pct), "expected ~44.47, got {pct}");
        assert!(p.spec_decoding_active);
        let Some(tps) = p.tps else {
            panic!("expected a windowed tps on tick 3");
        };
        // dt is at least the two sleeps minus the (small) pre-scrape
        // work of the seeding tick and at most the two sleeps plus the
        // other ticks' work — so a full window yields ~500/6s ≈ 72-88
        // tps, while a shortened window would yield ~500/3s ≈ 139-185.
        // The 55.0 floor is deliberately looser than the expected ~72-88
        // to absorb slow-machine overhead; it still excludes the
        // shortened-window hypothesis (~139-185) by a wide margin.
        assert!(
            (55.0..100.0).contains(&tps),
            "expected ~500/6s (the failed attempt must not shorten the window), got {tps}"
        );
    }

    /// A stale observation (31s old) blanks ALL FIVE wire fields; the same
    /// observation with a fresh timestamp stamps them (positive control —
    /// without it the staleness test would pass even if `fresh` were
    /// computed backwards). The scrape itself fails (nothing listening), so
    /// only the seeded state decides.
    #[test]
    fn test_tick_stale_observation_blanks_all_fields() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let mut collector = StatsCollector::new(test_state()).with_scrape_interval(Duration::ZERO);
        let now_ms = unix_now_ms();
        let seeded_obs = engine_metrics::WindowObs {
            tps: Some(42.0),
            prompt_tps: Some(120.0),
            cache_hit_pct: Some(25.0),
            spec_accept_pct: Some(44.5),
            spec_active: true,
            had_traffic: true,
        };
        // 31s old: every field must blank.
        collector.engine.insert(
            "m".to_string(),
            EngineState {
                prev: None,
                kind: None,
                last_scrape: None,
                last_parse: None,
                last_obs: Some(seeded_obs),
                last_obs_ms: now_ms - 31_000,
            },
        );
        let out = collector.tick(vec![spec_process(endpoint.clone())]);
        assert_eq!(out.processes.len(), 1);
        assert_eq!(out.processes[0].tps, None, "stale observation blanks tps");
        assert_eq!(
            out.processes[0].prompt_tps, None,
            "stale observation blanks prompt_tps"
        );
        assert_eq!(
            out.processes[0].cache_hit_pct, None,
            "stale observation blanks cache_hit_pct"
        );
        assert_eq!(
            out.processes[0].spec_accept_pct, None,
            "stale observation blanks spec_accept_pct"
        );
        assert!(
            !out.processes[0].spec_decoding_active,
            "stale observation un-sticks the active flag"
        );

        // Positive control: the SAME observation with a fresh timestamp
        // stamps every field.
        collector.engine.insert(
            "m".to_string(),
            EngineState {
                prev: None,
                kind: None,
                last_scrape: None,
                last_parse: None,
                last_obs: Some(seeded_obs),
                last_obs_ms: now_ms,
            },
        );
        let out = collector.tick(vec![spec_process(endpoint)]);
        assert_eq!(out.processes[0].tps, Some(42.0));
        assert_eq!(out.processes[0].prompt_tps, Some(120.0));
        assert_eq!(out.processes[0].cache_hit_pct, Some(25.0));
        assert_eq!(out.processes[0].spec_accept_pct, Some(44.5));
        assert!(out.processes[0].spec_decoding_active);
    }

    /// An idle window (a successful parse with UNCHANGED counters →
    /// `observe()` → `None`) advances the `prev`/`last_parse` anchors
    /// WITHOUT refreshing `last_obs_ms`: the last observation goes stale
    /// (30s) and blanks every field, and the next counter advance is
    /// diffed over the window since the idle tick — not the stale anchor.
    /// Seeded state + a live mock: ticks 1 and 2 serve unchanged counters
    /// (idle), tick 3 advances them.
    #[test]
    fn test_tick_idle_window_advances_anchors_without_refreshing_observation() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // Idle mock: the unchanged counters; advance mock: decode +500.
        let idle = rt.block_on(MockServer::start());
        let advance = rt.block_on(MockServer::start());
        let idle_body = vllm_body(100.0, 40.0, 10.0, 20.0, 60.0, 30.0);
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(ResponseTemplate::new(200).set_body_string(idle_body.clone()))
                .mount(&idle),
        );
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(vllm_body(600.0, 40.0, 10.0, 20.0, 60.0, 30.0)),
                )
                .mount(&advance),
        );

        // Seed a successful parse 10s ago (the `prev`/`last_parse`
        // anchor) with a traffic-bearing observation 31s old (stale).
        let (kind, prev) =
            engine_metrics::parse_engine_metrics(&idle_body).expect("vllm body parses");
        let now_ms = unix_now_ms();
        let stale_ms = now_ms - 31_000;
        let seeded_obs = engine_metrics::WindowObs {
            tps: Some(42.0),
            prompt_tps: Some(120.0),
            cache_hit_pct: Some(25.0),
            spec_accept_pct: Some(44.5),
            spec_active: true,
            had_traffic: true,
        };
        let mut collector =
            StatsCollector::new(test_state()).with_scrape_interval(Duration::from_millis(50));
        collector.engine.insert(
            "m".to_string(),
            EngineState {
                prev: Some(prev),
                kind: Some(kind),
                last_scrape: None,
                last_parse: Some(Instant::now() - Duration::from_secs(10)),
                last_obs: Some(seeded_obs),
                last_obs_ms: stale_ms,
            },
        );

        // Ticks run on the plain test thread (not in an async context
        // — the `rt.block_on` enter-guard is released before the blocking
        // scrape), so no `std::thread::spawn` wrapper is needed.
        // Tick 1: the counters are unchanged → idle → observe() → None.
        // The 31s-old observation is NOT refreshed: every field blanks.
        let before = Instant::now();
        let first = collector.tick(vec![spec_process(idle.uri())]);
        let p = &first.processes[0];
        assert_eq!(
            p.tps, None,
            "idle window leaves the stale observation blank"
        );
        assert_eq!(p.prompt_tps, None);
        assert_eq!(p.cache_hit_pct, None);
        assert_eq!(p.spec_accept_pct, None);
        assert!(!p.spec_decoding_active);
        let s = collector.engine.get("m").unwrap();
        assert_eq!(
            s.last_obs_ms, stale_ms,
            "an idle window must not refresh last_obs_ms"
        );
        assert!(
            s.last_parse.is_some_and(|t| t >= before),
            "an idle (successful) parse advances the last_parse dt anchor"
        );

        // Tick 2: still idle — the blanking persists across ticks.
        std::thread::sleep(Duration::from_millis(2000));
        let second = collector.tick(vec![spec_process(idle.uri())]);
        assert_eq!(second.processes[0].tps, None, "still idle, still blank");
        assert_eq!(
            collector.engine.get("m").unwrap().last_obs_ms,
            stale_ms,
            "the idle anchor extension persists across ticks"
        );

        // Tick 3: the counters advance — the window is diffed over the
        // ~1s since the last successful parse (the idle tick), NOT the
        // 10s+ anchor (which would yield ~500/13s ≈ 38 tps). The window
        // is at least the 1s sleep (the parse stamps are ordered), so
        // the rate is at most 500 tps and at least ~500/2.5s ≈ 200.
        // The 150.0 floor is deliberately looser than the expected
        // ~200-500 to absorb slow-machine overhead; it still excludes
        // the wrong-anchor hypothesis (~38 tps) by a wide margin.
        std::thread::sleep(Duration::from_millis(1000));
        let third = collector.tick(vec![spec_process(advance.uri())]);
        let p = &third.processes[0];
        let Some(tps) = p.tps else {
            panic!("expected a windowed tps after the counter advance");
        };
        assert!(
            (150.0..500.0).contains(&tps),
            "expected ~500/1s (the window since the idle tick), got {tps}"
        );
        // Only decode advanced: prompt_tps is a stamped 0.0 and the spec
        // counters are idle again.
        assert_eq!(p.prompt_tps, Some(0.0));
        assert_eq!(p.cache_hit_pct, None);
        assert_eq!(p.spec_accept_pct, None);
        assert!(!p.spec_decoding_active);
    }

    /// Non-vLLM body (llama.cpp-style metrics) → defaults on both ticks;
    /// a non-READY process is untouched regardless of engine.
    #[test]
    fn test_tick_spec_scrape_non_vllm_negative() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = rt.block_on(MockServer::start());
        let body = "llamacpp_duration_s{status_stage=\"0\",lifespan_stage=\"0\",vram_stage=\"3\"} 2.5\nllamacpp_inference_duration_s{model=\"m\"} 1.0\n";
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(ResponseTemplate::new(200).set_body_string(body))
                .mount(&server),
        );

        let (first, second) = std::thread::spawn(move || {
            let mut collector =
                StatsCollector::new(test_state()).with_scrape_interval(Duration::from_millis(1));
            let first = collector.tick(vec![spec_process(server.uri())]);

            // A stopped process is never scraped and keeps its defaults.
            let mut stopped = spec_process(server.uri());
            stopped.status = "stopped".to_string();
            stopped.alive = false;
            let second = collector.tick(vec![spec_process(server.uri()), stopped]);
            (first, second)
        })
        .join()
        .unwrap();

        assert!(first.processes[0].spec_accept_pct.is_none());
        assert!(!first.processes[0].spec_decoding_active);
        assert!(second.processes[0].spec_accept_pct.is_none());
        assert!(!second.processes[0].spec_decoding_active);
        assert!(second.processes[1].spec_accept_pct.is_none());
        assert!(!second.processes[1].spec_decoding_active);
    }

    /// Dead engine: nothing listening on the endpoint → the scrape fails,
    /// the tick completes, and the process keeps its defaults.
    #[test]
    fn test_tick_spec_scrape_dead_endpoint() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let mut collector =
            StatsCollector::new(test_state()).with_scrape_interval(Duration::from_millis(1));

        let out = collector.tick(vec![spec_process(endpoint)]);
        assert_eq!(out.processes.len(), 1);
        assert!(out.processes[0].spec_accept_pct.is_none());
        assert!(!out.processes[0].spec_decoding_active);
    }

    /// Budget-preflight regression: when a slow first scrape leaves less
    /// than `PER_SCRAPE_TIMEOUT` of headroom, the next scrape in the same
    /// tick is refused (never sent) instead of only being checked after
    /// it has (possibly needlessly) finished. Deterministic: verified via
    /// the mock's request log — one 1.1s-slow engine spends 1.1s, leaving
    /// <2s of the 3s tick budget, so the second model's scrape is refused
    /// and exactly ONE request reaches `/metrics`.
    #[test]
    fn test_tick_spec_scrape_budget_preflight_refuses_overshoot() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // Engine answers /metrics after ~1.1s — under the 2s
        // per-scrape timeout, but enough that a second scrape would push
        // the tick past the 3s budget.
        let slow = rt.block_on(MockServer::start());
        rt.block_on(
            Mock::given(method("GET"))
                .and(path("/metrics"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(vllm_body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                        .set_delay(Duration::from_millis(1100)),
                )
                .mount(&slow),
        );
        let uri = slow.uri();

        // Two ready+alive models against the same slow endpoint; the
        // per-endpoint scrape throttle is disabled so only the per-tick
        // budget decides.
        let a = spec_process(uri.clone());
        let mut b = spec_process(uri.clone());
        b.model_name = "m2".to_string();

        std::thread::spawn(move || {
            let mut collector =
                StatsCollector::new(test_state()).with_scrape_interval(Duration::ZERO);
            collector.tick(vec![a, b]);
        })
        .join()
        .unwrap();

        // Exactly one scrape was sent: model A spends ~1.1s, leaving
        // 3s - 1.1s < 2s (PER_SCRAPE_TIMEOUT), so model B's scrape is
        // refused by the preflight and never reaches the mock.
        let reqs = rt
            .block_on(slow.received_requests())
            .expect("request recording is on");
        let metrics_hits = reqs.iter().filter(|r| r.url.path() == "/metrics").count();
        assert_eq!(
            metrics_hits, 1,
            "preflight must refuse a scrape that could push the tick past its budget"
        );
    }
}
