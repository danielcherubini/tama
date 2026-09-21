//! Tamad-side lifecycle: spawn/health/unload/restart of backend processes
//! (plan-191 Task 5).
//!
//! Thin orchestrator over the local `process` module (plan-191 Task 10: moved
//! Tamad is a dumb executor (ADR-0010): it spawns whatever fully-resolved
//! launch spec the proxy sends in `LoadModelRequest`, health-polls it, and
//! records the process in the in-memory [`ProcessTable`]. No database, no
//! model registry.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tracing::{debug, error, info, warn};

use crate::process::{
    configure_backend_command, configure_process_group, force_kill_process_group,
    is_process_group_alive, kill_process_group, wait_group_dead,
};
use tama_core::tamad::{LoadModelRequest, LoadModelResponse, ProcessInfo, ProviderInfo};

use crate::process_table::{ProcessEntry, ProcessTable};
use crate::state::store::{Store, StoredProcess, DEFAULT_MAX_RESTARTS};
use crate::state::TamadState;

/// The canonical lifecycle status words for [process lines](ProcessEntry)
/// (the single host-side home; plan-193 runs with things here).
///
/// The four existing on-the-wire words (`starting`, `ready`, `failed`,
/// `unloading`) are joined here by the two of T2: `restarting` and
/// `budget_exhausted`. An empty word is out-of-spec legislation (the
/// table never writes them; T3's e2e asserts that the set of observed
/// words is ⊆ these six). Every write site in the
/// `lifecycle`/`process-table` uses just these constants — no
/// off-spec words.
pub mod status {
    /// Process startup began, but the health gate hasn't settled yet.
    pub const STARTING: &str = "starting";
    /// The backend is healthy and serving.
    pub const READY: &str = "ready";
    /// The backend died and a respawn is in flight — the row is defunct,
    /// and the dead-missing-pid reaper is, synchronously,
    /// standing a replacement back up (the reaper sets this BEFORE the
    /// respawn; `load()` flips it back to `starting` at the moment of
    /// launch, reflecting the normal entry-point walk).
    pub const RESTARTING: &str = "restarting";
    /// The process failed: the startup failed, became unhealthy in time,
    /// or died without a desired/flagged store row to respawn from.
    pub const FAILED: &str = "failed";
    /// The window of restart budget was blown: auto-respawn stops, and
    /// the store's `user_flagged` is set alongside. Nothing anywhere
    /// auto-clears the flag — the clean recovery is an operator
    /// `unload` of the model (the `UnloadModel` wire kills the host
    /// store row, and the flag with it) followed by a manual `load`
    /// (a fresh, un-flagged row with a fresh counter).
    pub const BUDGET_EXHAUSTED: &str = "budget_exhausted";
    /// The process is being brought down by an operator. The row comes
    /// down on unload completion; the words serve so that no
    /// reaper can race in mid-unload.
    pub const UNLOADING: &str = "unloading";

    /// Whether `s` is one of the six words (the single validator;
    /// the set of words observed on `process` info must stay ⊆).
    pub fn is_accepted(s: &str) -> bool {
        matches!(
            s,
            STARTING | READY | RESTARTING | FAILED | BUDGET_EXHAUSTED | UNLOADING
        )
    }
}

/// Span of the sliding restart budget window (seconds; the default
/// per-key budget `DEFAULT_MAX_RESTARTS` lives — T2 reuses — in
/// `crate::state::store`) .
pub const RESTART_WINDOW_SECS: u64 = 300;

/// Env key the proxy uses to carry the *proxy's* models directory in the
/// launch spec. The tamad rewrites any arg that references it (a path
/// prefix under the proxy's models dir) to its own `models_dir` so the
/// same spec works when proxy and tamad host the weights in different
/// places. The key is stripped before spawning.
pub const PROXY_MODELS_DIR_ENV: &str = "TAMA_MODELS_DIR";

/// The single host-side builder for the wire `[ProcessInfo]` of one
/// process line (plan-193 T3). The six legacy fields map straight off the
/// table `entry` (with `alive` folded from status+pid); the three T3
/// wire fields come from the persistent store row when one exists, else
/// fall back to the wire-spec defaults (not desired, zero restarts, the
/// default restart budget). Both write sites (the lifecycle `list()` and
/// the server `stream_stats` path) route through this one function so a
/// loaded model always reports `desired=true` and its restart counters
/// on the wire.
pub fn to_process_info(entry: &ProcessEntry, store_row: Option<&StoredProcess>) -> ProcessInfo {
    ProcessInfo {
        model_name: entry.model_name.clone(),
        provider_name: entry.provider_name.clone(),
        pid: entry.pid as i32,
        alive: entry.status != crate::lifecycle::status::FAILED
            && crate::process::is_process_alive(entry.pid),
        endpoint_url: entry.endpoint_url.clone(),
        status: entry.status.clone(),
        desired: store_row.map(|r| r.desired).unwrap_or(false),
        restart_count: entry.restart_count,
        max_restarts: store_row
            .map(|r| r.max_restarts)
            .unwrap_or(DEFAULT_MAX_RESTARTS),
        // Spec-decode observation (wire fields 10-11): defaults until the
        // tamad backend /metrics scrape populates them (plan-194 Task 2).
        spec_accept_pct: None,
        spec_decoding_active: false,
        // Inference stats (wire fields 12-14, ADR-0014): defaults until
        // the tamad's /metrics scrape window populates them.
        tps: None,
        prompt_tps: None,
        cache_hit_pct: None,
    }
}

/// Persist the budget trip's `user_flagged` mark with a bounded
/// retry: three attempts, 100 ms apart. The store write is an atomic
/// temp+rename, and a momentary disk flap must never drop the mark —
/// dropped, the in-memory mirror loses it on restart, and the boot
/// sweep would REPLAY the budget-exhausted model — a silent hole in a
/// terminal state.
///
/// Returns `Ok` on the first success; `Err` once all attempts are
/// exhausted. The caller must `error!` (carrying the operator
/// recovery: `tama admin unload <key>` then `load`) and stop — that
/// caller-side `error!` is the recovery contract; the state below has
/// not changed beyond the table row write that preceded it.
async fn persist_tripped(
    store: &Store,
    model_name: &str,
    persisted_restart_count: u32,
) -> Result<()> {
    for attempt in 1..=3u32 {
        match store.set_tripped(model_name, persisted_restart_count) {
            Ok(()) => return Ok(()),
            Err(e) => {
                warn!(
                    model = %model_name,
                    attempt,
                    error = %e,
                    "persisting the budget trip mark (user_flagged + tally) failed; retrying"
                );
            }
        }
        if attempt < 3 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    Err(anyhow!(
        "persisting the trip mark (user_flagged + at-cap tally) for '{model_name}' failed after all 3 attempts"
    ))
}

/// Operator-facing refusal line for the boot sweep's at-cap, unmarked
/// manifest (pre-policy durability, round-2 P1): the persisted trip tally
/// reached the disk; its `user_flagged` mark did not (the ENOSPC-at-trip shape).
/// The recovery sentence is word-for-word the trip's own fatal `error!`
/// sentence — the sweep LOGS; the operator FIXES.
fn at_cap_skip_note(model_name: &str, persisted: u32, cap: u32) -> String {
    format!(
        "persisted restart tally is at cap ({persisted} >= {cap}) — the boot sweep \
         will NOT replay '{model_name}' (replaying would re-arm the crash loop); \
         the manifest is left exactly as found. recovery = `tama admin unload \
{model_name}` then `load` (clean re-arm)"
    )
}

/// Tamad-side lifecycle over the process table.
pub struct TamadLifecycle {
    /// In-memory table of spawned backend processes.
    pub table: Arc<ProcessTable>,
    /// The per-model on-host-disk persistent store (plan-193 T1) — the
    /// source of truth for respawns and the boot replay.
    pub store: Arc<Store>,
    /// Runtime state (models_dir for path remapping).
    pub state: Arc<TamadState>,
    /// Queue into the respawn supervisor.
    respawn_tx: tokio::sync::mpsc::UnboundedSender<(String, LoadModelRequest, u32)>,
    /// The other end of that queue, handed to the supervisor task.
    respawn_rx: Arc<
        tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<(String, LoadModelRequest, u32)>>,
    >,
}

impl TamadLifecycle {
    /// Create a lifecycle backed by `table`, the persistent `store`,
    /// and runtime `state`.
    pub fn new(table: Arc<ProcessTable>, store: Arc<Store>, state: Arc<TamadState>) -> Self {
        let (respawn_tx, respawn_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            table,
            store,
            state,
            respawn_tx,
            respawn_rx: Arc::new(tokio::sync::Mutex::new(respawn_rx)),
        }
    }

    /// A state-sharing copy of this lifecycle for detached tasks (the
    /// reaper wait, the boot-sweep loads). The struct is just a trio of
    /// shared Arcs, so a copy IS the state itself — no identity data
    /// needs to be preserved.
    fn shared_copy(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
            store: Arc::clone(&self.store),
            state: Arc::clone(&self.state),
            respawn_tx: self.respawn_tx.clone(),
            respawn_rx: Arc::clone(&self.respawn_rx),
        }
    }

    /// Start the respawn supervisor (plan-193 T2). The dead-PID reaper
    /// itself only queues jobs: the relaunched spawn-back must not run
    /// inside the wait task — it cannot reenter `load` from there (the
    /// spawned task would have to prove `load`\'s future `Send` through
    /// its own spawn, which is circular) — so the supervisor, a
    /// detached task started once at boot, consumes the queue and
    /// dispatches each respawn through
    /// [`load`](Self::load). It exits when the lifecycle that owns it
    /// (and its test clones) are dropped.
    pub fn start_respawn_supervisor(
        lifecycle: &Arc<TamadLifecycle>,
    ) -> tokio::task::JoinHandle<()> {
        let lifecycle = Arc::clone(lifecycle);
        let queue = Arc::clone(&lifecycle.respawn_rx);
        tokio::spawn(async move {
            loop {
                let job = {
                    let mut guard = queue.lock().await;
                    guard.recv().await
                };
                let Some((model_name, req, restart_count)) = job else {
                    break; // lifecycle dropped
                };
                match lifecycle.load(&req).await {
                    Ok(resp) => info!(
                        model = %model_name,
                        new_pid = resp.pid,
                        restart_count,
                        "respawn supervisor relaunched the backend (gate detached)"
                    ),
                    Err(e) => warn!(
                        model = %model_name,
                        error = %e,
                        restart_count,
                        "respawn attempt failed; the row saw its outcome"
                    ),
                }
            }
        })
    }

    /// Start the detached reconciliation sweep for orphaned `starting`
    /// rows (plan-194). Runs every 5 seconds until the lifecycle drops.
    ///
    /// Defense in depth: rows can still strand in `starting` within a live
    /// tamad — bugs in older deployed binaries, a spawn-frame race, or any
    /// future path that forgets the detached-settle discipline. Each tick
    /// delegates to [`Self::reconcile_once`], which processes ALL eligible
    /// `starting` rows per pass: healthy orphans are adopted verified-ready
    /// and rows far past their health deadline are torn down (see that
    /// method for the exact semantics).
    pub fn start_starting_reconciler(
        lifecycle: &Arc<TamadLifecycle>,
    ) -> tokio::task::JoinHandle<()> {
        let lifecycle = Arc::clone(lifecycle);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(5));
            loop {
                ticker.tick().await;
                lifecycle.reconcile_once().await;
            }
        })
    }

    /// One reconciliation pass over all `starting` rows. Exposed
    /// (pub(crate)) for deterministic testing; production knobs are
    /// grace = 10s and min_deadline = 120_000ms.
    pub(crate) async fn reconcile_once(&self) {
        self.reconcile_once_with(Duration::from_secs(10), Duration::from_millis(120_000))
            .await;
    }

    /// Test-injectable core of [`Self::reconcile_once`]. Production:
    /// `grace = 10s`, `min_deadline = 120_000ms`. Both knobs are
    /// parameters so tests never sleep real production timeouts.
    ///
    /// Concurrency with live gates: after `grace` elapses this sweep runs
    /// CONCURRENTLY with any detached health gate that may still be settling
    /// the same row (`grace ≪ gate timeout`). Arbitration between the two
    /// writers is entirely via [`Self::owns_starting_row`] — both sides
    /// re-check it immediately before every teardown and terminal insert,
    /// so whichever actor sees the row leave `starting` first stands down.
    ///
    /// For each `starting` row:
    /// - gate-less specs (empty `health_url` or `health_timeout_ms == 0`)
    ///   are skipped entirely (they were meant to be instant-ready);
    /// - rows younger than `grace` are skipped — an active detached gate
    ///   owns them until then (both writers guard with `owns_row`, but
    ///   skipping avoids redundant pings);
    /// - a row older than `max(2 × health_timeout_ms, min_deadline)` is
    ///   torn down first (kill process group / stop+remove container,
    ///   mirroring the settle-path teardown) and then recorded `failed`
    ///   when it still owns its row — corpses are judged BEFORE probing
    ///   so they die promptly even though their port would refuse anyway;
    /// - otherwise a health probe runs: on success the row is adopted as
    ///   verified-ready with the same bookkeeping as a real gate pass
    ///   (including the persisted-tally reset).
    pub(crate) async fn reconcile_once_with(&self, grace: Duration, min_deadline: Duration) {
        for entry in self.table.list().await {
            if entry.status != status::STARTING {
                continue;
            }
            // The full launch spec is stored on every entry.
            let spec = entry.spec.clone();
            // Gate-less specs were meant to be instant-ready; a lingering
            // one indicates an in-flight operation too young to judge.
            if spec.health_url.is_empty() || spec.health_timeout_ms == 0 {
                continue;
            }
            let age = entry.started_at.elapsed();
            // Grace period: an active detached gate from the spawn owns the
            // row until then; the sweep must not race it.
            if age < grace {
                continue;
            }

            // Deadline breach FIRST — evaluate before the probe so a corpse
            // is torn down promptly even though its port would refuse.
            let deadline =
                Duration::from_millis(2u64.saturating_mul(spec.health_timeout_ms.max(0) as u64))
                    .max(min_deadline);
            if age > deadline {
                warn!(
                    model = %entry.model_name,
                    pid = entry.pid,
                    age_ms = age.as_millis() as u64,
                    deadline_ms = deadline.as_millis() as u64,
                    "starting row is past its health deadline; reconciling"
                );
                // Status-aware ownership FIRST: a stale snapshot row may have
                // been re-adopted by a detached gate or the reaper since the
                // pass started; only tear down a row that is still THIS
                // process AND still awaiting a verdict.
                if !self.owns_starting_row(&entry.model_name, entry.pid).await {
                    continue;
                }
                if !spec.docker_config_json.is_empty() {
                    // The ProcessEntry stores the launch spec but NOT the
                    // spawned container id, so this teardown cannot bind to
                    // the id captured at spawn the way `settle_container_gate`
                    // does (that gate tears down strictly by the container id
                    // it was handed, which survives name reuse). The PID
                    // inspection below is the ONLY thing standing between the
                    // owns_starting_row gate above and docker's kill: we
                    // verify the running container's host PID still matches
                    // entry.pid before acting (the same identity discipline as
                    // the settle gates), then stop/remove by the INSPECTED
                    // container id — closing the residual millisecond window
                    // where an unload→reload could legally rebind the
                    // deterministic name (`tama-{model}`) between inspect and
                    // kill. On any inspect failure or PID mismatch we stand
                    // down rather than kill an instance this row doesn't own.
                    let container_name =
                        crate::host_installs::docker::runner::container_name_for(&entry.model_name);
                    match crate::host_installs::docker::runner::inspect_container(
                        self.state.container_runtime,
                        &container_name,
                    )
                    .await
                    {
                        Ok(Some(inspect)) if inspect.state.pid == Some(entry.pid as u64) => {
                            // Teardown by inspected id when available; fall back
                            // to the deterministic name only if docker didn't
                            // report one.
                            let teardown_target =
                                inspect.id.clone().unwrap_or_else(|| container_name.clone());
                            let _ = crate::host_installs::docker::runner::stop_container(
                                self.state.container_runtime,
                                &teardown_target,
                            )
                            .await;
                            let _ = crate::host_installs::docker::runner::remove_container(
                                self.state.container_runtime,
                                &teardown_target,
                            )
                            .await;
                        }
                        other => {
                            warn!(
                                model = %entry.model_name,
                                expected_pid = entry.pid,
                                inspected_pid = ?other.ok().flatten().and_then(|i| i.state.pid),
                                "skipping container teardown: container identity unverified"
                            );
                        }
                    }
                } else {
                    // Mirror the native settle tail: SIGTERM, brief grace,
                    // SIGKILL escalation.
                    let _ = kill_process_group(entry.pid).await;
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    if is_process_group_alive(entry.pid) {
                        let _ = force_kill_process_group(entry.pid).await;
                    }
                }
                // Re-check AFTER teardown too: the kill window is where a
                // concurrent writer (unload / reaper / gate) is most likely
                // to have moved the row on.
                if self.owns_starting_row(&entry.model_name, entry.pid).await {
                    let previous = self.table.get(&entry.model_name).await;
                    self.table
                        .insert(Self::entry_for(
                            &spec,
                            entry.pid,
                            status::FAILED,
                            false,
                            &previous,
                        ))
                        .await;
                    warn!(
                        model = %entry.model_name,
                        "reconciler tore down orphaned starting row past health deadline"
                    );
                }
                continue;
            }

            // Health probe: adopt a backend that answers as verified ready.
            if let Ok(response) = crate::process::check_health(&spec.health_url, Some(5)).await {
                if response.status().is_success() {
                    // Re-check immediately before the insert (not just at
                    // loop top): the probe takes up to 5 s, ample time for a
                    // detached gate to land its own verdict. NOTE: single-
                    // probe adoption trades strictness for simplicity — two
                    // consecutive probes would be stricter, but the
                    // in-memory 300s restart window still bounds any crash
                    // loop this could mask.
                    if self.owns_starting_row(&entry.model_name, entry.pid).await {
                        let previous = self.table.get(&entry.model_name).await;
                        self.table
                            .insert(Self::entry_for(
                                &spec,
                                entry.pid,
                                status::READY,
                                false,
                                &previous,
                            ))
                            .await;
                        info!(
                            model = %entry.model_name,
                            pid = entry.pid,
                            "reconciler adopted orphaned starting row as ready"
                        );
                        self.reset_persisted_tally(&entry.model_name).await;
                    }
                }
            }
        }
    }

    /// Spawn the backend described by `req` and record the process in
    /// the table. The spawn itself is synchronous; the health gate is not:
    ///
    /// - With a real gate configured (non-empty `health_url` AND
    ///   `health_timeout_ms > 0`) the gate is detached into its own tokio
    ///   task, which polls until success or timeout and records the
    ///   terminal `ready`/`failed` row — this RPC returns `starting`
    ///   within seconds of the spawn regardless of how long the backend
    ///   takes to boot. The caller's disappearance can no longer strand a
    ///   `starting` row (the settle survives even if this future is
    ///   cancelled).
    /// - Without a gate (`health_timeout_ms == 0` or empty `health_url`)
    ///   the process is considered ready immediately and `ready` is
    ///   returned synchronously.
    ///
    /// `provider_name == "compaction"` → the proxy ships the generic
    /// `uv run uvicorn ...` shape and this tamad injects its own embedded
    /// server directory (`--project`), because the Python source is
    /// bundled in this binary (plan-191 Task 10).
    pub async fn load(&self, req: &LoadModelRequest) -> Result<LoadModelResponse> {
        let (args, env) = self.resolve_launch(req).await?;

        // Docker-backed engines (e.g. vLLM-radiance): the proxy shipped a
        // DockerConfig in `docker_config_json`; spawn a container instead of
        // a host binary (plan-080 style runner restored in tamad).
        if !req.docker_config_json.is_empty() {
            return self.load_container(req, args, env).await;
        }

        info!(
            model = %req.model_name,
            command = %req.command,
            "spawning backend process"
        );

        let mut command = tokio::process::Command::new(&req.command);
        command.args(&args);
        for (key, value) in &env {
            command.env(key, value);
        }
        // Same isolation as the proxy's former native path: companion .so
        // resolution next to the binary + own process group (so unload can
        // SIGTERM the whole tree).
        let binary_path = std::path::PathBuf::from(&req.command);
        configure_backend_command(&mut command, binary_path.as_path());
        configure_process_group(&mut command);
        command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to spawn backend '{}' for model '{}'",
                req.command, req.model_name
            )
        })?;
        let pid = child
            .id()
            .ok_or_else(|| anyhow!("failed to get PID for model '{}'", req.model_name))?;
        // Reaped-by-dead-PID reaper: the tamad owns the process, so it
        // must reap it — otherwise a crashed backend lingers as a zombie
        // that still answers `kill(pid, 0)`. On exit the reaper
        // consults the persistent store and the restart budget
        // (plan-193 T2) — see `reap_dead_pid`.
        let model_name = req.model_name.clone();
        {
            let lifecycle = self.shared_copy();
            tokio::spawn(async move {
                let _ = child.wait().await;
                lifecycle.reap_dead_pid(&model_name, pid).await;
            });
        }

        info!(model = %req.model_name, pid, "backend process spawned");

        // Track the new process as `starting` — the normal entry-point
        // visibility. A respawn that comes in through the reaper arrives
        // on a `restarting`-row (round-trip tally already posted into
        // this store); its failed-terminal row retains that tally so
        // an aborted attempt still consumes its budget.
        let previous = self.table.get(&req.model_name).await;
        let inheriting_attempt = previous
            .as_ref()
            .is_some_and(|p| p.status == status::RESTARTING);
        self.table
            .insert(Self::entry_for(
                req,
                pid,
                status::STARTING,
                inheriting_attempt,
                &previous,
            ))
            .await;

        // Health gating: when a real gate is configured, detach it into
        // its own task via `settle_native_gate` (it records the terminal
        // READY/FAILED row internally; `Err` means the gate settled
        // unhealthy). Detaching keeps the RPC response independent of the
        // boot duration AND of the caller's survival: if this future is
        // cancelled mid-boot (caller gone), the spawned settle still runs
        // to completion instead of stranding a forever-`starting` row.
        // Without a gate we fall through to the synchronous instant-ready
        // branch.
        if !req.health_url.is_empty() && req.health_timeout_ms != 0 {
            let timeout = Duration::from_millis(req.health_timeout_ms.max(0) as u64);
            let lc = self.shared_copy();
            let req2 = req.clone();
            let model_name2 = req.model_name.clone();
            tokio::spawn(async move {
                if let Err(e) = lc
                    .settle_native_gate(req2, pid, inheriting_attempt, timeout)
                    .await
                {
                    warn!(
                        model = %model_name2,
                        error = %e,
                        "detached health gate settled unhealthy"
                    );
                }
            });
            return Ok(LoadModelResponse {
                endpoint_url: Self::endpoint_from_health_url(&req.health_url),
                pid: pid as i32,
                status: status::STARTING.to_string(),
            });
        }

        // No gate configured → instant ready (unchanged synchronous path).
        let endpoint_url = Self::endpoint_from_health_url(&req.health_url);
        if self.owns_row(&req.model_name, pid).await {
            // The INSTANT ready (no gate configured) is not a verified
            // success — it preserves the row's bookkeeping so a backend
            // that exits 0.3 s after every start can still trip the
            // budget directive. No persisted-tally reset here either:
            // an unverified ready must keep presenting its at-cap tally
            // to the boot sweep.
            self.table
                .insert(Self::entry_for(
                    req,
                    pid,
                    status::READY,
                    inheriting_attempt,
                    &previous,
                ))
                .await;
        }

        Ok(LoadModelResponse {
            endpoint_url,
            pid: pid as i32,
            status: status::READY.to_string(),
        })
    }

    /// Settles the native-path health gate: polls `wait_for_health`, then
    /// records the terminal row (READY or FAILED) with the exact
    /// bookkeeping semantics of the former inline tail, including the
    /// verified-ready persisted-tally reset. Returns `Err` when the gate
    /// settled unhealthy (the teardown has already run by then).
    async fn settle_native_gate(
        &self,
        req: LoadModelRequest,
        pid: u32,
        inheriting_attempt: bool,
        timeout: Duration,
    ) -> Result<()> {
        let healthy = self.wait_for_health(&req.health_url, timeout).await;

        if !healthy {
            // Status-aware ownership FIRST (checked BEFORE any teardown): a
            // pid match alone does not mean this row still wants our verdict.
            // A concurrent unload / reaper / reconciler may have spun the key
            // around — someone else owns the outcome now; do nothing.
            if !self.owns_starting_row(&req.model_name, pid).await {
                debug!(
                    model = %req.model_name,
                    pid,
                    "gate timed out but no longer owns the starting row; standing down"
                );
                return Ok(());
            }
            warn!(
                model = %req.model_name,
                timeout_ms = timeout.as_millis() as u64,
                "backend failed to become healthy; killing process group"
            );
            let _ = kill_process_group(pid).await;
            tokio::time::sleep(Duration::from_millis(250)).await;
            if is_process_group_alive(pid) {
                let _ = force_kill_process_group(pid).await;
            }
            // Record the failed attempt (status `failed`). Re-check ownership
            // AFTER the kill too — the teardown window is where a concurrent
            // writer is most likely to have moved the row on. Only when the
            // line is still THIS process AND still `starting` may our
            // terminal verdict land.
            if self.owns_starting_row(&req.model_name, pid).await {
                // Bookkeeping reflects the CURRENT row at insert time (e.g.
                // budget round-trip tallies a reaper posted mid-gate are kept).
                // If a restart attempt burns out hard, retain the (already
                // recorded) round-trip tally; a plain load that fails from
                // scratch starts the record clean.
                let previous = self.table.get(&req.model_name).await;
                self.table
                    .insert(Self::entry_for(
                        &req,
                        pid,
                        status::FAILED,
                        inheriting_attempt,
                        &previous,
                    ))
                    .await;
            }
            return Err(anyhow!(
                "backend '{}' for model '{}' failed to become healthy within {}ms",
                req.provider_name,
                req.model_name,
                req.health_timeout_ms
            ));
        }

        // Same status-aware gate on the READY verdict: never overwrite a row
        // that another writer (reaper → RESTARTING/FAILED, unload → gone)
        // has taken over since the poll started.
        if !self.owns_starting_row(&req.model_name, pid).await {
            debug!(
                model = %req.model_name,
                pid,
                "gate went healthy but no longer owns the starting row; standing down"
            );
            return Ok(());
        }
        // The verified-`ready` verdict resets the round-trip record
        // (plan-193 T2): a successful HEALTH GATE clears the in-window trip
        // tally, so a healthy stretch of life restarts the budget clock.
        // The INSTANT ready (no gate configured) is no verified success and
        // keeps its bookkeeping instead. `user_flagged` is NOT auto-cleared
        // on success, either.
        let previous = self.table.get(&req.model_name).await;
        self.table
            .insert(Self::entry_for(&req, pid, status::READY, false, &previous))
            .await;
        info!(model = %req.model_name, pid, "backend became healthy (detached gate)");
        self.reset_persisted_tally(&req.model_name).await;

        Ok(())
    }

    /// Ownership check — does the row for `model_name` still stand
    /// for this very process (pid match)? The terminal rows (failed / ready)
    /// overwrite it their one; an unload or reaper race in between has
    /// already moved this key's own process, and a late
    /// terminal must not clobber it.
    async fn owns_row(&self, model_name: &str, pid: u32) -> bool {
        self.table
            .get(model_name)
            .await
            .is_some_and(|e| e.pid == pid)
    }

    /// Status-aware ownership: is the row still THIS process AND still
    /// awaiting a verdict (`starting`)? Detached writers must gate BOTH any
    /// teardown AND their terminal insert on this — a pid match alone does
    /// not mean the row still wants our verdict (a reaper may have flipped
    /// it to `restarting`, a reconciler adopted it, an unload removed it).
    async fn owns_starting_row(&self, model_name: &str, pid: u32) -> bool {
        self.table
            .get(model_name)
            .await
            .is_some_and(|e| e.pid == pid && e.status == crate::lifecycle::status::STARTING)
    }

    /// Durable half of the verified-ready reset (round-2 P1): a
    /// verified-ready zeroes the on-disk at-cap tally so the manifest does
    /// not keep presenting as at-cap and the next boot sweep will replay a
    /// perfectly fine key. `user_flagged` stays untouched — success never
    /// re-claims the operator's mark. Failures are warned with the operator
    /// recovery sentence; they must not fail the load path itself.
    async fn reset_persisted_tally(&self, model_name: &str) {
        if let Err(e) = self.store.zero_persisted_restart_count(model_name) {
            warn!(
                model = %model_name,
                error = %e,
                "verified-ready success reset could not zero the persisted \
                 at-cap tally; a later boot sweep may still treat this key \
                 as at-cap — recovery = `tama admin unload <key>` then \
                 `load` (clean re-arm)"
            );
        }
    }

    /// Single entry builder for the lifespan (process entry) —
    /// keeps the round-trip tally + the `user_flagged` mirror the
    /// same discipline everywhere: `ready` resets the tally while
    /// retaining the mirror; a failed restart attempt retains
    /// both (the attempt was counted); everything else starts clean.
    fn entry_for(
        req: &LoadModelRequest,
        pid: u32,
        entry_status: &str,
        inherit_bookkeeping: bool,
        previous: &Option<ProcessEntry>,
    ) -> ProcessEntry {
        let (restart_count, window_starts, user_flagged) =
            match (inherit_bookkeeping, previous.as_ref()) {
                (true, Some(prev)) => (
                    prev.restart_count,
                    prev.window_starts.clone(),
                    prev.user_flagged,
                ),
                _ => (
                    0,
                    Vec::new(),
                    previous.as_ref().map(|p| p.user_flagged).unwrap_or(false),
                ),
            };
        ProcessEntry {
            model_name: req.model_name.clone(),
            provider_name: req.provider_name.clone(),
            pid,
            endpoint_url: Self::endpoint_from_health_url(&req.health_url),
            status: entry_status.to_string(),
            started_at: Instant::now(),
            spec: req.clone(),
            restart_count,
            window_starts,
            user_flagged,
        }
    }

    /// Spawn a Docker-backed backend (container).
    ///
    /// The proxy ships a serialized [`DockerConfig`] in `req.docker_config_json`
    /// (the tamad owns no DB). We pull the image if missing, rewrite the
    /// already path-remapped args to the container's mounted model dir, then
    /// `docker run` with the mount/device/shm/capability config.
    ///
    /// Like the native path, the health gate is detached when configured:
    /// with a real gate (`health_url` + positive timeout) this returns
    /// `starting` within seconds while a detached task polls to ready/timeout
    /// and records the terminal row (on failure the container is stopped+
    /// removed and a `failed` entry recorded). Without a gate the container
    /// is considered ready immediately and `ready` is returned synchronously.
    async fn load_container(
        &self,
        req: &LoadModelRequest,
        args: Vec<String>,
        env: Vec<(String, String)>,
    ) -> Result<LoadModelResponse> {
        let config =
            serde_json::from_str::<tama_core::installations::DockerConfig>(&req.docker_config_json)
                .map_err(|e| anyhow!("invalid docker_config_json: {}", e))?;

        // Image presence: pull on first load (images can be large — allow a
        // generous timeout). Fail when the host genuinely can't fetch it.
        let runtime = self.state.container_runtime;
        if !crate::host_installs::docker::runner::is_image_present(runtime, &config.image).await? {
            info!(
                model = %req.model_name,
                image = %config.image,
                "pulling docker image"
            );
            crate::host_installs::docker::runner::pull_image(runtime, &config.image, 1800)
                .await
                .with_context(|| format!("pulling docker image '{}'", config.image))?;
        }

        let local_models = self.state.models_dir.clone();
        let container_models = config.model_mount.container_path.clone();
        let mut container_args = crate::host_installs::docker::runner::rewrite_args_for_container(
            &args,
            &local_models,
            &container_models,
        )?;

        // Inside the container, the backend must listen on 0.0.0.0 and the
        // internal container_port (e.g. 8000). Docker maps the host_port to it.
        tama_core::process::override_arg(&mut container_args, "--host", "0.0.0.0");
        tama_core::process::override_arg(
            &mut container_args,
            "--port",
            &config.container_port.to_string(),
        );

        // Host-side port: the proxy aliases it into the health URL
        // (`http://127.0.0.1:<n>/health`). Docker forwards host_port -> container_port.
        let host_port =
            Self::port_from_health_url(&req.health_url).unwrap_or(config.container_port);

        let env_strs: Vec<String> = env
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();

        info!(
            model = %req.model_name,
            image = %config.image,
            host_port,
            container_port = config.container_port,
            "spawning backend container"
        );

        let container = crate::host_installs::docker::runner::spawn_container(
            runtime,
            &req.model_name,
            &config,
            host_port,
            container_args,
            &env_strs,
            &local_models,
        )
        .await
        .with_context(|| {
            format!(
                "failed to spawn container '{}' for model '{}'",
                config.image, req.model_name
            )
        })?;

        let pid = container.pid;

        // Same entry discipline as the native path: a `starting` row at
        // launch, ownership-guarded terminal rows on settle, and the
        // reaper's round-trip rental carries through for respawns.
        let previous = self.table.get(&req.model_name).await;
        let inheriting_attempt = previous
            .as_ref()
            .is_some_and(|p| p.status == status::RESTARTING);
        self.table
            .insert(Self::entry_for(
                req,
                pid,
                status::STARTING,
                inheriting_attempt,
                &previous,
            ))
            .await;

        // Health gating: when a real gate is configured, detach it into
        // its own task via `settle_container_gate` (same rationale as the
        // native path — the terminal row must land even if the RPC caller
        // vanishes). Without a gate we fall through to the synchronous
        // instant-ready branch.
        if !req.health_url.is_empty() && req.health_timeout_ms != 0 {
            let timeout = Duration::from_millis(req.health_timeout_ms.max(0) as u64);
            let lc = self.shared_copy();
            let req2 = req.clone();
            // Identity-safe teardown: carry the Docker-assigned container id
            // (not the deterministic `tama-{model}` name) into the gate, so a
            // stale gate can never stop/remove a freshly RELOADED container
            // that merely reused the deterministic name.
            let container_id = container.id.clone();
            let model_name2 = req.model_name.clone();
            tokio::spawn(async move {
                if let Err(e) = lc
                    .settle_container_gate(req2, pid, inheriting_attempt, timeout, container_id)
                    .await
                {
                    warn!(
                        model = %model_name2,
                        error = %e,
                        "detached health gate settled unhealthy"
                    );
                }
            });
            return Ok(LoadModelResponse {
                endpoint_url: Self::endpoint_from_health_url(&req.health_url),
                pid: pid as i32,
                status: status::STARTING.to_string(),
            });
        }

        // No gate configured → instant ready (unchanged synchronous path).
        let endpoint_url = Self::endpoint_from_health_url(&req.health_url);
        if self.owns_row(&req.model_name, pid).await {
            // An instant ready (no gate configured) is not a verified
            // success — it preserves the row's bookkeeping and its persisted
            // at-cap tally, exactly like the native no-gate branch.
            self.table
                .insert(Self::entry_for(
                    req,
                    pid,
                    status::READY,
                    inheriting_attempt,
                    &previous,
                ))
                .await;
        }

        Ok(LoadModelResponse {
            endpoint_url,
            pid: pid as i32,
            status: status::READY.to_string(),
        })
    }

    /// Same shape as [`Self::settle_native_gate`] for the Docker path:
    /// polls `wait_for_health`, records the terminal row (READY or
    /// FAILED), and on unhealthy stops and removes the container instead
    /// of killing a process group. Teardown addresses the container BY
    /// INSTANCE ID (the Docker-assigned hash captured at spawn), never by
    /// the deterministic name — a name can be reused by a newer load of the
    /// same model while this stale gate was polling. Returns `Err` when the
    /// gate settled unhealthy (the teardown has already run by then).
    async fn settle_container_gate(
        &self,
        req: LoadModelRequest,
        pid: u32,
        inheriting_attempt: bool,
        timeout: Duration,
        container_id: String,
    ) -> Result<()> {
        let healthy = self.wait_for_health(&req.health_url, timeout).await;

        if !healthy {
            // Status-aware ownership FIRST (checked BEFORE any teardown): a
            // pid match alone does not mean this row still wants our verdict.
            if !self.owns_starting_row(&req.model_name, pid).await {
                debug!(
                    model = %req.model_name,
                    pid,
                    "gate timed out but no longer owns the starting row; standing down"
                );
                return Ok(());
            }
            warn!(
                model = %req.model_name,
                "container failed to become healthy; tearing down"
            );
            // docker stop/rm accept full container ids — teardown is bound to
            // THIS instance even if the deterministic name was reused.
            let _ = crate::host_installs::docker::runner::stop_container(
                self.state.container_runtime,
                &container_id,
            )
            .await;
            let _ = crate::host_installs::docker::runner::remove_container(
                self.state.container_runtime,
                &container_id,
            )
            .await;
            // Re-check ownership AFTER teardown too — the kill window is where
            // a concurrent writer is most likely to have moved the row on.
            if self.owns_starting_row(&req.model_name, pid).await {
                let previous = self.table.get(&req.model_name).await;
                self.table
                    .insert(Self::entry_for(
                        &req,
                        pid,
                        status::FAILED,
                        inheriting_attempt,
                        &previous,
                    ))
                    .await;
            }
            return Err(anyhow!(
                "container for model '{}' failed to become healthy within {}ms",
                req.model_name,
                req.health_timeout_ms
            ));
        }

        // Same status-aware gate on the READY verdict: never overwrite a row
        // that another writer has taken over since the poll started.
        if !self.owns_starting_row(&req.model_name, pid).await {
            debug!(
                model = %req.model_name,
                pid,
                "gate went healthy but no longer owns the starting row; standing down"
            );
            return Ok(());
        }
        // Verified-ready reset rule as the host path: a successful container
        // health gate clears the trip tally (`user_flagged` untouched).
        let previous = self.table.get(&req.model_name).await;
        self.table
            .insert(Self::entry_for(&req, pid, status::READY, false, &previous))
            .await;
        info!(model = %req.model_name, pid, "backend became healthy (detached gate)");
        self.reset_persisted_tally(&req.model_name).await;

        Ok(())
    }

    /// Extract the host-side port from a health URL like
    /// `http://127.0.0.1:8080/health`. Falls back to None.
    fn port_from_health_url(url: &str) -> Option<u16> {
        if url.is_empty() {
            return None;
        }
        url::Url::parse(url).ok().and_then(|u| u.port())
    }

    /// Kill the process group for `model_name` and remove the entry.
    ///
    /// Returns an error when the model is unknown to this tamad.
    pub async fn unload(&self, model_name: &str) -> Result<()> {
        let entry = self
            .table
            .remove(model_name)
            .await
            .ok_or_else(|| anyhow!("model '{}' is not loaded on this tamad", model_name))?;

        info!(model = %model_name, pid = entry.pid, "unloading backend process");

        // Docker backend: the "pid" is the container's host process. Kill it
        // and also stop+remove the managed container so it doesn't linger or
        // auto-restart.
        if !entry.spec.docker_config_json.is_empty() {
            let _ = kill_process_group(entry.pid).await;
            let name = crate::host_installs::docker::runner::container_name_for(model_name);
            let _ = crate::host_installs::docker::runner::stop_container(
                self.state.container_runtime,
                &name,
            )
            .await;
            let _ = crate::host_installs::docker::runner::remove_container(
                self.state.container_runtime,
                &name,
            )
            .await;
            info!(model = %model_name, "docker backend container unloaded");
            return Ok(());
        }

        let _ = kill_process_group(entry.pid).await;

        // SIGTERM → wait up to 5s → SIGKILL.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            tokio::time::sleep(Duration::from_millis(250)).await;
            if !is_process_group_alive(entry.pid) {
                break;
            }
            if Instant::now() >= deadline {
                warn!(model = %model_name, pid = entry.pid, "SIGTERM ignored; sending SIGKILL");
                let _ = force_kill_process_group(entry.pid).await;
                // Reap it properly — a lingering zombie would keep
                // answering `kill(pid, 0)`.
                if let Err(e) = wait_group_dead(entry.pid).await {
                    warn!(model = %model_name, pid = entry.pid, error = %e, "group not fully reaped after SIGKILL");
                }
                break;
            }
        }

        info!(model = %model_name, "backend process unloaded");
        Ok(())
    }

    /// Kill every loaded backend (their whole process groups): SIGTERM
    /// each, grace, SIGKILL escalation, then drop the entries — the
    /// in-memory inventory must not outlive the daemon (plan-191
    /// follow-up A: a SIGTERM to tamad leaves no orphaned backends).
    pub async fn kill_all(&self) -> Result<()> {
        let entries = self.table.list().await;
        for entry in entries {
            info!(
                model = %entry.model_name,
                pid = entry.pid,
                "kill_all: stopping backend process group"
            );
            // Reuse the graceful per-model path (SIGTERM → 5s → SIGKILL
            // → entry removal). One model's stall must not block the
            // rest — errors are logged, not returned.
            if let Err(e) = self.unload(&entry.model_name).await {
                warn!(
                    model = %entry.model_name,
                    error = %e,
                    "kill_all: unload failed; continuing with other backends"
                );
            }
        }
        Ok(())
    }

    /// Unload then re-load using the stored launch spec (the original
    /// `LoadModelRequest` that started the process).
    pub async fn restart(&self, model_name: &str) -> Result<LoadModelResponse> {
        let entry = self
            .table
            .get(model_name)
            .await
            .ok_or_else(|| anyhow!("model '{}' is not loaded on this tamad", model_name))?;
        let spec = entry.spec.clone();
        self.unload(model_name).await?;
        self.load(&spec).await
    }

    /// Group table entries by provider name.
    ///
    /// `engine`/`version`/`gpu_variant`/`status` are empty/"unknown": the
    /// tamad has no database — the proxy's DB is the source of truth for
    /// those fields.
    pub async fn list(&self) -> Vec<ProviderInfo> {
        let mut by_provider: std::collections::BTreeMap<String, Vec<ProcessInfo>> =
            std::collections::BTreeMap::new();
        for entry in self.table.list().await {
            let info = to_process_info(&entry, self.store.get(&entry.model_name).as_ref());
            by_provider
                .entry(entry.provider_name)
                .or_default()
                .push(info);
        }
        by_provider
            .into_iter()
            .map(|(name, loaded_models)| ProviderInfo {
                name,
                engine: String::new(),
                version: String::new(),
                status: "unknown".to_string(),
                gpu_variant: String::new(),
                loaded_models,
            })
            .collect()
    }

    /// Dead-PID reaper arm (plan-193 T2): runs when the reaped child
    /// exits. Two outcomes.
    ///
    /// *Legacy arm* — the key has no store row, or the row is not
    /// `desired`, or someone already gave this key up (either on-disk or
    /// in-memory `user_flagged`): mark the row `failed`, exactly the
    /// pre-T2 behavior.
    ///
    /// *Respawn arm* — desired, unflagged store row and the budget
    /// still has room: set the row to `restarting`, post one replay
    /// into the sliding heavy window, and re-fire the saved spec via
    /// [`load`](Self::load). If the class is a downward of
    /// `restart_count >= max` (the budget gate's refusal), set the row
    /// to `budget_exhausted` and burn in the operator's mark on the store.
    /// No auto-respawn after that.
    async fn reap_dead_pid(&self, model_name: &str, pid: u32) {
        let entry = match self.table.get(model_name).await {
            Some(e) if e.pid == pid => e,
            _ => return, // Already-unloaded, or a gone reaper from a previous round —
                         // the pid guard makes it a no-op.
        };
        debug!(
            model = %model_name,
            pid,
            status = %entry.status,
            "dead-PID reaper fired"
        );

        let stored = self.store.get(model_name);
        let respawn = stored
            .as_ref()
            .is_some_and(|sp| sp.desired && !sp.user_flagged)
            && !entry.user_flagged;
        if !respawn {
            // Legacy arm: no store row to restart from (or given up) —
            // nothing we can do, and no round-trip is charged.
            self.table.mark_failed(model_name, pid).await;
            return;
        }
        let stored = stored.expect("just-checked respawn");
        let max_restarts = stored.max_restarts;

        // One death → one round-trip in the sliding window, trimmed
        // and re-published.
        let Some(restart_count) = self
            .table
            .record_restart_window(model_name, Self::now_unix_ms())
            .await
        else {
            return; // The rows moved on (unload) — no-op.
        };

        if restart_count >= max_restarts {
            // Budget gate's refusal: a terminal state on the row, the
            // operator's mark burned in-disk, no auto-respawn after.
            if self.table.mark_budget_exhausted(model_name, pid).await {
                warn!(
                    model = %model_name,
                    pid,
                    restart_count,
                    max_restarts,
                    "restart budget exhausted; stopping auto-respawn for this key"
                );
                if let Err(e) = persist_tripped(&self.store, model_name, max_restarts).await {
                    error!(
                        model = %model_name,
                        pid,
                        restart_count,
                        max_restarts,
                        error = %e,
                        "budget tripped but the trip persistence (the mark + the at-cap \
                        tally) did not reach disk — the boot sweep 's at-cap refusal \
                        still covers it; recovery = `tama admin unload <key>` then \
                        `load` (clean re-arm)"
                    );
                }
            }
            return;
        }

        // Round-trip is coming up: set the row `restarting` FIRST
        // (the row's background is starting up, and it's defunct before
        // the replacement even boots). Only act when the line of this
        // process is still standing (the same guard the table method
        // applies) — the reaper bails on one move.
        if !self.table.mark_restarting(model_name, pid).await {
            return;
        }
        let req: LoadModelRequest = (&stored).into();
        // Hand the job to the supervisor. The reaper dispatches, so the
        // row stands `restarting` here; `load` flips it to `starting`
        // the moment the replacement launches.
        if self
            .respawn_tx
            .send((model_name.to_string(), req, restart_count))
            .is_err()
        {
            warn!(
                model = %model_name,
                restart_count,
                max_restarts,
                "respawn supervisor gone; row left `restarting` (daemon unwinding)"
            );
        }
    }

    /// Boot sweep (plan-193 T2's entry-point): re-fire every store
    /// row that's `desired` and not `user_flagged` as a live process.
    ///
    /// Per file: rows already live under the same key skip it (row
    /// wins over file); a Load that fails is logged and left desired
    /// (boot must never fail because one model failed). Bounded
    /// parallelism: at most two Loads in flight at any moment.
    /// (A serial loop would chain 30–300 s health polling
    /// latencies across all desired models.)
    ///
    /// `--no-replay-desired` turns the whole sweep off (this is the
    /// only operational switch in the plan).
    ///
    /// Pre-policy durability addition (round-2 P1): a row whose PERSISTED
    /// at-cap tally reached the disk but whose trip mark did not (the
    /// ENOSPC-trip shape) is also refused — replaying it would re-arm the
    /// crash loop. The log carries the recovery sentence (`tama admin
    /// unload` then `load`); the manifest is not rewritten.
    pub async fn replay_desired(&self, enabled: bool) {
        if !enabled {
            info!("boot sweep: replay-desired is disabled (--no-replay-desired); nothing replayed");
            return;
        }
        let sem = Arc::new(tokio::sync::Semaphore::new(2));
        let mut inflight: Vec<(String, tokio::task::JoinHandle<Result<LoadModelResponse>>)> =
            Vec::new();
        for stored in self.store.list() {
            let key = stored.model_name.clone();
            if !stored.desired {
                continue;
            }
            if stored.user_flagged {
                debug!(model = %key, "boot sweep: skipping user-flagged model");
                continue;
            }
            // Pre-policy durability (round-2 P1): a manifest whose PERSISTED
            // at-cap tally reached the disk but whose trip mark did not
            // (the ENOSPC not-on-disk shape) is also refused — replaying it
            // would re-arm the crash loop. The log carries the operator
            // recovery sentence; the manifest stays byte-for-byte as found.
            if stored.persisted_restart_count >= stored.max_restarts {
                warn!(
                    model = %key,
                    persisted = stored.persisted_restart_count,
                    cap = stored.max_restarts,
                    "boot sweep: {}",
                    at_cap_skip_note(&key, stored.persisted_restart_count, stored.max_restarts)
                );
                continue;
            }
            if self
                .table
                .get(&key)
                .await
                .filter(|e| crate::process::is_process_alive(e.pid))
                .is_some()
            {
                // Already running in-process under this key — the row
                // wins over the file. No double-spawn.
                debug!(model = %key, "boot sweep: model already alive; skipping");
                continue;
            }
            let Ok(permit) = sem.clone().acquire_owned().await else {
                warn!("boot sweep: semaphore closed; aborting remaining replays");
                break;
            };
            let lifecycle = self.shared_copy();
            inflight.push((
                key,
                tokio::spawn(async move {
                    let _permit = permit;
                    let req: LoadModelRequest = (&stored).into();
                    lifecycle.load(&req).await
                }),
            ));
        }
        let mut replayed = 0usize;
        for (key, task) in inflight {
            match task.await {
                Ok(Ok(resp)) => {
                    replayed += 1;
                    // With a configured health gate this fires at
                    // `starting` time — the detached gate records the
                    // terminal row asynchronously.
                    info!(
                        model = %key,
                        pid = resp.pid,
                        status = %resp.status,
                        "boot sweep: replayed desired model (gate detached; row settles async)"
                    );
                }
                Ok(Err(e)) => {
                    warn!(
                        model = %key,
                        error = %e,
                        "boot sweep: load failed; model left desired on disk"
                    );
                }
                Err(e) => {
                    warn!(model = %key, error = %e, "boot sweep: load task failed");
                }
            }
        }
        if replayed > 0 {
            info!(replayed, "boot sweep complete");
        }
    }

    /// Current unix time in milliseconds (budget-window bookkeeping;
    /// mirroring the store's own clock beyond a boundary — the store
    /// keeps `updated_at_ms` in the same unit).
    fn now_unix_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// Health-poll `url` (200–399) every 500ms until `timeout`.
    async fn wait_for_health(&self, url: &str, timeout: Duration) -> bool {
        let start = Instant::now();
        loop {
            if start.elapsed() >= timeout {
                return false;
            }
            let healthy = crate::process::check_health(url, Some(5))
                .await
                .map(|resp| resp.status().is_success())
                .unwrap_or(false);
            if healthy {
                debug!(url, "health check passed");
                return true;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Resolve the spawn args/env for a request:
    /// 1. remap model paths from the proxy's models dir (carried in
    ///    `PROXY_MODELS_DIR_ENV`) to the tamad's own `models_dir`,
    /// 2. strip the helper env key before spawning,
    /// 3. for `provider_name == "compaction"`, inject the embedded compaction
    ///    server directory (`--project` after `run`).
    async fn resolve_launch(
        &self,
        req: &LoadModelRequest,
    ) -> Result<(Vec<String>, Vec<(String, String)>)> {
        let mut env: Vec<(String, String)> = Vec::new();
        let mut proxy_models_dir: Option<String> = None;
        for (key, value) in &req.env {
            if key == PROXY_MODELS_DIR_ENV {
                proxy_models_dir = Some(value.clone());
            } else {
                env.push((key.clone(), value.clone()));
            }
        }

        // GPU isolation env vars are resolved on THIS HOST against this
        // daemon's own hardware (the proxy sends the configured device +
        // variant — ADR-0010: it never samples local hardware). Explicit
        // entries from the installation's `default_env` win over the
        // resolved vendor var.
        let device = req.gpu_device.trim();
        if !device.is_empty() {
            if let Ok(variant) = req.gpu_variant.parse::<tama_core::gpu::GpuVariant>() {
                let dev = device.to_string();
                match tokio::task::spawn_blocking(move || {
                    crate::gpu::env::resolve_gpu_env(&dev, &variant)
                })
                .await
                {
                    Ok(Some((key, value))) => {
                        if !env.iter().any(|(k, _)| k == &key) {
                            info!(
                                model = %req.model_name,
                                env = %key,
                                value = %value,
                                "resolved GPU isolation env on this host"
                            );
                            env.push((key, value));
                        }
                    }
                    Ok(None) => {
                        debug!(
                            model = %req.model_name,
                            device,
                            variant = %req.gpu_variant,
                            "no GPU env var for this device/variant (no matching local GPU)"
                        );
                    }
                    Err(e) => {
                        warn!(
                            model = %req.model_name,
                            error = %e,
                            "GPU env resolution panicked; launching without isolation env"
                        );
                    }
                }
            } else {
                warn!(
                    model = %req.model_name,
                    variant = %req.gpu_variant,
                    "unknown gpu_variant; skipping GPU env resolution"
                );
            }
        }

        let mut args = req.args.clone();
        if let Some(ref proxy_dir) = proxy_models_dir {
            let local_dir = self.state.models_dir.to_string_lossy().to_string();
            if proxy_dir.as_str() != local_dir {
                let prefix = proxy_dir.trim_end_matches('/').to_string() + "/";
                args = args
                    .into_iter()
                    .map(|arg| Self::remap_path_prefix(&arg, &prefix, &local_dir))
                    .collect();
                info!(
                    "remapped model paths from '{}' to '{}'",
                    proxy_dir, local_dir
                );
            }
        }

        // Compaction: the Python server is embedded in this binary — inject
        // the `--project` dir the proxy cannot know about.
        if req.provider_name == "compaction" {
            let server_dir = crate::compaction_server::get_server_dir(&self.state.data_dir)
                .with_context(|| "resolving embedded compaction server dir")?;
            let project = server_dir.to_string_lossy().into_owned();
            let mut new_args = Vec::with_capacity(args.len() + 1);
            let mut injected = false;
            for arg in args.into_iter() {
                if !injected && arg == "run" {
                    new_args.push(arg);
                    new_args.push("--project".to_string());
                    new_args.push(project.clone());
                    injected = true;
                } else {
                    new_args.push(arg);
                }
            }
            if !injected {
                new_args.insert(1, "--project".to_string());
                new_args.insert(2, project);
            }
            args = new_args;
        }

        Ok((args, env))
    }

    /// Rewrite an arg that is a path (optionally shell-quoted) under the
    /// proxy's models dir to the local equivalent. Non-paths pass through.
    fn remap_path_prefix(arg: &str, prefix: &str, local_dir: &str) -> String {
        let (quoted, inner) =
            if let Some(stripped) = arg.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
                (true, stripped)
            } else if let Some(stripped) = arg.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                (true, stripped)
            } else {
                (false, arg)
            };

        if let Some(rel) = inner.strip_prefix(prefix) {
            let remapped = if rel.is_empty() {
                local_dir.to_string()
            } else {
                format!("{}/{}", local_dir.trim_end_matches('/'), rel)
            };
            if quoted {
                format!("'{}'", remapped)
            } else {
                remapped
            }
        } else {
            arg.to_string()
        }
    }

    /// Derive the base endpoint URL from the health URL (strip the path).
    fn endpoint_from_health_url(health_url: &str) -> String {
        url::Url::parse(health_url)
            .ok()
            .map(|mut u| {
                u.set_path("");
                u.set_query(None);
                u.set_fragment(None);
                u.to_string().trim_end_matches('/').to_string()
            })
            .unwrap_or_else(|| health_url.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::test_state;
    use ::prost::Message;

    /// A healthy store succeeds on the first attempt: the cheap path is a
    /// single atomic temp+rename — the helper must never loop, and the
    /// mark is persisted.
    #[tokio::test]
    async fn test_persist_tripped_healthy_dir_ok_first_attempt() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path()).unwrap();
        store
            .insert("pf-ok", &make_req("pf-ok", "sh", &[], 0), true)
            .unwrap();

        persist_tripped(&store, "pf-ok", 10)
            .await
            .expect("healthy dir: Ok on the 1st attempt (no retry swallows)");

        assert!(
            store.get("pf-ok").unwrap().user_flagged,
            "the mark is persisted (terminal state is on disk)"
        );
        assert_eq!(
            store.get("pf-ok").unwrap().persisted_restart_count,
            10,
            "the at-cap tally is persisted in the same bounded write"
        );
    }

    /// With the state directory gone, EVERY retry fails and the helper
    /// gives up with `Err` — the retry loop runs and bails instead of
    /// panic/hang, reaching the caller's fatal branch. The caller's
    /// `error!` (with the operator recovery text) is the recovery
    /// contract; this unit pins the error-shaped path — the same shape
    /// that only its caller can convert into the fatal log.
    #[tokio::test]
    async fn test_persist_tripped_gives_up_when_dir_gone() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path()).unwrap();
        store
            .insert("pf-gone", &make_req("pf-gone", "sh", &[], 0), true)
            .unwrap();
        // Kill the store out from under itself: the in-memory manifest
        // remains, but every temp open + rename fails — all three
        // attempts are exhausted.
        std::fs::remove_dir_all(dir.path()).unwrap();

        let err = persist_tripped(&store, "pf-gone", 10)
            .await
            .expect_err("missing dir: every retry fails → the helper gives up with Err");
        assert!(
            err.to_string().contains("pf-gone"),
            "the failed key is carried for the caller's error!"
        );
    }

    fn make_req(
        model: &str,
        command: &str,
        args: &[&str],
        health_timeout_ms: i64,
    ) -> LoadModelRequest {
        let env = std::collections::HashMap::new();
        let health_url = if health_timeout_ms > 0 {
            format!("http://127.0.0.1:59{}/health", model.len() % 100)
        } else {
            String::new()
        };
        LoadModelRequest {
            provider_name: "llama_cpp".to_string(),
            model_path: format!("owner/repo/{}.gguf", model),
            gpu_variant: "cpu".to_string(),
            params: std::collections::HashMap::new(),
            model_name: model.to_string(),
            command: command.to_string(),
            args: args.iter().map(|a| a.to_string()).collect(),
            env,
            health_url,
            health_timeout_ms,
            gpu_device: String::new(),
            docker_config_json: String::new(),
        }
    }

    /// load (health skipped) → alive process in the table; restart → new
    /// pid; unload → entry gone and process dead.
    #[tokio::test]
    async fn test_load_restart_unload() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        // health_timeout_ms = 0 → immediately ready, no health polling.
        let resp = lc
            .load(&make_req("sleepy", "sh", &["-c", "sleep 30"], 0))
            .await
            .expect("load should succeed");
        assert_ne!(resp.pid, 0);
        assert_eq!(resp.status, "ready");

        let entry = table.get("sleepy").await.expect("entry recorded");
        assert_eq!(entry.status, "ready");
        assert!(
            crate::process::is_process_alive(entry.pid),
            "spawned process must be alive"
        );
        assert_eq!(entry.pid as i32, resp.pid);

        // Restart → new pid, old process gone.
        let old_pid = entry.pid;
        let resp2 = lc.restart("sleepy").await.expect("restart should succeed");
        assert_ne!(resp2.pid, old_pid as i32, "restart must spawn a new pid");
        // Old process should be dead (poll briefly).
        for _ in 0..40 {
            if !crate::process::is_process_alive(old_pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        assert!(
            !crate::process::is_process_alive(old_pid),
            "old process must be dead after restart"
        );
        let new_pid = table.get("sleepy").await.expect("entry replaced").pid;
        assert_eq!(new_pid as i32, resp2.pid);
        assert!(crate::process::is_process_alive(new_pid));

        // Unload → entry gone, process dead.
        lc.unload("sleepy").await.expect("unload should succeed");
        assert!(table.get("sleepy").await.is_none(), "entry removed");
        for _ in 0..40 {
            if !crate::process::is_process_alive(new_pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        assert!(
            !crate::process::is_process_alive(new_pid),
            "process must be dead after unload"
        );
    }

    /// unload of an unknown model fails.
    #[tokio::test]
    async fn test_unload_unknown_fails() {
        let (state, _dir) = test_state();
        let lc = TamadLifecycle::new(
            Arc::new(ProcessTable::default()),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );
        assert!(lc.unload("nope").await.is_err());
    }

    /// restart of an unknown model fails.
    #[tokio::test]
    async fn test_restart_unknown_fails() {
        let (state, _dir) = test_state();
        let lc = TamadLifecycle::new(
            Arc::new(ProcessTable::default()),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );
        assert!(lc.restart("nope").await.is_err());
    }

    /// Health polling with a configured gate: the RPC returns `starting`
    /// immediately (the gate runs detached) and the row flips to `ready`
    /// asynchronously once the sniffer answers 200.
    #[tokio::test]
    async fn test_load_with_health_check() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 512];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                        .await;
                });
            }
        });

        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        let mut req = make_req("healthy", "sh", &["-c", "sleep 30"], 0);
        req.health_url = format!("http://127.0.0.1:{port}/health");
        req.health_timeout_ms = 10_000;
        let resp = lc.load(&req).await.expect("health load should succeed");
        assert_eq!(resp.status, status::STARTING);
        assert_eq!(resp.endpoint_url, format!("http://127.0.0.1:{port}"));

        // The detached gate settles the row asynchronously.
        let entry = poll_until_status(&table, "healthy", status::READY, Duration::from_secs(5))
            .await
            .expect("row must reach ready via the detached gate");
        assert_eq!(entry.endpoint_url, format!("http://127.0.0.1:{port}"));

        lc.unload("healthy").await.ok();
    }

    /// Poll `table` until `model` reaches `want`, returning its entry.
    /// Gives up after `budget` and returns `None`.
    async fn poll_until_status(
        table: &Arc<ProcessTable>,
        model: &str,
        want: &str,
        budget: Duration,
    ) -> Option<ProcessEntry> {
        let deadline = tokio::time::Instant::now() + budget;
        while tokio::time::Instant::now() < deadline {
            if let Some(entry) = table.get(model).await {
                if entry.status == want {
                    return Some(entry);
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        None
    }

    /// Health timeout: unreachable health URL → `Ok(starting)` from the
    /// RPC, then the detached gate times out, kills the process group and
    /// records the failed entry asynchronously.
    #[tokio::test]
    async fn test_load_health_timeout() {
        use crate::process::is_process_group_alive;
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        let mut req = make_req("unhealthy", "sh", &["-c", "sleep 30"], 0);
        // A port nothing listens on.
        req.health_url = "http://127.0.0.1:1/health".to_string();
        req.health_timeout_ms = 1_500;
        let resp = lc
            .load(&req)
            .await
            .expect("RPC returns starting; gate detached");
        assert_eq!(resp.status, status::STARTING);

        // The detached gate settles the row to `failed` after its timeout
        // (1.5 s) + teardown margin. Generous budget: loaded CI machines
        // have been observed starving sub-5s polls.
        let entry = poll_until_status(&table, "unhealthy", status::FAILED, Duration::from_secs(10))
            .await
            .expect("row must reach failed via the detached gate");

        // Process group must have been killed.
        for _ in 0..40 {
            if !is_process_group_alive(entry.pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        assert!(
            !is_process_group_alive(entry.pid),
            "process group must be dead after the detached gate timed out"
        );
    }

    /// A backend that exits on its own is reaped by the tamad and marked
    /// "failed" in the table (the reap task is the authoritative liveness
    /// signal — a zombie pid would otherwise read as alive via kill(pid,0)
    /// and the auto-load path would never restart it).
    #[tokio::test]
    async fn test_load_marks_failed_when_backend_crashes() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        // No health check: load succeeds immediately, then the process
        // exits on its own.
        let resp = lc
            .load(&make_req("crashy", "sh", &["-c", "exit 1"], 0))
            .await
            .expect("load should succeed");
        assert_eq!(resp.status, "ready");

        // Poll until the reap task has marked the entry failed.
        for _ in 0..40 {
            if let Some(entry) = table.get("crashy").await {
                if entry.status == "failed" {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let entry = table.get("crashy").await.expect("entry kept");
        assert_eq!(
            entry.status, "failed",
            "crashed backend must be marked failed"
        );

        // The snapshot hands back the entry; the caller's alive fold
        // reports it dead (status "failed" → not alive).
        let snap = table.snapshot().await;
        let e = snap
            .iter()
            .find(|e| e.model_name == "crashy")
            .expect("crashy in snapshot");
        assert_eq!(e.status, "failed", "crashed backend marked failed");
    }

    /// Seed the row as an in-flight respawn (`restarting`) that already
    /// carries an in-window round-trip tally — the shape a respawning row
    /// has when `load()` re-injects the replacement.
    async fn seed_restarting_table(table: &Arc<ProcessTable>, model: &str) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        table
            .insert(ProcessEntry {
                model_name: model.to_string(),
                provider_name: "llama_cpp".to_string(),
                pid: 1,
                endpoint_url: String::new(),
                status: status::RESTARTING.to_string(),
                started_at: Instant::now(),
                spec: make_req(model, "sh", &["-c", "sleep 30"], 0),
                restart_count: 3,
                window_starts: vec![now_ms - 2_000, now_ms - 1_000, now_ms],
                user_flagged: false,
            })
            .await;
    }

    /// A load that reaches `ready` must reset the in-window restart tally
    /// ONLY when the health gate actually fired (`health_url` present AND
    /// `health_timeout_ms` positive). A spec with a non-empty `health_url`
    /// but `health_timeout_ms == 0` SKIPS the gate (`load`'s `healthy`
    /// branch), so that row is NOT verified: it preserves the in-flight
    /// round-trip tally instead of clearing it.
    ///
    /// Before the double-negation fix, `verified` was `A || B` rather than
    /// `A && B`, so this family was misread as verified and the tally was
    /// reset (restart_count read 0) — failing the two asserts below.
    #[tokio::test]
    async fn test_load_ready_url_zero_timeout_is_unverified_and_preserves_window() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        seed_restarting_table(&table, "win").await;

        // Non-empty URL but zero timeout: the health gate is skipped, so
        // the row is never verified.
        let mut req = make_req("win", "sh", &["-c", "sleep 30"], 0);
        req.health_url = "http://127.0.0.1:1/health".to_string();
        let resp = lc
            .load(&req)
            .await
            .expect("ready without health polling (gate skipped)");
        assert_eq!(resp.status, status::READY);

        let entry = table.get("win").await.expect("row present after ready");
        assert_eq!(entry.status, status::READY);
        assert_eq!(
            entry.restart_count,
            3,
            "URL + zero-timeout skips the health gate: not verified, so the round-trip tally is preserved"
        );
        assert_eq!(
            entry.window_starts.len(),
            3,
            "gate was not verified: the restart window must not be reset"
        );
        lc.unload("win").await.ok();
    }

    /// Behaviour the fix must NOT break: a concrete `health_url` AND a
    /// positive `health_timeout_ms` round-trips a verified ready, so the
    /// successful health gate resets the in-window restart tally (the
    /// `restarting` row's tallies are cleared from the verdict).
    #[tokio::test]
    async fn test_load_ready_url_positive_timeout_is_verified_and_resets_window() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 512];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                        .await;
                });
            }
        });

        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        seed_restarting_table(&table, "gated").await;

        let mut req = make_req("gated", "sh", &["-c", "sleep 30"], 0);
        req.health_url = format!("http://127.0.0.1:{port}/health");
        req.health_timeout_ms = 10_000;
        let resp = lc
            .load(&req)
            .await
            .expect("health-gated load should return starting immediately");
        assert_eq!(resp.status, status::STARTING);

        // Await the terminal row transition driven by the detached gate.
        let entry = poll_until_status(&table, "gated", status::READY, Duration::from_secs(5))
            .await
            .expect("verified ready must land via the detached gate");
        assert_eq!(
            entry.restart_count, 0,
            "the verified health-gated ready resets the round-trip tally"
        );
        assert!(
            entry.window_starts.is_empty(),
            "verified ready resets the restart window"
        );
        lc.unload("gated").await.ok();
    }

    /// "No `health_url` + no timeout" is the canonically unverified ready
    /// (the gate never fires), and is the baseline a "URL + zero-timeout"
    /// spec (the previous test) must match: the tally is preserved, not
    /// reset.
    #[tokio::test]
    async fn test_load_ready_no_url_zero_timeout_is_unverified_and_preserves_window() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        seed_restarting_table(&table, "nourl").await;

        // `make_req` with `health_timeout_ms == 0` leaves `health_url` empty.
        let req = make_req("nourl", "sh", &["-c", "sleep 30"], 0);
        let resp = lc
            .load(&req)
            .await
            .expect("ready without health polling (no gate)");
        assert_eq!(resp.status, status::READY);

        let entry = table.get("nourl").await.expect("row present after ready");
        assert_eq!(entry.status, status::READY);
        assert_eq!(
            entry.restart_count, 3,
            "no health gate: unverified, so the round-trip tally is preserved"
        );
        assert_eq!(
            entry.window_starts.len(),
            3,
            "no gate fired: the restart window must not be reset"
        );
        lc.unload("nourl").await.ok();
    }

    /// list() groups entries by provider_name with empty engine/version.
    #[tokio::test]
    async fn test_list_groups_by_provider() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        let mut req_a = make_req("alpha", "sh", &["-c", "sleep 30"], 0);
        req_a.provider_name = "llama_cpp".to_string();
        let mut req_b = make_req("beta", "sh", &["-c", "sleep 30"], 0);
        req_b.provider_name = "vllm".to_string();
        lc.load(&req_a).await.unwrap();
        lc.load(&req_b).await.unwrap();

        let providers = lc.list().await;
        assert_eq!(providers.len(), 2);
        let llama = providers
            .iter()
            .find(|p| p.name == "llama_cpp")
            .expect("llama_cpp group");
        assert_eq!(llama.loaded_models.len(), 1);
        assert_eq!(llama.loaded_models[0].model_name, "alpha");
        assert!(llama.engine.is_empty());
        assert_eq!(llama.status, "unknown");
        let vllm = providers
            .iter()
            .find(|p| p.name == "vllm")
            .expect("vllm group");
        assert_eq!(vllm.loaded_models[0].model_name, "beta");

        let _ = lc.unload("alpha").await;
        let _ = lc.unload("beta").await;
    }

    /// Model paths under the proxy's models dir are remapped to the
    /// tamad's own models dir via PROXY_MODELS_DIR_ENV.
    #[tokio::test]
    async fn test_models_dir_remap() {
        let (state, _dir) = test_state();
        let lc = TamadLifecycle::new(
            Arc::new(ProcessTable::default()),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        let proxy_dir = "/srv/proxy/models";
        let local_dir = state.models_dir.to_string_lossy().to_string();
        assert_ne!(proxy_dir, local_dir, "fixture must differ from local dir");

        let mut req = make_req(
            "pathy",
            "sh",
            &[
                "-c",
                "sleep 30",
                "-m",
                "/srv/proxy/models/owner/repo/m.gguf",
            ],
            0,
        );
        req.env
            .insert(PROXY_MODELS_DIR_ENV.to_string(), proxy_dir.to_string());

        let (args, env) = lc.resolve_launch(&req).await.unwrap();
        // The -m value was remapped to the local models dir.
        let mut it = args.iter();
        let m_pos = it.position(|a| a == "-m").expect("-m flag present");
        let remapped = &args[m_pos + 1];
        assert_eq!(
            remapped,
            &format!("{local_dir}/owner/repo/m.gguf"),
            "path must be remapped"
        );
        // The helper env key was stripped.
        assert!(
            !env.iter().any(|(k, _)| k == PROXY_MODELS_DIR_ENV),
            "PROXY_MODELS_DIR_ENV must not be passed to the process"
        );
        // A non-path arg is untouched.
        assert!(args.contains(&"sleep 30".to_string()));
    }

    /// Paths already under the local models dir are left untouched.
    #[test]
    fn test_remap_path_prefix() {
        assert_eq!(
            TamadLifecycle::remap_path_prefix(
                "/srv/models/a/b.gguf",
                "/srv/models/",
                "/local/models"
            ),
            "/local/models/a/b.gguf"
        );
        // Quoted path.
        assert_eq!(
            TamadLifecycle::remap_path_prefix(
                "'/srv/models/a b/c.gguf'",
                "/srv/models/",
                "/local/models"
            ),
            "'/local/models/a b/c.gguf'"
        );
        // Non-path passthrough.
        assert_eq!(
            TamadLifecycle::remap_path_prefix("-ngl", "/srv/models/", "/local/models"),
            "-ngl"
        );
        assert_eq!(
            TamadLifecycle::remap_path_prefix("99", "/srv/models/", "/local/models"),
            "99"
        );
    }

    /// `resolve_launch` GPU-env wiring: the isolation env vars are resolved
    /// on this host (ADR-0010) — on a GPU-less host the resolution must not
    /// fail and no vendor env appears; explicit `default_env` entries are
    /// always preserved; the cpu variant never gains a vendor env. The
    /// device→env mapping itself is covered in `gpu::env` tests.
    #[tokio::test]
    async fn test_resolve_launch_gpu_env_wiring() {
        let (state, _dir) = test_state();
        let lc = TamadLifecycle::new(
            Arc::new(ProcessTable::default()),
            state.store.clone(),
            state,
        );

        // GPU device + cuda variant with an explicit env entry.
        let mut req = make_req("gpu-env-a", "/bin/true", &[], 0);
        req.gpu_device = "GPU0".to_string();
        req.gpu_variant = "cuda".to_string();
        req.env.insert("MY_ENV".to_string(), "1".to_string());
        let (_args, env) = lc.resolve_launch(&req).await.expect("resolve must succeed");
        let env: std::collections::BTreeMap<String, String> = env.into_iter().collect();
        assert_eq!(env.get("MY_ENV"), Some(&"1".to_string()));
        if crate::gpu::system::detect_gpu_devices().is_empty() {
            assert!(
                !env.contains_key("CUDA_VISIBLE_DEVICES"),
                "no local GPU → no isolation env var"
            );
        } else {
            assert!(
                env.contains_key("CUDA_VISIBLE_DEVICES"),
                "CUDA host with device set must gain the isolation env var"
            );
        }

        // CPU variant on a device: never a vendor env var.
        let mut req = make_req("gpu-env-b", "/bin/true", &[], 0);
        req.gpu_device = "GPU0".to_string();
        req.gpu_variant = "cpu".to_string();
        let (_args, env) = lc.resolve_launch(&req).await.unwrap();
        for key in env.iter().map(|(k, _)| k.as_str()) {
            assert!(
                !key.to_uppercase().contains("VISIBLE"),
                "cpu variant must not gain a vendor env var, got {key}"
            );
        }

        // Unknown variant folder: warn path, still resolves.
        let mut req = make_req("gpu-env-c", "/bin/true", &[], 0);
        req.gpu_device = "GPU0".to_string();
        req.gpu_variant = "not-a-variant".to_string();
        assert!(lc.resolve_launch(&req).await.is_ok());
    }

    /// `kill_all` must terminate EVERY loaded backend's process group —
    /// including grandchildren of the spawned leader (the orphan case:
    /// tamad dies while backends run, plan-191 follow-up A).
    #[tokio::test]
    #[cfg(unix)]
    async fn test_kill_all_kills_every_backend_group() {
        use crate::process::{is_process_group_alive, wait_group_dead};

        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(Arc::clone(&table), state.store.clone(), state);

        // Child (sh) + grandchild (sleep) — the grandchild keeps running
        // even after sh exits; only a *group* kill removes it.
        let req = make_req("ghost-1", "/bin/sh", &["-c", "sleep 120"], 0);
        lc.load(&req).await.expect("load ghost-1");
        let req = make_req("ghost-2", "/bin/sh", &["-c", "sleep 120"], 0);
        lc.load(&req).await.expect("load ghost-2");

        let entries = table.list().await;
        assert_eq!(entries.len(), 2, "two backends loaded before kill_all");
        let pids: Vec<u32> = entries.iter().map(|e| e.pid).collect();
        for p in &pids {
            assert!(
                is_process_group_alive(*p),
                "group leader {p} alive before kill"
            );
        }

        lc.kill_all().await.expect("kill_all succeeds");

        // Every group (leader + grandchildren in the group) must be gone.
        for p in &pids {
            wait_group_dead(*p).await.expect("group reaped");
            assert!(
                !is_process_group_alive(*p),
                "group {p} must be dead after kill_all"
            );
        }
        assert!(table.list().await.is_empty(), "table cleared by kill_all");
    }

    // ── plan-193 T2: respawn + restart budget + boot sweep ────────────

    /// Rewrite the persisted `max_restarts` of a key's manifest and return
    /// a FRESH store that sees it (the probed store in the lifecycle is
    /// the one built here — `state.store` holds the pre-rewrite view).
    fn rekey_stored_max(
        state: &TamadState,
        key: &str,
        max: u32,
    ) -> Arc<crate::state::store::Store> {
        use crate::state::store::StoredProcess;
        let path = state.data_dir.join("state").join(format!("{key}.json"));
        let mut sp: StoredProcess =
            serde_json::from_slice(&std::fs::read(&path).expect("manifest read"))
                .expect("manifest parse");
        sp.max_restarts = max;
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&sp).expect("manifest write"),
        )
        .expect("manifest rewrite");
        Arc::new(crate::state::store::Store::new(&state.data_dir).expect("store reload"))
    }

    /// (window trim) four seeded respawns + 301 s elapse → the window
    /// trims to the single fresh replay: the trimmed result keeps the
    /// survivor, not the raw history.
    #[tokio::test]
    async fn test_reap_window_trims_after_300s() {
        let (_state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let t: i64 = 1_000_000;
        let window_ms = RESTART_WINDOW_SECS as i64 * 1000;
        table
            .insert(ProcessEntry {
                model_name: "trim".to_string(),
                provider_name: "llama_cpp".to_string(),
                pid: std::process::id(),
                endpoint_url: String::new(),
                status: "ready".to_string(),
                started_at: Instant::now(),
                spec: LoadModelRequest::default(),
                restart_count: 4,
                window_starts: vec![t, t, t, t],
                user_flagged: false,
            })
            .await;
        // +301 s past the seeded stamps → all four have left the 300 s
        // window; only the death the trim itself records survives.
        let count = table
            .record_restart_window("trim", t + window_ms + 1)
            .await
            .expect("entry present");
        assert_eq!(count, 1, "the four older replays must trim off");
        let e = table.get("trim").await.expect("entry still here");
        assert_eq!(e.restart_count, 1);
        assert_eq!(e.window_starts.len(), 1);

        // A death that happens … just after? stays inside the window
        // against the fresh stamp — still one.
        let count2 = table
            .record_restart_window("trim", t + window_ms + 2)
            .await
            .expect("entry present");
        assert_eq!(count2, 2, "in-window replays keep accumulating");
    }

    /// (budget trip) with `max_restarts = 2`, two failing respawns
    /// exhaust the budget: the row lands in `budget_exhausted`, the
    /// operator mark persists on the disk store, and the
    /// auto-respawned there (no third attempt ever comes up).
    #[tokio::test]
    async fn test_reap_budget_trip_flags_and_refuses() {
        let _s = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let req = make_req("budget-shot", "sh", &["-c", "sleep 0.3; exit 2"], 0);
        state
            .store
            .insert(&req.model_name, &req, true)
            .expect("seed desired row");
        let store = rekey_stored_max(&state, &req.model_name, 2);
        let lc = Arc::new(TamadLifecycle::new(
            Arc::clone(&table),
            store,
            Arc::clone(&state),
        ));
        let _supervisor = TamadLifecycle::start_respawn_supervisor(&lc);

        lc.load(&req).await.expect("initial load");
        let by = tokio::time::Instant::now() + Duration::from_secs(15);
        for _ in 0..100 {
            if let Some(e) = table.get(&req.model_name).await {
                if e.status == status::BUDGET_EXHAUSTED {
                    break;
                }
            }
            assert!(tokio::time::Instant::now() < by, "no budget trip in time");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let e = table
            .get(&req.model_name)
            .await
            .expect("row kept after the trip");
        assert_eq!(
            e.status,
            status::BUDGET_EXHAUSTED,
            "trip on the row (count={} window={:?} pid={})",
            e.restart_count,
            e.window_starts.len(),
            e.pid
        );
        assert_eq!(e.restart_count, 2, "both respawns rented out the window");
        assert_eq!(e.window_starts.len(), 2);
        assert!(e.user_flagged, "row mirror carried the mark at trip");
        // The flag went to disk (a fresh store reload sees it).
        let reloaded = crate::state::store::Store::new(&state.data_dir).expect("fresh store");
        assert!(
            reloaded
                .get(&req.model_name)
                .expect("manifest present")
                .user_flagged
        );
        // The trip's tally went to disk WITH the mark — one write:
        // the boot sweep's at-cap refusal (round-2 P1) reads exactly
        // this pairing, so a count without a flag (or vice versa)
        // can never be half-persisted.
        assert_eq!(
            reloaded
                .get(&req.model_name)
                .expect("manifest present")
                .persisted_restart_count,
            2,
            "the at-cap tally persisted with the mark (one write)"
        );
        // And it STOPS: row + dead process stay put a little while.
        let dead_pid = e.pid;
        tokio::time::sleep(Duration::from_millis(600)).await;
        let after = table.get(&req.model_name).await.expect("row still here");
        assert_eq!(after.status, status::BUDGET_EXHAUSTED);
        assert_eq!(after.pid, dead_pid, "no third respawn came");
        assert!(!crate::process::is_process_alive(dead_pid));
    }

    /// (success reset) with `max_restarts = 3`, the crasher fails
    /// twice (count 2), the next respawn then comes from a healthy
    /// spec — the stored row is rewritten mid-flight — and the ready
    /// row resets the counter to zero without more touching the
    /// (`user_flagged` is never auto-cleared, but it was not set).
    #[tokio::test]
    async fn test_reap_success_resets_counter() {
        // Local health sniffer (same pattern as `test_load_with_health_check`):
        // tokio TCP listener answering one 200 per connection.
        let sniffer = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let health_port = sniffer.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = sniffer.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 512];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                        .await;
                });
            }
        });
        let health_url = format!("http://127.0.0.1:{health_port}/health");

        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let crashing = make_req("reset-me", "sh", &["-c", "sleep 0.5; exit 2"], 0);
        state
            .store
            .insert(&crashing.model_name, &crashing, true)
            .expect("seed desired row");
        let store = rekey_stored_max(&state, &crashing.model_name, 3);
        let lc = Arc::new(TamadLifecycle::new(
            Arc::clone(&table),
            store,
            Arc::clone(&state),
        ));
        let _supervisor = TamadLifecycle::start_respawn_supervisor(&lc);

        lc.load(&crashing).await.expect("initial (crashing) load");
        let initial_pid = table
            .get(&crashing.model_name)
            .await
            .expect("row present")
            .pid;
        // At the moment the first respawn shows up (the table reports a
        // different pid), swap the stored spec to a healthy sleeper —
        // still safely before that one can die (0.5 s window deadline,
        // we detect within one 50 ms poll and the next crash reads the
        // already-rewritten row).
        let by = tokio::time::Instant::now() + Duration::from_secs(15);
        loop {
            let e = table.get(&crashing.model_name).await.expect("row present");
            if e.pid != initial_pid {
                // A healthy replacement whose health gate PASSES (local
                // sniffer) → verified ready → lever reset. The swap
                // goes into the store the REAPER sees (not the test's
                // original pre-load view).
                let mut healthy = make_req("reset-me", "sh", &["-c", "sleep 30"], 0);
                healthy.health_url = health_url.clone();
                healthy.health_timeout_ms = 15_000;
                lc.store
                    .insert(&crashing.model_name, &healthy, true)
                    .expect("mid-flight swap");
                break;
            }
            assert!(
                tokio::time::Instant::now() < by,
                "first respawn did not show"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // The healthy respawn then lands a `ready` row that resets the
        // bookkeeping (keeps `user_flagged`, which was false).
        let by = tokio::time::Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(e) = table.get(&crashing.model_name).await {
                if e.status == "ready" && e.restart_count == 0 {
                    break;
                }
            }
            assert!(tokio::time::Instant::now() < by, "no reset in time");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let e = table.get(&crashing.model_name).await.expect("ready row");
        assert_eq!(e.restart_count, 0, "success zeroes the counter");
        assert!(e.window_starts.is_empty(), "success clears the window");
        assert!(!e.user_flagged, "flag untouched (it was never set)");
        assert!(crate::process::is_process_alive(e.pid), "respawn is live");
        // The verified-healthy respawn also zeroed the ON-DISK tally
        // (round-2 P1): without it, the manifest would still present
        // as at-cap and the next boot sweep would refuse to replay a
        // perfectly healthy key.
        assert_eq!(
            lc.store
                .get(&crashing.model_name)
                .unwrap()
                .persisted_restart_count,
            0,
            "the verified-ready reset zeroed the persisted tally"
        );
        lc.unload(&crashing.model_name).await.expect("cleanup");
    }

    /// (sweep) two desired, unflagged models replay (in parallel), a
    /// flagged one is skipped, and a model that is already alive
    /// under the same key loses to its row — exactly two new spawns
    /// in total.
    #[tokio::test]
    async fn test_boot_sweep_parallel_and_skips() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );
        let a = make_req("sw-a", "sh", &["-c", "sleep 40"], 0);
        let b = make_req("sw-b", "sh", &["-c", "sleep 40"], 0);
        let c = make_req("sw-c", "sh", &["-c", "sleep 40"], 0);
        let d = make_req("sw-d", "sh", &["-c", "sleep 40"], 0);
        state.store.insert(&a.model_name, &a, true).expect("a");
        state.store.insert(&b.model_name, &b, true).expect("b");
        state.store.insert(&c.model_name, &c, true).expect("c");
        state.store.insert(&d.model_name, &d, true).expect("d");
        state
            .store
            .set_user_flagged(&c.model_name, true)
            .expect("flag c");
        // `d` already has a row alive under it (test-own pid as its
        // standing) — the row must win over the file.
        table
            .insert(ProcessEntry {
                model_name: d.model_name.clone(),
                provider_name: "llama_cpp".to_string(),
                pid: std::process::id(),
                endpoint_url: String::new(),
                status: "ready".to_string(),
                started_at: Instant::now(),
                spec: d.clone(),
                restart_count: 0,
                window_starts: Vec::new(),
                user_flagged: false,
            })
            .await;

        lc.replay_desired(true).await;

        let ea = table.get("sw-a").await.expect("a replayed");
        let eb = table.get("sw-b").await.expect("b replayed");
        assert!(
            crate::process::is_process_alive(ea.pid) && crate::process::is_process_alive(eb.pid),
            "both spawns are alive at the same time (2 in-flight)"
        );
        assert!(table.get("sw-c").await.is_none(), "flagged skips");
        let ed = table.get("sw-d").await.expect("row stands");
        assert_eq!(ed.pid, std::process::id(), "row wins; no second spawn");
        assert_eq!(
            table.list().await.len(),
            3,
            "a + b new; c skipped; d stands"
        );
        assert_eq!(lc.store.list().len(), 4, "store untouched by the sweep");
        lc.unload("sw-a").await.expect("cleanup a");
        lc.unload("sw-b").await.expect("cleanup b");
    }

    /// (round-2 P1, pre-policy durability) A trip-persisted row whose
    /// MARK never reached disk (its tally write succeeded, its mark
    /// write did not — the C3-warned-not-on-disk form): the boot
    /// sweep must NOT replay it (replay re-arms the crash loop),
    /// must leave the manifest the way it found it (tally at cap,
    /// flag absent), and must hand the operator the recovery
    /// sentence the trip trip would have — the sweep LOGS; the
    /// operator FIXES.
    #[tokio::test]
    async fn test_boot_sweep_refuses_at_cap_persisted_unflagged() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let m = make_req("cap-unflagged", "sh", &["-c", "sleep 40"], 0);
        state
            .store
            .insert(&m.model_name, &m, true)
            .expect("seed desired row");
        let _rekeyed = rekey_stored_max(&state, &m.model_name, 2);
        // Witness the trip-time ENOSPC (the C3 warning that the mark
        // could not reach disk): tally at cap, mark absent.
        let path = state
            .data_dir
            .join("state")
            .join(format!("{}.json", m.model_name));
        let mut sp: crate::state::store::StoredProcess =
            serde_json::from_slice(&std::fs::read(&path).expect("manifest read"))
                .expect("manifest parse");
        assert_eq!(sp.max_restarts, 2, "the re-keyed budget must be in place");
        sp.persisted_restart_count = 2; // == max_restarts
        sp.user_flagged = false; // the flag did not reach disk
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&sp).expect("manifest serialize"),
        )
        .expect("manifest write");
        // A dead daemon means its in-memory store died with it — the store at boot
        // is re-read FRESH from the disk the crash left (the partial write
        // survives; the in-memory mirror of whatever wrote it does not).
        let store = Arc::new(
            crate::state::store::Store::new(&state.data_dir)
                .expect("boot-fresh store reads the crashed write"),
        );
        let lc = TamadLifecycle::new(Arc::clone(&table), store, Arc::clone(&state));

        lc.replay_desired(true).await;

        assert!(
            table.list().await.is_empty(),
            "a cap-tally + unflagged row must not be replayed — replay = the crash loop re-armed"
        );
        // The manifest is exactly capped as found: the sweep REFUSED
        // the replay; it did not rewrite.
        let got = lc
            .store
            .get(&m.model_name)
            .expect("manifest present in store");
        assert_eq!(got.persisted_restart_count, 2, "the tally stays at cap");
        assert!(!got.user_flagged, "the flag stays un-persisted");
        // The operator line carries the trip's recovery sentence.
        let note = at_cap_skip_note(&m.model_name, 2, 2);
        assert!(
            note.contains("will NOT replay"),
            "the refusal is explicit: {note}"
        );
        assert!(
            note.contains("recovery = `tama admin unload ")
                && note.contains("then `load` (clean re-arm)"),
            "the recovery sentence = the trip's: {note}"
        );
    }

    /// (no-op) with replay disabled, the sweep does nothing at all:
    /// no spawns, no file changes.
    #[tokio::test]
    async fn test_boot_sweep_disabled_is_noop() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );
        let m = make_req("no-sweep", "sh", &["-c", "sleep 40"], 0);
        state
            .store
            .insert(&m.model_name, &m, true)
            .expect("seed desired row");
        lc.replay_desired(false).await;
        assert!(table.list().await.is_empty(), "no spawns with replay off");
        assert_eq!(lc.store.list().len(), 1, "manifest files stay put");
        assert!(
            lc.store.get(&m.model_name).expect("still there").desired,
            "desired is preserved"
        );
    }

    /// (double-issue guard) a model that stands with an alive row
    /// during the sweep under the same key stays exactly one
    /// process — the row wins over the file (the table key cannot
    /// admit a second entry in the same key).
    #[tokio::test]
    async fn test_boot_sweep_double_issue_guard() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );
        let m = make_req("double-key", "sh", &["-c", "sleep 40"], 0);
        state
            .store
            .insert(&m.model_name, &m, true)
            .expect("seed desired row");
        let live = ProcessEntry {
            model_name: m.model_name.clone(),
            provider_name: "llama_cpp".to_string(),
            pid: std::process::id(),
            endpoint_url: String::new(),
            status: "ready".to_string(),
            started_at: Instant::now(),
            spec: m.clone(),
            restart_count: 0,
            window_starts: Vec::new(),
            user_flagged: false,
        };
        let live_pid = live.pid;
        table.insert(live).await;

        lc.replay_desired(true).await;

        let after = table.get(&m.model_name).await.expect("row kept");
        assert_eq!(after.pid, live_pid, "row untouched — no second spawn");
        assert!(crate::process::is_process_alive(after.pid));
        assert_eq!(table.list().await.len(), 1);
    }

    /// Back-compat regression (plan-193 T3): an OLD wire frame — only the
    /// six legacy fields, none of 7/8/9 — decodes with the NEW prost to
    /// the new fields at their defaults (zeros). Encoding a struct whose
    /// 7/8/9 are all zero skips them on the proto, so the emitted bytes
    /// are exactly the old shape; the new decoder must still yield a
    /// valid `ProcessInfo` with `desired=false`, `restart_count=0` and
    /// `max_restarts=0`.
    #[test]
    fn old_frame_decodes() {
        let old = ProcessInfo {
            model_name: "m".to_string(),
            provider_name: "p".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://x".to_string(),
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
        let bytes = old.encode_to_vec();
        let decoded = ProcessInfo::decode(::prost::bytes::Bytes::from(bytes)).unwrap();
        assert!(!decoded.desired, "old frame decodes desired=false");
        assert_eq!(
            decoded.restart_count, 0,
            "old frame decodes restart_count=0"
        );
        assert_eq!(decoded.max_restarts, 0, "old frame decodes max_restarts=0");
        assert_eq!(decoded.status, "ready", "the six legacy fields survive");
    }

    /// The T3 builder folds the store row: an inserted desired row makes
    /// the wire info report `desired=true` and the default restart budget.
    #[tokio::test]
    async fn test_to_process_info_reads_store() {
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );
        let m = make_req("wire", "sh", &["-c", "sleep 30"], 0);
        state
            .store
            .insert(&m.model_name, &m, true)
            .expect("seed desired row");
        let entry = ProcessEntry {
            model_name: m.model_name.clone(),
            provider_name: m.provider_name.clone(),
            pid: std::process::id(),
            endpoint_url: String::new(),
            status: "ready".to_string(),
            started_at: Instant::now(),
            spec: m.clone(),
            restart_count: 0,
            window_starts: Vec::new(),
            user_flagged: false,
        };
        let info = to_process_info(&entry, lc.store.get(&entry.model_name).as_ref());
        assert!(info.desired, "store row makes the wire info desired");
        assert_eq!(info.restart_count, 0);
        assert_eq!(info.max_restarts, DEFAULT_MAX_RESTARTS);
    }

    // ---- plan-194 Task 4: reconciliation sweep for orphaned `starting`
    // rows ---------------------------------------------------------------

    /// A TCP sniffer that answers every connection with a bare
    /// HTTP/1.1 200 (the same pattern as `test_load_with_health_check`).
    /// Returns the bound host port.
    async fn start_sniffer() -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 512];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                        .await;
                });
            }
        });
        port
    }

    /// Spawn a real child in its own process group and return a guard
    /// holding its pid — the reconciler's native teardown has something
    /// genuinely killable, and the guard kills the WHOLE group (SIGKILL)
    /// deterministically even when a test assertion fails first.
    fn spawn_group_child() -> GroupChildGuard {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.process_group(0);
        cmd.arg("-c").arg("sleep 30");
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut child = cmd.spawn().expect("spawn sleep child");
        let pid = child.id().expect("child pid");
        tokio::spawn(async move {
            let _ = child.wait().await;
        });
        GroupChildGuard { pid }
    }

    /// Drop guard around [`Self::spawn_group_child`]: SIGKILLs the child's
    /// process group on scope exit so no `sleep 30` survives a failed
    /// assertion or an early return.
    struct GroupChildGuard {
        pid: u32,
    }

    impl GroupChildGuard {
        fn pid(&self) -> u32 {
            self.pid
        }
    }

    impl Drop for GroupChildGuard {
        fn drop(&mut self) {
            // SAFETY: negative PID targets the process group led by `pid`,
            // which this test itself created via process_group(0). SIGKILL
            // cannot access invalid memory; ESRCH (already dead) is ignored.
            unsafe {
                libc::kill(-(self.pid as libc::pid_t), libc::SIGKILL);
            }
        }
    }

    /// Insert a `starting` row whose age is `age` (via `started_at`
    /// backdating — a pub `Instant` field).
    async fn seed_starting_row(
        table: &Arc<ProcessTable>,
        spec: &LoadModelRequest,
        pid: u32,
        age: Duration,
    ) {
        let started_at = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
        table
            .insert(ProcessEntry {
                model_name: spec.model_name.clone(),
                provider_name: spec.provider_name.clone(),
                pid,
                endpoint_url: String::new(),
                status: status::STARTING.to_string(),
                started_at,
                spec: spec.clone(),
                restart_count: 0,
                window_starts: Vec::new(),
                user_flagged: false,
            })
            .await;
    }

    /// An orphaned `starting` row whose backend answers its health URL is
    /// adopted as a verified ready within one pass, with the verified-success
    /// bookkeeping: the persisted at-cap tally is zeroed on disk.
    #[tokio::test]
    async fn test_reconciler_adopts_healthy_orphan() {
        let port = start_sniffer().await;
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        let mut req = make_req("orphan", "sh", &["-c", "sleep 30"], 0);
        req.health_url = format!("http://127.0.0.1:{port}/health");
        req.health_timeout_ms = 10_000;

        // Seed the persisted tally non-zero so the verified-ready reset is
        // observable (the adopt path must run the same bookkeeping as the
        // detached settle path).
        state
            .store
            .insert(&req.model_name, &req, true)
            .expect("seed desired row");
        state
            .store
            .set_tripped(&req.model_name, 5)
            .expect("seed at-cap tally");

        let child = spawn_group_child();
        let pid = child.pid();
        seed_starting_row(&table, &req, pid, Duration::from_secs(11)).await;

        lc.reconcile_once_with(Duration::from_secs(10), Duration::from_secs(120))
            .await;

        let entry = poll_until_status(&table, "orphan", status::READY, Duration::from_secs(5))
            .await
            .expect("reconciler must adopt the healthy orphan as ready");
        assert_eq!(entry.pid, pid, "adopted row keeps its process");
        let stored = state
            .store
            .get(&req.model_name)
            .expect("store row survives");
        assert_eq!(
            stored.persisted_restart_count, 0,
            "verified-ready adoption zeroes the persisted at-cap tally"
        );

        lc.unload("orphan").await.ok();
    }

    /// A native-path `starting` row far past its health deadline (age >
    /// max(2 × health_timeout_ms, min_deadline)) is torn down — process
    /// group killed — and recorded `failed`. The deadline check runs
    /// BEFORE the probe, so the corpse dies even with a refusing port.
    #[tokio::test]
    async fn test_reconciler_fails_deadline_breach() {
        use crate::process::is_process_group_alive;
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        // Port 1 refuses: nothing listens there (same shape as
        // `test_load_health_timeout`). Deadline floor 100ms vs
        // 2 × 500ms timeout → breach past 1s; the row is aged well past it.
        let mut req = make_req("corpse", "sh", &["-c", "sleep 30"], 0);
        req.health_url = "http://127.0.0.1:1/health".to_string();
        req.health_timeout_ms = 500;

        let child = spawn_group_child();
        let pid = child.pid();
        seed_starting_row(&table, &req, pid, Duration::from_secs(5)).await;

        // Small injectable knobs: grace already passed, deadline floor tiny.
        lc.reconcile_once_with(Duration::from_millis(50), Duration::from_millis(100))
            .await;

        let entry = poll_until_status(&table, "corpse", status::FAILED, Duration::from_secs(5))
            .await
            .expect("deadline-breach row must be recorded failed");
        assert_eq!(entry.pid, pid, "failed row keeps its process identity");

        // Native teardown must have killed the process group.
        for _ in 0..40 {
            if !is_process_group_alive(pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        assert!(
            !is_process_group_alive(pid),
            "process group must be dead after the reconciler tore down the breach"
        );
    }

    /// Rows the sweep must never touch: a gate-less spec (`health_url`
    /// empty AND `health_timeout_ms == 0`) is skipped entirely even when
    /// old, and a gated row younger than the grace period stays untouched
    /// (an active detached gate owns it until then).
    #[tokio::test]
    async fn test_reconciler_skips_young_and_gainless() {
        use crate::process::is_process_group_alive;
        let port = start_sniffer().await;
        let (state, _dir) = test_state();
        let table = Arc::new(ProcessTable::default());
        let lc = TamadLifecycle::new(
            Arc::clone(&table),
            Arc::clone(&state.store),
            Arc::clone(&state),
        );

        // Gate-less spec, aged far beyond any grace: skip entirely.
        let gateless = make_req("gainless", "sh", &["-c", "sleep 30"], 0);
        let gateless_child = spawn_group_child();
        let gateless_pid = gateless_child.pid();
        seed_starting_row(&table, &gateless, gateless_pid, Duration::from_secs(60)).await;

        // Gated spec but fresh (< grace): an active detached gate owns it.
        let mut gated = make_req("young", "sh", &["-c", "sleep 30"], 0);
        gated.health_url = format!("http://127.0.0.1:{port}/health");
        gated.health_timeout_ms = 10_000;
        let gated_child = spawn_group_child();
        let gated_pid = gated_child.pid();
        seed_starting_row(&table, &gated, gated_pid, Duration::from_millis(0)).await;

        // Production-shaped knobs: 10s grace means both rows fall inside
        // protected territory this tick.
        lc.reconcile_once_with(Duration::from_secs(10), Duration::from_millis(100))
            .await;

        let g = table.get("gainless").await.expect("gateless row kept");
        assert_eq!(
            g.status,
            status::STARTING,
            "gate-less row must be skipped entirely"
        );
        let y = table.get("young").await.expect("young row kept");
        assert_eq!(
            y.status,
            status::STARTING,
            "row inside the grace period must not be judged yet"
        );
        assert!(
            is_process_group_alive(gateless_pid),
            "skipped gate-less process must stay alive"
        );
        assert!(
            is_process_group_alive(gated_pid),
            "skipped young process must stay alive"
        );

        lc.unload("gainless").await.ok();
        lc.unload("young").await.ok();
    }
}
