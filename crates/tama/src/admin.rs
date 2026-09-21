//! Headless administration of the proxy side — `tama admin` (plan-193 T6).
//!
//! Boots the SAME prep chain as the server (bootstrap config → Postgres pool
//! → migrations → seed → DB config → `ProxyState` + tamad pool) but NOT
//! the SSR `Web` state, and runs the four verbs against the row-sourced
//! lifecycle. No new RPC: the three row verbs call the existing handlers
//! (T2/T4/T5). It is a CLI, not an SSR thing — the dispatch happens
//! before any `ssr` feature gate, in `main()`.
//!
//! | Verb                    | Path                                                          |
//! | ----------------------- | ------------------------------------------------------------- |
//! | `status`                | all live rows, one line of JSON (T4 `Rows::all()`)             |
//! | `load <key>`             | `ensure_model_loaded` (idempotent: an already-alive row ⇒ no   |
//! |                         | second `LoadModel`) — 503-mapped clue ⇒ exit `13`               |
//! | `unload <key>`            | the existing `ProxyState::unload_model` path                    |
//! | `logs <key>`              | a tail of `TamadHandle::logs` (the pool gRPC; container-       |
//! |                         | engine only — the wire tails the `tama-<key>` container)         |
//!
//! Exit codes (`0` = success): `2` not-found (the key has no row —
//! offline, never loaded, or not a model at all; odd keys such as `..`
//! or `/`-leading keys are not-found, NOT treated as mis-use); `13`
//! budget-exhausted (the CLI literal matching the wire word
//! `budget_exhausted` — there is no numeric "code" on the wire);
//! `1` any other error.
//!
//! `status` / `models_loaded` semantics (asserted by the T6 tests,
//! T4/T5c enforce): the wire name survived (gauge + JSON contract) and
//! is now the live set's `Rows::ready_count()` — the current
//! number of ready rows. There is no monotonically-increasing counter
//! anywhere; a reload is one, not two.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use std::sync::Arc;

use tama_core::config::Config;
use tama_core::proxy::lifecycle::{ensure_model_loaded, BudgetExhausted};
use tama_core::proxy::{live_rows, ProxyState};
use tama_core::tamad::{pool::TamadHandle, LogsRequest};

type FilterHandle =
    tracing_subscriber::reload::Handle<tracing_subscriber::EnvFilter, tracing_subscriber::Registry>;

/// An admin verb failure is an explicit CLI exit-code contract
/// (plan-193 T6): `2` not-found / `13` budget-exhausted / `1`
/// anything else (`0` is success — not represented here).
#[derive(Debug)]
pub struct AdminError {
    message: String,
    /// The exit code for `std::process::exit` (0 / 2 / 13, and 1
    /// otherwise).
    pub exit_code: i32,
}

impl AdminError {
    fn message(message: impl Into<String>, exit_code: i32) -> Self {
        Self {
            message: message.into(),
            exit_code,
        }
    }

    /// The key was not found (no live row, unknown model, offline host)
    /// → `2`.
    pub fn not_found(key: &str) -> Self {
        Self::message(
            format!("model '{key}' not found (no live row or process)"),
            2,
        )
    }

    /// The model's restart budget is exhausted (a `budget_exhausted`
    /// row) → `13` — the literal that corresponds to the 503 +
    /// `retry-after: 60` clue for loading.
    pub fn budget_exhausted(key: &str) -> Self {
        Self::message(
            format!("model '{key}' exhausted its restart budget; retry in ~60 seconds"),
            13,
        )
    }

    /// Any error other than not-found/budget-exhausted → `1`.
    pub fn other(message: impl Into<String>) -> Self {
        Self::message(message, 1)
    }
}

impl std::fmt::Display for AdminError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AdminError {}

/// A weird key (such as `..` or one that starts with `/`) is not a model
/// key — it is treated as not-found (surfaced at `2`), not as a flag
/// for CLI misuse.
fn key_is_model_key(key: &str) -> bool {
    !key.is_empty() && key != ".." && !key.starts_with('/')
}

// plan-193: `NONE` of these fall outside the plan's
// magic-constant allow-list (`RESTART_WINDOW_SECS` / `DEFAULT_MAX_RESTARTS` /
// `RETRY_AFTER` / `LIVE_FRAME_MAX_AGE`) — the two marked CLI-bound
// constants below are recorded intentional exceptions
// (each tagged in its doc) for next-plan gate adjudication.
/// The pool gRPC host-mapping frame-freshness window (5 s) — the same
/// wire staleness contract as `live`'s row handling: we only trust a
/// host's newest frame for "who hosts this key."
const LOGS_FRESH_FRAME: Duration = Duration::from_secs(5);

/// CLI tail bound (plan-193 magic-constant EXCEPTION — not in the
/// allow-list). Bounds the whole one-shot `logs` stream read, not a
/// per-message timeout; no write-side mirror on the tamad (the host
/// streams a finite tail once and closes). Carry to docs/plans for
/// the next plan to adjudicate.
const LOGS_STREAM_CAP: Duration = Duration::from_secs(30);

/// CLI tail cap (plan-193 magic-constant EXCEPTION — not in the
/// allow-list). Value mirrors the host-side
/// `tail_container_logs(&container, 200)` in `crates/tamad/src/server.rs`
/// (signature `tail_container_logs(container_name: &str, max_lines: usize)`);
/// keep them in sync if one moves; verified at T6.
const LOGS_LINE_CAP: usize = 200;

/// The bootstrap chain the admin verb shares with all the
/// servers, minus the SSR web state and the tracing file writer (the
/// admin persists console-level presets and applies the DB-derived
/// `log_level`).
async fn bootstrap_proxy_state(filter_handle: &FilterHandle) -> Result<Arc<ProxyState>> {
    let config_dir = Config::config_dir().context("Failed to determine config directory")?;
    let db_bootstrap =
        tama_core::config::database::load_bootstrap(&config_dir)?.ok_or_else(|| {
            anyhow!(
                "v3 requires a [database] section in config.toml (host/port/name/user/password). \
                 The app config now lives in Postgres — run `tama migrate` to copy your v2 data."
            )
        })?;
    db_bootstrap
        .resolved_password()
        .with_context(|| "failed to resolve Postgres password")?;

    let pool = tama_core::db::pool::create_pool(&db_bootstrap)
        .await
        .context("creating Postgres pool")?;
    // A CLI should not retry forever like a daemon: give up after
    // 10 attempts (a common case — a slow foreground host) and
    // emit an exit message.
    tama_core::db::pool::connect_with_retry_capped(&pool, Duration::from_secs(1), Some(10))
        .await
        .context("connecting to Postgres")?;
    tama_core::db::postgres::run_migrations(&pool)
        .await
        .context("applying Postgres migrations")?;
    tama_core::db::queries::seed_defaults(&pool)
        .await
        .context("seeding default app config")?;

    let config = Config::load_from_pool(&pool)
        .await
        .context("loading app config from Postgres")?;
    // Apply the DB-derived log_level + log_directives to the live console
    // filter (the JSON file writer stays in discarding mode; admin does
    // not log to a file).
    let log_directives = config.general.log_directives.clone().unwrap_or_default();
    match tama_core::logstore::filter::build_log_filter(&config.general.log_level, &log_directives)
    {
        Ok(new_filter) => {
            if let Err(e) = filter_handle.modify(|f| *f = new_filter) {
                tracing::warn!("failed to apply the DB log level: {e}");
            }
        }
        Err(e) => tracing::warn!("failed to build the DB log level filter: {e}"),
    }
    crate::setup_hf_token(&config);

    let db_pool: Arc<sqlx::PgPool> = Arc::new(pool);
    let state = Arc::new(ProxyState::new(config, Some(config_dir), db_pool));

    // Load the registered tamads into the pool (needed for the verbs; not
    // fatal — a verb that logs will surface a pool with not-enough-as-data).
    if let Err(e) = state.tamad_pool().load_all().await {
        tracing::error!("Failed to load tamad pool at admin start: {e}");
    }
    Ok(state)
}

/// `status` — every live-model row, at once, one line of JSON (T4
/// `Rows::all()`).
async fn cmd_status(state: &Arc<ProxyState>) -> std::result::Result<(), AdminError> {
    let rows = live_rows(state.tamad_pool().as_ref()).await;
    let values = rows
        .all()
        .iter()
        .map(|row| {
            serde_json::to_value(row)
                .map_err(|e| AdminError::other(format!("serializing row failed: {e}")))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let line = serde_json::to_string(&values)
        .map_err(|e| AdminError::other(format!("serializing rows failed: {e}")))?;
    println!("{line}");
    Ok(())
}

/// `load <config_key>` — the idempotent `ensure_model_loaded` path (a
/// headless CLI runs in-process the same API path the proxy does:
/// alias resolution → the 503 budget check → the already-alive-row fast
/// path (no second `LoadModel`) → the wire `LoadModel`).
///
/// **Key contract (uniform across all verbs):** a config key or alias;
/// resolution happens at the verb entry, and the not-found determination
/// uses the alias-resolved key (the exit-code mapping is unchanged).
async fn cmd_load(
    state: &Arc<ProxyState>,
    config_key: &str,
) -> std::result::Result<(), AdminError> {
    if !key_is_model_key(config_key) {
        return Err(AdminError::not_found(config_key));
    }
    // not-found determination: the key (after alias resolution) must
    // be a known model — a key that is not a model is `2` rather
    // than an ambiguous load failure.
    let resolved = state.resolve_alias(config_key).await;
    if !state.has_model_config(&resolved).await {
        return Err(AdminError::not_found(config_key));
    }
    // A load failure is surfaced as-is (no HTTP response here; the
    // CLI maps it to its exit code).
    let on_load_error = |name: &str, e: anyhow::Error| -> Result<String> {
        Err(anyhow!("model '{name}' failed to load: {e}"))
    };
    match ensure_model_loaded(state, config_key, on_load_error).await {
        Ok(backend) => {
            // The loaded backend's name — readiness is up to the wire to
            // declare (the row shows it in `status`).
            println!("{backend}");
            Ok(())
        }
        Err(e) if e.is::<BudgetExhausted>() => Err(AdminError::budget_exhausted(&resolved)),
        Err(e) => Err(AdminError::other(format!(
            "loading '{config_key}' failed: {e}"
        ))),
    }
}

/// `unload <config_key>` — the existing unload path (no row ⇒
/// not-found; a status that cannot be terminated is a normal error).
///
/// **Key contract (uniform across all verbs):** a config key or alias;
/// resolution happens at the verb entry, and the not-found determination
/// uses the alias-resolved key (the exit-code mapping is unchanged).
async fn cmd_unload(
    state: &Arc<ProxyState>,
    config_key: &str,
) -> std::result::Result<(), AdminError> {
    if !key_is_model_key(config_key) {
        return Err(AdminError::not_found(config_key));
    }
    // Alias resolution at the verb entry (the uniform contract): the
    // row lookup, the unload, and the report all use the resolved key.
    let resolved = state.resolve_alias(config_key).await;
    if live_rows(state.tamad_pool().as_ref())
        .await
        .row(&resolved)
        .is_none()
    {
        return Err(AdminError::not_found(config_key));
    }
    state
        .unload_model(&resolved)
        .await
        .map_err(|e| AdminError::other(format!("unloading '{resolved}' failed: {e}")))?;
    println!("unloaded {resolved}");
    Ok(())
}

/// `logs <config_key>` — a tail on `TamadHandle::logs` (the pool gRPC
/// — reused; no new RPC). Container engine only: the wire `Logs`
/// tails the `tama-<key>` container, and a native-backend engine is
/// not in scope for this plan.
///
/// **Key contract (uniform across all verbs):** a config key or alias;
/// resolution happens at the verb entry, and the not-found determination
/// uses the alias-resolved key (the exit-code mapping is unchanged).
async fn cmd_logs(
    state: &Arc<ProxyState>,
    config_key: &str,
) -> std::result::Result<(), AdminError> {
    if !key_is_model_key(config_key) {
        return Err(AdminError::not_found(config_key));
    }
    // Alias resolution at the verb entry (the uniform contract): the
    // row check, the host-map iteration, and the log request all use
    // the resolved key.
    let resolved = state.resolve_alias(config_key).await;
    let pool = state.tamad_pool();
    // The row must exist (offline / not loaded / hook already
    // collected ⇒ not-found).
    if live_rows(pool.as_ref()).await.row(&resolved).is_none() {
        return Err(AdminError::not_found(config_key));
    }
    // Host mapping: pick the handle reporting the key in its newest
    // frame (rows dedupe by key across hosts; per-host attribution is
    // resolved from the pool here as well).
    let mut host: Option<Arc<TamadHandle>> = None;
    for handle in pool.list_handles().await {
        let Some(stats) = handle.latest_fresh(LOGS_FRESH_FRAME).await else {
            continue;
        };
        if stats.processes.iter().any(|p| p.model_name == resolved) {
            if host.is_none() {
                host = Some(handle);
            }
            continue;
        }
    }
    let Some(host) = host else {
        // The row was present, but between reads the frame went to the
        // old one: operationally not-found.
        return Err(AdminError::not_found(config_key));
    };
    let req = LogsRequest {
        provider_name: String::new(),
        model_name: resolved.clone(),
    };
    let mut stream = host
        .logs(&req)
        .await
        .map_err(|e| AdminError::other(format!("logs RPC for '{config_key}' failed: {e}")))?;
    let mut printed = 0usize;
    while printed < LOGS_LINE_CAP {
        let next = match tokio::time::timeout(LOGS_STREAM_CAP, stream.next()).await {
            Ok(next) => next,
            Err(_) => {
                return Err(AdminError::other(format!(
                    "logs stream for '{config_key}' hung (>{LOGS_STREAM_CAP:?} per message)"
                )))
            }
        };
        match next {
            Some(Ok(entry)) => {
                let line = entry.message.trim_end_matches('\n');
                if !line.is_empty() {
                    println!("{line}");
                    printed += 1;
                }
            }
            Some(Err(e)) => {
                return Err(AdminError::other(format!(
                    "logs stream for '{config_key}' failed: {e}"
                )))
            }
            None => break, // tail complete (the tamad closed the stream)
        }
    }
    Ok(())
}

/// Run a parsed admin verb (the public entry point `main()` targets).
pub async fn run(
    args: tama_web::cli::AdminArgs,
    filter_handle: FilterHandle,
) -> std::result::Result<(), AdminError> {
    let state = bootstrap_proxy_state(&filter_handle)
        .await
        .map_err(|e| AdminError::other(format!("proxy bootstrap failed: {e}")))?;
    match args.verb {
        tama_web::cli::AdminVerb::Status => cmd_status(&state).await,
        tama_web::cli::AdminVerb::Load { config_key } => cmd_load(&state, &config_key).await,
        tama_web::cli::AdminVerb::Unload { config_key } => cmd_unload(&state, &config_key).await,
        tama_web::cli::AdminVerb::Logs { config_key } => cmd_logs(&state, &config_key).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// not-found is `2`: no live row, or not a model at all.
    #[test]
    fn test_admin_not_found_exits_2() {
        let e = AdminError::not_found("nope");
        assert_eq!(e.exit_code, 2);
        assert!(e.to_string().contains("not found"));
    }

    /// Budget exhaustion is `13` — the CLI literal that matches the
    /// wire word `budget_exhausted` for the 503-clue mapping.
    #[test]
    fn test_admin_budget_exhausted_exits_13() {
        let e = AdminError::budget_exhausted("small-llama");
        assert_eq!(e.exit_code, 13);
        assert!(e.to_string().contains("exhausted"));
    }

    /// Everything else is the normal non-zero `1`.
    #[test]
    fn test_admin_other_errors_exit_1() {
        assert_eq!(AdminError::other("boom").exit_code, 1);
    }

    /// A weird key (`..` or `/`-leading) is *not* a model key: it
    /// surfaces as not-found, not as misuse/flag.
    #[test]
    fn test_odd_keys_are_not_model_keys() {
        assert!(!key_is_model_key(""));
        assert!(!key_is_model_key(".."));
        assert!(!key_is_model_key("/abs/path"));
        assert!(key_is_model_key("qwen3"));
        assert!(key_is_model_key("tts_kokoro"));
    }

    /// Admin path (plan-193 T5c/T6): `admin unload` on a `budget_exhausted`
    /// row must succeed — the row is live (the 503 reads it) so the
    /// not-found check at exit `2` does not fire, and `unload_model` — now
    /// admitted for `budget_exhausted` — returns `Ok`, which this verb maps
    /// to exit `0` (the pre-fix gate made it an `other` error → exit `1`).
    ///
    /// Driven at the `cmd_unload` level (the real exit-code contract):
    /// an in-memory `ProxyState` is enough, since `cmd_unload` itself does
    /// not bootstrap Postgres (only `bootstrap_proxy_state` does).
    #[tokio::test]
    async fn test_cmd_unload_budget_exhausted_exit_0() {
        let config = Config::default();
        let state = Arc::new(ProxyState::new(
            config,
            None,
            tama_test_support::test_dummy_pool(),
        ));

        // Seed a live `budget_exhausted` row for the key (the host stub
        // frame the T5c proxy test drives off: same shape, same key).
        let key = "model.gguf";
        let proc = tama_core::tamad::ProcessInfo {
            model_name: key.to_string(),
            provider_name: "llama-cpp".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://127.0.0.1:8080".to_string(),
            status: "budget_exhausted".to_string(),
            desired: true,
            restart_count: 0,
            max_restarts: 3,
            spec_accept_pct: None,
            spec_decoding_active: false,
            tps: None,
            prompt_tps: None,
            cache_hit_pct: None,
        };
        let stats = tama_core::tamad::pool::test_support::stats_full(1.5, vec![], vec![proc]);
        state
            .tamad_pool()
            .insert_raw_handle(
                key,
                Arc::new(
                    tama_core::tamad::pool::test_support::handle_with_latest(
                        std::time::Instant::now(),
                        stats,
                    )
                    .await,
                ),
            )
            .await;

        // The row must be live so cmd_unload's not-found check passes
        // (budget_exhausted is deliberately an eligible wire row).
        assert!(
            live_rows(state.tamad_pool().as_ref())
                .await
                .row(key)
                .is_some(),
            "precondition: the budget_exhausted row is live"
        );

        let result = cmd_unload(&state, key).await;
        assert!(
            result.is_ok(),
            "admin unload of a budget_exhausted row must map to exit 0 (Ok), got: {result:?}"
        );
    }
}
