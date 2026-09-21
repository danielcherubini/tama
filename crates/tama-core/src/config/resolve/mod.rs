use super::types::{BackendConfig, Config, ModelConfig};
use crate::models::repo_path;
use anyhow::Result;

/// Empty BackendConfig used as fallback when backend is not in TOML config
/// (e.g. after migration to provider_configs DB table cleared the [backends] section).
static EMPTY_BACKEND_CONFIG: BackendConfig = BackendConfig {
    path: None,
    version: None,
    gpu_variant: None,
};

impl Config {
    pub fn resolve_backend<'a>(
        &'a self,
        models: &'a std::collections::HashMap<String, ModelConfig>,
        name: &str,
    ) -> Result<(&'a ModelConfig, &'a BackendConfig)> {
        use anyhow::Context;

        // First, search by api_name to avoid config key precedence issues.
        // Comparison is case-insensitive (OpenAI API model IDs are
        // case-insensitive) while the stored api_name preserves the
        // original case used by the user.
        let mut api_name_matches: Vec<_> = models
            .values()
            .filter(|s| {
                s.api_name
                    .as_deref()
                    .is_some_and(|n| n.eq_ignore_ascii_case(name))
            })
            .collect();

        let server = if api_name_matches.len() == 1 {
            // Single api_name match - use it
            api_name_matches.pop().unwrap()
        } else if api_name_matches.len() > 1 {
            // Ambiguous api_name - error out
            anyhow::bail!(
                "Ambiguous api_name '{}': multiple models share this api_name",
                name
            );
        } else if let Some(server) = models.get(name) {
            // No api_name match, try direct config key lookup
            server
        } else {
            // Fall back to searching model field
            models
                .values()
                .find(|s| s.model.as_deref() == Some(name))
                .with_context(|| format!("Model '{}' not found in config", name))?
        };

        let backend = match self.backends.get(&server.backend) {
            Some(b) => b,
            None => {
                tracing::debug!(
                    "Backend '{}' not in TOML [backends] section; using DB-backed defaults",
                    server.backend
                );
                &EMPTY_BACKEND_CONFIG
            }
        };

        Ok((server, backend))
    }

    pub fn resolve_backends_for_model<'a>(
        &'a self,
        models: &'a std::collections::HashMap<String, ModelConfig>,
        model_name: &str,
    ) -> Vec<(String, &'a ModelConfig, &'a BackendConfig)> {
        let mut results = Vec::new();

        for (config_name, model_config) in models {
            if !model_config.enabled {
                continue;
            }
            // Use TOML backend config if present, otherwise empty default.
            // After migration to provider_configs table, the [backends] TOML
            // section may be empty — backend data (default_args, health URL)
            // now lives in the DB, not TOML.
            let backend = match self.backends.get(&model_config.backend) {
                Some(b) => b,
                None => {
                    tracing::debug!(
                        "Backend '{}' not in TOML [backends] section; using DB-backed defaults",
                        model_config.backend
                    );
                    &EMPTY_BACKEND_CONFIG
                }
            };

            // Match on api_name (highest priority), then config key, then model field.
            // Comparisons are case-insensitive for api_name and model (OpenAI API
            // model IDs are case-insensitive), but config_name is case-sensitive.
            let api_name_match = model_config
                .api_name
                .as_deref()
                .is_some_and(|n| n.eq_ignore_ascii_case(model_name));
            let model_match = model_config
                .model
                .as_deref()
                .is_some_and(|n| n.eq_ignore_ascii_case(model_name));
            if api_name_match || config_name == model_name || model_match {
                results.push((config_name.clone(), model_config, backend));
            }
        }

        results
    }

    /// Resolve the health check URL for a model_config, taking into account:
    /// 1. Pre-resolved health_check_url if available (from DB via InstallationManager)
    /// 2. Server's custom port if set
    /// 3. Fallback to http://localhost:{port}/health
    ///
    /// Does not require the backend to exist in TOML [backends] section.
    /// After migration to provider_configs DB table, the [backends] section
    /// may be empty — this function resolves purely from the provided URL
    /// parameter and model_config port.
    pub fn resolve_health_url(
        &self,
        server: &ModelConfig,
        health_check_url: Option<&str>,
    ) -> Option<String> {
        // If pre-resolved health_check_url is provided, use it (and replace port if server.port is set)
        if let Some(backend_url) = health_check_url {
            if let Some(port) = server.port {
                let mut url = url::Url::parse(backend_url).ok()?;
                url.set_port(Some(port)).ok()?;
                return Some(url.to_string());
            }
            return Some(backend_url.to_string());
        }

        // health_check_url is None, try server.port fallback
        if let Some(port) = server.port {
            return Some(format!("http://localhost:{}/health", port));
        }

        // Neither health_check_url nor server.port present
        None
    }

    /// Resolve the backend URL (without /health) for a server.
    ///
    /// Does not require the backend to exist in TOML [backends] section.
    /// After migration to provider_configs DB table, the [backends] section
    /// may be empty — this function resolves purely from the provided URL
    /// parameter and server port.
    pub fn resolve_backend_url(
        &self,
        server: &ModelConfig,
        health_check_url: Option<&str>,
    ) -> Option<String> {
        // If pre-resolved health_check_url is provided, derive the base URL from it
        if let Some(health_url) = health_check_url {
            let mut url = url::Url::parse(health_url).ok()?;

            // Override port if the server specifies one
            if let Some(port) = server.port {
                url.set_port(Some(port)).ok()?;
            }

            // Strip the path to get the base origin (scheme + host + port)
            url.set_path("");
            url.set_query(None);
            url.set_fragment(None);
            let base = url.to_string().trim_end_matches('/').to_string();
            return Some(base);
        }

        // health_check_url is None, try server.port fallback
        if let Some(port) = server.port {
            return Some(format!("http://localhost:{}", port));
        }

        // Neither health_check_url nor server.port present
        None
    }

    /// Build the merged arg list for a server, returning **flat tokens**
    /// suitable for `Command::args`.
    ///
    /// Merging order: pre-resolved `default_args` → `server.args` →
    /// `server.sampling.to_args()`. Each later layer's flags fully replace
    /// the same flag in the earlier layers via `merge_args`.
    pub fn build_args(
        &self,
        server: &ModelConfig,
        // Kept for API stability; default_args now resolved externally.
        #[allow(dead_code)] _backend: &BackendConfig,
        default_args: &[String],
    ) -> Vec<String> {
        // Normalize the backend's `default_args` (which may be stored as flat
        // tokens, e.g. `["--mamba-cache-mode", "align"]`) into grouped form.
        // `merge_args` dedupes by flag name across layers; if the base were left
        // flat, a model override would drop the flag token but leave its value
        // token behind as an orphaned positional (e.g. a stray `align`).
        let base = crate::config::group_legacy_flat_args(default_args).0;
        let mut grouped = crate::config::merge_args(&base, &server.args);
        if let Some(sampling) = &server.sampling {
            if !sampling.is_empty() {
                grouped = crate::config::merge_args(&grouped, &sampling.to_args());
            }
        }
        crate::config::flatten_args(&grouped)
    }

    /// Build the full argument list for a model, including model config args
    /// (`-m`, `-c`, `-ngl`) and sampling. Returns **flat tokens** suitable for
    /// `Command::args`.
    ///
    /// Merging order:
    /// 1. Pre-resolved `default_args`
    /// 2. `server.args`     (replaces same-flag entries from #1)
    /// 3. Injected `-m`/`-c`/`-ngl` (only if not already present after #1+#2)
    /// 4. `server.vllm.to_args()` (transformers only, replaces same-flag entries)
    /// 5. `server.sampling.to_args()` (replaces same-flag entries from #1-#4)
    ///
    /// **Invariant:** the returned `Vec<String>` is always flat (one token
    /// per element). Callers like `proxy/lifecycle.rs::override_arg` and
    /// `bench/runner.rs::_override_arg` depend on this. The final
    /// `flatten_args` call enforces it; the `debug_assert!` makes accidental
    /// regressions visible in test/debug builds.
    pub fn build_full_args(
        &self,
        server: &ModelConfig,
        // Kept for API stability; default_args now resolved externally.
        #[allow(dead_code)] _backend: &BackendConfig,
        ctx_override: Option<u32>,
        default_args: &[String],
    ) -> Result<Vec<String>> {
        // Normalize the backend's `default_args` (which may be stored as flat
        // tokens, e.g. `["--mamba-cache-mode", "align"]`) into grouped form.
        // `merge_args` dedupes by flag name across layers; if the base were left
        // flat, a model override would drop the flag token but leave its value
        // token behind as an orphaned positional (e.g. a stray `align`).
        let base = crate::config::group_legacy_flat_args(default_args).0;
        let mut grouped = crate::config::merge_args(&base, &server.args);

        // Determine the HuggingFace format for format-aware arg injection.
        let is_transformers = server.hf_format.as_deref() == Some("transformers");

        // ── Positional model path for transformers format ──────────────
        // vLLM (and other transformers backends) expect the model path
        // as the first positional arg.  For GGUF / llama.cpp the model
        // path is emitted via `-m <file>` below.
        if is_transformers {
            if let Some(ref model_id) = server.model {
                let models_dir = self.models_dir()?;
                let model_path = repo_path(&models_dir, model_id);
                let path_str = model_path.to_string_lossy();

                // Dedup: skip if the user already has a positional model path
                // or --model flag in their args. Mirrors the already_has_m pattern.
                let already_has_positional = grouped
                    .iter()
                    .any(|e| matches!(crate::config::flag_name(e), Some("--model")))
                    || grouped.iter().any(|e| {
                        // Check if any entry matches the resolved model path
                        *e == path_str
                            || e.starts_with(&format!("\"{}", path_str))
                            || e.starts_with(&format!("'{}", path_str))
                    });

                if !already_has_positional {
                    // Insert after any subcommand (e.g. "serve") and before
                    // the first `--` flag.  Entries are still grouped at this
                    // point (flatten_args runs later), so we scan for the first
                    // flag-like entry to find the insertion position.
                    let path_token = crate::config::quote_value(&path_str);
                    // Find insertion point: after subcommand, before first flag.
                    let insert_at = grouped.iter().position(|e| {
                        // End of grouped entries: stop at the first flag-like entry
                        e.starts_with('-')
                    });
                    match insert_at {
                        Some(pos) => grouped.insert(pos, path_token.to_string()),
                        None => grouped.push(path_token.to_string()),
                    }
                }
            }

            // vLLM serves the model under the positional path by default, so
            // OpenAI clients addressing it by name get a 404 from the
            // backend (the proxy only rewrites the model field in responses,
            // not requests). Name the served model after the Tama api_name
            // (falling back to the repo id) so both sides agree.
            let served_name = server.api_name.as_deref().or(server.model.as_deref());
            if let Some(name) = served_name {
                grouped =
                    crate::config::merge_args(&grouped, &[format!("--served-model-name {}", name)]);
            }

            // Emit vLLM-specific flags from typed config.
            // Uses merge_args so typed config wins over user args on collision.
            if !server.vllm.is_empty() {
                grouped = crate::config::merge_args(&grouped, &server.vllm.to_args());
            }
        }

        // ── llama.cpp-only flags — gate on non-transformers format ─────
        let is_llama_cpp_backend = backend_is_llama_cpp(&server.backend);

        // Inject -m from model card, only if not transformers format.
        if !is_transformers {
            if let (Some(ref model_id), Some(ref quant_name)) = (&server.model, &server.quant) {
                if let Some(quant_entry) = server.quants.get(quant_name.as_str()) {
                    let models_dir = self.models_dir()?;
                    let model_path = repo_path(&models_dir, model_id).join(&quant_entry.file);
                    let already_has_m = grouped.iter().any(|e| {
                        matches!(crate::config::flag_name(e), Some("-m") | Some("--model"))
                    });
                    if !already_has_m {
                        let path_str = model_path.to_string_lossy();
                        let quoted = crate::config::quote_value(&path_str);
                        grouped.push(format!("-m {}", quoted));
                    }
                } else {
                    tracing::warn!(
                        "Quant '{}' not found in ModelConfig for model '{}'",
                        quant_name,
                        model_id
                    );
                }
            }
        } // end !is_transformers guard for -m

        // Inject --mmproj from model card, only if not transformers format.
        // The mmproj entry must exist in `server.quants` and have kind = Mmproj.
        if !is_transformers {
            if let (Some(ref model_id), Some(ref mmproj_name)) = (&server.model, &server.mmproj) {
                if let Some(mmproj_entry) = server.quants.get(mmproj_name.as_str()) {
                    if mmproj_entry.kind == crate::config::QuantKind::Mmproj {
                        let models_dir = self.models_dir()?;
                        let mmproj_path = repo_path(&models_dir, model_id).join(&mmproj_entry.file);
                        let already_has_mmproj = grouped
                            .iter()
                            .any(|e| matches!(crate::config::flag_name(e), Some("--mmproj")));
                        if !already_has_mmproj {
                            let path_str = mmproj_path.to_string_lossy();
                            let quoted = crate::config::quote_value(&path_str);
                            grouped.push(format!("--mmproj {}", quoted));
                        }
                    } else {
                        tracing::warn!(
                            "mmproj '{}' for model '{}' has kind={:?}, expected Mmproj",
                            mmproj_name,
                            model_id,
                            mmproj_entry.kind
                        );
                    }
                } else {
                    tracing::warn!(
                        "mmproj '{}' not found in ModelConfig for model '{}'",
                        mmproj_name,
                        model_id
                    );
                }
            }
        }

        // Inject --spec-draft-model from model card, only if not transformers format.
        // 1. mtp_model is set
        // 2. The referenced quant has kind = Mtp
        // 3. draft-mtp is in spec_decoding.spec_types (user enabled it)
        if !is_transformers {
            if let (Some(ref model_id), Some(ref mtp_name)) = (&server.model, &server.mtp_model) {
                let has_draft_mtp = server
                    .spec_decoding
                    .spec_types
                    .iter()
                    .any(|t| t == "draft-mtp");
                if has_draft_mtp {
                    if let Some(mtp_entry) = server.quants.get(mtp_name.as_str()) {
                        if mtp_entry.kind == crate::config::QuantKind::Mtp {
                            let models_dir = self.models_dir()?;
                            let mtp_path = repo_path(&models_dir, model_id).join(&mtp_entry.file);
                            let already_has_draft = grouped.iter().any(|e| {
                                matches!(crate::config::flag_name(e), Some("--spec-draft-model"))
                            });
                            if !already_has_draft {
                                let path_str = mtp_path.to_string_lossy();
                                let quoted = crate::config::quote_value(&path_str);
                                grouped.push(format!("--spec-draft-model {}", quoted));
                            }
                        } else {
                            tracing::warn!(
                                "mtp_model '{}' for model '{}' has kind={:?}, expected Mtp",
                                mtp_name,
                                model_id,
                                mtp_entry.kind
                            );
                        }
                    } else {
                        tracing::warn!(
                            "mtp_model '{}' not found in ModelConfig for model '{}'",
                            mtp_name,
                            model_id
                        );
                    }
                }
            }
        } // end !is_transformers guard for --spec-draft-model

        // Inject -c (context length) only if not transformers format.
        if !is_transformers {
            let ctx = ctx_override.or(server.context_length).or_else(|| {
                server
                    .quant
                    .as_ref()
                    .and_then(|q| server.quants.get(q).and_then(|qe| qe.context_length))
            });

            if let Some(ctx) = ctx {
                let already_has_c = grouped.iter().any(|e| {
                    matches!(crate::config::flag_name(e), Some("-c") | Some("--ctx-size"))
                });
                if !already_has_c {
                    let slots = server.num_parallel.unwrap_or(1).max(1); // 0 = auto, treat as 1 for ctx calc
                    let total_ctx = if is_llama_cpp_backend && server.kv_unified {
                        // Unified KV: all slots share one pool, -c = per-slot context
                        ctx
                    } else {
                        // Non-unified: each slot gets dedicated region, -c = per_slot * slots
                        ctx.saturating_mul(slots)
                    };
                    grouped.push(format!("-c {}", total_ctx));
                }
            }
        }

        // Inject -np (number of parallel slots) if set, >= 1, and not transformers.
        // 0 means auto (don't set the flag).
        if !is_transformers {
            if let Some(slots) = server.num_parallel {
                if slots >= 1 {
                    let already_has_np = grouped.iter().any(|e| {
                        matches!(
                            crate::config::flag_name(e),
                            Some("-np") | Some("--parallel")
                        )
                    });
                    if !already_has_np {
                        grouped.push(format!("-np {}", slots));
                    }
                }
            }
        }

        // Inject -b (batch size). Typed field wins over any leftover
        // `-b`/`--batch-size` in args via merge_args dedup. Not for transformers.
        if !is_transformers {
            if let Some(b) = server.n_batch {
                grouped = crate::config::merge_args(&grouped, &[format!("-b {}", b)]);
            }
        }

        // Inject -ub (ubatch size). Typed field wins over any leftover
        // `-ub`/`--ubatch-size` in args via merge_args dedup. Not for transformers.
        if !is_transformers {
            if let Some(ub) = server.n_ubatch {
                grouped = crate::config::merge_args(&grouped, &[format!("-ub {}", ub)]);
            }
        }

        // Inject -ngl only if not transformers format and not already present.
        if !is_transformers {
            if let Some(ngl) = server.gpu_layers {
                let already_has_ngl = grouped.iter().any(|e| {
                    matches!(
                        crate::config::flag_name(e),
                        Some("-ngl") | Some("--n-gpu-layers")
                    )
                });
                if !already_has_ngl {
                    grouped.push(format!("-ngl {}", ngl));
                }
            }
        }

        // Inject --kv-unified flag when enabled and backend supports it.
        if is_llama_cpp_backend && server.kv_unified {
            let already_has_kv_unified = grouped
                .iter()
                .any(|e| matches!(crate::config::flag_name(e), Some("--kv-unified")));
            if !already_has_kv_unified {
                grouped.push("--kv-unified".to_string());
            }
        }

        // Inject --cache-type-k only if set and backend supports it.
        if is_llama_cpp_backend {
            if let Some(ref ct_k) = server.cache_type_k {
                let trimmed = ct_k.trim();
                if !trimmed.is_empty() {
                    let already_has_ctk = grouped.iter().any(|e| {
                        matches!(
                            crate::config::flag_name(e),
                            Some("-ctk") | Some("--cache-type-k")
                        )
                    });
                    if !already_has_ctk {
                        grouped.push(format!("-ctk {}", crate::config::quote_value(trimmed)));
                    }
                }
            }
        }

        // Inject --cache-type-v only if set and backend supports it.
        if is_llama_cpp_backend {
            if let Some(ref ct_v) = server.cache_type_v {
                let trimmed = ct_v.trim();
                if !trimmed.is_empty() {
                    let already_has_ctv = grouped.iter().any(|e| {
                        matches!(
                            crate::config::flag_name(e),
                            Some("-ctv") | Some("--cache-type-v")
                        )
                    });
                    if !already_has_ctv {
                        grouped.push(format!("-ctv {}", crate::config::quote_value(trimmed)));
                    }
                }
            }
        }

        // Inject spec-decoding flags when configured (llama.cpp backends only).
        if is_llama_cpp_backend && !server.spec_decoding.spec_types.is_empty() {
            let sd = &server.spec_decoding;

            let already_has_spec_type = grouped
                .iter()
                .any(|e| matches!(crate::config::flag_name(e), Some("--spec-type")));
            if !already_has_spec_type {
                grouped.push(format!("--spec-type {}", sd.spec_types.join(",")));
            }

            if let Some(n) = sd.n_max {
                let already_has = grouped
                    .iter()
                    .any(|e| matches!(crate::config::flag_name(e), Some("--spec-draft-n-max")));
                if !already_has {
                    grouped.push(format!("--spec-draft-n-max {}", n));
                }
            }

            if let Some(n) = sd.n_min {
                let already_has = grouped
                    .iter()
                    .any(|e| matches!(crate::config::flag_name(e), Some("--spec-draft-n-min")));
                if !already_has {
                    grouped.push(format!("--spec-draft-n-min {}", n));
                }
            }

            if sd.spec_types.iter().any(|t| t == "draft-mtp") {
                if let Some(spec_ngl) = sd.draft_ngl {
                    let already_has = grouped
                        .iter()
                        .any(|e| matches!(crate::config::flag_name(e), Some("--spec-draft-ngl")));
                    if !already_has {
                        grouped.push(format!("--spec-draft-ngl {}", spec_ngl));
                    }
                }
            }
        }

        // Inject --alias for model identification in /v1/models responses.
        // This allows the merge logic to match backend entries (by filename)
        // against config entries (by api_name) via the aliases array.
        // Only inject for llama.cpp backends, and only if not already set.
        if is_llama_cpp_backend {
            let alias_value = server
                .api_name
                .clone()
                .or_else(|| server.model.clone())
                .unwrap_or_default();
            if !alias_value.is_empty() {
                let already_has_alias = grouped
                    .iter()
                    .any(|e| matches!(crate::config::flag_name(e), Some("--alias") | Some("-a")));
                if !already_has_alias {
                    grouped.push(format!("--alias {}", alias_value));
                }
            }
        }

        // Managed flag (ADR-0014): the tamad scrapes the engine's /metrics endpoint
        // for inference stats; llama.cpp serves it only with --metrics (501 without).
        // Inject for llama.cpp backends, presence-checked (user args never overridden).
        if is_llama_cpp_backend {
            crate::config::llama_cpp_args::ensure_metrics(&mut grouped);
        }

        // Sampling: each sampling flag fully replaces the same flag in
        // anything injected so far.
        if let Some(sampling) = &server.sampling {
            if !sampling.is_empty() {
                grouped = crate::config::merge_args(&grouped, &sampling.to_args());
            }
        }

        let flat = crate::config::flatten_args(&grouped);
        // INVARIANT: build_full_args returns flat tokens. Callers like
        // proxy/lifecycle.rs::override_arg depend on this. The check
        // catches the failure mode where a *grouped* entry (e.g.
        // "-b 4096") leaks through unflattened: such an element starts
        // with '-' AND contains whitespace AND is not quoted.
        // Legitimate value-side tokens like "system: hi" or
        // "/path with space/m.gguf" contain whitespace but do NOT start
        // with '-', so they pass. We also allow tokens that start with a
        // quote character (escaped quotes from shlex unquoting edge cases).
        debug_assert!(
            flat.iter().all(|t| {
                !t.starts_with('-')
                    || !t.contains(char::is_whitespace)
                    || t.starts_with('"')
                    || t.starts_with('\'')
            }),
            "build_full_args invariant violated: element looks like a grouped entry (flag + space + value): {:?}",
            flat
        );
        Ok(flat)
    }

    pub fn service_name(backend_name: &str) -> String {
        format!("tama-{}", backend_name)
    }

    /// Build the proxy base URL from config, e.g. `http://0.0.0.0:11411`.
    /// Always returns a URL since the proxy may be running even if not
    /// marked as enabled in config (e.g. started manually via `tama serve`).
    pub fn proxy_url(&self) -> String {
        format!("http://{}:{}", self.proxy.host, self.proxy.port)
    }

    /// Resolve the filesystem path for a named backend binary.
    ///
    /// Priority:
    /// 1. Model-level `gpu_variant` (passed as `model_variant`) — most specific
    /// 2. Global config `[backends.<name>].gpu_variant`
    /// 3. Discover from InstallationManager (first active installation, or first variant found)
    /// 4. Default "cpu"
    ///
    /// Then:
    /// 1. If `config.backends[name].version` is pinned, look up that exact version
    ///    in the InstallationManager for the resolved variant.
    /// 2. Otherwise, use the active (latest) installation for that variant.
    /// 3. Fallback to `path` field in the [backends] section.
    ///
    /// `manager` is `None` when no Postgres pool is configured (e.g. in tests):
    /// DB lookups are then skipped and only the config-path fallback applies
    /// (a pinned version still fails hard when missing).
    pub async fn resolve_backend_path(
        &self,
        name: &str,
        model_variant: Option<&crate::gpu::GpuVariant>,
        manager: Option<&crate::installations::InstallationManager>,
    ) -> Result<std::path::PathBuf> {
        // Determine the gpu_variant to use (model > config > "cpu")
        let gpu_variant: &str = model_variant
            .map(|v| v.variant_folder())
            .or_else(|| {
                self.backends
                    .get(name)
                    .and_then(|b| b.gpu_variant.as_ref())
                    .map(|v| v.variant_folder())
            })
            .unwrap_or("cpu");

        // Check if a specific version is pinned in config
        if let Some(pinned_version) = self.backends.get(name).and_then(|b| b.version.as_deref()) {
            let pinned_path = if let Some(manager) = manager {
                // Try the specified variant first
                if let Some(info) = manager
                    .get_by_version(name, gpu_variant, pinned_version)
                    .await?
                {
                    Some(info.path)
                } else {
                    // If not found, try all variants of this backend for the pinned version
                    manager
                        .list_versions(name, None)
                        .await?
                        .and_then(|versions| {
                            versions
                                .into_iter()
                                .find(|v| v.version == pinned_version)
                                .map(|v| v.path)
                        })
                }
            } else {
                None
            };
            if let Some(path) = pinned_path {
                return Ok(path);
            }
            anyhow::bail!(
                "Backend '{}' version '{}' not found in DB. Run `tama backend install {}` first.",
                name,
                pinned_version,
                name
            );
        }

        // No version pin — try to find the active installation.
        if let Some(manager) = manager {
            // First, try the specific variant (model > config > "cpu").
            if let Some(info) = manager.get_active(name, gpu_variant).await? {
                return Ok(info.path);
            }

            // If not found for the specific variant, try all active variants
            // for this backend. This handles the case where the user selects
            // a backend that's installed for a different GPU variant (e.g.,
            // "rocm" instead of "cpu").
            if let Some(info) = manager
                .list_active()
                .await?
                .into_iter()
                .find(|v| v.name == name)
            {
                return Ok(info.path);
            }
        }

        // Fallback to config path (for custom/manual installs)
        self.backends
            .get(name)
            .and_then(|b| b.path.as_deref())
            .map(std::path::PathBuf::from)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Backend '{}' has no installed path. Run `tama backend install {}` first.",
                    name,
                    name
                )
            })
    }
}

/// Check if a backend name refers to a llama.cpp-compatible backend.
/// Used to gate llama.cpp-specific flags like `--kv-unified`.
pub fn backend_is_llama_cpp(backend_name: &str) -> bool {
    backend_name.starts_with("llama")
}

#[cfg(test)]
mod tests;
