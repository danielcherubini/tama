pub mod api_keys;
pub mod auth;
pub mod forward;
mod handlers;
pub mod lifecycle;
pub mod pull_jobs;
pub mod pull_queue;
mod remote;
mod rename;
pub mod scope_middleware;
pub mod server;
mod state;
pub mod status;
pub mod tama_handlers;
mod types;

pub use crate::process::override_arg;
pub use api_keys::{ApiKeyRecord, AuthSubject, Scope};
pub use forward::forward_request;
pub use handlers::chat::{handle_chat_completions, handle_stream_chat_completions};
pub use handlers::forward::{
    forward_to_backend, handle_fallback, handle_forward_get, handle_forward_post,
};
pub use handlers::models::{
    fetch_models_from_backend, handle_get_model, handle_list_models, parse_models_response,
    BackendModelEntry,
};
pub use handlers::status::{handle_health, handle_metrics, handle_reload_configs, handle_status};
pub use handlers::tts::{
    handle_audio_models, handle_audio_speech, handle_audio_stream, handle_audio_voices,
};
pub use handlers::{json_error, json_error_response};
pub use server::ProxyServer;
pub use state::repo_pull::{RepoPullError, RepoPullStart, RepoPullStatusDto};
pub use state::rows::{live as live_rows, ModelRow, Rows};
pub use types::{ProxyMetrics, ProxyState};

#[cfg(test)]
mod tests {
    mod restart_test;

    use super::*;
    use crate::config::{BackendConfig, Config, ModelConfig};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_proxy_state_new() {
        let config = Config::default();
        let state = ProxyState::new(config.clone(), None, crate::db::pool::test_dummy_pool());
        assert!(state.registry.last_accessed.read().await.is_empty());
        assert_eq!(
            state.config.read().await.proxy.idle_timeout_secs,
            config.proxy.idle_timeout_secs
        );
    }

    #[tokio::test]
    async fn test_no_available_server_for_unknown_model() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());
        let result = state.get_available_backend_for_model("nonexistent").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_build_status_response() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        let response = state.build_status_response().await;
        let v = serde_json::to_value(&response).unwrap();

        // live wire shape: vram/gpu_utilization_pct are OMITTED when None
        assert!(v.get("vram").is_none() || !v["vram"].is_null());

        // auto_unload and idle_timeout_secs at top level per spec
        assert_eq!(
            v.get("auto_unload").and_then(|v| v.as_bool()),
            Some(false),
            "auto_unload should be a boolean (default false)"
        );
        assert!(
            v.get("idle_timeout_secs")
                .and_then(|v| v.as_u64())
                .is_some(),
            "idle_timeout_secs should be a number"
        );

        // models is an object keyed by model name
        assert!(v.get("models").unwrap().is_object());

        // metrics is an object
        assert!(v.get("metrics").unwrap().is_object());
    }

    #[tokio::test]
    async fn test_build_status_response_model_fields() {
        // Create a config with a model for testing
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Add a model to the registry
        {
            let mut model_configs = state.registry.model_configs.write().await;
            model_configs.insert(
                "test-model".to_string(),
                ModelConfig {
                    backend: "llama_cpp".to_string(),
                    model: Some("test/model".to_string()),
                    enabled: true,
                    ..Default::default()
                },
            );
        }
        let state = state;

        let response = state.build_status_response().await;

        // models is an object, check that our test model is present
        let models = &response.models;
        assert!(!models.is_empty(), "config should have at least one model");

        let (_, first_model) = models.iter().next().unwrap();

        // Per spec: flat fields, not nested in runtime
        assert_eq!(first_model.backend, "llama_cpp");
        assert!(first_model.enabled);
        // Unloaded model should have state = Idle
        assert_eq!(
            first_model.state,
            crate::proxy::status::StatusModelState::Idle
        );
    }

    #[tokio::test]
    async fn test_rename_model_success() {
        let _temp_dir = tempfile::tempdir().unwrap();
        let config = Config {
            ..Default::default()
        };
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Add a model to the registry
        {
            let mut model_configs = state.registry.model_configs.write().await;
            model_configs.insert(
                "old-name".to_string(),
                ModelConfig {
                    enabled: true,
                    num_parallel: Some(1),
                    ..ModelConfig::test_config("llama_cpp")
                },
            );
        }

        // Rename should succeed
        state.rename_model("old-name", "new-name").await.unwrap();

        // Verify old name is gone, new name exists
        let model_configs = state.registry.model_configs.read().await;
        assert!(!model_configs.contains_key("old-name"));
        assert!(model_configs.contains_key("new-name"));
    }

    #[tokio::test]
    async fn test_rename_model_new_name_taken() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Add models to the registry
        {
            let mut model_configs = state.registry.model_configs.write().await;
            model_configs.insert(
                "old-name".to_string(),
                ModelConfig {
                    enabled: true,
                    num_parallel: Some(1),
                    ..ModelConfig::test_config("llama_cpp")
                },
            );
            model_configs.insert(
                "new-name".to_string(),
                ModelConfig {
                    enabled: true,
                    num_parallel: Some(1),
                    ..ModelConfig::test_config("llama_cpp")
                },
            );
        }

        // Rename should fail because new name is taken
        let result = state.rename_model("old-name", "new-name").await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "model name 'new-name' already taken"
        );
    }

    #[tokio::test]
    async fn test_rename_model_old_name_not_found() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Add a model to the registry
        {
            let mut model_configs = state.registry.model_configs.write().await;
            model_configs.insert(
                "existing-name".to_string(),
                ModelConfig {
                    enabled: true,
                    num_parallel: Some(1),
                    ..ModelConfig::test_config("llama_cpp")
                },
            );
        }

        // Rename should fail because old name doesn't exist
        let result = state.rename_model("non-existent", "new-name").await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "model 'non-existent' does not exist"
        );
    }

    #[tokio::test]
    async fn test_rename_model_empty_name() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Add a model to the registry
        {
            let mut model_configs = state.registry.model_configs.write().await;
            model_configs.insert(
                "old-name".to_string(),
                ModelConfig {
                    enabled: true,
                    num_parallel: Some(1),
                    ..ModelConfig::test_config("llama_cpp")
                },
            );
        }

        // Rename should fail because new name is empty
        let result = state.rename_model("old-name", "").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "new name cannot be empty");
    }

    #[tokio::test]
    async fn test_rename_model_same_name() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Add a model to the registry
        {
            let mut model_configs = state.registry.model_configs.write().await;
            model_configs.insert(
                "same-name".to_string(),
                ModelConfig {
                    enabled: true,
                    num_parallel: Some(1),
                    ..ModelConfig::test_config("llama_cpp")
                },
            );
        }

        // Rename should fail because old and new name are the same
        let result = state.rename_model("same-name", "same-name").await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "old name and new name must differ"
        );
    }

    #[tokio::test]
    async fn test_proxy_state_shutdown_clears_access_map() {
        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // plan-193 T5c: there is no model mirror to clear anymore — the
        // shutdown contract that survives is the per-key access map.
        state.update_last_accessed("test-model").await;
        state.shutdown().await;

        assert!(state.registry.last_accessed.read().await.is_empty());
    }
    #[tokio::test]
    async fn test_build_status_response_golden_shape() {
        use crate::gpu::{SystemMetrics, VramInfo};

        // Shared fixture
        let mut config = Config::default();
        config.backends.insert(
            "llama_cpp".to_string(),
            BackendConfig {
                path: Some("/opt/llama/llama-server".into()),
                version: None,
                gpu_variant: None,
            },
        );
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Two model configs: one idle, one ready (with db_id).
        {
            let mut mc = state.registry.model_configs.write().await;
            mc.insert(
                "idle-model".to_string(),
                ModelConfig {
                    backend: "llama_cpp".to_string(),
                    enabled: true,
                    ..Default::default()
                },
            );
            mc.insert(
                "ready-model".to_string(),
                ModelConfig {
                    backend: "llama_cpp".to_string(),
                    display_name: Some("Ready Model".to_string()),
                    model: Some("test/model".to_string()),
                    db_id: Some(7),
                    enabled: true,
                    ..Default::default()
                },
            );
        }

        // State now comes from the TAMAD's row (plan-193 T5c: the
        // mirror is gone): a wire-ready row (pid feeds backend_pid)
        // plus the proxy-side access entry (last_accessed field).
        {
            let proc = crate::tamad::ProcessInfo {
                model_name: "ready-model".to_string(),
                provider_name: "llama.cpp".to_string(),
                pid: 4242,
                alive: true,
                endpoint_url: "http://127.0.0.1:18910".to_string(),
                status: "ready".to_string(),
                desired: true,
                restart_count: 0,
                max_restarts: 3,
                spec_accept_pct: None,
                spec_decoding_active: false,
                tps: None,
                prompt_tps: None,
                cache_hit_pct: None,
            };
            let stats = crate::tamad::pool::test_support::stats_full(1.5, vec![], vec![proc]);
            state
                .tamad_pool()
                .insert_raw_handle(
                    "t1",
                    Arc::new(
                        crate::tamad::pool::test_support::handle_with_latest(
                            std::time::Instant::now(),
                            stats,
                        )
                        .await,
                    ),
                )
                .await;
            state.update_last_accessed("ready-model").await;
        }

        // Block 1: vram + gpu_utilization present
        {
            state
                .metrics
                .set_system_metrics(SystemMetrics {
                    cpu_usage_pct: 12.5,
                    ram_used_mib: 0,
                    ram_total_mib: 0,
                    gpu_utilization_pct: Some(75),
                    vram: Some(VramInfo {
                        used_mib: 100,
                        total_mib: 200,
                    }),
                    ..Default::default()
                })
                .await;
        }
        let mut v = serde_json::to_value(state.build_status_response().await).unwrap();
        // Mask the volatile last_accessed_secs_ago field.
        v["models"]["ready-model"]["last_accessed_secs_ago"] = serde_json::json!(0);
        assert_eq!(
            v,
            serde_json::json!({
                "cpu_usage_pct": 12.5,
                "ram_used_mib": 0,
                "ram_total_mib": 0,
                "gpu_utilization_pct": 75,
                "vram": { "used_mib": 100, "total_mib": 200 },
                "auto_unload": false,
                "idle_timeout_secs": 300,
                "metrics": {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "models_loaded": 1,
                    "models_unloaded": 0
                },
                "models": {
                    "idle-model": {
                        "id": null,
                        "display_name": null,
                        "backend": "llama_cpp",
                        "backend_path": "/opt/llama/llama-server",
                        "model": null,
                        "quant": null,
                        "context_length": null,
                        "enabled": true,
                        "api_name": null,
                        "state": "idle",
                        "backend_pid": null,
                        "load_time_secs": null,
                        "last_accessed_secs_ago": null,
                        "idle_timeout_remaining_secs": null,
                        "consecutive_failures": null
                    },
                    "ready-model": {
                        "id": 7,
                        "display_name": "Ready Model",
                        "backend": "llama_cpp",
                        "backend_path": "/opt/llama/llama-server",
                        "model": "test/model",
                        "quant": null,
                        "context_length": null,
                        "enabled": true,
                        "api_name": null,
                        "state": "ready",
                        "backend_pid": 4242,
                        "load_time_secs": null,
                        "last_accessed_secs_ago": 0,
                        "idle_timeout_remaining_secs": null,
                        "consecutive_failures": null
                    }
                }
            })
        );

        // Block 2: vram + gpu_utilization None -> keys omitted on the wire
        // live wire shape: vram/gpu_utilization_pct are OMITTED when None
        {
            state
                .metrics
                .set_system_metrics(SystemMetrics {
                    cpu_usage_pct: 12.5,
                    ram_used_mib: 0,
                    ram_total_mib: 0,
                    gpu_utilization_pct: None,
                    vram: None,
                    ..Default::default()
                })
                .await;
        }
        let mut v = serde_json::to_value(state.build_status_response().await).unwrap();
        v["models"]["ready-model"]["last_accessed_secs_ago"] = serde_json::json!(0);
        assert_eq!(
            v,
            serde_json::json!({
                "cpu_usage_pct": 12.5,
                "ram_used_mib": 0,
                "ram_total_mib": 0,
                "auto_unload": false,
                "idle_timeout_secs": 300,
                "metrics": {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "models_loaded": 1,
                    "models_unloaded": 0
                },
                "models": {
                    "idle-model": {
                        "id": null,
                        "display_name": null,
                        "backend": "llama_cpp",
                        "backend_path": "/opt/llama/llama-server",
                        "model": null,
                        "quant": null,
                        "context_length": null,
                        "enabled": true,
                        "api_name": null,
                        "state": "idle",
                        "backend_pid": null,
                        "load_time_secs": null,
                        "last_accessed_secs_ago": null,
                        "idle_timeout_remaining_secs": null,
                        "consecutive_failures": null
                    },
                    "ready-model": {
                        "id": 7,
                        "display_name": "Ready Model",
                        "backend": "llama_cpp",
                        "backend_path": "/opt/llama/llama-server",
                        "model": "test/model",
                        "quant": null,
                        "context_length": null,
                        "enabled": true,
                        "api_name": null,
                        "state": "ready",
                        "backend_pid": 4242,
                        "load_time_secs": null,
                        "last_accessed_secs_ago": 0,
                        "idle_timeout_remaining_secs": null,
                        "consecutive_failures": null
                    }
                }
            })
        );
    }

    #[tokio::test]
    async fn test_config_write_semaphore_allows_concurrent_acquisitions() {
        use std::time::Duration;

        let config = Config::default();
        let state = ProxyState::new(config, None, crate::db::pool::test_dummy_pool());

        // Verify the semaphore has capacity 4
        assert_eq!(state.config_write_semaphore.available_permits(), 4);

        // Acquire 4 permits concurrently — all should succeed quickly
        let start = std::time::Instant::now();
        let mut handles = vec![];
        for i in 0..4 {
            let sem = Arc::clone(&state.config_write_semaphore);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("acquire should succeed");
                // Hold the permit briefly to simulate work
                tokio::time::sleep(Duration::from_millis(50)).await;
                (i, start.elapsed())
            }));
        }

        // All 4 should complete within ~150ms (50ms work + some overhead)
        let results: Vec<(usize, Duration)> = futures_util::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(results.len(), 4);

        // Verify each completed within a reasonable time (all ran concurrently)
        for (idx, elapsed) in &results {
            assert!(
                *elapsed < Duration::from_millis(200),
                "Concurrent permit {} took {:?}, expected < 200ms",
                idx,
                elapsed
            );
        }

        // Now verify that exceeding capacity blocks try_acquire.
        // Acquire all 4 permits in the main task.
        let mut held_permits = vec![];
        for _ in 0..4 {
            let p = state
                .config_write_semaphore
                .acquire()
                .await
                .expect("acquire should succeed");
            held_permits.push(p);
        }
        assert_eq!(state.config_write_semaphore.available_permits(), 0);

        // try_acquire should return Err when all permits are exhausted
        assert!(
            state.config_write_semaphore.try_acquire().is_err(),
            "try_acquire should return Err when semaphore is full"
        );

        // Release one permit — now try_acquire should succeed
        drop(held_permits.pop());
        let permit = state
            .config_write_semaphore
            .try_acquire()
            .expect("try_acquire should succeed after releasing a permit");

        // Release the remaining permits and the acquired one
        drop(held_permits);
        drop(permit);
    }
}
