pub mod client;
pub mod pool;
pub mod protocol;
pub mod stream_logs;

pub mod tamad_service {
    include!(concat!(env!("OUT_DIR"), "/tamad.rs"));
}

pub use tamad_service::tamad_service_client::TamadServiceClient;
pub use tamad_service::tamad_service_server::{TamadService, TamadServiceServer};

// Re-export generated message types for convenience
pub use tamad_service::{
    CancelJobRequest, CancelJobResponse, Empty, GpuInfo, HealthResponse, InstallProviderRequest,
    JobEvent, JobIdResponse, JobRequest, ListProvidersResponse, LoadModelRequest,
    LoadModelResponse, LogEntry, LoggedLine, LogsRequest, ProcessInfo, ProviderInfo,
    PullModelRequest, RemoveProviderRequest, RestartProviderRequest, RunBenchmarkRequest,
    StatsRequest, StreamInit, StreamLogMessage, SystemStats, UnloadModelRequest,
    UpdateProviderRequest,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-shape test: construct the key v2 messages with all fields and
    /// assert round-trip, proving the generated types match plan-191.
    #[test]
    fn test_v2_message_shapes() {
        let stats = SystemStats {
            cpu_usage_percent: 42.5,
            memory_total_bytes: 1 << 40,
            memory_used_bytes: 1 << 39,
            swap_total_bytes: 1 << 38,
            swap_used_bytes: 1 << 37,
            disk_total_bytes: 1 << 41,
            disk_free_bytes: 1 << 40,
            gpus: vec![GpuInfo {
                index: 0,
                name: "NVIDIA RTX 4090".to_string(),
                driver_version: "570.0".to_string(),
                vram_total_bytes: 24 * 1024 * 1024 * 1024,
                vram_used_bytes: 12 * 1024 * 1024 * 1024,
                utilization_percent: 99.9,
                temperature_c: 71.0,
                power_w: 350.5,
                fan_percent: 39.5,
            }],
            processes: vec![ProcessInfo {
                model_name: "qwen3".to_string(),
                provider_name: "llama.cpp".to_string(),
                pid: 1234,
                alive: true,
                endpoint_url: "http://127.0.0.1:8080".to_string(),
                status: "ready".to_string(),
                desired: false,
                restart_count: 0,
                max_restarts: 0,
                spec_accept_pct: None,
                spec_decoding_active: false,
                tps: None,
                prompt_tps: None,
                cache_hit_pct: None,
                last_obs_ms: None,
            }],
        };
        assert_eq!(stats.cpu_usage_percent, 42.5);
        assert_eq!(stats.gpus[0].vram_total_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(stats.processes[0].model_name, "qwen3");
        assert!(stats.processes[0].alive);

        let event = JobEvent {
            job_id: "job-1".to_string(),
            kind: "pull".to_string(),
            progress: 55,
            message: "downloading".to_string(),
            status: "running".to_string(),
            result_json: String::new(),
            error: String::new(),
            bytes_downloaded: 0,
            total_bytes: 0,
        };
        assert_eq!(event.progress, 55);
        assert_eq!(event.kind, "pull");
        assert!(event.result_json.is_empty());

        let pull = PullModelRequest {
            repo_id: "owner/repo".to_string(),
            quants: vec!["Q4_K_M".to_string()],
            model_name: "my-model".to_string(),
            backend: "llama_cpp".to_string(),
            hf_token: String::new(),
            repo_pull: true,
            dest_dir: String::new(),
        };
        assert_eq!(pull.repo_id, "owner/repo");
        assert_eq!(pull.quants, vec!["Q4_K_M".to_string()]);
        assert!(pull.repo_pull);
        assert!(pull.hf_token.is_empty());

        // plan-191 follow-up B: job-cancel RPC wire shape (idempotent flag).
        let cancel = CancelJobRequest {
            job_id: "job-1".to_string(),
        };
        let response = CancelJobResponse { cancelled: true };
        assert_eq!(cancel.job_id, "job-1");
        assert!(response.cancelled);
    }

    /// Compile-shape test: LoadModelRequest carries the full launch spec
    /// (fields 5-10) and ProviderInfo carries loaded model processes.
    #[test]
    fn test_v2_extended_message_shapes() {
        let req = LoadModelRequest {
            provider_name: "llama.cpp".to_string(),
            model_path: "/models/qwen3".to_string(),
            gpu_variant: "cuda".to_string(),
            params: std::collections::HashMap::new(),
            model_name: "qwen3".to_string(),
            command: "/usr/bin/llama-server".to_string(),
            args: vec!["--port".to_string(), "8080".to_string()],
            env: std::iter::once(("CUDA_VISIBLE_DEVICES".to_string(), "0".to_string())).collect(),
            health_url: "http://127.0.0.1:8080/health".to_string(),
            health_timeout_ms: 30000,
            gpu_device: String::new(),
            docker_config_json: String::new(),
        };
        assert_eq!(req.model_name, "qwen3");
        assert_eq!(req.args.len(), 2);
        assert_eq!(req.env["CUDA_VISIBLE_DEVICES"], "0");
        assert_eq!(req.health_timeout_ms, 30000);

        let provider = ProviderInfo {
            name: "llama.cpp".to_string(),
            engine: "llama_cpp".to_string(),
            version: "b6000".to_string(),
            status: "installed".to_string(),
            gpu_variant: "cuda".to_string(),
            loaded_models: vec![ProcessInfo {
                model_name: "qwen3".to_string(),
                provider_name: "llama.cpp".to_string(),
                pid: 42,
                alive: true,
                endpoint_url: "http://127.0.0.1:8080".to_string(),
                status: "ready".to_string(),
                desired: false,
                restart_count: 0,
                max_restarts: 0,
                spec_accept_pct: None,
                spec_decoding_active: false,
                tps: None,
                prompt_tps: None,
                cache_hit_pct: None,
                last_obs_ms: None,
            }],
        };
        assert_eq!(provider.loaded_models.len(), 1);
        assert_eq!(provider.loaded_models[0].pid, 42);
    }

    /// Compile-shape test (plan-191 Task 8): `RunBenchmarkRequest` carries
    /// the per-kind config plus the tamad-relative execution paths.
    #[test]
    fn test_v2_run_benchmark_request_shape() {
        let req = RunBenchmarkRequest {
            model_name: "qwen3".to_string(),
            kind: "llama_bench".to_string(),
            config_json: r#"{"pp_sizes":[512]}"#.to_string(),
            model_path_rel: "qwen/qwen3/model-Q4_K_M.gguf".to_string(),
            binary_path_rel: "llama_cpp/cuda/b6000/llama-server".to_string(),
        };
        assert_eq!(req.kind, "llama_bench");
        assert_eq!(req.model_path_rel, "qwen/qwen3/model-Q4_K_M.gguf");
        assert_eq!(req.binary_path_rel, "llama_cpp/cuda/b6000/llama-server");
        assert!(!req.config_json.is_empty());
    }

    /// Compile-shape test (plan-191 Task 7): the install/update/remove
    /// request messages carry the execution parameters appended in the
    /// append-only extension (force/git_url, engine/gpu_variant, version).
    #[test]
    fn test_v2_install_update_remove_message_shapes() {
        let install = InstallProviderRequest {
            name: "llama_cpp".to_string(),
            engine: "llama_cpp".to_string(),
            version: "latest".to_string(),
            gpu_variant: "cpu".to_string(),
            force: true,
            git_url: String::new(),
        };
        assert!(install.force);
        assert!(install.git_url.is_empty());

        let update = UpdateProviderRequest {
            name: "llama_cpp".to_string(),
            version: "b9123".to_string(),
            engine: "llama_cpp".to_string(),
            gpu_variant: "cuda".to_string(),
            git_url: "https://github.com/ggml-org/llama.cpp.git".to_string(),
        };
        assert_eq!(update.version, "b9123");
        assert!(!update.git_url.is_empty());

        let remove = RemoveProviderRequest {
            name: "llama_cpp".to_string(),
            engine: "llama_cpp".to_string(),
            gpu_variant: String::new(),
            version: String::new(),
        };
        assert!(remove.gpu_variant.is_empty());
        assert!(remove.version.is_empty());
    }

    /// ADR-0014: the new inference-stats fields (ProcessInfo 12-15) use
    /// explicit proto3 presence — `Some(0.0)` round-trips as `Some(0.0)`
    /// (0.0 is NOT the same as absent), and an unset field round-trips
    /// as `None`. An old tamad's frame (fields omitted) decodes to `None`.
    #[test]
    fn test_process_info_inference_stats_explicit_presence() {
        use prost::Message;

        let p = ProcessInfo {
            model_name: "qwen3".to_string(),
            provider_name: "vllm".to_string(),
            pid: 1,
            alive: true,
            endpoint_url: "http://127.0.0.1:8000".to_string(),
            status: "ready".to_string(),
            desired: false,
            restart_count: 0,
            max_restarts: 0,
            spec_accept_pct: None,
            spec_decoding_active: false,
            tps: Some(0.0),
            prompt_tps: None,
            cache_hit_pct: None,
            last_obs_ms: Some(1234),
        };
        let bytes = p.encode_to_vec();
        let back = ProcessInfo::decode(&bytes[..]).expect("round-trip decode");
        assert_eq!(back.tps, Some(0.0), "explicit 0.0 is not absent");
        assert_eq!(back.prompt_tps, None, "unset field round-trips as None");
        assert_eq!(back.cache_hit_pct, None, "unset field round-trips as None");
        assert_eq!(back.last_obs_ms, Some(1234), "explicit value round-trips");

        // Unset last_obs_ms (an old tamad's frame) decodes to None.
        let mut p2 = p;
        p2.last_obs_ms = None;
        let bytes2 = p2.encode_to_vec();
        let back2 = ProcessInfo::decode(&bytes2[..]).expect("round-trip decode");
        assert_eq!(back2.last_obs_ms, None, "unset field round-trips as None");
    }
}
