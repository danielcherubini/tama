use crate::config::BackendConfig;

mod aliases;
mod basic;
mod batch_args;
mod context_np;
mod gpu_device;
mod kv_cache_types;
mod metrics_flag;
mod path_resolution;
mod server_resolution;
mod spec_decoding;
mod test_helpers;
mod transformers_format;
mod unified_slots;

fn make_test_config(llama_cpp_path: Option<&str>) -> crate::config::Config {
    let mut config = crate::config::Config::default();
    if let Some(path) = llama_cpp_path {
        config.backends.insert(
            "llama_cpp".to_string(),
            BackendConfig {
                path: Some(path.to_string()),
                version: None,
                gpu_variant: None,
            },
        );
    } else {
        config.backends.insert(
            "llama_cpp".to_string(),
            BackendConfig {
                path: None,
                version: None,
                gpu_variant: None,
            },
        );
    }
    config
}
