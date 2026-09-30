pub mod api_keys;
pub mod backend_logs;
pub mod logs_api;
pub(crate) mod models;
mod pull;
mod system;
mod types;

#[cfg(test)]
mod system_tests;
#[cfg(test)]
mod tests;

pub use api_keys::{
    handle_tama_api_keys_create, handle_tama_api_keys_list, handle_tama_api_keys_revoke,
    handle_tama_api_keys_update,
};
pub use logs_api::*;
pub use models::{
    capitalize_first, generate_display_name, handle_litellm_model_info,
    handle_opencode_list_models, handle_tama_cancel_load, handle_tama_get_model,
    handle_tama_list_models, handle_tama_load_model, handle_tama_unload_model, ModelEntry,
    ModelLimit, OpencodeModelsResponse,
};
pub use pull::{
    enqueue_pull, handle_pull_job_stream, handle_tama_get_pull_job, handle_tama_pull_model,
    start_pull_from_queue,
};
pub use system::{
    handle_hf_list_quants, handle_system_metrics_stream, handle_tama_system_gpu_devices,
    handle_tama_system_gpu_devices_refresh, handle_tama_system_health, handle_tama_system_restart,
};
pub use types::{
    max_concurrent_pulls, ListModelsResponse, ListedModelResponse, ModelMutationResponse,
    ModelResponse, OkResponse, PullRequest, PullResponse, QuantEntry, QuantPullSpec,
};
