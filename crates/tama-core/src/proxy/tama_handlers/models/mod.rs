mod handlers;
mod litellm;
mod opencode;
mod utils;

/// Internal capability flags for a loaded backend.
#[derive(Debug, Clone, Copy, Default)]
struct ModelCapabilities {
    tool_call: bool,
    reasoning: bool,
}

// Re-export only the public handlers — internal helpers stay private.
pub use handlers::{
    handle_tama_cancel_load, handle_tama_get_model, handle_tama_list_models,
    handle_tama_load_model, handle_tama_unload_model,
};
pub use litellm::handle_litellm_model_info;
pub use opencode::handle_opencode_list_models;
pub use utils::{
    capitalize_first, generate_display_name, ModelEntry, ModelLimit, OpencodeModelsResponse,
};

// Re-exported so the /v1/models handler tree (proxy::handlers::models) can
// build the canonical reasoning_options field without duplicating the logic.
pub(crate) use utils::reasoning_options_from_levels;

#[cfg(test)]
mod tests {
    mod cancel;
    mod capabilities;
    mod helpers;
    mod litellm;
    mod model_handlers;
    mod opencode;
}
