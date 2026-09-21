use crate::config::resolve::tests::test_helpers as h;

/// Test that --metrics is injected for a GGUF llama.cpp backend when the
/// user did not set it (ADR-0014: the tamad scrapes the engine's
/// /metrics endpoint for inference stats).
#[test]
fn test_build_full_args_injects_metrics_for_llama_cpp() {
    let (_temp_dir, models_dir) = h::temp_model_dir();
    let config = h::sample_config(models_dir);

    let server = h::sample_server(|s| {
        s.backend = "llama.cpp".to_string();
        s.hf_format = None; // non-transformers (GGUF)
        s.args = vec!["--port".to_string(), "8080".to_string()];
    });

    let backend = h::sample_backend();

    let args = config
        .build_full_args(&server, &backend, None, &[])
        .expect("build_full_args failed");

    // --metrics injected exactly once
    let count = args.iter().filter(|a| *a == "--metrics").count();
    assert_eq!(
        count, 1,
        "Expected exactly one --metrics for llama.cpp backend, got {} in: {:?}",
        count, args
    );
}

/// Test that --metrics is NOT duplicated when the user already set it in
/// args (presence-checked: user args are never overridden).
#[test]
fn test_build_full_args_respects_user_metrics() {
    let (_temp_dir, models_dir) = h::temp_model_dir();
    let config = h::sample_config(models_dir);

    let server = h::sample_server(|s| {
        s.backend = "llama.cpp".to_string();
        s.hf_format = None; // non-transformers (GGUF)
        s.args = vec!["--metrics".to_string()];
    });

    let backend = h::sample_backend();

    let args = config
        .build_full_args(&server, &backend, None, &[])
        .expect("build_full_args failed");

    // Still exactly one --metrics (the user's), not a duplicate
    let count = args.iter().filter(|a| *a == "--metrics").count();
    assert_eq!(
        count, 1,
        "Expected exactly one --metrics when the user set it, got {} in: {:?}",
        count, args
    );
}

/// Test that --metrics is NOT injected for non-llama.cpp backends
/// (vLLM serves /metrics by default — no flag needed).
#[test]
fn test_build_full_args_no_metrics_for_non_llama_cpp() {
    let (_temp_dir, models_dir) = h::temp_model_dir();
    let config = h::sample_config(models_dir);

    let server = h::sample_server(|s| {
        s.backend = "vllm".to_string();
        s.hf_format = Some("transformers".to_string());
        s.model = None;
        s.quant = None;
        s.quants = std::collections::BTreeMap::new();
    });

    let backend = h::sample_backend();

    let args = config
        .build_full_args(&server, &backend, None, &[])
        .expect("build_full_args failed");

    assert!(
        !args.contains(&"--metrics".to_string()),
        "Expected no --metrics for non-llama.cpp backend, got: {:?}",
        args
    );
}
