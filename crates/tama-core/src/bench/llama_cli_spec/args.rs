//! CLI-argument construction for the llama-cli binary with speculative decoding.
//!
//! Isolated from the orchestrator so the flag-encoding rules can be exercised
//! as pure unit tests without needing a live llama-cli process.

use super::{SpecBenchConfig, SpecType};

/// Build command-line arguments for a speculative decoding run.
///
/// Uses `--spec-default` (llama.cpp's built-in n-gram spec decoding) plus
/// `--draft-max` / `--draft-min` to tune draft behavior. `--single-turn` makes
/// llama-cli exit after one generation instead of entering interactive mode.
///
/// Note: llama.cpp b8893+ removed the old `--spec-type`, `--spec-ngram-size-*`,
/// and `--draft-modalism` flags. All spec decoding is now configured via
/// `--spec-default` combined with `--draft-max` / `--draft-min`. The spec_type,
/// ngram_n, and ngram_m parameters are accepted for API compatibility but have
/// no effect on the generated args.
///
/// Always included: `-m`, `-p`, `-n`, `--n-gpu-layers` (if set), `-fa`,
/// `--single-turn`, `--no-display-prompt`, `--spec-default`, `--draft-min`,
/// `--draft-max`.
///
/// The prompt is generated via `crate::bench::build_prompt(512)`.
pub(super) fn build_args(
    config: &SpecBenchConfig,
    _spec_type: SpecType,
    draft_max: u32,
    _ngram_n: Option<u32>,
    _ngram_m: Option<u32>,
) -> Vec<String> {
    let mut args = Vec::new();

    // Model path
    args.push("-m".to_string());
    args.push(config.model_path.to_string_lossy().to_string());

    // Prompt (512 tokens)
    let prompt = crate::bench::build_prompt(512);
    args.push("-p".to_string());
    args.push(prompt);

    // Generation length
    args.push("-n".to_string());
    args.push(config.gen_tokens.to_string());

    // GPU layers (if specified)
    if let Some(ngl) = config.ngl {
        args.push("--n-gpu-layers".to_string());
        args.push(ngl.to_string());
    }

    // Flash attention
    args.push("-fa".to_string());
    args.push(if config.flash_attn { "1" } else { "0" }.to_string());

    // Non-interactive: exit after single generation.
    args.push("--single-turn".to_string());
    // Suppress prompt echoing.
    args.push("--no-display-prompt".to_string());

    // Spec-decoding flags — llama.cpp b8893+ uses --spec-default for all n-gram
    // variants, tuned via --draft-max / --draft-min.
    let draft_min = (draft_max / 2).max(1);
    args.push("--spec-default".to_string());
    args.push("--draft-min".to_string());
    args.push(draft_min.to_string());
    args.push("--draft-max".to_string());
    args.push(draft_max.to_string());

    args
}

/// Build command-line arguments for a baseline run (no speculative decoding).
///
/// Includes `--single-turn` and `--no-display-prompt` for non-interactive batch
/// mode. Omits all `--spec-*` and `--draft-*` flags entirely. Still includes
/// model, prompt, generation length, GPU layers, and flash attention settings.
pub(super) fn build_baseline_args(config: &SpecBenchConfig) -> Vec<String> {
    let mut args = Vec::new();

    // Model path
    args.push("-m".to_string());
    args.push(config.model_path.to_string_lossy().to_string());

    // Prompt (512 tokens)
    let prompt = crate::bench::build_prompt(512);
    args.push("-p".to_string());
    args.push(prompt);

    // Generation length
    args.push("-n".to_string());
    args.push(config.gen_tokens.to_string());

    // GPU layers (if specified)
    if let Some(ngl) = config.ngl {
        args.push("--n-gpu-layers".to_string());
        args.push(ngl.to_string());
    }

    // Flash attention
    args.push("-fa".to_string());
    args.push(if config.flash_attn { "1" } else { "0" }.to_string());

    // Non-interactive: exit after single generation.
    args.push("--single-turn".to_string());
    // Suppress prompt echoing.
    args.push("--no-display-prompt".to_string());

    args
}

/// Find the value following a flag in an argument list.
#[cfg(test)]
fn find_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .map(|idx| args.get(idx + 1).map(|s| s.as_str()).unwrap_or(""))
}

/// Check if a flag is present in an argument list.
#[cfg(test)]
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> SpecBenchConfig {
        SpecBenchConfig {
            model_path: std::path::PathBuf::from("/test/model.gguf"),
            spec_types: vec![],
            draft_max_values: vec![],
            ngram_n_values: vec![],
            ngram_m_values: vec![],
            ngram_min_hits: 1,
            gen_tokens: 256,
            runs: 3,
            ngl: None,
            flash_attn: true,
        }
    }

    #[test]
    fn test_baseline_args_model() {
        let config = make_config();
        let args = build_baseline_args(&config);
        assert_eq!(find_arg(&args, "-m"), Some("/test/model.gguf"));
    }

    #[test]
    fn test_baseline_args_generation_length() {
        let config = make_config();
        let args = build_baseline_args(&config);
        assert_eq!(find_arg(&args, "-n"), Some("256"));
    }

    #[test]
    fn test_baseline_args_gpu_layers() {
        let mut config = make_config();
        config.ngl = Some(99);
        let args = build_baseline_args(&config);
        assert_eq!(find_arg(&args, "--n-gpu-layers"), Some("99"));
    }

    #[test]
    fn test_baseline_args_flash_attn_enabled() {
        let mut config = make_config();
        config.flash_attn = true;
        let args = build_baseline_args(&config);
        assert_eq!(find_arg(&args, "-fa"), Some("1"));
    }

    #[test]
    fn test_baseline_args_flash_attn_disabled() {
        let mut config = make_config();
        config.flash_attn = false;
        let args = build_baseline_args(&config);
        assert_eq!(find_arg(&args, "-fa"), Some("0"));
    }

    #[test]
    fn test_baseline_args_single_turn() {
        let config = make_config();
        let args = build_baseline_args(&config);
        assert!(has_flag(&args, "--single-turn"));
    }

    #[test]
    fn test_baseline_args_no_display_prompt() {
        let config = make_config();
        let args = build_baseline_args(&config);
        assert!(has_flag(&args, "--no-display-prompt"));
    }

    #[test]
    fn test_baseline_args_no_spec_flags() {
        let config = make_config();
        let args = build_baseline_args(&config);
        assert!(!has_flag(&args, "--spec-default"));
        assert!(!has_flag(&args, "--draft-max"));
        assert!(!has_flag(&args, "--spec-type"));
    }

    #[test]
    fn test_baseline_args_has_prompt() {
        let config = make_config();
        let args = build_baseline_args(&config);
        let prompt_arg = find_arg(&args, "-p").unwrap();
        assert!(!prompt_arg.is_empty(), "Prompt should not be empty");
    }

    #[test]
    fn test_spec_args_spec_default() {
        let config = make_config();
        let args = build_args(&config, SpecType::NgramSimple, 16, None, None);
        assert!(has_flag(&args, "--spec-default"));
    }

    #[test]
    fn test_spec_args_draft_max() {
        let config = make_config();
        let args = build_args(&config, SpecType::NgramSimple, 32, None, None);
        assert_eq!(find_arg(&args, "--draft-max"), Some("32"));
    }

    #[test]
    fn test_spec_args_draft_min() {
        let config = make_config();
        // draft_min = draft_max / 2
        let args = build_args(&config, SpecType::NgramSimple, 32, None, None);
        assert_eq!(find_arg(&args, "--draft-min"), Some("16"));
    }

    #[test]
    fn test_spec_args_draft_min_minimum_one() {
        let config = make_config();
        // draft_min = max(draft_max / 2, 1) = max(2, 1) = 2
        let args = build_args(&config, SpecType::NgramSimple, 4, None, None);
        assert_eq!(find_arg(&args, "--draft-min"), Some("2"));
    }

    #[test]
    fn test_spec_args_single_turn() {
        let config = make_config();
        let args = build_args(&config, SpecType::NgramSimple, 16, None, None);
        assert!(has_flag(&args, "--single-turn"));
    }

    #[test]
    fn test_spec_args_no_spec_type() {
        let config = make_config();
        let args = build_args(&config, SpecType::NgramSimple, 16, None, None);
        // --spec-type is removed in b8893+
        assert!(!has_flag(&args, "--spec-type"));
    }

    #[test]
    fn test_spec_args_no_old_flags() {
        let config = make_config();
        let args = build_args(&config, SpecType::NgramMod, 16, Some(12), Some(48));
        // Old flags should not be present in b8893+
        assert!(!has_flag(&args, "--spec-type"));
        assert!(!has_flag(&args, "--spec-ngram-size-n"));
        assert!(!has_flag(&args, "--spec-ngram-size-m"));
        assert!(!has_flag(&args, "--draft-modalism"));
    }

    #[test]
    fn test_spec_args_gpu_layers() {
        let mut config = make_config();
        config.ngl = Some(99);
        let args = build_args(&config, SpecType::NgramSimple, 16, None, None);
        assert_eq!(find_arg(&args, "--n-gpu-layers"), Some("99"));
    }

    #[test]
    fn test_spec_args_flash_attn() {
        let mut config = make_config();
        config.flash_attn = true;
        let args = build_args(&config, SpecType::NgramSimple, 16, None, None);
        assert_eq!(find_arg(&args, "-fa"), Some("1"));
    }

    #[test]
    fn test_spec_args_has_prompt() {
        let config = make_config();
        let args = build_args(&config, SpecType::NgramSimple, 16, None, None);
        let prompt_arg = find_arg(&args, "-p").unwrap();
        assert!(!prompt_arg.is_empty(), "Prompt should not be empty");
    }
}
