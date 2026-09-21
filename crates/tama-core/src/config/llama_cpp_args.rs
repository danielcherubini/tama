//! Managed llama.cpp flags (ADR-0014).
//!
//! `--metrics` enables llama.cpp's Prometheus `/metrics` endpoint (off by
//! default upstream — the endpoint answers 501 without the flag). The
//! tamad scrapes that endpoint for inference stats (tps, prompt_tps,
//! cache_hit_pct, spec fields), so the proxy injects the flag for
//! llama.cpp backends. Presence-checked: a user-supplied `--metrics` is
//! never duplicated or overridden.

use crate::config::flag_name;

/// Append `--metrics` to grouped args if not already present.
///
/// Returns `true` if the flag was added.
pub fn ensure_metrics(grouped: &mut Vec<String>) -> bool {
    let present = grouped
        .iter()
        .any(|e| matches!(flag_name(e), Some("--metrics")));
    if present {
        false
    } else {
        grouped.push("--metrics".to_string());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_metrics;

    /// `--metrics` is appended when absent.
    #[test]
    fn test_ensure_metrics_adds_when_absent() {
        let mut grouped = vec!["--port".to_string(), "8080".to_string()];
        assert!(ensure_metrics(&mut grouped));
        assert_eq!(grouped, vec!["--port", "8080", "--metrics"]);
    }

    /// A grouped `--metrics` entry suppresses the injection.
    #[test]
    fn test_ensure_metrics_idempotent_when_present_grouped() {
        let mut grouped = vec!["--metrics".to_string()];
        assert!(!ensure_metrics(&mut grouped));
        assert_eq!(grouped, vec!["--metrics"]);
    }

    /// An inline `--metrics=...` entry suppresses the injection.
    #[test]
    fn test_ensure_metrics_idempotent_when_present_inline() {
        let mut grouped = vec!["--metrics=true".to_string()];
        assert!(!ensure_metrics(&mut grouped));
        assert_eq!(grouped, vec!["--metrics=true"]);
    }

    /// Pre-existing entries keep their order and values.
    #[test]
    fn test_ensure_metrics_does_not_touch_other_args() {
        let mut grouped = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-c".to_string(),
            "4096".to_string(),
        ];
        assert!(ensure_metrics(&mut grouped));
        assert_eq!(grouped, vec!["-m", "model.gguf", "-c", "4096", "--metrics"]);
    }
}
