use super::*;

#[test]
fn test_process_sse_line_rewrites_model_in_data() {
    let mut out = String::new();
    process_sse_line(
        "data: {\"model\": \"backend-model\", \"choices\": []}",
        Some("user-model"),
        &mut out,
    );
    // serde_json serializes without spaces by default
    assert!(out.contains("\"model\""), "output: {}", out);
    assert!(out.contains("user-model"), "output: {}", out);
}

#[test]
fn test_process_sse_line_skips_rewrite_when_none() {
    let mut out = String::new();
    process_sse_line(
        "data: {\"model\": \"backend-model\", \"choices\": []}",
        None,
        &mut out,
    );
    // Model should NOT be rewritten when model_name is None
    assert!(out.contains("backend-model"), "output: {}", out);
    assert!(!out.contains("user-model"), "output: {}", out);
}

#[test]
fn test_process_sse_line_passes_done_unchanged() {
    let mut out = String::new();
    process_sse_line("data: [DONE]", Some("any-model"), &mut out);
    // DONE is pushed as-is (no trailing newline added by this function)
    assert_eq!(out, "data: [DONE]");
}

#[test]
fn test_process_sse_line_passes_comment_unchanged() {
    let mut out = String::new();
    process_sse_line(": heartbeat", Some("any-model"), &mut out);
    assert_eq!(out, ": heartbeat");
}

#[test]
fn test_process_sse_line_passes_empty_line_unchanged() {
    let mut out = String::new();
    process_sse_line("", Some("any-model"), &mut out);
    assert_eq!(out, "");
}

#[test]
fn test_process_sse_line_handles_invalid_json() {
    let mut out = String::new();
    process_sse_line("data: not valid json {", Some("any-model"), &mut out);
    assert_eq!(out, "data: not valid json {");
}

#[test]
fn test_process_sse_line_handles_non_data_lines() {
    let mut out = String::new();
    process_sse_line("event: message", Some("any-model"), &mut out);
    assert_eq!(out, "event: message");
}

#[test]
fn test_process_sse_line_multiline_buffer() {
    // A single call to process_sse_line processes one line at a time.
    // Lines without trailing newline are not processed as complete SSE lines.
    let mut out = String::new();
    // First line with newline - should be processed
    process_sse_line("data: {\"model\": \"a\"}\n", Some("user"), &mut out);
    assert!(out.contains("user"), "output: {}", out);
}
