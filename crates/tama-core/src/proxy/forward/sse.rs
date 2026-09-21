use serde_json::Value as JsonValue;

/// Process a complete SSE line, rewriting the `model` field in JSON data lines.
pub(super) fn process_sse_line(line: &str, model_name: Option<&str>, out: &mut String) {
    if let Some(data_content) = line.strip_prefix("data: ") {
        let trimmed = data_content.trim_end();
        if trimmed == "[DONE]" {
            out.push_str(line);
        } else if let Ok(mut json_value) = serde_json::from_str::<JsonValue>(trimmed) {
            if let Some(name) = model_name {
                if !name.is_empty() {
                    json_value["model"] = JsonValue::String(name.to_string());
                }
            }
            out.push_str("data: ");
            out.push_str(
                &serde_json::to_string(&json_value).unwrap_or_else(|_| trimmed.to_string()),
            );
            out.push('\n');
        } else {
            out.push_str(line);
        }
    } else {
        // Comments, empty lines, and other lines pass through unchanged
        out.push_str(line);
    }
}
