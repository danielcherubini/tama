use wiremock::matchers::method;
use wiremock::matchers::path;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;

use super::super::utils::{extract_capabilities, fetch_capabilities_from_backend};

/// Valid response with supports_tool_calls: true → (true, false)
#[test]
fn test_extract_capabilities_tool_calls_true() {
    let body = r#"{
        "chat_template_caps": {
            "supports_tool_calls": true
        }
    }"#;
    let (tool_call, reasoning) = extract_capabilities(body.as_bytes());
    assert!(tool_call, "tool_call should be true");
    assert!(!reasoning, "reasoning should default to false");
}

/// Valid response with supports_preserve_reasoning: true → (true, true)
#[test]
fn test_extract_capabilities_preserve_reasoning_true() {
    let body = r#"{
        "chat_template_caps": {
            "supports_tool_calls": true,
            "supports_preserve_reasoning": true
        }
    }"#;
    let (tool_call, reasoning) = extract_capabilities(body.as_bytes());
    assert!(tool_call);
    assert!(
        reasoning,
        "reasoning should be true from supports_preserve_reasoning"
    );
}

/// Valid response with reasoning_format: "xml" → (true, true)
#[test]
fn test_extract_capabilities_reasoning_format_xml() {
    let body = r#"{
        "default_generation_settings": {
            "params": {
                "reasoning_format": "xml"
            }
        }
    }"#;
    let (tool_call, reasoning) = extract_capabilities(body.as_bytes());
    assert!(tool_call, "tool_call should default to true");
    assert!(
        reasoning,
        "reasoning should be true from reasoning_format != none"
    );
}

/// Missing chat_template_caps → (true, false) defaults
#[test]
fn test_extract_capabilities_missing_chat_template_caps() {
    let body = r#"{}"#;
    let (tool_call, reasoning) = extract_capabilities(body.as_bytes());
    assert!(tool_call, "tool_call should default to true");
    assert!(!reasoning, "reasoning should default to false");
}

/// Invalid JSON → (true, false) defaults
#[test]
fn test_extract_capabilities_invalid_json() {
    let body = b"not json at all";
    let (tool_call, reasoning) = extract_capabilities(body);
    assert!(tool_call, "tool_call should default to true on parse error");
    assert!(
        !reasoning,
        "reasoning should default to false on parse error"
    );
}

/// Empty body → (true, false) defaults
#[test]
fn test_extract_capabilities_empty_body() {
    let body = b"";
    let (tool_call, reasoning) = extract_capabilities(body);
    assert!(tool_call, "tool_call should default to true on empty body");
    assert!(
        !reasoning,
        "reasoning should default to false on empty body"
    );
}

// ── fetch_capabilities_from_backend HTTP failure tests ────────────────

/// Backend returns 500 → safe defaults (true, false).
#[tokio::test]
async fn test_fetch_capabilities_backend_500_returns_defaults() {
    let mock_server = MockServer::start().await;

    // Mock /props returning a 500 Internal Server Error
    Mock::given(method("GET"))
        .and(path("/props"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let (tool_call, reasoning) = fetch_capabilities_from_backend(&client, &mock_server.uri()).await;

    assert!(tool_call, "tool_call should default to true on 500 error");
    assert!(!reasoning, "reasoning should default to false on 500 error");
}

/// Backend unreachable (no mock) → safe defaults (true, false) with timeout.
#[tokio::test]
async fn test_fetch_capabilities_unreachable_backend_returns_defaults() {
    // Use a local address with no listener — the 3-second timeout prevents
    // hanging indefinitely.
    let unreachable_url = "http://127.0.0.1:19999";

    let client = reqwest::Client::new();
    let (tool_call, reasoning) = fetch_capabilities_from_backend(&client, unreachable_url).await;

    assert!(
        tool_call,
        "tool_call should default to true on unreachable backend"
    );
    assert!(
        !reasoning,
        "reasoning should default to false on unreachable backend"
    );
}
