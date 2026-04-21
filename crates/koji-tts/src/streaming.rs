//! Streaming utilities for TTS audio output.

use base64::Engine;

use crate::config::AudioChunk;

/// Convert an AudioChunk to an SSE-formatted string.
///
/// Format: `event: audio\ndata: <base64>\n\n` (or with `event: end` for final chunks).
pub fn audio_chunk_to_sse(chunk: &AudioChunk) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(&chunk.data);
    if chunk.is_final {
        format!("event: audio\ndata: {}\n\nevent: end\n\n", encoded)
    } else {
        format!("event: audio\ndata: {}\n\n", encoded)
    }
}

/// Convert an error to an SSE-formatted error string.
pub fn error_chunk_to_sse(error: &anyhow::Error) -> String {
    format!("event: error\ndata: {}\n\n", error.to_string())
}
