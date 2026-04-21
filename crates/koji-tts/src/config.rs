/// A TTS synthesis request.
#[derive(Debug, Clone, Default)]
pub struct TtsRequest {
    /// Text to synthesize.
    pub text: String,
    /// Voice ID to use (e.g., "af_sky" for Kokoro).
    pub voice: String,
    /// Speech speed: 0.5 = half speed, 1.0 = normal, 2.0 = double speed.
    pub speed: f32,
    /// Output audio format.
    pub format: AudioFormat,
}

/// Supported audio output formats.
#[derive(Debug, Clone, Default)]
pub enum AudioFormat {
    #[default]
    Mp3,
    Wav,
    Ogg,
}

/// Metadata about a single available voice.
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    /// Unique voice identifier (e.g., "af_sky").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Language code (e.g., "en").
    pub language: String,
    /// Gender if known.
    pub gender: Option<String>,
}

/// A chunk of audio data from streaming synthesis.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw audio bytes.
    pub data: Vec<u8>,
    /// Whether this is the final chunk of the stream.
    pub is_final: bool,
}
