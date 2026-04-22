//! Piper TTS engine wrapper.
//!
//! Wraps the `piper-rs` crate for text-to-speech synthesis.

use std::path::Path;
use std::pin::Pin;

use anyhow::{anyhow, Result};
use futures_core::Stream;

use crate::config::{AudioChunk, TtsRequest, VoiceInfo};
use crate::TtsEngine;

/// A loaded Piper TTS engine.
#[derive(Debug, Clone)]
pub struct PiperEngine {
    /// The ONNX model path for reference.
    #[allow(dead_code)]
    model_path: std::path::PathBuf,
    /// Voice metadata.
    voice: VoiceInfo,
}

impl PiperEngine {
    /// Create a new PiperEngine from a model file path or directory.
    pub async fn new(model_path: &Path) -> Result<Self> {
        // model_path may be a file (e.g., en_US-lessac-medium.onnx) or a directory
        let base = if model_path.is_dir() {
            model_path.to_path_buf()
        } else {
            model_path
                .parent()
                .ok_or_else(|| anyhow!("Failed to get parent of model path"))?
                .to_path_buf()
        };

        // Piper files are named {voice_id}.onnx and {voice_id}.onnx.json
        // Search for .onnx file in the model directory
        let mut onnx_files: Vec<_> = std::fs::read_dir(&base)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
            .collect();

        if onnx_files.is_empty() {
            return Err(anyhow!("No Piper ONNX model found in {}", base.display()));
        }

        let model_file = onnx_files.remove(0).path();
        let voice_name = model_file
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .trim_end_matches(".onnx")
            .to_string();

        // Try to load voice config from JSON if available
        let _config_file = base.join(format!("{}.onnx.json", voice_name));

        Ok(Self {
            model_path: model_path.to_path_buf(),
            voice: VoiceInfo {
                id: voice_name.clone(),
                name: format!("Piper - {}", voice_name),
                language: "en".to_string(),
                gender: Some("female".to_string()),
            },
        })
    }
}

#[async_trait::async_trait]
impl TtsEngine for PiperEngine {
    fn name(&self) -> &str {
        "piper"
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        vec![self.voice.clone()]
    }

    async fn synthesize(&self, req: &TtsRequest) -> Result<Vec<u8>> {
        // NOTE: piper-rs API may differ. This is a placeholder that
        // needs to be adapted to the actual crate API.
        //
        // The real implementation will call piper-rs's synthesize function
        // with the text and voice settings.

        if !self.model_path.exists() {
            return Err(anyhow!("Piper model file not found"));
        }

        // Placeholder: return empty audio for now — actual synthesis
        // requires adapting to piper-rs's real API surface.
        // TODO: Integrate with actual piper-rs synthesize call
        let _ = req;
        Ok(vec![])
    }

    async fn synthesize_stream(
        &self,
        req: &TtsRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<AudioChunk>> + Send>>> {
        // For non-streaming engines like Piper, return the full result as one chunk.
        let audio = self.synthesize(req).await?;
        let stream = async_stream::stream! {
            yield Ok(AudioChunk {
                data: audio,
                is_final: true,
            });
        };
        Ok(Box::pin(stream))
    }
}
