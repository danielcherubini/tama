pub mod download;
pub mod paths;

use super::{BackendInfo, BackendRegistry, BackendSource, BackendType, ProgressSink};
use anyhow::{anyhow, Context};

/// Install the Kokoro TTS backend: download model + voices, register in registry.
pub async fn install_tts_kokoro(
    registry: &mut BackendRegistry,
    progress: Box<dyn ProgressSink>,
) -> anyhow::Result<()> {
    let p = std::sync::Arc::from(progress);

    // Download model and voices
    download::download_all(&p).await?;

    // Register in the backend registry — path points to model file (not directory)
    let base_dir = crate::backends::backends_dir()?;
    let info = BackendInfo {
        name: "tts_kokoro".to_string(),
        backend_type: BackendType::TtsKokoro,
        version: "0.0.1".to_string(),
        path: paths::model_file(&base_dir),
        installed_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64),
        gpu_type: None,
        source: Some(BackendSource::Prebuilt {
            version: "0.0.1".to_string(),
        }),
    };

    registry
        .add(info)
        .with_context(|| "Failed to register Kokoro backend")?;

    Ok(())
}

/// Verify the installed Kokoro backend has all required files.
pub fn verify_tts_kokoro(info: &BackendInfo) -> anyhow::Result<()> {
    // info.path is now the model file path
    if !info.path.exists() {
        return Err(anyhow::anyhow!(
            "Kokoro model file not found: {}",
            info.path.display()
        ));
    }

    // Voices are in the same directory as the model file
    let voices = info
        .path
        .parent()
        .ok_or_else(|| anyhow!("Failed to get parent of model path"))?
        .join("voices");
    if !voices.is_dir() {
        return Err(anyhow::anyhow!(
            "Kokoro voices directory not found: {}",
            voices.display()
        ));
    }

    // Check that at least one voice file exists
    let voice_count = std::fs::read_dir(&voices)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "pt" || ext == "onnx")
        })
        .count();

    if voice_count == 0 {
        return Err(anyhow::anyhow!(
            "No Kokoro voice files found in {}",
            voices.display()
        ));
    }

    Ok(())
}
