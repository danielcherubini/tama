pub mod download;
pub mod paths;

use super::{BackendInfo, BackendRegistry, BackendSource, BackendType, ProgressSink};
use anyhow::Context;

/// Install the Piper TTS backend: download default voice, register in registry.
pub async fn install_tts_piper(
    registry: &mut BackendRegistry,
    progress: Box<dyn ProgressSink>,
) -> anyhow::Result<()> {
    let p = std::sync::Arc::from(progress);

    // Download the default voice model and config
    download::download_piper_model(&p).await?;

    // Register in the backend registry
    let base_dir = crate::backends::backends_dir()?;
    let info = BackendInfo {
        name: "tts_piper".to_string(),
        backend_type: BackendType::TtsPiper,
        version: "0.0.1".to_string(),
        path: paths::models_dir(&base_dir),
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
        .with_context(|| "Failed to register Piper backend")?;

    Ok(())
}

/// Verify the installed Piper backend has all required files.
pub fn verify_tts_piper(info: &BackendInfo) -> anyhow::Result<()> {
    let model = paths::model_file(&info.path);
    if !model.exists() {
        return Err(anyhow::anyhow!(
            "Piper model file not found: {}",
            model.display()
        ));
    }

    let config = paths::config_file(&info.path);
    if !config.exists() {
        return Err(anyhow::anyhow!(
            "Piper config file not found: {}",
            config.display()
        ));
    }

    Ok(())
}
