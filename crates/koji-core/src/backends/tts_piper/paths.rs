use std::path::{Path, PathBuf};

/// Piper voice ID used for versioned directory naming.
pub const VOICE_ID: &str = "en_US-lessac-medium";

/// Return the base directory for tts_piper models: `<backends_dir>/tts_piper`.
pub fn base_dir(base: &Path) -> PathBuf {
    base.join("tts_piper")
}

/// Return the versioned model directory: `<backends_dir>/tts_piper/<VOICE_ID>/`.
pub fn models_dir(base: &Path) -> PathBuf {
    base_dir(base).join(VOICE_ID)
}

/// Return the path to the Piper ONNX model file.
pub fn model_file(base: &Path) -> PathBuf {
    models_dir(base).join(format!("{}.onnx", VOICE_ID))
}

/// Return the path to the Piper JSON config file (contains voice settings).
pub fn config_file(base: &Path) -> PathBuf {
    models_dir(base).join(format!("{}.onnx.json", VOICE_ID))
}
