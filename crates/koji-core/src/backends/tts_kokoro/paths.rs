use std::path::{Path, PathBuf};

/// Kokoro model version used for versioned directory naming.
pub const MODEL_VERSION: &str = "kokoro-v1_0";

/// Return the base directory for tts_kokoro models: `<backends_dir>/tts_kokoro`.
pub fn base_dir(base: &Path) -> PathBuf {
    base.join("tts_kokoro")
}

/// Return the versioned model directory: `<backends_dir>/tts_kokoro/<MODEL_VERSION>/`.
pub fn models_dir(base: &Path) -> PathBuf {
    base_dir(base).join(MODEL_VERSION)
}

/// Return the path to the Kokoro PyTorch model file.
pub fn model_file(base: &Path) -> PathBuf {
    models_dir(base).join("kokoro-v1_0.pth")
}

/// Return the path to the voices directory.
pub fn voices_dir(base: &Path) -> PathBuf {
    models_dir(base).join("voices")
}

/// Return the path to a specific voice PyTorch file.
pub fn voice_file(base: &Path, name: &str) -> PathBuf {
    voices_dir(base).join(format!("{name}.pt"))
}
