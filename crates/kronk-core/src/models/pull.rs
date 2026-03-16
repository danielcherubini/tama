use crate::models::card::ModelCard;
use anyhow::{Context, Result};
use hf_hub::api::tokio::Api;
use std::path::PathBuf;
use tokio::sync::OnceCell;

static HF_API: OnceCell<Api> = OnceCell::const_new();

/// Get or create the shared HuggingFace API client.
async fn hf_api() -> Result<&'static Api> {
    HF_API
        .get_or_try_init(|| async {
            Api::new().context("Failed to initialise HuggingFace API client")
        })
        .await
}

/// Information about a GGUF file in a HuggingFace repo.
#[derive(Debug, Clone)]
pub struct RemoteGguf {
    /// Filename, e.g. "OmniCoder-8B-Q4_K_M.gguf"
    pub filename: String,
    /// Inferred quant type from filename, e.g. "Q4_K_M"
    pub quant: Option<String>,
}

/// List GGUF files available in a HuggingFace model repository.
/// Returns the resolved repo_id (which may differ from input if `-GGUF` was appended)
/// and the list of available GGUF files.
///
/// Auto-resolves repos: if `repo_id` doesn't end with `-GGUF` and the initial
/// fetch finds no GGUF files (or the repo doesn't exist), retries with `-GGUF` appended.
pub async fn list_gguf_files(repo_id: &str) -> Result<(String, Vec<RemoteGguf>)> {
    let api = hf_api().await?;

    // Try the repo_id as given first
    let candidates = if repo_id.to_uppercase().ends_with("-GGUF") {
        vec![repo_id.to_string()]
    } else {
        vec![repo_id.to_string(), format!("{}-GGUF", repo_id)]
    };

    let mut last_error: Option<String> = None;

    for candidate in &candidates {
        let repo = api.model(candidate.clone());
        match repo.info().await {
            Ok(info) => {
                let ggufs: Vec<RemoteGguf> = info
                    .siblings
                    .into_iter()
                    .filter(|s| s.rfilename.ends_with(".gguf"))
                    .map(|s| {
                        let quant = infer_quant_from_filename(&s.rfilename);
                        RemoteGguf {
                            filename: s.rfilename,
                            quant,
                        }
                    })
                    .collect();

                if !ggufs.is_empty() {
                    return Ok((candidate.clone(), ggufs));
                }
                // Repo exists but no GGUFs — try next candidate
                last_error = Some(format!(
                    "'{}' exists but contains no .gguf files",
                    candidate
                ));
            }
            Err(e) => {
                last_error = Some(format!("'{}': {}", candidate, e));
                continue;
            }
        }
    }

    let detail = last_error.unwrap_or_else(|| "unknown error".to_string());
    anyhow::bail!(
        "No GGUF files found. Tried: {}\nLast error: {}",
        candidates.join(", "),
        detail
    )
}

/// Result of downloading a GGUF file.
pub struct DownloadResult {
    /// Local path to the file (in the model directory)
    pub path: PathBuf,
    /// File size in bytes (from the hf-hub cache, always accurate)
    pub size_bytes: u64,
}

/// Download a specific GGUF file from a HuggingFace repo to the given model directory.
/// Returns the local path and file size.
/// Downloads directly via reqwest with timeouts and retry (bypasses hf-hub's downloader).
pub async fn download_gguf(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
) -> Result<DownloadResult> {
    use futures_util::StreamExt;
    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::Write;
    use std::time::Duration;

    // Ensure the full directory path exists (sync — confirmed works on Windows)
    let dest_path = dest_dir.join(filename);
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(30))
        .build()
        .context("Failed to create HTTP client")?;

    // Try HEAD to get file size for skip-if-exists check
    let head_size = client
        .head(&url)
        .send()
        .await
        .ok()
        .and_then(|r| r.content_length());

    // Check if already downloaded with matching size
    if let Some(total) = head_size {
        if dest_path.exists() {
            if let Ok(meta) = std::fs::metadata(&dest_path) {
                if meta.len() == total {
                    return Ok(DownloadResult {
                        path: dest_path,
                        size_bytes: total,
                    });
                }
            }
            // Size mismatch — re-download
            std::fs::remove_file(&dest_path).ok();
        }
    }

    // Download with progress bar and retry
    const MAX_RETRIES: u32 = 3;
    let mut attempt = 0;
    let mut downloaded: u64 = 0;
    let mut total_size: u64 = 0;
    let mut pb: Option<ProgressBar> = None;

    loop {
        attempt += 1;

        // Resume from where we left off
        let mut request = client.get(&url);
        if downloaded > 0 {
            request = request.header("Range", format!("bytes={}-", downloaded));
        }

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) if attempt <= MAX_RETRIES => {
                println!(
                    "  Download stalled (attempt {}/{}), retrying... ({})",
                    attempt, MAX_RETRIES, e
                );
                tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
                continue;
            }
            Err(e) => {
                if let Some(ref p) = pb {
                    p.finish_and_clear();
                }
                return Err(e).with_context(|| {
                    format!(
                        "Failed to download '{}' after {} attempts",
                        filename, attempt
                    )
                });
            }
        };

        if !response.status().is_success() && response.status().as_u16() != 206 {
            if attempt <= MAX_RETRIES {
                println!(
                    "  Server returned {}, retrying ({}/{})...",
                    response.status(),
                    attempt,
                    MAX_RETRIES
                );
                tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
                continue;
            }
            if let Some(ref p) = pb {
                p.finish_and_clear();
            }
            anyhow::bail!(
                "Download failed with status {} after {} attempts",
                response.status(),
                attempt
            );
        }

        // Create progress bar from GET response Content-Length (available after redirect)
        if pb.is_none() {
            if let Some(content_len) = response.content_length() {
                total_size = content_len + downloaded; // account for resumed bytes
                let p = ProgressBar::new(total_size);
                p.set_style(
                    ProgressStyle::default_bar()
                        .template("{msg} [{elapsed_precise}] [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})")
                        .unwrap_or_else(|_| ProgressStyle::default_bar())
                        .progress_chars("=>-"),
                );
                p.set_message(filename.to_string());
                p.set_position(downloaded);
                pb = Some(p);
            } else {
                // No Content-Length at all — use a bytes counter
                let p = ProgressBar::new_spinner();
                p.set_style(
                    ProgressStyle::default_spinner()
                        .template("{msg} [{elapsed_precise}] {bytes} ({bytes_per_sec})")
                        .unwrap_or_else(|_| ProgressStyle::default_spinner()),
                );
                p.set_message(filename.to_string());
                pb = Some(p);
            }
        }

        // Open file for append (resume) or create — use sync I/O for Windows compat
        let mut file = if downloaded > 0 {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&dest_path)
                .with_context(|| format!("Failed to open {} for append", dest_path.display()))?
        } else {
            std::fs::File::create(&dest_path)
                .with_context(|| format!("Failed to create {}", dest_path.display()))?
        };

        let mut stream = response.bytes_stream();
        let mut stream_failed = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    file.write_all(&chunk)
                        .with_context(|| format!("Failed to write to {}", dest_path.display()))?;
                    downloaded += chunk.len() as u64;
                    if let Some(ref p) = pb {
                        p.set_position(downloaded);
                    }
                }
                Err(e) if attempt <= MAX_RETRIES => {
                    if let Some(ref p) = pb {
                        p.suspend(|| {
                            println!(
                                "  Stream interrupted at {:.1} MiB (attempt {}/{}), resuming... ({})",
                                downloaded as f64 / 1_048_576.0,
                                attempt,
                                MAX_RETRIES,
                                e
                            );
                        });
                    }
                    stream_failed = true;
                    break;
                }
                Err(e) => {
                    if let Some(ref p) = pb {
                        p.finish_and_clear();
                    }
                    return Err(e.into());
                }
            }
        }

        file.flush().ok();
        drop(file);

        if stream_failed {
            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
            continue;
        }

        // Stream ended cleanly — check if download is complete
        if total_size == 0 || downloaded >= total_size {
            break;
        }
        // Unexpected short read — retry
        if attempt <= MAX_RETRIES {
            if let Some(ref p) = pb {
                p.suspend(|| {
                    println!(
                        "  Short read ({:.1}/{:.1} MiB), retrying ({}/{})...",
                        downloaded as f64 / 1_048_576.0,
                        total_size as f64 / 1_048_576.0,
                        attempt,
                        MAX_RETRIES
                    );
                });
            }
            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt - 1))).await;
            continue;
        }
        anyhow::bail!(
            "Download incomplete: got {} of {} bytes",
            downloaded,
            total_size
        );
    }

    if let Some(ref p) = pb {
        p.finish_and_clear();
    }

    let final_size = std::fs::metadata(&dest_path)
        .map(|m| m.len())
        .unwrap_or(downloaded);

    Ok(DownloadResult {
        path: dest_path,
        size_bytes: final_size,
    })
}

const MODELCARDS_BASE_URL: &str =
    "https://raw.githubusercontent.com/danielcherubini/kronk/main/modelcards";

/// Try to fetch a community model card from the kronk repository.
///
/// Attempts several name variants derived from the repo_id:
/// 1. Exact: `{company}/{model}.toml` (e.g. `Tesslate/OmniCoder-9B-GGUF.toml`)
/// 2. Strip `-GGUF` suffix: `Tesslate/OmniCoder-9B.toml`
/// 3. Strip `-gguf` suffix (lowercase)
///
/// Returns `None` silently on network errors or 404s.
pub async fn fetch_community_card(repo_id: &str) -> Option<ModelCard> {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    if parts.len() != 2 {
        return None;
    }
    let (company, model) = (parts[0], parts[1]);

    // Build candidate names: exact, then stripped variants
    let mut candidates = vec![model.to_string()];
    for suffix in ["-GGUF", "-gguf", "-Gguf"] {
        if let Some(stripped) = model.strip_suffix(suffix) {
            candidates.push(stripped.to_string());
        }
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;

    for name in &candidates {
        let url = format!("{}/{}/{}.toml", MODELCARDS_BASE_URL, company, name);
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(body) = resp.text().await {
                    if let Ok(card) = toml::from_str::<ModelCard>(&body) {
                        return Some(card);
                    }
                }
            }
        }
    }

    None
}

/// Try to infer the quantisation type from a GGUF filename.
/// Common patterns: "Model-Q4_K_M.gguf", "model.Q8_0.gguf", "model-q4_k_m.gguf"
pub fn infer_quant_from_filename(filename: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf")?;

    // Ordered longest-first so "Q4_K_M" matches before "Q4_K"
    let quant_patterns = [
        "IQ2_XXS", "IQ3_XXS", "IQ1_S", "IQ1_M", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XS", "IQ3_S",
        "IQ3_M", "IQ4_XS", "IQ4_NL", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M",
        "Q4_K_L", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q4_0",
        "Q4_1", "Q5_0", "Q5_1", "Q6_0", "Q8_0", "Q8_1", "F16", "F32", "BF16",
    ];

    let stem_upper = stem.to_uppercase();
    for pattern in &quant_patterns {
        if stem_upper.ends_with(pattern)
            || stem_upper.contains(&format!("-{}", pattern))
            || stem_upper.contains(&format!(".{}", pattern))
            || stem_upper.contains(&format!("_{}", pattern))
        {
            return Some(pattern.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_quant_q4_k_m() {
        assert_eq!(
            infer_quant_from_filename("OmniCoder-8B-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn test_infer_quant_q8_0() {
        assert_eq!(
            infer_quant_from_filename("model-Q8_0.gguf"),
            Some("Q8_0".to_string())
        );
    }

    #[test]
    fn test_infer_quant_lowercase() {
        assert_eq!(
            infer_quant_from_filename("model-q4_k_m.gguf"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn test_infer_quant_f16() {
        assert_eq!(
            infer_quant_from_filename("model-F16.gguf"),
            Some("F16".to_string())
        );
    }

    #[test]
    fn test_infer_quant_none() {
        assert_eq!(infer_quant_from_filename("model.gguf"), None);
    }

    #[test]
    fn test_infer_quant_dot_separator() {
        assert_eq!(
            infer_quant_from_filename("Llama-3.2-1B-Instruct.Q6_K.gguf"),
            Some("Q6_K".to_string())
        );
    }

    #[test]
    fn test_infer_quant_iq() {
        assert_eq!(
            infer_quant_from_filename("model-IQ4_NL.gguf"),
            Some("IQ4_NL".to_string())
        );
    }
}
