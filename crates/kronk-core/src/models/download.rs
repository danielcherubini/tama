use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;


const MIN_CHUNK_SIZE: u64 = 5 * 1024 * 1024; // 5 MiB

/// Download a file using parallel HTTP Range requests.
/// Falls back to single-stream if Range is not supported.
pub async fn download_chunked(
    url: &str,
    dest: &Path,
    connections: usize,
) -> Result<u64> {
    let client = Client::new();

    // HEAD request to get Content-Length and check Range support
    let head = client
        .head(url)
        .send()
        .await
        .with_context(|| format!("HEAD request failed for {}", url))?;

    let total_size = head
        .content_length()
        .with_context(|| "Server did not return Content-Length")?;

    let accept_ranges = head
        .headers()
        .get("accept-ranges")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("none");

    let use_chunked = accept_ranges != "none" && total_size > MIN_CHUNK_SIZE;
    let num_connections = if use_chunked {
        connections.min((total_size / MIN_CHUNK_SIZE) as usize).max(1)
    } else {
        1
    };

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .unwrap()
            .progress_chars("=>-"),
    );

    if num_connections == 1 {
        // Single-stream fallback
        download_single(&client, url, dest, total_size, &pb).await?;
    } else {
        // Parallel chunked download
        download_parallel(&client, url, dest, total_size, num_connections, &pb).await?;
    }

    pb.finish_with_message("done");
    Ok(total_size)
}

async fn download_single(
    client: &Client,
    url: &str,
    dest: &Path,
    _total_size: u64,
    pb: &ProgressBar,
) -> Result<()> {
    use futures_util::StreamExt;

    let resp = client.get(url).send().await?;
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(dest).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    Ok(())
}

async fn download_parallel(
    client: &Client,
    url: &str,
    dest: &Path,
    total_size: u64,
    num_connections: usize,
    pb: &ProgressBar,
) -> Result<()> {
    use futures_util::StreamExt;

    let chunk_size = total_size / num_connections as u64;

    // Download each chunk to a temp file
    let tmp_dir = dest.parent().unwrap_or(Path::new("."));
    let mut handles = Vec::new();

    for i in 0..num_connections {
        let start = i as u64 * chunk_size;
        let end = if i == num_connections - 1 {
            total_size - 1
        } else {
            (i as u64 + 1) * chunk_size - 1
        };

        let client = client.clone();
        let url = url.to_string();
        let tmp_path = tmp_dir.join(format!(
            ".{}.part{}",
            dest.file_name().unwrap().to_string_lossy(),
            i
        ));
        let pb = pb.clone();

        let handle = tokio::spawn(async move {
            let range = format!("bytes={}-{}", start, end);
            let resp = client
                .get(&url)
                .header("Range", &range)
                .send()
                .await
                .with_context(|| format!("Range request failed for chunk {}", i))?;

            let mut stream = resp.bytes_stream();
            let mut file = tokio::fs::File::create(&tmp_path).await?;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk).await?;
                pb.inc(chunk.len() as u64);
            }

            file.flush().await?;
            Ok::<PathBuf, anyhow::Error>(tmp_path)
        });

        handles.push(handle);
    }

    // Wait for all chunks
    let mut chunk_paths = Vec::new();
    for handle in handles {
        let path = handle.await??;
        chunk_paths.push(path);
    }

    // Reassemble chunks into final file
    let mut dest_file = tokio::fs::File::create(dest).await?;
    for chunk_path in &chunk_paths {
        let chunk_data = tokio::fs::read(chunk_path).await?;
        dest_file.write_all(&chunk_data).await?;
        tokio::fs::remove_file(chunk_path).await.ok();
    }
    dest_file.flush().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_download_single_small_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("test.txt");

        // Use a small known file from HuggingFace (a config.json)
        let url = "https://huggingface.co/gpt2/resolve/main/config.json";
        let size = download_chunked(url, &dest, 1).await.unwrap();

        assert!(dest.exists());
        assert!(size > 0);
        assert_eq!(std::fs::metadata(&dest).unwrap().len(), size);
    }

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_download_parallel_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("test.bin");

        // Use a larger file to test parallel downloads
        let url = "https://huggingface.co/gpt2/resolve/main/merges.txt";
        let size = download_chunked(url, &dest, 4).await.unwrap();

        assert!(dest.exists());
        assert!(size > 0);
        assert_eq!(std::fs::metadata(&dest).unwrap().len(), size);
    }
}