use std::fs;
use std::path::{Path, PathBuf};

use crate::downloader;
use crate::{Error, Result};

/// Download a model weight file into `dest_dir`, verifying its SHA-256.
/// Returns the installed path. Downloads to a temp file first so a failed
/// or mismatched download never leaves a partial/corrupt weight behind.
pub fn download_model(
    url: &str,
    dest_dir: &Path,
    filename: &str,
    expected_sha256: Option<&str>,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    log::info!("model download: {url} -> {}", dest_dir.join(filename).display());

    let temp = downloader::download_to_temp(url, &dest_dir.join(".tmp"), filename, expected_sha256, on_progress)?;
    let out = dest_dir.join(filename);
    fs::rename(&temp, &out).map_err(|e| Error::command_failed(e.to_string()))?;
    Ok(out)
}
