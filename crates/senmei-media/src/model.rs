use std::fs;
use std::path::{Path, PathBuf};

use crate::downloader;
use crate::{Error, Result};

fn verify_checksum(actual: &str, expected: &str) -> Result<()> {
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(Error::command_failed(format!(
            "model checksum mismatch (expected {expected}, got {actual})"
        )))
    }
}

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

    fs::create_dir_all(dest_dir).map_err(|e| Error::command_failed(e.to_string()))?;
    let temp_dir = dest_dir.join(".tmp");
    fs::create_dir_all(&temp_dir).map_err(|e| Error::command_failed(e.to_string()))?;
    let temp = temp_dir.join(filename);

    downloader::fetch(url, &temp, on_progress)?;

    if let Some(expected) = expected_sha256 {
        let actual = downloader::sha256_hex(&temp)?;
        verify_checksum(&actual, expected)?;
    }

    let out = dest_dir.join(filename);
    fs::rename(&temp, &out).map_err(|e| Error::command_failed(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_match_and_mismatch() {
        let actual = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        verify_checksum(actual, actual).unwrap();
        assert!(verify_checksum(actual, "deadbeef").is_err());
    }
}
