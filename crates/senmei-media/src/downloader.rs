use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::{Error, Result};

/// Download `url` to `dest`, reporting progress as (downloaded, total).
pub fn fetch(url: &str, dest: &Path, on_progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| Error::Command(format!("download failed: {e}")))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut file = fs::File::create(dest).map_err(Error::from)?;
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(Error::from)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(Error::from)?;
        downloaded += n as u64;
        on_progress(downloaded, total);
    }
    Ok(())
}

/// Pull a single file out of a zip or tar.xz archive by path suffix.
pub fn extract_binary(archive: &Path, out: &Path, suffix: &str) -> Result<()> {
    let file = fs::File::open(archive).map_err(Error::from)?;
    let found = match archive.extension().and_then(|e| e.to_str()) {
        Some("zip") => {
            let mut zip = zip::ZipArchive::new(file).map_err(Error::from)?;
            for i in 0..zip.len() {
                let mut entry = zip.by_index(i).map_err(Error::from)?;
                if entry.name().ends_with(suffix) {
                    let mut f = fs::File::create(out).map_err(Error::from)?;
                    std::io::copy(&mut entry, &mut f).map_err(Error::from)?;
                    return Ok(());
                }
            }
            Err(Error::Command("binary not found in archive".into()))
        }
        _ => {
            let xz = liblzma::read::XzDecoder::new(file);
            let mut ar = tar::Archive::new(xz);
            for entry in ar.entries().map_err(Error::from)? {
                let mut entry = entry.map_err(Error::from)?;
                let name = entry
                    .path()
                    .map_err(Error::from)?
                    .to_string_lossy()
                    .into_owned();
                if name.ends_with(suffix) {
                    let mut f = fs::File::create(out).map_err(Error::from)?;
                    std::io::copy(&mut entry, &mut f).map_err(Error::from)?;
                    return Ok(());
                }
            }
            Err(Error::Command("binary not found in archive".into()))
        }
    };
    found
}

/// Hex SHA-256 of a file's contents.
pub fn sha256_hex(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// Case-insensitive SHA-256 comparison against an expected value.
pub fn verify_checksum(actual: &str, expected: &str) -> Result<()> {
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(Error::Command(format!(
            "checksum mismatch (expected {expected}, got {actual})"
        )))
    }
}

/// Download `url` into `temp_dir/<filename>`, verifying SHA-256 when `expected`
/// is provided. On mismatch the temp file is removed. Returns the temp path.
pub fn download_to_temp(
    url: &str,
    temp_dir: &Path,
    filename: &str,
    expected: Option<&str>,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<std::path::PathBuf> {
    fs::create_dir_all(temp_dir).map_err(Error::from)?;
    let temp = temp_dir.join(filename);
    fetch(url, &temp, on_progress)?;
    if let Some(expected) = expected {
        let actual = sha256_hex(&temp)?;
        if let Err(e) = verify_checksum(&actual, expected) {
            let _ = fs::remove_file(&temp);
            return Err(e);
        }
    }
    Ok(temp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        let path = std::env::temp_dir().join("senmei-sha256-test.txt");
        fs::write(&path, b"hello").unwrap();
        assert_eq!(sha256_hex(&path).unwrap(), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
        fs::remove_file(path).unwrap();
    }
}
