use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

const ENV_PATH: &str = "SENMEI_FFMPEG";
const ARCHIVE_DIR: &str = "temp";
// SHA-256 of the pinned BtbN build; update when bumping the download.
const FFMPEG_SHA256: &str = "e0ae9c7c76dd029457ac54d8d6f95742bd398c8ed5ac434ad313a1e99136278e";

fn system_ffmpeg_works() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn resolve(data_dir: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os(ENV_PATH) {
        let p = PathBuf::from(p);
        if p.exists() {
            return p;
        }
    }
    if system_ffmpeg_works() {
        return PathBuf::from("ffmpeg");
    }
    let bundled = data_dir.join("bin").join("ffmpeg");
    if bundled.exists() {
        return bundled;
    }
    PathBuf::from("ffmpeg")
}

#[derive(Debug, Clone, Default, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct FfmpegInfo {
    pub found: bool,
    pub path: Option<String>,
    pub version: Option<String>,
    pub encoders: Vec<String>,
    pub decoders: Vec<String>,
}

fn run_args(ffmpeg: &Path, args: &[&str]) -> Option<String> {
    Command::new(ffmpeg)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
}

fn parse_caps(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|l| !l.starts_with("Encoders") && !l.starts_with("Decoders"))
        .filter_map(|l| {
            let mut t = l.split_whitespace();
            let _ = t.next();
            t.next().map(String::from)
        })
        .collect()
}

pub fn probe(ffmpeg: &Path) -> FfmpegInfo {
    let version = run_args(ffmpeg, &["-version"]).and_then(|s| {
        s.lines().next().and_then(|l| {
            l.strip_prefix("ffmpeg version")
                .map(|v| v.trim().trim_end_matches("Copyright").trim().to_string())
        })
    });

    if version.is_none() {
        return FfmpegInfo::default();
    }

    let encoders = run_args(ffmpeg, &["-hide_banner", "-encoders"])
        .map(|s| parse_caps(&s))
        .unwrap_or_default();
    let decoders = run_args(ffmpeg, &["-hide_banner", "-decoders"])
        .map(|s| parse_caps(&s))
        .unwrap_or_default();

    FfmpegInfo {
        found: true,
        path: Some(ffmpeg.to_string_lossy().into_owned()),
        version,
        encoders,
        decoders,
    }
}

fn archive_url() -> Option<&'static str> {
    match std::env::consts::OS {
        "linux" if std::env::consts::ARCH == "x86_64" => Some(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
        ),
        "windows" if std::env::consts::ARCH == "x86_64" => Some(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        ),
        _ => None,
    }
}

fn fetch_file(url: &str, dest: &Path, on_progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let resp = ureq::get(url).call().map_err(|e| Error::command_failed(format!("download failed: {e}")))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut file = fs::File::create(dest).map_err(|e| Error::command_failed(e.to_string()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::command_failed(e.to_string()))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| Error::command_failed(e.to_string()))?;
        downloaded += n as u64;
        on_progress(downloaded, total);
    }
    Ok(())
}

fn extract_binary(
    archive: &Path,
    out: &Path,
    bin_name: &str,
) -> Result<()> {
    let file = fs::File::open(archive).map_err(|e| Error::command_failed(e.to_string()))?;
    let target = format!("/bin/{bin_name}");
    let found = match archive.extension().and_then(|e| e.to_str()) {
        Some("zip") => {
            let mut zip = zip::ZipArchive::new(file)
                .map_err(|e| Error::command_failed(e.to_string()))?;
            for i in 0..zip.len() {
                let mut entry = zip
                    .by_index(i)
                    .map_err(|e| Error::command_failed(e.to_string()))?;
                if entry.name().ends_with(&target) {
                    let mut f = fs::File::create(out)
                        .map_err(|e| Error::command_failed(e.to_string()))?;
                    std::io::copy(&mut entry, &mut f)
                        .map_err(|e| Error::command_failed(e.to_string()))?;
                    return Ok(());
                }
            }
            Err(Error::command_failed("ffmpeg binary not found in archive".into()))
        }
        _ => {
            let xz = xz2::read::XzDecoder::new(file);
            let mut ar = tar::Archive::new(xz);
            for entry in ar
                .entries()
                .map_err(|e| Error::command_failed(e.to_string()))?
            {
                let mut entry = entry.map_err(|e| Error::command_failed(e.to_string()))?;
                let name = entry
                    .path()
                    .map_err(|e| Error::command_failed(e.to_string()))?
                    .to_string_lossy()
                    .into_owned();
                if name.ends_with(&target) {
                    let mut f = fs::File::create(out)
                        .map_err(|e| Error::command_failed(e.to_string()))?;
                    std::io::copy(&mut entry, &mut f)
                        .map_err(|e| Error::command_failed(e.to_string()))?;
                    return Ok(());
                }
            }
            Err(Error::command_failed("ffmpeg binary not found in archive".into()))
        }
    };
    found
}

fn sha256_hex(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub fn download(data_dir: &Path, mut on_progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    let url = archive_url().ok_or_else(|| {
        Error::command_failed("no prebuilt FFmpeg for this platform yet (macOS: TODO)".into())
    })?;
    log::info!("ffmpeg download from {url}");

    let temp = data_dir.join(ARCHIVE_DIR);
    fs::create_dir_all(&temp).map_err(|e| Error::command_failed(e.to_string()))?;
    let archive = temp.join(
        url.rsplit('/').next().unwrap_or("ffmpeg-archive"),
    );

    fetch_file(url, &archive, &mut on_progress)?;

    let actual = sha256_hex(&archive)?;
    if !actual.eq_ignore_ascii_case(FFMPEG_SHA256) {
        let _ = fs::remove_file(&archive);
        return Err(Error::command_failed(format!(
            "ffmpeg checksum mismatch (expected {FFMPEG_SHA256}, got {actual}); update FFMPEG_SHA256 for the new build"
        )));
    }

    let bin_dir = data_dir.join("bin");
    fs::create_dir_all(&bin_dir).map_err(|e| Error::command_failed(e.to_string()))?;
    let bin_name = if std::env::consts::OS == "windows" {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let out = bin_dir.join(bin_name);
    extract_binary(&archive, &out, bin_name)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&out, fs::Permissions::from_mode(0o755))
            .map_err(|e| Error::command_failed(e.to_string()))?;
    }

    let _ = fs::remove_file(&archive);
    log::info!("ffmpeg installed to {}", out.display());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_caps_extracts_names() {
        let out = "Encoders:\n V..... libx264              H.264 (codec h264)\n V..... libx265              HEVC (codec hevc)\n A..... libopus              Opus\n";
        let names = parse_caps(out);
        assert_eq!(names, vec!["libx264", "libx265", "libopus"]);
    }

    #[test]
    fn probe_system_ffmpeg() {
        if !run_args(&PathBuf::from("ffmpeg"), &["-version"]).is_some() {
            eprintln!("ffmpeg not found, skipping");
            return;
        }
        let info = probe(&PathBuf::from("ffmpeg"));
        assert!(info.found);
        assert!(info.version.is_some());
        assert!(info.encoders.contains(&"libx264".to_string()));
    }
}
