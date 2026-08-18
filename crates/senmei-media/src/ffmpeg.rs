use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::{downloader, process};
use crate::{Error, Result};

const ENV_PATH: &str = "SENMEI_FFMPEG";
const ARCHIVE_DIR: &str = "temp";
// Pinned BtbN LGPL builds (autobuild 2026-08-17, N-126188) — LGPL-only per the
// license policy (no GPL components, so no libx264; the encoder picks
// libopenh264 instead). Pinned to a dated tag so the SHA is stable; bump URL
// and SHA together.
const LINUX_LGPL_URL: &str = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-17-13-05/ffmpeg-N-126188-g426841da9d-linux64-lgpl.tar.xz";
const LINUX_LGPL_SHA256: &str = "0afc3d4d9728587ae1a4af1062c80f11dfdf82833b003b0f4fdf8027e9bf5c53";
const WINDOWS_LGPL_URL: &str = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-17-13-05/ffmpeg-N-126188-g426841da9d-win64-lgpl.zip";
const WINDOWS_LGPL_SHA256: &str = "fdf4fcb4797762e8b4cc3eccdedfedad1e4a345fe9bd8f6a44a20ebf57718c7a";

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
    let version = process::command_output(ffmpeg.to_str().unwrap_or("ffmpeg"), &["-version"]).and_then(|s| {
        s.lines().next().and_then(|l| {
            l.strip_prefix("ffmpeg version")
                .map(|v| v.trim().trim_end_matches("Copyright").trim().to_string())
        })
    });

    if version.is_none() {
        return FfmpegInfo::default();
    }

    let encoders = process::command_output(ffmpeg.to_str().unwrap_or("ffmpeg"), &["-hide_banner", "-encoders"])
        .map(|s| parse_caps(&s))
        .unwrap_or_default();
    let decoders = process::command_output(ffmpeg.to_str().unwrap_or("ffmpeg"), &["-hide_banner", "-decoders"])
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

fn archive() -> Option<(&'static str, &'static str)> {
    match std::env::consts::OS {
        "linux" if std::env::consts::ARCH == "x86_64" => Some((LINUX_LGPL_URL, LINUX_LGPL_SHA256)),
        "windows" if std::env::consts::ARCH == "x86_64" => {
            Some((WINDOWS_LGPL_URL, WINDOWS_LGPL_SHA256))
        }
        _ => None,
    }
}

pub fn download(data_dir: &Path, mut on_progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    let (url, sha256) = archive().ok_or_else(|| {
        Error::Command("no prebuilt FFmpeg for this platform yet (macOS: TODO)".into())
    })?;
    log::info!("ffmpeg download from {url}");

    let archive = downloader::download_to_temp(
        url,
        &data_dir.join(ARCHIVE_DIR),
        url.rsplit('/').next().unwrap_or("ffmpeg-archive"),
        Some(sha256),
        &mut on_progress,
    )?;

    let bin_dir = data_dir.join("bin");
    fs::create_dir_all(&bin_dir).map_err(Error::from)?;
    let bin_name = if std::env::consts::OS == "windows" {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let out = bin_dir.join(bin_name);
    downloader::extract_binary(&archive, &out, &format!("/bin/{bin_name}"))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&out, fs::Permissions::from_mode(0o755))
            .map_err(Error::from)?;
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
        if !process::command_output("ffmpeg", &["-version"]).is_some() {
            eprintln!("ffmpeg not found, skipping");
            return;
        }
        let info = probe(&PathBuf::from("ffmpeg"));
        assert!(info.found);
        assert!(info.version.is_some());
        assert!(info.encoders.contains(&"libx264".to_string()));
    }
}
