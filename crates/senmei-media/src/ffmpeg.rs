use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::{downloader, process};
use crate::{Error, Result};

const ENV_PATH: &str = "SENMEI_FFMPEG";
const ARCHIVE_DIR: &str = "temp";
// Pinned BtbN LGPL builds (autobuild 2026-09-01, N-126386) — LGPL-only per the
// license policy (no GPL components, so no libx264; the encoder picks
// libopenh264 instead). Pinned to a dated tag so the SHA is stable; bump URL
// and SHA together — BtbN purges old autobuild tags (~2 weeks).
const LINUX_LGPL_URL: &str = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-01-13-13/ffmpeg-N-126386-gc27482a18d-linux64-lgpl.tar.xz";
const LINUX_LGPL_SHA256: &str = "8fee5342057184e7ec32a40beed9b069fef6af1ec9c82c18725a2f040fd02abb";
const WINDOWS_LGPL_URL: &str = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-01-13-13/ffmpeg-N-126386-gc27482a18d-win64-lgpl.zip";
const WINDOWS_LGPL_SHA256: &str =
    "b14a959412a27e2404019d72179b75f9a4dee1656aba0d042f4febb5ccb8e392";

pub const fn ffmpeg_bin_name() -> &'static str {
    if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    }
}

pub const fn ffprobe_bin_name() -> &'static str {
    if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    }
}

/// `ffprobe` binary next to the resolved `ffmpeg` (portable builds ship both).
pub fn ffprobe_next_to(ffmpeg: &Path) -> PathBuf {
    match ffmpeg.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(ffprobe_bin_name()),
        _ => PathBuf::from(ffprobe_bin_name()),
    }
}

fn system_ffmpeg_works() -> bool {
    Command::new(ffmpeg_bin_name())
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
        return PathBuf::from(ffmpeg_bin_name());
    }
    let bundled = data_dir.join("bin").join(ffmpeg_bin_name());
    if bundled.exists() {
        return bundled;
    }
    PathBuf::from(ffmpeg_bin_name())
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
    let version = process::command_output(ffmpeg.to_str().unwrap_or("ffmpeg"), &["-version"])
        .and_then(|s| {
            s.lines().next().and_then(|l| {
                l.strip_prefix("ffmpeg version")
                    .map(|v| v.trim().trim_end_matches("Copyright").trim().to_string())
            })
        });

    if version.is_none() {
        return FfmpegInfo::default();
    }

    let encoders = process::command_output(
        ffmpeg.to_str().unwrap_or("ffmpeg"),
        &["-hide_banner", "-encoders"],
    )
    .map(|s| parse_caps(&s))
    .unwrap_or_default();
    let decoders = process::command_output(
        ffmpeg.to_str().unwrap_or("ffmpeg"),
        &["-hide_banner", "-decoders"],
    )
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
        Error::Command(if cfg!(target_os = "macos") {
            "no LGPL-compatible portable FFmpeg for macOS; install it via Homebrew (brew install ffmpeg)"
                .into()
        } else {
            format!(
                "no prebuilt FFmpeg for {}-{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        })
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
    let ffmpeg_bin = bin_dir.join(ffmpeg_bin_name());
    downloader::extract_binary(
        &archive,
        &ffmpeg_bin,
        &format!("/bin/{}", ffmpeg_bin_name()),
    )?;
    let ffprobe_bin = bin_dir.join(ffprobe_bin_name());
    downloader::extract_binary(
        &archive,
        &ffprobe_bin,
        &format!("/bin/{}", ffprobe_bin_name()),
    )?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for bin in [&ffmpeg_bin, &ffprobe_bin] {
            fs::set_permissions(bin, fs::Permissions::from_mode(0o755)).map_err(Error::from)?;
        }
    }

    let _ = fs::remove_file(&archive);
    log::info!("ffmpeg installed to {}", ffmpeg_bin.display());
    Ok(ffmpeg_bin)
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
