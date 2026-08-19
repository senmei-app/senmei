//! Transport-agnostic Senmei service. No Tauri, no webview — only
//! `senmei-media` / `senmei-ml` (+ `senmei-pipeline` for render). Every
//! adapter (MCP, later HTTP) calls into here, so license/confirm gates live
//! once.

use std::path::{Path, PathBuf};

/// App data dir (`$XDG_DATA_HOME/senmei`), same convention as the GUI.
pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".local")
                .join("share")
        });
    base.join("senmei")
}

/// Resolve the models dir: dev repo checkout first, then the data dir
/// (packaged install). Mirrors the GUI's resolution order.
pub fn models_dir() -> PathBuf {
    let anchored = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    if anchored.join("metadata.json").is_file() {
        return anchored;
    }
    let data_models = data_dir().join("models");
    if data_models.join("metadata.json").is_file() {
        return data_models;
    }
    data_models
}

/// Load the model registry from the resolved models dir.
pub fn load_registry() -> Result<(senmei_ml::Registry, PathBuf), String> {
    let dir = models_dir();
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&dir).map_err(|e| e.to_string())?;
    Ok((registry, dir))
}

/// Resolved system/portable FFmpeg binary.
pub fn ffmpeg() -> PathBuf {
    senmei_media::resolve(&data_dir())
}

pub fn ffmpeg_status() -> senmei_media::FfmpegInfo {
    senmei_media::probe_ffmpeg(&ffmpeg())
}

pub fn probe_video(input: &str) -> Result<senmei_media::VideoInfo, String> {
    let ffprobe = senmei_media::ffprobe_next_to(&ffmpeg());
    senmei_media::probe(&ffprobe, Path::new(input)).map_err(|e| e.to_string())
}

pub fn list_models() -> Vec<senmei_ml::ModelMetadata> {
    load_registry()
        .map(|(registry, _)| registry.models().to_vec())
        .unwrap_or_default()
}
