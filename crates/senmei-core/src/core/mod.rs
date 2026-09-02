//! Transport-agnostic Senmei core. No Tauri, no webview — only
//! `senmei-media` / `senmei-ml` (+ `senmei-pipeline` for render). Every
//! adapter (MCP, HTTP, GUI) calls into here, so license/confirm gates live
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

/// Media extensions the thumbnail command will serve (defense-in-depth: the
/// IPC/HTTP surface must not read arbitrary files off disk).
const THUMBNAIL_EXTS: &[&str] = &[
    "mp4", "mkv", "mov", "webm", "avi", "m4v", "mpg", "mpeg", "ts", "flv", "wmv", "m2ts", "jpg",
    "jpeg", "png", "webp",
];

/// Small JPEG thumbnail of `input` as a `data:image/jpeg;base64,…` URL plus
/// the source probe (transport-agnostic — works over Tauri IPC and HTTP
/// alike; the probe lets the caller skip a second `probe_video` call).
/// Rejects non-media paths so the command can't read arbitrary files.
pub fn thumbnail(input: &str, max_w: u32) -> Result<(String, senmei_media::VideoInfo), String> {
    use base64::Engine as _;
    let path = Path::new(input);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !THUMBNAIL_EXTS.iter().any(|e| *e == ext) {
        return Err(format!("thumbnail: unsupported media type .{ext}"));
    }
    let thumb = senmei_media::thumbnail(&ffmpeg(), path, max_w).map_err(|e| e.to_string())?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&thumb.jpeg);
    Ok((format!("data:image/jpeg;base64,{b64}"), thumb.info))
}

/// List registry models, annotating `downloaded` from the on-disk weight files.
pub fn list_models() -> Vec<senmei_ml::ModelMetadata> {
    match load_registry() {
        Ok((registry, dir)) => registry
            .models()
            .iter()
            .cloned()
            .map(|mut m| {
                m.downloaded = m
                    .weights
                    .as_ref()
                    .and_then(|w| w.first())
                    .map(|w| dir.join(w).is_file())
                    .unwrap_or(false);
                m
            })
            .collect(),
        Err(e) => {
            log::warn!("list_models: registry load failed: {e}");
            Vec::new()
        }
    }
}

/// Recursively list videos under `dir` (batch folder scan over HTTP).
pub fn scan_folder(dir: &str) -> Result<Vec<String>, String> {
    senmei_media::find_videos(std::path::Path::new(dir), true)
        .map(|v| {
            v.into_iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect()
        })
        .map_err(|e| e.to_string())
}

mod compare;
mod config;
#[cfg(feature = "render")]
mod download;
#[cfg(feature = "render")]
mod render;
mod suggest;

pub use compare::compare_sample;
pub use config::{settings_schema, FilterConfig, RenderConfig};
#[cfg(feature = "render")]
pub use download::download_model;
#[cfg(feature = "render")]
pub use render::{
    cancel_render, confirm_render, engine_for_model, propose_render, render, render_sample,
    render_status, RenderOpts, RenderProgress, RenderStatus, StepTimingInfo,
};
pub use suggest::suggest_pipeline;
