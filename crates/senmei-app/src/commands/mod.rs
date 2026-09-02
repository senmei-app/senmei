use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::Manager;

use crate::models::load_registry;
use crate::preview::{read_frame_inner, FrameMeta, FramePixels};
use crate::store;
use senmei_core::core;

/// Shared cancellation flag for the active render (set by `cancel_render`).
static CANCEL_RENDER: OnceLock<Arc<AtomicBool>> = OnceLock::new();
/// Shared pause flag for the active render (set by `pause_render`).
static PAUSE_RENDER: OnceLock<Arc<AtomicBool>> = OnceLock::new();

#[tauri::command]
#[specta::specta]
pub fn health_check() -> String {
    "ok".to_string()
}

#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: u64,
}

#[tauri::command]
#[specta::specta]
pub fn get_ffmpeg_status() -> senmei_media::FfmpegInfo {
    core::ffmpeg_status()
}

#[tauri::command]
#[specta::specta]
pub async fn download_ffmpeg(on_progress: Channel<DownloadProgress>) -> Result<String, String> {
    log::info!("downloading portable ffmpeg");
    let dir = store::data_dir();
    tauri::async_runtime::spawn_blocking(move || {
        senmei_media::download(&dir, |downloaded, total| {
            let _ = on_progress.send(DownloadProgress { downloaded, total });
        })
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[specta::specta]
pub fn list_models() -> Vec<senmei_ml::ModelMetadata> {
    core::list_models()
}

/// One model's on-disk weight info (size + sha256 check).
#[derive(serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ModelFileInfo {
    pub id: String,
    pub file: String,
    pub size: u64,
    pub verified: bool,
}

/// List installed weight files with size + sha256 verification.
#[tauri::command]
#[specta::specta]
pub fn model_files() -> Vec<ModelFileInfo> {
    let Ok((registry, dir)) = load_registry() else {
        return Vec::new();
    };
    registry
        .models()
        .iter()
        .filter_map(|m| {
            let file = m.weights.as_ref()?.first()?;
            let path = dir.join(file);
            let Ok(meta) = std::fs::metadata(&path) else {
                return None;
            };
            if !meta.is_file() {
                return None;
            }
            let verified = match m.sha256.as_deref() {
                Some(expected) => senmei_media::sha256_hex(&path)
                    .map(|a| a.eq_ignore_ascii_case(expected))
                    .unwrap_or(false),
                None => true,
            };
            Some(ModelFileInfo {
                id: m.id.clone(),
                file: file.clone(),
                size: meta.len(),
                verified,
            })
        })
        .collect()
}

/// Delete a model's weight files to free disk space.
#[tauri::command]
#[specta::specta]
pub fn delete_model_file(id: String) -> Result<(), String> {
    let (registry, dir) = load_registry()?;
    let Some(model) = registry.models().iter().find(|m| m.id == id) else {
        return Err(format!("model {id} not found"));
    };
    for w in model.weights.as_deref().unwrap_or_default() {
        let path = dir.join(w);
        if path.is_file() {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
            log::info!("delete_model_file: removed {}", path.display());
        }
    }
    Ok(())
}

/// Download a model's weights (`.pth`, sha256-verified when pinned) and
/// convert them to the app's f16 `.bpk` burnpack.
#[tauri::command]
#[specta::specta]
pub async fn download_model(
    model_id: String,
    on_progress: Channel<DownloadProgress>,
) -> Result<String, String> {
    log::info!("download_model: {model_id}");
    tauri::async_runtime::spawn_blocking(move || {
        core::download_model(&model_id, |d, t| {
            let _ = on_progress.send(DownloadProgress {
                downloaded: d,
                total: t,
            });
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[specta::specta]
pub fn probe_video(
    input: String,
    app: tauri::AppHandle,
) -> Result<senmei_media::VideoInfo, String> {
    log::info!("probe_video: {input}");
    // Let the webview load this file via the asset protocol (native <video>).
    let _ = app
        .state::<tauri::scope::Scopes>()
        .allow_file(std::path::Path::new(&input));
    core::probe_video(&input)
}

/// JPEG data-URL + source probe from the `thumbnail` command — one round trip
/// so the library tile doesn't need a second `probe_video` call.
#[derive(serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ThumbnailResult {
    pub data: String,
    pub info: senmei_media::VideoInfo,
}

/// Small JPEG thumbnail (data URL) for the media library tiles.
#[tauri::command]
#[specta::specta]
pub fn thumbnail(input: String, max_w: Option<u32>) -> Result<ThumbnailResult, String> {
    log::info!("thumbnail: {input}");
    let (data, info) = core::thumbnail(&input, max_w.unwrap_or(160))?;
    Ok(ThumbnailResult { data, info })
}

/// Probe content and suggest a default pipeline (content-aware defaults).
/// Lives in `senmei-core` so Tauri and HTTP share one implementation; returns
/// a JSON string (`{ anime, steps: [...] }`) for specta's TS export.
#[tauri::command]
#[specta::specta]
pub fn suggest_pipeline(input: String) -> Result<String, String> {
    core::suggest_pipeline(&input)
}

#[tauri::command]
#[specta::specta]
pub async fn read_frame(
    input: String,
    position_ms: f64,
    on_meta: Channel<FrameMeta>,
    on_frame: Channel<FramePixels>,
) -> Result<(), String> {
    log::info!("read_frame: {input} @ {position_ms:.0}ms");
    // Decode off the main thread so the UI never freezes per frame.
    let frame = tauri::async_runtime::spawn_blocking(move || read_frame_inner(&input, position_ms))
        .await
        .map_err(|e| e.to_string())??;
    // Meta (JSON) first, then the raw RGB24 bytes (ArrayBuffer on the JS side)
    // — no base64 over the IPC.
    on_meta
        .send(FrameMeta {
            width: frame.width,
            height: frame.height,
        })
        .map_err(|e| e.to_string())?;
    on_frame
        .send(FramePixels(frame.data))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Keep only the `keep` newest sample render files in `dir` (deletes older
/// video files so the sample folder never grows unbounded).
#[tauri::command]
#[specta::specta]
pub fn prune_samples(dir: String, keep: usize) -> Result<(), String> {
    store::ensure_within_data_dir(std::path::Path::new(&dir))?;
    let keep = keep.max(1);
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| {
                    matches!(
                        x.to_string_lossy().to_lowercase().as_str(),
                        "mkv" | "mp4" | "webm" | "mov"
                    )
                })
                .unwrap_or(false)
        })
        .collect();
    // Oldest first by modification time, so `keep` always retains the newest
    // files regardless of filename (range-tagged names don't sort chronologically).
    files.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH)
    });
    for p in files.iter().take(files.len().saturating_sub(keep)) {
        let _ = std::fs::remove_file(p);
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn import_folder(dir: String) -> Result<Vec<String>, String> {
    let found =
        senmei_media::find_videos(std::path::Path::new(&dir), false).map_err(|e| e.to_string())?;
    Ok(found
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
}

/// Recursively collect all videos under `dir` (batch folder processing).
#[tauri::command]
#[specta::specta]
pub fn scan_folder(dir: String) -> Result<Vec<String>, String> {
    core::scan_folder(&dir)
}

#[tauri::command]
#[specta::specta]
pub fn get_settings() -> store::Settings {
    store::load_settings()
}

#[tauri::command]
#[specta::specta]
pub fn save_settings(settings: store::Settings) -> Result<(), String> {
    store::save_settings(&settings)
}

#[tauri::command]
#[specta::specta]
pub fn backend_info() -> senmei_ml::BackendInfo {
    senmei_ml::backend_info()
}

#[tauri::command]
#[specta::specta]
pub fn hardware_status() -> crate::resources::HardwareSnapshot {
    crate::resources::sample_hardware()
}

#[tauri::command]
#[specta::specta]
pub fn list_projects() -> Vec<store::ProjectEntry> {
    store::list_projects()
}

#[tauri::command]
#[specta::specta]
pub fn create_project(name: String) -> Result<String, String> {
    store::create_project(&name)
}

#[tauri::command]
#[specta::specta]
pub fn delete_project(path: String) -> Result<(), String> {
    store::delete_project(&path)
}

#[tauri::command]
#[specta::specta]
pub fn export_project(src: String, dest: String) -> Result<(), String> {
    store::ensure_within_data_dir(std::path::Path::new(&src))?;
    store::export_project(&src, &dest)
}

/// Package logs + system info into a `.tar.xz` (diagnose export).
#[tauri::command]
#[specta::specta]
pub fn export_diagnostics(dest: String) -> Result<(), String> {
    crate::diagnostics::export(&dest)
}

#[tauri::command]
#[specta::specta]
pub fn open_project(file: String) -> Result<String, String> {
    store::open_project(&file)
}

#[tauri::command]
#[specta::specta]
pub fn load_project_settings(path: String) -> store::ProjectSettings {
    store::load_project_settings(&PathBuf::from(path))
}

#[tauri::command]
#[specta::specta]
pub fn save_project_settings(path: String, settings: store::ProjectSettings) -> Result<(), String> {
    store::save_project_settings(&PathBuf::from(path), &settings)
}

#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct RenderProgress {
    pub frames_processed: u64,
    pub total_frames: u64,
    /// Per-step ms/frame + fps; empty during the run, populated on the final
    /// event once the render finishes (the FPS benchmark report).
    pub steps: Vec<StepTimingInfo>,
}

/// One pipeline step's timing (FPS benchmark).
#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct StepTimingInfo {
    pub name: String,
    pub frames: u64,
    pub ms_per_frame: f64,
    pub fps: f64,
}

/// Optional reference filter steps (denoise/deblur/dedup) for a render.
#[derive(Clone, Debug, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterParams {
    pub denoise_radius: Option<u32>,
    /// Optional ML denoiser model (DRUNet); when set the denoise step runs the
    /// model instead of the CPU box blur.
    pub denoise_model_id: Option<String>,
    pub deblur_amount: Option<f32>,
    /// Optional ML deblur model (NAFNet); when set the deblur step runs the
    /// model instead of the CPU unsharp mask.
    pub deblur_model_id: Option<String>,
    pub dedup_threshold: Option<f32>,
    /// Free-form FFmpeg `-vf` filter graph applied per frame (frame-preserving
    /// 1:1 only; runs after the reference/ML filters).
    pub ffmpeg_filter: Option<String>,
}

/// All render knobs in one struct (specta caps command arity at 10 args).
#[derive(Clone, Debug, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderConfig {
    pub scale: Option<u32>,
    pub model_id: Option<String>,
    pub resize: Option<f32>,
    pub filter: Option<FilterParams>,
    /// Optional ML decompress model (RealPLKSR 1×); runs a scale-1 pass
    /// (de-artifact/de-JPEG/de-H.264) ahead of the step chain.
    pub decompress_model_id: Option<String>,
    pub output_resize: Option<f32>,
    pub fps_multiplier: Option<u32>,
    pub interp_model: Option<String>,
    /// Pre-split ffmpeg args (the frontend parses the custom field).
    pub ffmpeg_args: Option<Vec<String>>,
    /// HDR→SDR tonemapping: "auto" | "always" | "off" (default auto).
    pub tonemap: Option<String>,
    /// Render only a time range (start ms, end ms; None end = to the end).
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

#[tauri::command]
#[specta::specta]
pub async fn render(
    input: String,
    output: String,
    config: RenderConfig,
    on_progress: Channel<RenderProgress>,
) -> Result<String, String> {
    log::info!("render start: {input} -> {output} (config {config:?})");
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let settings = store::load_settings();
        let cfg = core::RenderConfig {
            input,
            output,
            scale: config.scale,
            model_id: config.model_id,
            decompress_model_id: config.decompress_model_id,
            resize: config.resize,
            filter: config.filter.map(filter_to_core),
            output_resize: config.output_resize,
            fps_multiplier: config.fps_multiplier,
            interp_model: config.interp_model,
            ffmpeg_args: config.ffmpeg_args,
            tonemap: config.tonemap,
            start_ms: config.start_ms,
            end_ms: config.end_ms,
        };
        let opts = core::RenderOpts {
            tile_size: settings.tile_size.unwrap_or(0),
            pipeline_depth: settings.pipeline_depth.unwrap_or(0) as usize,
            backend: settings.backend.unwrap_or_default(),
            gpu_index: settings.gpu_index.unwrap_or(0),
            cancel: Some(
                CANCEL_RENDER
                    .get_or_init(|| Arc::new(AtomicBool::new(false)))
                    .clone(),
            ),
            pause: Some(
                PAUSE_RENDER
                    .get_or_init(|| Arc::new(AtomicBool::new(false)))
                    .clone(),
            ),
        };
        let (processed, total) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let (p_ref, t_ref) = (processed.clone(), total.clone());
        let progress_tx = on_progress.clone();
        let steps = core::render(&cfg, &opts, move |p| {
            p_ref.store(p.frames_processed, Ordering::Relaxed);
            t_ref.store(p.total_frames, Ordering::Relaxed);
            let _ = on_progress.send(RenderProgress {
                frames_processed: p.frames_processed,
                total_frames: p.total_frames,
                steps: Vec::new(),
            });
        })?;
        // Final event carries the per-step benchmark (only steps that ran).
        let steps: Vec<StepTimingInfo> = steps
            .into_iter()
            .map(|t| StepTimingInfo {
                name: t.name,
                frames: t.frames,
                ms_per_frame: t.ms_per_frame,
                fps: t.fps,
            })
            .collect();
        let _ = progress_tx.send(RenderProgress {
            frames_processed: processed.load(Ordering::Relaxed),
            total_frames: total.load(Ordering::Relaxed),
            steps,
        });
        Ok("ok".to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Map the IPC filter params onto the shared core filter config (same fields).
fn filter_to_core(f: FilterParams) -> core::FilterConfig {
    core::FilterConfig {
        denoise_radius: f.denoise_radius,
        denoise_model_id: f.denoise_model_id,
        deblur_amount: f.deblur_amount,
        deblur_model_id: f.deblur_model_id,
        dedup_threshold: f.dedup_threshold,
        ffmpeg_filter: f.ffmpeg_filter,
    }
}

/// Abort the active render (the pipeline checks the flag between frames).
#[tauri::command]
#[specta::specta]
pub fn cancel_render() {
    if let Some(c) = CANCEL_RENDER.get() {
        c.store(true, Ordering::Relaxed);
        log::info!("render cancelled (flag set)");
    }
}

/// Pause/resume the active render (the pipeline waits between frames).
#[tauri::command]
#[specta::specta]
pub fn pause_render(paused: bool) {
    if let Some(p) = PAUSE_RENDER.get() {
        p.store(paused, Ordering::Relaxed);
    }
}

/// Return `path` if free, else `{stem}_2.{ext}`, `{stem}_3.{ext}`, … first
/// free name, so batch renders never overwrite an existing file.
#[tauri::command]
#[specta::specta]
pub fn unique_path(path: String) -> Result<String, String> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Ok(path);
    }
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    let ext = p
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_default();
    let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
    for n in 2..10_000u32 {
        let name = if ext.is_empty() {
            format!("{stem}_{n}")
        } else {
            format!("{stem}_{n}.{ext}")
        };
        let candidate = parent.join(&name);
        if !candidate.exists() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
    }
    Err("no free output name found".into())
}

/// Persist the batch queue state (JSON) so a crash doesn't lose it.
#[tauri::command]
#[specta::specta]
pub fn save_batch_queue(state: String) -> Result<(), String> {
    let path = store::data_dir().join("batch-queue.json");
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&path, state).map_err(|e| e.to_string())
}

/// Load the persisted batch queue state, if any.
#[tauri::command]
#[specta::specta]
pub fn load_batch_queue() -> Result<Option<String>, String> {
    let path = store::data_dir().join("batch-queue.json");
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Drop the persisted batch queue state.
#[tauri::command]
#[specta::specta]
pub fn clear_batch_queue() -> Result<(), String> {
    let path = store::data_dir().join("batch-queue.json");
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests;
