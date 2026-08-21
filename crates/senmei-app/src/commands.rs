use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::Manager;

use crate::models::{engine_for_model, load_registry};
use crate::preview::{extract_audio_inner, probe_video_inner, read_frame_inner};
use crate::store;

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
    let dir = store::data_dir();
    let ffmpeg = senmei_media::resolve(&dir);
    senmei_media::probe_ffmpeg(&ffmpeg)
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
    load_registry()
        .map(|(registry, dir)| {
            registry
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
                .collect()
        })
        .unwrap_or_default()
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
    let (registry, dir) = load_registry()?;
    let meta = registry
        .models()
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("model not found: {model_id}"))?
        .clone();
    // The 5th convert arg is num_block (realesrgan) or feature_channels (span).
    let convert_arg = registry
        .resolve(&model_id, &dir)
        .map(|m| if m.arch == "span" { m.feature_channels } else { m.num_block })
        .unwrap_or(4);
    let layer_norm = registry
        .resolve(&model_id, &dir)
        .map(|m| m.layer_norm)
        .unwrap_or(false);
    if meta.license_blocked() {
        return Err(format!(
            "model {model_id} has an unconfirmed/restrictive license ({}); refusing download",
            meta.license.as_deref().unwrap_or("none")
        ));
    }
    if !meta.loadable {
        return Err(format!("model {model_id} has no loadable arch yet"));
    }
    let url = meta
        .download_url
        .clone()
        .ok_or_else(|| format!("model {model_id} has no download_url"))?;
    let weight = meta
        .weights
        .as_ref()
        .and_then(|w| w.first())
        .cloned()
        .ok_or_else(|| format!("model {model_id} has no weights"))?;
    let is_ncnn = weight.ends_with(".bin");
    if !weight.ends_with(".bpk") && !is_ncnn {
        return Err(format!("expected f16 burnpack or ncnn weight, got {weight}"));
    }
    let is_archive = url.ends_with(".zip");
    // Multi-model archives (e.g. the nihui rife release zip bundles every
    // version) need a version-specific entry; default to the weight filename.
    let extract_suffix = meta
        .metadata
        .get("extract_suffix")
        .and_then(|v| v.as_str())
        .unwrap_or(&weight)
        .to_string();
    let target = dir.join(&weight);
    if target.is_file() {
        log::info!("download_model: {model_id} already present, skipping");
        return Ok(target.to_string_lossy().into_owned());
    }
    let onnx = std::path::Path::new(&url)
        .extension()
        .and_then(|e| e.to_str())
        == Some("onnx");
    let ext = if onnx {
        "onnx"
    } else if is_archive {
        "zip"
    } else {
        "pth"
    };
    log::info!("download_model: {model_id} <- {url} -> {}", dir.display());
    tauri::async_runtime::spawn_blocking(move || {
        let progress = on_progress;
        let base = weight.trim_end_matches(".f16.bpk");
        let source = senmei_media::download_to_temp(
            &url,
            &dir,
            &format!("{base}.{ext}"),
            meta.sha256.as_deref(),
            &mut |d, t| {
                let _ = progress.send(DownloadProgress {
                    downloaded: d,
                    total: t,
                });
            },
        )
        .map_err(|e| {
            log::error!("download_model {model_id}: download failed: {e}");
            e.to_string()
        })?;
        log::info!(
            "download_model: {model_id} downloaded to {}",
            source.display()
        );
        // RIFE ships ncnn weights: the .bin is inside a release zip, or a raw
        // .bin — either way no burnpack conversion.
        if is_archive {
            senmei_media::extract_binary(&source, &target, &extract_suffix).map_err(|e| {
                log::error!("download_model {model_id}: extract failed: {e}");
                e.to_string()
            })?;
            let _ = std::fs::remove_file(&source);
            log::info!("download_model: {model_id} wrote {}", target.display());
            return Ok::<String, String>(target.to_string_lossy().into_owned());
        }
        if is_ncnn {
            std::fs::rename(&source, &target).map_err(|e| {
                log::error!("download_model {model_id}: rename failed: {e}");
                e.to_string()
            })?;
            log::info!("download_model: {model_id} wrote {}", target.display());
            return Ok::<String, String>(target.to_string_lossy().into_owned());
        }
        let conv = if onnx {
            senmei_ml::convert_onnx_to_bpk(&meta.arch, &source, &target, meta.scale, convert_arg)
        } else {
            senmei_ml::convert_pth_to_bpk(
                &meta.arch,
                &source,
                &target,
                meta.scale,
                convert_arg,
                layer_norm,
            )
        };
        if let Err(e) = conv {
            log::error!("download_model {model_id}: conversion failed: {e}");
            return Err(e.to_string());
        }
        let _ = std::fs::remove_file(&source);
        log::info!("download_model: {model_id} wrote {}", target.display());
        Ok::<String, String>(target.to_string_lossy().into_owned())
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
    probe_video_inner(&input)
}

#[tauri::command]
#[specta::specta]
pub async fn read_frame(
    input: String,
    position_ms: f64,
    project_dir: Option<String>,
    app: tauri::AppHandle,
) -> Result<String, String> {
    log::info!("read_frame: {input} @ {position_ms:.0}ms");
    // Decode off the main thread so the UI never freezes per frame.
    let path = tauri::async_runtime::spawn_blocking(move || {
        read_frame_inner(&input, position_ms, project_dir.as_deref())
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = app
        .state::<tauri::scope::Scopes>()
        .allow_file(std::path::Path::new(&path));
    Ok(path)
}

#[tauri::command]
#[specta::specta]
pub async fn extract_audio(
    input: String,
    project_dir: Option<String>,
    app: tauri::AppHandle,
) -> Result<String, String> {
    log::info!("extract_audio: {input}");
    let path = tauri::async_runtime::spawn_blocking(move || {
        extract_audio_inner(&input, project_dir.as_deref())
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = app
        .state::<tauri::scope::Scopes>()
        .allow_file(std::path::Path::new(&path));
    Ok(path)
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
    videos_under(&dir, false)
}

/// Recursively collect all videos under `dir` (batch folder processing).
#[tauri::command]
#[specta::specta]
pub fn scan_folder(dir: String) -> Result<Vec<String>, String> {
    videos_under(&dir, true)
}

fn videos_under(dir: &str, recursive: bool) -> Result<Vec<String>, String> {
    let found = senmei_media::find_videos(std::path::Path::new(dir), recursive)
        .map_err(|e| e.to_string())?;
    Ok(found
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect())
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
    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let RenderConfig {
            scale,
            model_id,
            resize,
            filter,
            decompress_model_id,
            output_resize,
            fps_multiplier,
            interp_model,
            ffmpeg_args,
            tonemap,
            start_ms,
            end_ms,
        } = config;
        senmei_ml::set_tile_size(store::load_settings().tile_size.unwrap_or(640));
        let ffmpeg = senmei_media::resolve(&store::data_dir());
        let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
            vec![Box::new(senmei_pipeline::Passthrough)];
        if let Some(f) = resize {
            steps.push(Box::new(senmei_pipeline::Resize::new(f)));
        }
        // Decompress pass runs first: scale-1 de-artifact (RealPLKSR 1×) ahead
        // of interpolation/upscaling. Skipped when the model can't be loaded.
        if let Some(id) = decompress_model_id {
            if !id.is_empty() {
                match engine_for_model(&id) {
                    Ok(engine) => {
                        steps.push(Box::new(senmei_pipeline::Upscale::new(1, Some(engine))));
                    }
                    Err(err) => log::warn!("decompress model {id} unavailable, skipping: {err}"),
                }
            }
        }
        if let Some(s) = scale {
            if s > 1 {
                // A selected model that cannot be loaded (missing weights,
                // unsupported format) falls back to the reference scaler.
                let engine = match model_id.as_deref() {
                    Some(id) => match engine_for_model(id) {
                        Ok(e) => Some(e),
                        Err(err) => {
                            log::warn!("model {id} unavailable, using reference scaler: {err}");
                            None
                        }
                    },
                    None => None,
                };
                steps.push(Box::new(senmei_pipeline::Upscale::new(s, engine)));
            }
        }
        if let Some(f) = filter {
            if let Some(r) = f.denoise_radius {
                if r > 0 {
                    // A selected denoise model that cannot be loaded falls back
                    // to the CPU box blur (engine stays None).
                    let engine = match f.denoise_model_id.as_deref() {
                        Some(id) => match engine_for_model(id) {
                            Ok(e) => Some(e),
                            Err(err) => {
                                log::warn!("denoise model {id} unavailable, using box blur: {err}");
                                None
                            }
                        },
                        None => None,
                    };
                    steps.push(Box::new(senmei_pipeline::Denoise::new(r, engine)));
                }
            }
            if let Some(a) = f.deblur_amount {
                if a > 0.0 {
                    // A selected deblur model that cannot be loaded falls back
                    // to the CPU unsharp mask (engine stays None).
                    let engine = match f.deblur_model_id.as_deref() {
                        Some(id) => match engine_for_model(id) {
                            Ok(e) => Some(e),
                            Err(err) => {
                                log::warn!(
                                    "deblur model {id} unavailable, using unsharp mask: {err}"
                                );
                                None
                            }
                        },
                        None => None,
                    };
                    steps.push(Box::new(senmei_pipeline::Deblur::new(a, engine)));
                }
            }
            if let Some(t) = f.dedup_threshold {
                if t > 0.0 {
                    steps.push(Box::new(senmei_pipeline::Dedup::new(t)));
                }
            }
            if let Some(filter) = f.ffmpeg_filter.as_deref() {
                if !filter.trim().is_empty() {
                    steps.push(Box::new(senmei_pipeline::Filter::new(filter, &ffmpeg)));
                }
            }
        }
        if let Some(f) = output_resize {
            steps.push(Box::new(senmei_pipeline::Resize::new(f)));
        }
        let mut pipeline = senmei_pipeline::Pipeline::new(steps);
        if start_ms.is_some() || end_ms.is_some() {
            pipeline.set_range(start_ms.unwrap_or(0), end_ms);
        }
        if let Some(args) = ffmpeg_args {
            if !args.is_empty() {
                pipeline.set_encoder_args(args);
            }
        }
        if let Some(t) = tonemap {
            pipeline.set_tonemap(match t.as_str() {
                "always" => senmei_media::Tonemap::Always,
                "off" => senmei_media::Tonemap::Off,
                _ => senmei_media::Tonemap::Auto,
            });
        }
        let cancel = CANCEL_RENDER
            .get_or_init(|| Arc::new(AtomicBool::new(false)))
            .clone();
        cancel.store(false, Ordering::Relaxed);
        pipeline.set_cancel(cancel);
        let pause = PAUSE_RENDER
            .get_or_init(|| Arc::new(AtomicBool::new(false)))
            .clone();
        pause.store(false, Ordering::Relaxed);
        pipeline.set_pause(pause);
        if let Some(f) = fps_multiplier {
            if f > 1 {
                // An interpolate model that cannot be loaded (missing weights,
                // unsupported arch) falls back to the reference blend.
                let interp = match interp_model.as_deref() {
                    Some(id) => match engine_for_model(id) {
                        Ok(e) => Some(senmei_pipeline::Interpolator::with_engine(f, e)),
                        Err(err) => {
                            log::warn!(
                                "interp model {id} unavailable, using reference blend: {err}"
                            );
                            None
                        }
                    },
                    None => None,
                };
                let interpolator = interp.unwrap_or_else(|| senmei_pipeline::Interpolator::new(f));
                pipeline.set_interpolator(interpolator);
            }
        }

        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let processed = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let total = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (p_ref, t_ref) = (processed.clone(), total.clone());
        let progress_tx = on_progress.clone();
        let run = pipeline.run(&ffmpeg, &input, &output, move |p| {
            p_ref.store(p.frames_processed, Ordering::Relaxed);
            t_ref.store(p.total_frames, Ordering::Relaxed);
            let _ = on_progress.send(RenderProgress {
                frames_processed: p.frames_processed,
                total_frames: p.total_frames,
                steps: Vec::new(),
            });
        });
        // Final event carries the per-step benchmark (only steps that ran).
        let steps: Vec<StepTimingInfo> = pipeline
            .step_timings()
            .iter()
            .filter(|t| t.frames > 0)
            .map(|t| StepTimingInfo {
                name: t.name.clone(),
                frames: t.frames,
                ms_per_frame: t.total.as_secs_f64() * 1000.0 / t.frames as f64,
                fps: t.frames as f64 / t.total.as_secs_f64(),
            })
            .collect();
        let _ = progress_tx.send(RenderProgress {
            frames_processed: processed.load(Ordering::Relaxed),
            total_frames: total.load(Ordering::Relaxed),
            steps,
        });
        if let Err(e) = &run {
            log::error!("render failed: {e}");
            // Drop the partial file on abort/error so it cannot be mistaken for a result.
            let _ = std::fs::remove_file(&output);
        }
        run.map(|_| "ok".to_string()).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Abort the active render (the pipeline checks the flag between frames).
#[tauri::command]
#[specta::specta]
pub fn cancel_render() {
    if let Some(c) = CANCEL_RENDER.get() {
        c.store(true, Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_commands_produce_png_and_info() {
        let dir = std::env::temp_dir().join("senmei-cmd-smoke");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.mp4");
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=160x120:rate=10",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&input)
            .status()
            .unwrap()
            .success();
        assert!(ok, "ffmpeg input generation failed");

        let info = probe_video_inner(&input.to_string_lossy()).expect("probe_video failed");
        assert_eq!((info.width, info.height), (160, 120));
        assert!(info.duration > 0.0);

        let file =
            read_frame_inner(&input.to_string_lossy(), 500.0, None).expect("read_frame failed");
        let png = std::fs::read(&file).unwrap();
        assert!(png.starts_with(&[0x89, 0x50, 0x4E, 0x47]), "not a PNG");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end proof of the app's render path: models_dir resolution +
    /// BurnEngine load + real 1080p→2160p upscale + ffmpeg encode.
    #[test]
    #[ignore = "requires burn Vulkan engine + ffmpeg; ~15s render"]
    fn app_render_upscales_real_model() {
        let dir = std::env::temp_dir().join("senmei-render-smoke");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.mp4");
        let output = dir.join("output.mp4");
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=1920x1080:rate=24",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&input)
            .status()
            .unwrap()
            .success();
        assert!(ok, "ffmpeg input generation failed");

        let engine = engine_for_model("real-cugan-x2").expect("engine_for_model");
        let ffmpeg = senmei_media::resolve(&store::data_dir());
        let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
            vec![Box::new(senmei_pipeline::Passthrough)];
        steps.push(Box::new(senmei_pipeline::Upscale::new(2, Some(engine))));
        let mut pipeline = senmei_pipeline::Pipeline::new(steps);
        // Custom ffmpeg args must override the default x264 encoder.
        pipeline.set_encoder_args(vec![
            "-c:v".into(),
            "libx265".into(),
            "-crf".into(),
            "18".into(),
            "-preset".into(),
            "ultrafast".into(),
            "-pix_fmt".into(),
            "yuv420p10le".into(),
        ]);
        pipeline
            .run(&ffmpeg, &input, &output, |_| {})
            .expect("render failed");

        let info = probe_video_inner(&output.to_string_lossy()).expect("probe output");
        assert_eq!((info.width, info.height), (3840, 2160));
        assert!(output.exists());
        let ffprobe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "csv=p=0",
            ])
            .arg(&output)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&ffprobe.stdout);
        assert!(
            stdout.contains("hevc") && stdout.contains("yuv420p10le"),
            "custom args not applied, got {stdout}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unique_path_numbers_collisions() {
        let dir = std::env::temp_dir().join("senmei-unique-smoke");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("out.mkv");
        let b = dir.join("out_2.mkv");
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"x").unwrap();
        let free = unique_path(a.to_string_lossy().into_owned()).unwrap();
        assert_eq!(free, dir.join("out_3.mkv").to_string_lossy().into_owned());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_samples_keeps_newest_by_mtime() {
        let _guard = crate::store::TEST_ENV_LOCK.lock().unwrap();
        let base =
            std::env::temp_dir().join(format!("senmei-prune-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::env::set_var("XDG_DATA_HOME", &base);
        let dir = store::data_dir().join("samples");
        std::fs::create_dir_all(&dir).unwrap();

        // The newest file's name sorts first; a lexical prune would wrongly
        // delete it, a mtime prune must keep it.
        let oldest = dir.join("b.mkv");
        let newest = dir.join("a.mkv");
        std::fs::write(&oldest, b"x").unwrap();
        std::fs::write(&newest, b"x").unwrap();
        let t0 = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
        let t1 = t0 + std::time::Duration::from_secs(10);
        let set = |p: &std::path::Path, t: std::time::SystemTime| {
            std::fs::File::options()
                .write(true)
                .open(p)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(t))
                .unwrap();
        };
        set(&oldest, t0);
        set(&newest, t1);

        prune_samples(dir.to_string_lossy().into_owned(), 1).unwrap();
        assert!(newest.exists(), "newest sample was pruned");
        assert!(!oldest.exists(), "oldest sample kept");
        let _ = std::fs::remove_dir_all(&base);
    }
}
