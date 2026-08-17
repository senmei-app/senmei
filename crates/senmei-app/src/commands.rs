use std::path::PathBuf;

use serde::Serialize;
use tauri::ipc::Channel;

use crate::store;

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
    if std::env::var_os("SENMEI_FORCE_FFMPEG_MISSING").is_some() {
        return senmei_media::FfmpegInfo::default();
    }
    let dir = store::data_dir();
    let ffmpeg = senmei_media::resolve(&dir);
    senmei_media::probe_ffmpeg(&ffmpeg)
}

#[tauri::command]
#[specta::specta]
pub async fn download_ffmpeg(
    on_progress: Channel<DownloadProgress>,
) -> Result<String, String> {
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
        .map(|(registry, _)| registry.models().to_vec())
        .unwrap_or_default()
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
    if !weight.ends_with(".bpk") {
        return Err(format!("expected f16 burnpack weight, got {weight}"));
    }
    // Sources host the f32 `.pth`; download it, convert to the f16 `.bpk`.
    let pth_name = format!("{}.pth", weight.trim_end_matches(".f16.bpk"));
    let bpk_path = dir.join(&weight);
    tauri::async_runtime::spawn_blocking(move || {
        let mut progress = on_progress;
        let pth = senmei_media::download_to_temp(
            &url,
            &dir,
            &pth_name,
            meta.sha256.as_deref(),
            &mut |d, t| {
                let _ = progress.send(DownloadProgress { downloaded: d, total: t });
            },
        )
        .map_err(|e| e.to_string())?;
        let num_block = meta
            .metadata
            .get("num_block")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as u32;
        senmei_ml::convert_pth_to_bpk(&meta.arch, &pth, &bpk_path, meta.scale, num_block)
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&pth);
        Ok::<String, String>(bpk_path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[specta::specta]
pub fn import_folder(dir: String) -> Result<Vec<String>, String> {
    const EXTS: [&str; 10] = [
        "mp4", "mkv", "mov", "webm", "avi", "m4v", "ts", "m2ts", "flv", "wmv",
    ];

    let mut result = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if EXTS.contains(&ext.to_ascii_lowercase().as_str()) {
            result.push(path.to_string_lossy().into_owned());
        }
    }
    result.sort();
    Ok(result)
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
pub fn remember_project(path: String) -> Result<(), String> {
    store::remember_project(&path)
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
}

fn models_dir() -> PathBuf {
    // Anchor to the repo checkout: cargo tauri dev runs the binary from the
    // crate dir, so CWD-relative paths can miss models/ at the repo root.
    let anchored = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    for dir in [anchored, PathBuf::from("models"), PathBuf::from("../models")] {
        if dir.is_dir() {
            return dir;
        }
    }
    PathBuf::from("models")
}

fn load_registry() -> Result<(senmei_ml::Registry, PathBuf), String> {
    let dir = models_dir();
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&dir).map_err(|e| e.to_string())?;
    Ok((registry, dir))
}

fn engine_for_model(model_id: &str) -> Result<Box<dyn senmei_ml::InferenceEngine>, String> {
    let (registry, dir) = load_registry()?;
    let meta = registry
        .models()
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("model not found: {model_id}"))?;
    if !meta.loadable {
        return Err(format!("model {model_id} has no loadable weights yet"));
    }
    let mref = registry
        .resolve(model_id, &dir)
        .ok_or_else(|| format!("model weights not resolved: {model_id}"))?;
    let mut engine = senmei_ml::engine_for_model(&mref).map_err(|e| e.to_string())?;
    engine.load(&mref).map_err(|e| e.to_string())?;
    Ok(engine)
}

#[tauri::command]
#[specta::specta]
pub async fn render(
    input: String,
    output: String,
    scale: Option<u32>,
    model_id: Option<String>,
    resize: Option<f32>,
    output_resize: Option<f32>,
    fps_multiplier: Option<u32>,
    on_progress: Channel<RenderProgress>,
) -> Result<String, String> {
    log::info!(
        "render start: {input} -> {output} (scale {scale:?}, model {model_id:?}, resize {resize:?}, output_resize {output_resize:?}, fps {fps_multiplier:?})"
    );
    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let ffmpeg = senmei_media::resolve(&store::data_dir());
        let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
            vec![Box::new(senmei_pipeline::Passthrough)];
        if let Some(f) = resize {
            steps.push(Box::new(senmei_pipeline::Resize::new(f)));
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
        if let Some(f) = output_resize {
            steps.push(Box::new(senmei_pipeline::Resize::new(f)));
        }
        let mut pipeline = senmei_pipeline::Pipeline::new(steps);
        if let Some(f) = fps_multiplier {
            if f > 1 {
                pipeline.set_interpolator(senmei_pipeline::Interpolator::new(f));
            }
        }

        pipeline
            .run(&ffmpeg, &input, &output, |p| {
                let _ = on_progress.send(RenderProgress {
                    frames_processed: p.frames_processed,
                    total_frames: p.total_frames,
                });
            })
            .map(|_| "ok".to_string())
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}
