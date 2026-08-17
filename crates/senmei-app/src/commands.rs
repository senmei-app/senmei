use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

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
        let progress = on_progress;
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
pub fn probe_video(input: String) -> Result<senmei_media::VideoInfo, String> {
    log::info!("probe_video: {input}");
    senmei_media::probe(std::path::Path::new(&input)).map_err(|e| {
        log::warn!("probe_video failed: {e}");
        e.to_string()
    })
}

/// Extract one frame at `position_ms` and return it as a base64 JPEG.
#[tauri::command]
#[specta::specta]
pub fn read_frame(input: String, position_ms: f64) -> Result<String, String> {
    use base64::Engine;
    log::info!("read_frame: {input} @ {position_ms:.0}ms");
    let jpeg = senmei_media::extract_frame(std::path::Path::new(&input), position_ms / 1000.0)
        .map_err(|e| {
            log::warn!("read_frame failed: {e}");
            e.to_string()
        })?;
    Ok(base64::engine::general_purpose::STANDARD.encode(jpeg))
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
pub fn delete_project(path: String) -> Result<(), String> {
    store::delete_project(&path)
}

#[tauri::command]
#[specta::specta]
pub fn remember_project(path: String) -> Result<(), String> {
    store::remember_project(&path)
}

#[tauri::command]
#[specta::specta]
pub fn save_project_as(src: String, name: String) -> Result<String, String> {
    store::save_project_as(&src, &name)
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

/// Optional reference filter steps (denoise/deblur/dedup) for a render.
#[derive(Clone, Debug, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterParams {
    pub denoise_radius: Option<u32>,
    pub deblur_amount: Option<f32>,
    pub dedup_threshold: Option<f32>,
}

/// All render knobs in one struct (specta caps command arity at 10 args).
#[derive(Clone, Debug, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderConfig {
    pub scale: Option<u32>,
    pub model_id: Option<String>,
    pub resize: Option<f32>,
    pub filter: Option<FilterParams>,
    pub output_resize: Option<f32>,
    pub fps_multiplier: Option<u32>,
    pub interp_model: Option<String>,
    pub ffmpeg_args: Option<String>,
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

/// Split a shell-style arg string into tokens (respects double quotes).
fn split_ffmpeg_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for c in s.chars() {
        match c {
            '"' => in_quote = !in_quote,
            ' ' | '\t' if !in_quote => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
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
    config: RenderConfig,
    on_progress: Channel<RenderProgress>,
) -> Result<String, String> {
    log::info!(
        "render start: {input} -> {output} (config {config:?})"
    );
    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let RenderConfig {
            scale,
            model_id,
            resize,
            filter,
            output_resize,
            fps_multiplier,
            interp_model,
            ffmpeg_args,
        } = config;
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
        if let Some(f) = filter {
            if let Some(r) = f.denoise_radius {
                if r > 0 {
                    steps.push(Box::new(senmei_pipeline::Denoise::new(r)));
                }
            }
            if let Some(a) = f.deblur_amount {
                if a > 0.0 {
                    steps.push(Box::new(senmei_pipeline::Deblur::new(a)));
                }
            }
            if let Some(t) = f.dedup_threshold {
                if t > 0.0 {
                    steps.push(Box::new(senmei_pipeline::Dedup::new(t)));
                }
            }
        }
        if let Some(f) = output_resize {
            steps.push(Box::new(senmei_pipeline::Resize::new(f)));
        }
        let mut pipeline = senmei_pipeline::Pipeline::new(steps);
        if let Some(args) = ffmpeg_args.as_deref() {
            if !args.trim().is_empty() {
                pipeline.set_encoder_args(split_ffmpeg_args(args));
            }
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
                            log::warn!("interp model {id} unavailable, using reference blend: {err}");
                            None
                        }
                    },
                    None => None,
                };
                let interpolator = interp.unwrap_or_else(|| senmei_pipeline::Interpolator::new(f));
                pipeline.set_interpolator(interpolator);
            }
        }

        let run = pipeline.run(&ffmpeg, &input, &output, move |p| {
            let _ = on_progress.send(RenderProgress {
                frames_processed: p.frames_processed,
                total_frames: p.total_frames,
            });
        });
        if run.is_err() {
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
    fn preview_commands_produce_jpeg_and_info() {
        let dir = std::env::temp_dir().join("senmei-cmd-smoke");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.mp4");
        let ok = std::process::Command::new("ffmpeg")
            .args([
                "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
                "-pix_fmt", "yuv420p",
            ])
            .arg(&input)
            .status()
            .unwrap()
            .success();
        assert!(ok, "ffmpeg input generation failed");

        let info = probe_video(input.to_string_lossy().into_owned()).expect("probe_video failed");
        assert_eq!((info.width, info.height), (160, 120));
        assert!(info.duration > 0.0);

        let b64 = read_frame(input.to_string_lossy().into_owned(), 500.0).expect("read_frame failed");
        use base64::Engine;
        let jpeg = base64::engine::general_purpose::STANDARD.decode(&b64).unwrap();
        assert!(jpeg.starts_with(&[0xFF, 0xD8]), "not a JPEG");
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

        let engine = engine_for_model("shuffle-cugan").expect("engine_for_model");
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

        let info = probe_video(output.to_string_lossy().into_owned()).expect("probe output");
        assert_eq!((info.width, info.height), (3840, 2160));
        assert!(output.exists());
        let ffprobe = std::process::Command::new("ffprobe")
            .args([
                "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0",
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
    fn split_ffmpeg_args_handles_quotes() {
        let args = split_ffmpeg_args("-vf \"scale=1920:1080,yadif=mode=0\" -c:v libx265 -crf 18");
        assert_eq!(
            args,
            vec![
                "-vf",
                "scale=1920:1080,yadif=mode=0",
                "-c:v",
                "libx265",
                "-crf",
                "18"
            ]
        );
    }
}
