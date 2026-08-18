use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::Manager;

use crate::store;

/// Shared cancellation flag for the active render (set by `cancel_render`).
static CANCEL_RENDER: OnceLock<Arc<AtomicBool>> = OnceLock::new();
/// Shared pause flag for the active render (set by `pause_render`).
static PAUSE_RENDER: OnceLock<Arc<AtomicBool>> = OnceLock::new();
/// Warm decode streams for the monitor (source + result).
static PREVIEW_CACHE: OnceLock<Mutex<Option<senmei_media::PreviewCache>>> = OnceLock::new();
/// Monotonic counter for unique preview frame names.
static PREVIEW_SEQ: AtomicU64 = AtomicU64::new(0);

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

fn probe_video_inner(input: &str) -> Result<senmei_media::VideoInfo, String> {
    senmei_media::probe(std::path::Path::new(input)).map_err(|e| {
        log::warn!("probe_video failed: {e}");
        e.to_string()
    })
}

#[tauri::command]
#[specta::specta]
pub fn probe_video(input: String, app: tauri::AppHandle) -> Result<senmei_media::VideoInfo, String> {
    log::info!("probe_video: {input}");
    // Let the webview load this file via the asset protocol (native <video>).
    let _ = app.state::<tauri::scope::Scopes>().allow_file(std::path::Path::new(&input));
    probe_video_inner(&input)
}

/// Short stable namespace for one input file, so original/result/compare frames
/// never share a prune bucket or filename.
fn frame_ns(input: &str) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Extract one frame at `position_ms` as a PNG file and return its path. Uses
/// a persistent decode stream (one ffmpeg per file) so playback reads frames
/// from the pipe instead of spawning a process per frame. Frames are written
/// under the project (`preview/`) when one is open, else the app data dir;
/// a non-writable project dir falls back to the data dir.
fn read_frame_inner(
    input: &str,
    position_ms: f64,
    project_dir: Option<&str>,
) -> Result<String, String> {
    let dir = project_dir
        .and_then(|p| {
            let d = std::path::Path::new(p).join("preview");
            std::fs::create_dir_all(&d).ok().map(|_| d)
        })
        .unwrap_or_else(|| {
            let d = store::data_dir().join("preview");
            let _ = std::fs::create_dir_all(&d);
            d
        });
    let ns = frame_ns(input);
    prune_preview_frames(&dir, &ns, 60);

    let ffmpeg = senmei_media::resolve(&store::data_dir());
    // Decode under the lock (fast pipe read) but encode outside it, so the two
    // compare sides don't serialize their (slow) PNG encode behind each other.
    let frame = {
        let mut cache = PREVIEW_CACHE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|e| e.to_string())?;
        if cache.is_none() {
            *cache = Some(senmei_media::PreviewCache::new(ffmpeg));
        }
        cache
            .as_mut()
            .unwrap()
            .frame(input, position_ms)
            .map_err(|e| {
                log::warn!("preview decode failed: {e}");
                e.to_string()
            })?
    };
    let png = senmei_media::encode_png(frame.width, frame.height, &frame.data)
        .map_err(|e| e.to_string())?;

    let seq = PREVIEW_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("frame_{ns}_{seq:08}.png"));
    std::fs::write(&path, &png).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Best-effort cap on leftover preview frames for one namespace (keep the
/// newest `keep`). Zero-padded counters make name order equal to write order.
fn prune_preview_frames(dir: &std::path::Path, ns: &str, keep: usize) {
    let prefix = format!("frame_{ns}_");
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut old: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().starts_with(&prefix))
                    .unwrap_or(false)
            })
            .collect();
        old.sort();
        for p in old.iter().take(old.len().saturating_sub(keep)) {
            let _ = std::fs::remove_file(p);
        }
    }
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
    // Let the webview load this frame through the asset protocol.
    let _ = app.state::<tauri::scope::Scopes>().allow_file(std::path::Path::new(&path));
    Ok(path)
}

/// Keep only the `keep` newest sample render files in `dir` (deletes older
/// video files so the sample folder never grows unbounded).
#[tauri::command]
#[specta::specta]
pub fn prune_samples(dir: String, keep: usize) -> Result<(), String> {
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
    files.sort();
    for p in files.iter().take(files.len().saturating_sub(keep)) {
        let _ = std::fs::remove_file(p);
    }
    Ok(())
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
pub fn export_project(src: String, dest: String) -> Result<(), String> {
    store::export_project(&src, &dest)
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
    /// Render only a time range (start ms, end ms; None end = to the end).
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
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
            start_ms,
            end_ms,
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
        if start_ms.is_some() || end_ms.is_some() {
            pipeline.set_range(start_ms.unwrap_or(0), end_ms);
        }
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

        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
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
    fn preview_commands_produce_png_and_info() {
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

        let info = probe_video_inner(&input.to_string_lossy()).expect("probe_video failed");
        assert_eq!((info.width, info.height), (160, 120));
        assert!(info.duration > 0.0);

        let file = read_frame_inner(&input.to_string_lossy(), 500.0, None).expect("read_frame failed");
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
