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

#[cfg(feature = "render")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "render")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "render")]
pub use senmei_pipeline::Progress as RenderProgress;

/// Hard cancel flag for the active render (checked between frames).
#[cfg(feature = "render")]
static CANCEL_RENDER: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Pending (proposed) render — starts only after an explicit confirm.
#[cfg(feature = "render")]
static PENDING_RENDER: OnceLock<Mutex<Option<RenderConfig>>> = OnceLock::new();

/// Shared status of the active render, updated from the worker thread.
#[cfg(feature = "render")]
static RENDER_STATUS: OnceLock<Arc<Mutex<RenderStatus>>> = OnceLock::new();

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterConfig {
    pub denoise_radius: Option<u32>,
    pub denoise_model_id: Option<String>,
    pub deblur_amount: Option<f32>,
    pub deblur_model_id: Option<String>,
    pub dedup_threshold: Option<f32>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderConfig {
    pub input: String,
    pub output: String,
    pub scale: Option<u32>,
    pub model_id: Option<String>,
    pub resize: Option<f32>,
    pub filter: Option<FilterConfig>,
    pub output_resize: Option<f32>,
    pub fps_multiplier: Option<u32>,
    pub interp_model: Option<String>,
    pub ffmpeg_args: Option<Vec<String>>,
    pub tonemap: Option<String>,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

/// Load a model engine, enforcing the license gate (hard). Missing weights or
/// an unloadable arch are errors here — build steps may still fall back to the
/// reference filter when a model is unavailable (like the GUI).
#[cfg(feature = "render")]
pub fn engine_for_model(model_id: &str) -> Result<Box<dyn senmei_ml::InferenceEngine>, String> {
    let (registry, dir) = load_registry()?;
    let meta = registry
        .models()
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("model not found: {model_id}"))?;
    if meta.license_blocked() {
        return Err(format!(
            "model {model_id} has an unconfirmed/restrictive license ({}); refusing to load weights",
            meta.license.as_deref().unwrap_or("none")
        ));
    }
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

/// Validate a render config: required paths, sane ranges, and every referenced
/// model must exist with a permissive license (never a blocked one).
#[cfg(feature = "render")]
pub fn validate(config: &RenderConfig) -> Result<(), String> {
    if config.input.is_empty() || config.output.is_empty() {
        return Err("input and output are required".into());
    }
    if config.scale.unwrap_or(1) > 4 {
        return Err("scale must be <= 4".into());
    }
    if let Some(f) = config.fps_multiplier {
        if !(1..=16).contains(&f) {
            return Err("fps_multiplier must be in 1..=16".into());
        }
    }
    let mut ids: Vec<&str> = Vec::new();
    for id in [config.model_id.as_deref(), config.interp_model.as_deref()] {
        if let Some(id) = id {
            ids.push(id);
        }
    }
    if let Some(f) = config.filter.as_ref() {
        for id in [f.denoise_model_id.as_deref(), f.deblur_model_id.as_deref()] {
            if let Some(id) = id {
                ids.push(id);
            }
        }
    }
    let (registry, _) = load_registry()?;
    for id in ids {
        let meta = registry
            .models()
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| format!("unknown model: {id}"))?;
        if meta.license_blocked() {
            return Err(format!(
                "model {id} is license-blocked ({}); refusing render",
                meta.license.as_deref().unwrap_or("none")
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "render")]
fn build_steps(config: &RenderConfig) -> Vec<Box<dyn senmei_pipeline::Step>> {
    let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Passthrough)];
    if let Some(f) = config.resize {
        steps.push(Box::new(senmei_pipeline::Resize::new(f)));
    }
    if let Some(s) = config.scale {
        if s > 1 {
            let engine = match config.model_id.as_deref() {
                Some(id) => engine_for_model(id).ok(),
                None => None,
            };
            steps.push(Box::new(senmei_pipeline::Upscale::new(s, engine)));
        }
    }
    if let Some(f) = config.filter.as_ref() {
        if let Some(r) = f.denoise_radius {
            if r > 0 {
                let engine = match f.denoise_model_id.as_deref() {
                    Some(id) => engine_for_model(id).ok(),
                    None => None,
                };
                steps.push(Box::new(senmei_pipeline::Denoise::new(r, engine)));
            }
        }
        if let Some(a) = f.deblur_amount {
            if a > 0.0 {
                let engine = match f.deblur_model_id.as_deref() {
                    Some(id) => engine_for_model(id).ok(),
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
    }
    if let Some(f) = config.output_resize {
        steps.push(Box::new(senmei_pipeline::Resize::new(f)));
    }
    steps
}

/// Run a render (blocking; call from spawn_blocking). Mirrors the GUI's
/// pipeline assembly, without Tauri.
#[cfg(feature = "render")]
pub fn render(
    config: &RenderConfig,
    on_progress: impl FnMut(RenderProgress) + Send + 'static,
) -> Result<(), String> {
    senmei_ml::set_tile_size(640);
    let ffmpeg = ffmpeg();
    let input = PathBuf::from(&config.input);
    let output = PathBuf::from(&config.output);
    let mut pipeline = senmei_pipeline::Pipeline::new(build_steps(config));
    if config.start_ms.is_some() || config.end_ms.is_some() {
        pipeline.set_range(config.start_ms.unwrap_or(0), config.end_ms);
    }
    if let Some(args) = config.ffmpeg_args.as_ref() {
        if !args.is_empty() {
            pipeline.set_encoder_args(args.clone());
        }
    }
    if let Some(t) = config.tonemap.as_deref() {
        pipeline.set_tonemap(match t {
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
    if let Some(f) = config.fps_multiplier {
        if f > 1 {
            let interp = match config.interp_model.as_deref() {
                Some(id) => engine_for_model(id)
                    .ok()
                    .map(|e| senmei_pipeline::Interpolator::with_engine(f, e)),
                None => None,
            };
            pipeline.set_interpolator(interp.unwrap_or_else(|| senmei_pipeline::Interpolator::new(f)));
        }
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let run = pipeline.run(&ffmpeg, &input, &output, on_progress);
    if let Err(e) = &run {
        log::error!("render failed: {e}");
        let _ = std::fs::remove_file(&output);
    }
    run.map_err(|e| e.to_string())
}

/// Propose a render: validates and parks it. Does NOT start — the confirm
/// gate requires `confirm_render` first.
#[cfg(feature = "render")]
pub fn propose_render(config: RenderConfig) -> Result<String, String> {
    validate(&config)?;
    let slot = PENDING_RENDER.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap() = Some(config);
    Ok("render proposed — call confirm_render to start".into())
}

/// Run the previously proposed render (confirmation gate).
/// Starts it on a worker thread and returns immediately — poll
/// [`render_status`] for progress; [`cancel_render`] aborts it.
#[cfg(feature = "render")]
pub fn confirm_render() -> Result<String, String> {
    let slot = PENDING_RENDER.get_or_init(|| Mutex::new(None));
    let config = slot
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "no pending render; propose_render first".to_string())?;
    let status = RENDER_STATUS
        .get_or_init(|| Arc::new(Mutex::new(RenderStatus::default())))
        .clone();
    {
        let mut s = status.lock().unwrap();
        if s.state == "running" {
            return Err("a render is already running".into());
        }
        *s = RenderStatus {
            state: "running".into(),
            ..Default::default()
        };
    }
    std::thread::spawn(move || {
        let progress_status = status.clone();
        let result = render(&config, move |p| {
            let mut s = progress_status.lock().unwrap();
            s.frames_processed = p.frames_processed;
            s.total_frames = p.total_frames;
        });
        let mut s = status.lock().unwrap();
        match result {
            Ok(()) => s.state = "done".into(),
            Err(e) => {
                s.state = "failed".into();
                s.error = Some(e);
            }
        }
    });
    Ok("render started — poll render_status".into())
}

/// Render lifecycle status (polled over MCP; no push notifications yet).
#[cfg(feature = "render")]
#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RenderStatus {
    /// idle | running | done | failed
    pub state: String,
    pub frames_processed: u64,
    pub total_frames: u64,
    pub error: Option<String>,
}

impl Default for RenderStatus {
    fn default() -> Self {
        Self {
            state: "idle".into(),
            frames_processed: 0,
            total_frames: 0,
            error: None,
        }
    }
}

/// Current render status (idle when nothing has run yet).
#[cfg(feature = "render")]
pub fn render_status() -> RenderStatus {
    RENDER_STATUS
        .get()
        .map(|s| s.lock().unwrap().clone())
        .unwrap_or_default()
}

/// Abort the active render (pipeline checks the flag between frames).
#[cfg(feature = "render")]
pub fn cancel_render() {
    if let Some(c) = CANCEL_RENDER.get() {
        c.store(true, Ordering::Relaxed);
    }
}
