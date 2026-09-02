//! Render execution: step assembly, lifecycle status, confirm gate (`render` feature).

use super::config::RenderConfig;
use super::{data_dir, ffmpeg, load_registry};
use std::path::{Path, PathBuf};

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

/// Serializes renders across transports (GUI command, MCP, HTTP): a new render
/// is rejected while one is still running — including its cleanup, so cancel +
/// immediate re-render never overlap two GPU engines.
#[cfg(feature = "render")]
static RENDER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// RAII guard that frees [`RENDER_ACTIVE`] on drop, including early `?` returns.
#[cfg(feature = "render")]
struct RenderGate;

#[cfg(feature = "render")]
impl RenderGate {
    fn acquire() -> Result<Self, String> {
        if RENDER_ACTIVE.swap(true, Ordering::SeqCst) {
            return Err("a render is already running".into());
        }
        Ok(RenderGate)
    }
}

#[cfg(feature = "render")]
impl Drop for RenderGate {
    fn drop(&mut self) {
        RENDER_ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// Extra knobs the caller may pass into [`render`]: the fused-RGB8 tile size
/// (0 = engine default 640) and the caller's own cancel/pause flags. When
/// `cancel`/`pause` are `None`, the shared core flags (used by
/// `confirm_render`/`cancel_render`) are used.
#[cfg(feature = "render")]
#[derive(Default)]
pub struct RenderOpts {
    pub tile_size: u32,
    /// Readback pipeline depth (batches kept in flight); 0 = default (2).
    pub pipeline_depth: usize,
    pub backend: senmei_ml::EngineBackend,
    /// Discrete-GPU index for inference (0 = first discrete GPU).
    pub gpu_index: u32,
    pub cancel: Option<Arc<AtomicBool>>,
    pub pause: Option<Arc<AtomicBool>>,
}

/// Load a model engine, enforcing the license gate (hard). Missing weights or
/// an unloadable arch are errors here — build steps may still fall back to the
/// reference filter when a model is unavailable (like the GUI).
#[cfg(feature = "render")]
pub fn engine_for_model(
    model_id: &str,
    backend: senmei_ml::EngineBackend,
) -> Result<Box<dyn senmei_ml::InferenceEngine>, String> {
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
    if !mref.path.is_file() {
        return Err(format!(
            "model {model_id} weights are not downloaded (expected {}); download the model first",
            mref.path.display()
        ));
    }
    let mut engine =
        senmei_ml::engine_for_model(&mref, backend, &data_dir()).map_err(|e| e.to_string())?;
    engine.load(&mref).map_err(|e| e.to_string())?;
    log::info!("engine: {model_id} weights loaded");
    Ok(engine)
}

/// Validate a render config: required paths, sane ranges (mirrors the settings
/// schema), and every referenced model must exist with a permissive license
/// (never a blocked one).
#[cfg(feature = "render")]
pub fn validate(config: &RenderConfig) -> Result<(), String> {
    if config.input.is_empty() || config.output.is_empty() {
        return Err("input and output are required".into());
    }
    if !(1..=4).contains(&config.scale.unwrap_or(1)) {
        return Err("scale must be in 1..=4".into());
    }
    if let Some(f) = config.resize {
        if f <= 0.0 {
            return Err("resize must be > 0".into());
        }
    }
    if let Some(f) = config.output_resize {
        if f <= 0.0 {
            return Err("output_resize must be > 0".into());
        }
    }
    if let Some(f) = config.fps_multiplier {
        if !(1..=16).contains(&f) {
            return Err("fps_multiplier must be in 1..=16".into());
        }
    }
    if let Some(t) = config.tonemap.as_deref() {
        if !matches!(t, "auto" | "always" | "off") {
            return Err("tonemap must be one of auto|always|off".into());
        }
    }
    if let (Some(s), Some(e)) = (config.start_ms, config.end_ms) {
        if e <= s {
            return Err("end_ms must be > start_ms".into());
        }
    }
    if let Some(f) = config.filter.as_ref() {
        if let Some(t) = f.dedup_threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err("dedup_threshold must be in 0..=1".into());
            }
        }
    }
    let mut ids: Vec<&str> = Vec::new();
    for id in [config.model_id.as_deref(), config.interp_model.as_deref()]
        .into_iter()
        .flatten()
    {
        ids.push(id);
    }
    if let Some(f) = config.filter.as_ref() {
        for id in [f.denoise_model_id.as_deref(), f.deblur_model_id.as_deref()]
            .into_iter()
            .flatten()
        {
            ids.push(id);
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
fn build_steps(
    config: &RenderConfig,
    backend: senmei_ml::EngineBackend,
) -> Result<Vec<Box<dyn senmei_pipeline::Step>>, String> {
    let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Passthrough)];
    if let Some(f) = config.resize {
        steps.push(Box::new(senmei_pipeline::Resize::new(f)));
    }
    // Optional aux models keep their reference fallback, but a load failure
    // is logged — never silent.
    let optional = |id: &str| match engine_for_model(id, backend) {
        Ok(e) => Some(e),
        Err(e) => {
            log::warn!("model {id} unavailable, using reference filter: {e}");
            None
        }
    };
    // Decompress pass runs first: scale-1 de-artifact (RealPLKSR 1×) ahead of
    // interpolation/upscaling. Skipped when the model can't be loaded.
    if let Some(id) = config.decompress_model_id.as_deref() {
        if !id.is_empty() {
            let engine = optional(id);
            steps.push(Box::new(senmei_pipeline::Upscale::new(1, engine)));
        }
    }
    if let Some(s) = config.scale {
        if s > 1 {
            // The main upscale model is mandatory: a missing/unloadable model
            // is a hard error, not a silent resize.
            let engine = match config.model_id.as_deref() {
                Some(id) if !id.is_empty() => Some(engine_for_model(id, backend)?),
                _ => None,
            };
            steps.push(Box::new(senmei_pipeline::Upscale::new(s, engine)));
        }
    }
    if let Some(f) = config.filter.as_ref() {
        if let Some(r) = f.denoise_radius {
            if r > 0 {
                let engine = match f.denoise_model_id.as_deref() {
                    Some(id) => optional(id),
                    None => None,
                };
                steps.push(Box::new(senmei_pipeline::Denoise::new(r, engine)));
            }
        }
        if let Some(a) = f.deblur_amount {
            if a > 0.0 {
                let engine = match f.deblur_model_id.as_deref() {
                    Some(id) => optional(id),
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
                steps.push(Box::new(senmei_pipeline::Filter::new(filter, ffmpeg())));
            }
        }
    }
    if let Some(f) = config.output_resize {
        steps.push(Box::new(senmei_pipeline::Resize::new(f)));
    }
    Ok(steps)
}

/// Run a render (blocking; call from spawn_blocking). Mirrors the GUI's
/// pipeline assembly, without Tauri.
#[cfg(feature = "render")]
pub fn render(
    config: &RenderConfig,
    opts: &RenderOpts,
    on_progress: impl FnMut(RenderProgress) + Send + 'static,
) -> Result<Vec<StepTimingInfo>, String> {
    let _gate = RenderGate::acquire()?;
    senmei_ml::set_tile_size(opts.tile_size);
    senmei_ml::set_gpu_index(opts.gpu_index);
    senmei_pipeline::set_pipeline_depth(opts.pipeline_depth);
    let cancel = match &opts.cancel {
        Some(c) => c.clone(),
        None => CANCEL_RENDER
            .get_or_init(|| Arc::new(AtomicBool::new(false)))
            .clone(),
    };
    // Clear before the (potentially slow) model load below, so a cancel
    // issued while models are loading isn't overwritten to false afterwards.
    cancel.store(false, Ordering::Relaxed);
    let ffmpeg = ffmpeg();
    let input = PathBuf::from(&config.input);
    let output = PathBuf::from(&config.output);
    let mut pipeline = senmei_pipeline::Pipeline::new(build_steps(config, opts.backend)?);
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
    pipeline.set_cancel(cancel);
    if let Some(p) = &opts.pause {
        p.store(false, Ordering::Relaxed);
        pipeline.set_pause(p.clone());
    }
    if let Some(f) = config.fps_multiplier {
        if f > 1 {
            let interp = match config.interp_model.as_deref() {
                Some(id) => match engine_for_model(id, opts.backend) {
                    Ok(e) => Some(senmei_pipeline::Interpolator::with_engine(f, e)),
                    Err(e) => {
                        log::warn!(
                            "interpolation model {id} unavailable, using CPU interpolator: {e}"
                        );
                        None
                    }
                },
                None => None,
            };
            pipeline
                .set_interpolator(interp.unwrap_or_else(|| senmei_pipeline::Interpolator::new(f)));
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
    let steps = pipeline
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
    run.map(|_| steps).map_err(|e| e.to_string())
}

/// Extract one frame as PNG (fast seek) — best effort.
#[cfg(feature = "render")]
fn extract_frame(ff: &Path, input: &str, at_secs: f64, out_png: &str) -> Result<(), String> {
    let status = std::process::Command::new(ff)
        .args([
            "-hide_banner",
            "-ss",
            &format!("{at_secs:.3}"),
            "-i",
            input,
            "-frames:v",
            "1",
            "-update",
            "1",
            "-y",
            out_png,
        ])
        .status()
        .map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("frame extraction failed for {input}"))
    }
}

/// Render a short sample range synchronously — no confirm gate (samples are
/// cheap). Returns the output path plus best-effort before/after PNG frames at
/// the range midpoint. Rejects while another render is running.
///
/// Samples are quality-check only, so audio is dropped (`-an`): the copied
/// audio input is exactly what needs `-ss`/`-t`/`-copyts`/`-shortest` mux
/// surgery on ranged renders (and has hung at 100% before). A single
/// rawvideo-pipe stream has no mux-sync hazard.
#[cfg(feature = "render")]
pub fn render_sample(config: RenderConfig) -> Result<serde_json::Value, String> {
    // The RenderGate inside render() serializes; no pre-check needed here.
    validate(&config)?;
    let (start, end) = match (config.start_ms, config.end_ms) {
        (Some(s), Some(e)) if e > s => (s, e),
        _ => return Err("render_sample requires start_ms < end_ms".into()),
    };
    let mut config = config;
    // Strip any caller audio codec (e.g. `-c:a copy`) then force `-an`.
    let args = config.ffmpeg_args.get_or_insert_with(Vec::new);
    args.retain(|a| a != "-an");
    if let Some(pos) = args.windows(2).position(|w| w[0] == "-c:a") {
        args.drain(pos..pos + 2);
    }
    args.push("-an".into());
    render(&config, &RenderOpts::default(), |_| {})?;

    let mid = start + (end - start) / 2;
    let ff = ffmpeg();
    let before = format!("{}.before.png", config.output);
    let after = format!("{}.after.png", config.output);
    let before_ok = extract_frame(&ff, &config.input, mid as f64 / 1000.0, &before).is_ok();
    let after_ok =
        extract_frame(&ff, &config.output, (mid - start) as f64 / 1000.0, &after).is_ok();

    Ok(serde_json::json!({
        "output": config.output,
        "beforeFrame": before_ok.then_some(before),
        "afterFrame": after_ok.then_some(after),
    }))
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
        let result = render(&config, &RenderOpts::default(), move |p| {
            let mut s = progress_status.lock().unwrap();
            s.frames_processed = p.frames_processed;
            s.total_frames = p.total_frames;
        });
        let mut s = status.lock().unwrap();
        match result {
            Ok(steps) => {
                s.state = "done".into();
                s.steps = steps;
            }
            Err(e) => {
                s.state = "failed".into();
                s.error = Some(e);
            }
        }
    });
    Ok("render started — poll render_status".into())
}

/// One pipeline step's timing (FPS benchmark; ms/frame + fps).
#[cfg(feature = "render")]
#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StepTimingInfo {
    pub name: String,
    pub frames: u64,
    pub ms_per_frame: f64,
    pub fps: f64,
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
    /// Per-step timing once the render finishes (FPS benchmark).
    pub steps: Vec<StepTimingInfo>,
}

#[cfg(feature = "render")]
impl Default for RenderStatus {
    fn default() -> Self {
        Self {
            state: "idle".into(),
            frames_processed: 0,
            total_frames: 0,
            error: None,
            steps: Vec::new(),
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
        log::info!("render cancelled (flag set)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::FilterConfig;

    #[test]
    fn render_gate_serializes() {
        let gate = RenderGate::acquire().unwrap();
        assert!(
            RenderGate::acquire().is_err(),
            "a second render must be rejected while one is active"
        );
        drop(gate);
        assert!(RenderGate::acquire().is_ok(), "gate must free on drop");
    }

    #[test]
    fn validate_rejects_bad_ranges() {
        let base = || RenderConfig {
            input: "in.mp4".into(),
            output: "out.mp4".into(),
            ..Default::default()
        };
        assert!(validate(&base()).is_ok());

        let bad = |cfg: RenderConfig| validate(&cfg).unwrap_err();
        assert!(bad(RenderConfig {
            scale: Some(5),
            ..base()
        })
        .contains("scale"));
        assert!(bad(RenderConfig {
            scale: Some(0),
            ..base()
        })
        .contains("scale"));
        assert!(bad(RenderConfig {
            fps_multiplier: Some(0),
            ..base()
        })
        .contains("fps_multiplier"));
        assert!(bad(RenderConfig {
            tonemap: Some("weird".into()),
            ..base()
        })
        .contains("tonemap"));
        assert!(bad(RenderConfig {
            resize: Some(0.0),
            ..base()
        })
        .contains("resize"));
        assert!(bad(RenderConfig {
            start_ms: Some(2000),
            end_ms: Some(1000),
            ..base()
        })
        .contains("end_ms"));
        assert!(bad(RenderConfig {
            filter: Some(FilterConfig {
                dedup_threshold: Some(1.5),
                ..Default::default()
            }),
            ..base()
        })
        .contains("dedup"));
    }
}
