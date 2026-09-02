//! Render execution: step assembly, lifecycle status, confirm gate (`render` feature).

mod lifecycle;

use super::config::RenderConfig;
use super::{data_dir, ffmpeg, load_registry};
use std::path::{Path, PathBuf};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub use lifecycle::{cancel_render, confirm_render, propose_render, render_status, RenderStatus};
pub use senmei_pipeline::Progress as RenderProgress;

/// Serializes renders across transports: a new render is rejected while one is
/// still running — including its cleanup.
static RENDER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// RAII guard that frees [`RENDER_ACTIVE`] on drop.
struct RenderGate;

impl RenderGate {
    fn acquire() -> Result<Self, String> {
        if RENDER_ACTIVE.swap(true, Ordering::SeqCst) {
            return Err("a render is already running".into());
        }
        Ok(RenderGate)
    }
}

impl Drop for RenderGate {
    fn drop(&mut self) {
        RENDER_ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// Extra knobs the caller may pass into [`render`].
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

/// One pipeline step's timing (FPS benchmark; ms/frame + fps).
#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StepTimingInfo {
    pub name: String,
    pub frames: u64,
    pub ms_per_frame: f64,
    pub fps: f64,
}

/// Load a model engine, enforcing the license gate (hard).
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

/// Validate a render config: required paths, sane ranges, and every referenced
/// model must exist with a permissive license.
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

fn build_steps(
    config: &RenderConfig,
    backend: senmei_ml::EngineBackend,
) -> Result<Vec<Box<dyn senmei_pipeline::Step>>, String> {
    let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Passthrough)];
    if let Some(f) = config.resize {
        steps.push(Box::new(senmei_pipeline::Resize::new(f)));
    }
    let optional = |id: &str| match engine_for_model(id, backend) {
        Ok(e) => Some(e),
        Err(e) => {
            log::warn!("model {id} unavailable, using reference filter: {e}");
            None
        }
    };
    if let Some(id) = config.decompress_model_id.as_deref() {
        if !id.is_empty() {
            let engine = optional(id);
            steps.push(Box::new(senmei_pipeline::Upscale::new(1, engine)));
        }
    }
    if let Some(s) = config.scale {
        if s > 1 {
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

/// Run a render (blocking; call from spawn_blocking).
pub fn render(
    config: &RenderConfig,
    opts: &RenderOpts,
    on_progress: impl FnMut(RenderProgress) + Send + 'static,
) -> Result<Vec<StepTimingInfo>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        render_inner(config, opts, on_progress)
    }))
    .unwrap_or_else(|p| {
        if !config.output.is_empty() {
            let _ = std::fs::remove_file(&config.output);
        }
        Err(format!("render panicked: {}", panic_message(&p)))
    })
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

fn render_inner(
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
        None => lifecycle::CANCEL_RENDER
            .get_or_init(|| Arc::new(AtomicBool::new(false)))
            .clone(),
    };
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
fn extract_frame(ff: &Path, input: &str, at_secs: f64, out_png: &str) -> Result<(), String> {
    let status = senmei_media::process::hidden(ff)
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

/// Render a short sample range synchronously — no confirm gate.
pub fn render_sample(config: RenderConfig) -> Result<serde_json::Value, String> {
    validate(&config)?;
    let (start, end) = match (config.start_ms, config.end_ms) {
        (Some(s), Some(e)) if e > s => (s, e),
        _ => return Err("render_sample requires start_ms < end_ms".into()),
    };
    let mut config = config;
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
    fn panic_message_extracts_str_and_string() {
        let s: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert_eq!(panic_message(&s), "boom");
        let owned: Box<dyn std::any::Any + Send> = Box::new("bang".to_string());
        assert_eq!(panic_message(&owned), "bang");
        let other: Box<dyn std::any::Any + Send> = Box::new(7);
        assert_eq!(panic_message(&other), "unknown panic");
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
