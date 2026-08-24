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

/// Extract one frame as a base64 PNG (fast seek via an ffmpeg pipe).
pub fn frame_png(input: &str, position_ms: f64) -> Result<String, String> {
    let ff = ffmpeg();
    let out = std::process::Command::new(&ff)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            &format!("{:.3}", position_ms / 1000.0),
            "-i",
            input,
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2pipe",
            "-",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("frame extraction failed for {input}"));
    }
    use base64::Engine as _;
    Ok(base64::engine::general_purpose::STANDARD.encode(out.stdout))
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

/// Download a model's weights (`.pth`/`.onnx`/ncnn `.bin`, sha256-verified when
/// pinned) and convert to the f16 `.bpk` burnpack. Handles RIFE's ncnn release
/// zips (extract one entry) and skips when the target already exists. Mirrors
/// the GUI's `download_model` without Tauri. Needs the `render` feature (burn
/// convert). `on_progress` receives (downloaded, total) bytes.
#[cfg(feature = "render")]
pub fn download_model(
    model_id: &str,
    mut on_progress: impl FnMut(u64, u64) + Send,
) -> Result<String, String> {
    let (registry, dir) = load_registry()?;
    let meta = registry
        .models()
        .iter()
        .find(|m| m.id == model_id)
        .cloned()
        .ok_or_else(|| format!("model not found: {model_id}"))?;
    let convert_arg = registry
        .resolve(model_id, &dir)
        .map(|m| match m.arch.as_str() {
            "span" => m.feature_channels,
            "srvgg" => m.num_conv,
            _ => m.num_block,
        })
        .unwrap_or(4);
    let layer_norm = registry.resolve(model_id, &dir).map(|m| m.layer_norm).unwrap_or(false);
    let dysample = registry.resolve(model_id, &dir).map(|m| m.dysample).unwrap_or(true);
    let shuffle = registry.resolve(model_id, &dir).map(|m| m.shuffle).unwrap_or(1);
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
    let onnx = std::path::Path::new(&url).extension().and_then(|e| e.to_str()) == Some("onnx");
    let st = std::path::Path::new(&url).extension().and_then(|e| e.to_str()) == Some("safetensors");
    let ext = if onnx {
        "onnx"
    } else if st {
        "safetensors"
    } else if is_archive {
        "zip"
    } else {
        "pth"
    };
    log::info!("download_model: {model_id} <- {url} -> {}", dir.display());
    let base = weight.trim_end_matches(".f16.bpk");
    let source = senmei_media::download_to_temp(
        &url,
        &dir,
        &format!("{base}.{ext}"),
        meta.sha256.as_deref(),
        &mut on_progress,
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
        return Ok(target.to_string_lossy().into_owned());
    }
    if is_ncnn {
        std::fs::rename(&source, &target).map_err(|e| {
            log::error!("download_model {model_id}: rename failed: {e}");
            e.to_string()
        })?;
        log::info!("download_model: {model_id} wrote {}", target.display());
        return Ok(target.to_string_lossy().into_owned());
    }
    let conv = if onnx {
        senmei_ml::convert_onnx_to_bpk(
            &meta.arch, &source, &target, meta.scale, convert_arg, shuffle,
        )
    } else if st {
        senmei_ml::convert_safetensors_to_bpk(&meta.arch, &source, &target, meta.scale)
    } else {
        senmei_ml::convert_pth_to_bpk(
            &meta.arch,
            &source,
            &target,
            meta.scale,
            convert_arg,
            layer_norm,
            dysample,
            shuffle,
        )
    };
    if let Err(e) = conv {
        log::error!("download_model {model_id}: conversion failed: {e}");
        return Err(e.to_string());
    }
    let _ = std::fs::remove_file(&source);
    log::info!("download_model: {model_id} wrote {}", target.display());
    Ok(target.to_string_lossy().into_owned())
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

#[cfg(feature = "render")]
impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            tile_size: 0,
            pipeline_depth: 0,
            backend: senmei_ml::EngineBackend::default(),
            gpu_index: 0,
            cancel: None,
            pause: None,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterConfig {
    /// Denoise box-blur radius; `denoise_model_id` overrides with a learned model.
    pub denoise_radius: Option<u32>,
    /// Learned denoise model id (kind=denoise); empty = reference filter only.
    pub denoise_model_id: Option<String>,
    /// Deblur unsharp-mask amount; `deblur_model_id` overrides with a learned model.
    pub deblur_amount: Option<f32>,
    /// Learned deblur model id (kind=deblur); empty = reference filter only.
    pub deblur_model_id: Option<String>,
    /// Dedup mean-pixel-diff threshold in [0,1]; drops near-duplicate frames.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub dedup_threshold: Option<f32>,
    /// Free-form FFmpeg `-vf` filter graph applied per frame (e.g.
    /// `"hue=h=10,unsharp"`). Frame-preserving only (1:1) — filters that change
    /// the output size are rejected. Runs between the reference/ML filters and
    /// the final `output_resize`.
    pub ffmpeg_filter: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderConfig {
    /// Input video path (required).
    pub input: String,
    /// Output video path (required); container guessed from extension.
    pub output: String,
    /// Integer upscale factor for the upscale step (1..=4).
    #[schemars(range(min = 1, max = 4))]
    pub scale: Option<u32>,
    /// Upscale model id (kind=upscale, license-gated); empty = reference upscale only.
    pub model_id: Option<String>,
    /// Decompress model id (scale-1 de-artifact pass, e.g. RealPLKSR 1×).
    pub decompress_model_id: Option<String>,
    /// Pre-upscale resize factor (>0, e.g. 0.5 to shrink first).
    pub resize: Option<f32>,
    /// Optional filter chain: denoise / deblur / dedup.
    pub filter: Option<FilterConfig>,
    /// Post-upscale resize factor (>0, e.g. 0.5 for a net 1x).
    pub output_resize: Option<f32>,
    /// Frame-rate multiplier for interpolation (1..=16).
    #[schemars(range(min = 1, max = 16))]
    pub fps_multiplier: Option<u32>,
    /// Interpolation model id (kind=interpolate); empty = linear blend.
    pub interp_model: Option<String>,
    /// Extra raw ffmpeg output args; appended last, override structured fields.
    pub ffmpeg_args: Option<Vec<String>>,
    /// HDR→SDR tonemapping for decode: "auto" | "always" | "off".
    pub tonemap: Option<String>,
    /// Render only from this timestamp (ms); pairs with `end_ms` for samples.
    pub start_ms: Option<u64>,
    /// Render only up to this timestamp (ms).
    pub end_ms: Option<u64>,
}

/// Enriched settings schema for agents — works without the `render` feature.
/// Returns the render-config JSON Schema (schemars), the model slots (which
/// registry models fill which config field) and the hard constraints (license
/// gate, ranges, confirm gate).
pub fn settings_schema() -> serde_json::Value {
    let config_schema = schemars::schema_for!(RenderConfig);
    let models = list_models();

    let slot = |field: &str, kind: &str| -> serde_json::Value {
        let kind_v = serde_json::Value::String(kind.to_string());
        let candidates: Vec<serde_json::Value> = models
            .iter()
            .filter(|m| serde_json::to_value(&m.kind).ok() == Some(kind_v.clone()))
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "kind": m.kind,
                    "scale": m.scale,
                    "arch": m.arch,
                    "loadable": m.loadable,
                    "license": m.license,
                    "licenseBlocked": m.license_blocked(),
                })
            })
            .collect();
        serde_json::json!({ "field": field, "kind": kind, "models": candidates })
    };

    serde_json::json!({
        "renderConfig": serde_json::to_value(&config_schema).unwrap_or_default(),
        "modelSlots": [
            slot("model_id", "upscale"),
            slot("interp_model", "interpolate"),
            slot("filter.denoise_model_id", "denoise"),
            slot("filter.deblur_model_id", "deblur"),
        ],
        "constraints": {
            "scale": "1..=4",
            "fpsMultiplier": "1..=16",
            "tonemap": ["auto", "always", "off"],
            "licenseGate": "every referenced model must be permissive + loadable (licenseBlocked/!loadable rejected)",
            "confirmGate": "full render starts only after propose_render + confirm_render",
        }
    })
}

/// Parse the last occurrence of `key` in ffmpeg's stderr summary lines
/// (PSNR `average:`, SSIM `All:`).
fn parse_after(stderr: &str, key: &str) -> Option<f64> {
    stderr.lines().rev().find_map(|l| {
        let rest = l.split_once(key)?.1.trim_start();
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        num.parse().ok()
    })
}

/// Run one FFmpeg metric filter (psnr/ssim) between two clips, scaling the
/// rendered clip back to the original resolution. Returns the parsed summary.
fn run_metric(
    ff: &Path,
    rendered: &str,
    original: &str,
    scale: &str,
    filter: &str,
    key: &str,
) -> Result<Option<f64>, String> {
    let lavfi = format!("[0:v]{scale}format=yuv420p[s];[1:v]format=yuv420p[r];[s][r]{filter}");
    let out = std::process::Command::new(ff)
        .args([
            "-hide_banner",
            "-i",
            rendered,
            "-i",
            original,
            "-lavfi",
            &lavfi,
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg {filter} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(parse_after(&String::from_utf8_lossy(&out.stderr), key))
}

/// Compare a rendered sample against its source: PSNR (dB) + SSIM, both on the
/// original resolution (the rendered clip is scaled back down). VMAF is null —
/// it needs a libvmaf FFmpeg build (§16.5).
pub fn compare_sample(original: &str, rendered: &str) -> Result<serde_json::Value, String> {
    let ff = ffmpeg();
    let ffprobe = senmei_media::ffprobe_next_to(&ff);
    let orig = senmei_media::probe(&ffprobe, Path::new(original)).map_err(|e| e.to_string())?;
    let rend = senmei_media::probe(&ffprobe, Path::new(rendered)).map_err(|e| e.to_string())?;

    let scale = if (orig.width, orig.height) != (rend.width, rend.height) {
        format!("scale={}:{}:flags=bicubic,", orig.width, orig.height)
    } else {
        String::new()
    };

    let psnr_db = run_metric(&ff, rendered, original, &scale, "psnr", "average:")?;
    let ssim = run_metric(&ff, rendered, original, &scale, "ssim", "All:")?;

    Ok(serde_json::json!({
        "original": { "path": original, "width": orig.width, "height": orig.height },
        "rendered": { "path": rendered, "width": rend.width, "height": rend.height },
        "psnrDb": psnr_db,
        "ssim": ssim,
        "vmaf": null,
        "note": "PSNR/SSIM on the original resolution (rendered downscaled); VMAF needs a libvmaf FFmpeg build",
    }))
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
                        log::warn!("interpolation model {id} unavailable, using CPU interpolator: {e}");
                        None
                    }
                },
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
    if render_status().state == "running" {
        return Err("a render is already running".into());
    }
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

    #[cfg(feature = "render")]
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
    fn settings_schema_has_slots_and_constraints() {
        let schema = settings_schema();
        let obj = schema.as_object().expect("schema is an object");
        assert!(obj.contains_key("renderConfig"));
        assert!(obj.contains_key("constraints"));

        let slots = obj["modelSlots"].as_array().expect("modelSlots is an array");
        assert_eq!(slots.len(), 4);
        for slot in slots {
            assert!(slot.get("field").is_some());
            assert!(slot.get("kind").is_some());
            assert!(slot.get("models").is_some());
        }

        let upscale = slots
            .iter()
            .find(|s| s["field"] == "model_id")
            .expect("upscale slot present");
        assert_eq!(upscale["kind"], "upscale");
        let ids: Vec<&str> = upscale["models"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(
            ids.contains(&"span-2x-nomosuni-ldl"),
            "SPAN registered: {ids:?}"
        );
    }

    #[test]
    fn parses_metric_summaries() {
        let psnr = "[Parsed_psnr_0 @ 0x55] PSNR y:37.0 u:41.0 v:41.0 average:38.234 min:36.1 max:40.2";
        assert_eq!(parse_after(psnr, "average:"), Some(38.234));
        let ssim = "[Parsed_ssim_0 @ 0x55] SSIM Y:0.98 (12.0) U:0.97 (11.0) V:0.96 (10.0) All:0.981234 (12.3)";
        assert_eq!(parse_after(ssim, "All:"), Some(0.981234));
        // summary = LAST matching line
        assert_eq!(parse_after("All:0.1 (1)\nAll:0.9 (2)", "All:"), Some(0.9));
    }

    #[cfg(feature = "render")]
    #[test]
    fn validate_rejects_bad_ranges() {
        let base = || RenderConfig {
            input: "in.mp4".into(),
            output: "out.mp4".into(),
            ..Default::default()
        };
        assert!(validate(&base()).is_ok());

        let bad = |cfg: RenderConfig| validate(&cfg).unwrap_err();
        assert!(bad(RenderConfig { scale: Some(5), ..base() }).contains("scale"));
        assert!(bad(RenderConfig { scale: Some(0), ..base() }).contains("scale"));
        assert!(bad(RenderConfig { fps_multiplier: Some(0), ..base() }).contains("fps_multiplier"));
        assert!(bad(RenderConfig { tonemap: Some("weird".into()), ..base() }).contains("tonemap"));
        assert!(bad(RenderConfig { resize: Some(0.0), ..base() }).contains("resize"));
        assert!(bad(RenderConfig { start_ms: Some(2000), end_ms: Some(1000), ..base() }).contains("end_ms"));
        assert!(bad(RenderConfig {
            filter: Some(FilterConfig { dedup_threshold: Some(1.5), ..Default::default() }),
            ..base()
        }).contains("dedup"));
    }
}
