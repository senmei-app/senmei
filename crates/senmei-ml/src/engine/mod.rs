#[cfg(any(feature = "burn", feature = "tch"))]
pub mod core;

use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use std::path::Path;

#[cfg(any(feature = "burn", feature = "tch"))]
pub use core::Model;

#[derive(Debug, Clone, Copy)]
pub struct EngineCaps {
    pub tiles: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct InferOptions {
    pub tile_size: Option<u32>,
}

/// A fused RGB8 batch whose GPU readback is still pending. `resolve` blocks
/// until the transfer completes and converts to packed rgb24 on the CPU. Split
/// from the forward so the caller can queue the next batch's GPU work before
/// resolving the previous readback (keeps the GPU busy during the transfer).
/// Raw decoded frames: (rgb24 bytes, width, height).
pub type Rgb8Frames = Vec<(Vec<u8>, u32, u32)>;

pub trait Rgb8Batch: Send {
    fn resolve(self: Box<Self>) -> Result<Vec<(Vec<u8>, u32, u32)>>;
}

pub trait InferenceEngine: Send + Sync {
    fn capabilities(&self) -> EngineCaps;
    fn load(&mut self, model: &ModelRef) -> Result<()>;

    /// Optional warmup: run one representative forward so the backend's
    /// autotune cache is populated before the first real frame (avoids
    /// first-render stutter). Default no-op.
    fn warmup(&mut self) {}

    /// The model's native upscale factor (1 for non-upscale models). Used to
    /// keep the fused RGB8 path when the requested scale differs.
    fn native_scale(&self) -> u32 {
        1
    }

    fn infer(&mut self, input: &Tensor, opts: &InferOptions) -> Result<Tensor>;

    /// Optional fused path: infer and hand back packed RGB8 bytes directly,
    /// bypassing the CPU f32 intermediate + `tensor_to_frame`. Returns `None`
    /// to fall back to `infer` + `tensor_to_frame`. `scale` is the requested
    /// upscale factor. Implementations must tile internally — a full-frame
    /// pass OOMs autotune on large matmuls (see docs/upstream-issues.md §2).
    fn infer_rgb8(&mut self, input: &Tensor, scale: u32) -> Option<Result<(Vec<u8>, u32, u32)>> {
        let _ = (input, scale);
        None
    }

    /// Optional fused multi-frame path: `n` same-shaped inputs, one RGB8
    /// result per input, bit-identical to `n` separate `infer_rgb8` calls.
    /// Returns `None` to fall back to per-frame `infer_rgb8`.
    fn infer_rgb8_batch(&mut self, inputs: &[Tensor], scale: u32) -> Option<Result<Rgb8Frames>> {
        let _ = (inputs, scale);
        None
    }

    /// Fused multi-frame RGB8, split for readback pipelining: queues the GPU
    /// work and returns a handle whose `resolve` performs the readback + CPU
    /// conversion. The caller keeps one batch in flight so the next forward
    /// overlaps the previous readback (the wgpu queue interleaves them).
    /// `None` falls back to `infer_rgb8_batch` / per-frame `infer_rgb8`.
    fn infer_rgb8_submit(
        &mut self,
        inputs: &[Tensor],
        scale: u32,
    ) -> Option<Result<Box<dyn Rgb8Batch>>> {
        let _ = (inputs, scale);
        None
    }

    /// Optional two-input frame interpolation (e.g. RIFE): produce the frame at
    /// time `t` in [0,1] between `a` and `b` (both NCHW). Returns `None` to
    /// fall back to the CPU blend.
    fn infer_interp(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        t: f32,
        opts: &InferOptions,
    ) -> Option<Result<Tensor>> {
        let _ = (a, b, t, opts);
        None
    }

    /// Optional single-input denoise: 3-channel NCHW `[0,1]` → 3-channel
    /// estimate, `sigma` = noise level in `[0,1]` (fed as the model's constant
    /// noise-level map). Returns `None` to fall back to the CPU reference.
    fn infer_denoise(
        &mut self,
        input: &Tensor,
        sigma: f32,
        opts: &InferOptions,
    ) -> Option<Result<Tensor>> {
        let _ = (input, sigma, opts);
        None
    }
}

/// Run an engine over a full input, tiling when the engine advertises tile support
/// and the input exceeds `opts.tile_size`. Tile outputs are stitched with overlap
/// averaging; the output canvas is scaled by the engine's per-tile scale factor.
///
/// Small inputs (≤ full-HD) run as a single full-frame pass — no per-tile GPU
/// sync at all. Larger inputs are padded to a uniform tile grid and processed in
/// small GPU batches (activations stay bounded, syncs drop to ~tiles/4).
pub fn infer_tiled(
    engine: &mut dyn InferenceEngine,
    input: &Tensor,
    opts: &InferOptions,
) -> Result<Tensor> {
    // Upscale models (convs) are translation-equivariant, so tiling is valid;
    // keep the 1080p full-frame threshold (GPU saturation).
    run_tiled(engine, input, opts, 1920 * 1080, |e, t, o| e.infer(t, o))
}

/// Tiled `infer_denoise`. U-Net denoisers (SCUNet/DRUNet) are **not**
/// translation-equivariant — window attention (Swin) re-partitions the grid per
/// tile and the ÷8 down/upsample pyramid has wide border effects, so tiled
/// output differs from full-frame globally (verified: scunet tiled vs full
/// mae ≈ 0.13) and reads as ghost copies on moving content. DnCNN/FFDNet tile
/// fine, but a shared threshold keeps this correct for the models that matter:
/// run full-frame up to 4K (the models pad internally to their stride) and only
/// fall back to tiling for very large inputs to bound memory.
pub fn infer_denoise_tiled(
    engine: &mut dyn InferenceEngine,
    input: &Tensor,
    sigma: f32,
    opts: &InferOptions,
) -> Result<Tensor> {
    run_tiled(engine, input, opts, 3840 * 2160, |e, t, o| {
        e.infer_denoise(t, sigma, o)
            .unwrap_or_else(|| Err(Error::new("engine has no denoise path")))
    })
}

fn run_tiled(
    engine: &mut dyn InferenceEngine,
    input: &Tensor,
    opts: &InferOptions,
    full_frame_pixels: usize,
    infer: impl Fn(&mut dyn InferenceEngine, &Tensor, &InferOptions) -> Result<Tensor>,
) -> Result<Tensor> {
    let caps = engine.capabilities();
    let Some(tile_size) = opts.tile_size else {
        return infer(engine, input, opts);
    };
    if !caps.tiles {
        return infer(engine, input, opts);
    }
    let tile = tile_size as usize;
    let h = input.shape[2];
    let w = input.shape[3];
    if h <= tile && w <= tile {
        return infer(engine, input, opts);
    }

    // Full-frame single pass keeps the GPU saturated (matches TensorRT-style
    // whole-frame inference); tile only above the caller's full-frame cap.
    if h * w <= full_frame_pixels {
        return infer(engine, input, opts);
    }

    let overlap = tile / 4;
    let step = tile - overlap;
    let num_y = (h.saturating_sub(tile)).div_ceil(step) + 1;
    let num_x = (w.saturating_sub(tile)).div_ceil(step) + 1;
    let ph = (num_y - 1) * step + tile;
    let pw = (num_x - 1) * step + tile;
    let padded = crate::pad_to(input, ph, pw);
    let tiles = crate::uniform_tile(&padded, tile, step);
    debug_assert!(tiles
        .iter()
        .all(|(_, _, t)| t.shape[2] == tile && t.shape[3] == tile));

    let c = input.shape[1];
    const BATCH: usize = 4;
    let mut out_tiles: Vec<(usize, usize, Tensor)> = Vec::with_capacity(tiles.len());
    for chunk in tiles.chunks(BATCH) {
        let n = chunk.len();
        let mut data = Vec::with_capacity(n * c * tile * tile);
        for (_, _, t) in chunk {
            data.extend_from_slice(&t.data);
        }
        let batch = Tensor::new(vec![n, c, tile, tile], data);
        let out_batch = infer(engine, &batch, opts)?;
        let oc = out_batch.shape[1];
        let oh = out_batch.shape[2];
        let ow = out_batch.shape[3];
        let per = oc * oh * ow;
        for (i, (x, y, _)) in chunk.iter().enumerate() {
            let s = i * per;
            out_tiles.push((
                *x,
                *y,
                Tensor::new(vec![1, oc, oh, ow], out_batch.data[s..s + per].to_vec()),
            ));
        }
    }

    let scale_h = out_tiles[0].2.shape[2] as f32 / tile as f32;
    let scale_w = out_tiles[0].2.shape[3] as f32 / tile as f32;
    let out_h = (ph as f32 * scale_h).round() as usize;
    let out_w = (pw as f32 * scale_w).round() as usize;
    let scaled: Vec<(usize, usize, Tensor)> = out_tiles
        .iter()
        .map(|(x, y, t)| {
            let sx = (*x as f32 * scale_w).round() as usize;
            let sy = (*y as f32 * scale_h).round() as usize;
            (sx, sy, t.clone())
        })
        .collect();
    let ov = (
        (overlap as f32 * scale_h).round() as usize,
        (overlap as f32 * scale_w).round() as usize,
    );
    let stitched = crate::stitch(&scaled, out_h, out_w, input.shape[1], ov);
    let out_h_target = (h as f32 * scale_h).round() as usize;
    let out_w_target = (w as f32 * scale_w).round() as usize;
    Ok(crate::crop(&stitched, out_h_target, out_w_target))
}

/// Which inference backend to use. `Auto` picks the best compiled one.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, specta::Type,
)]
#[serde(rename_all = "camelCase")]
pub enum EngineBackend {
    /// LibTorch if compiled, else Vulkan.
    #[default]
    Auto,
    Vulkan,
    LibTorch,
}

/// Runtime info about the compiled backends (settings UI).
#[derive(Debug, Clone, serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct BackendInfo {
    pub vulkan_compiled: bool,
    pub libtorch_compiled: bool,
    /// LibTorch version string when the `tch` feature is compiled.
    pub libtorch_version: Option<String>,
    /// CUDA/ROCm device present (only meaningful when libtorch is compiled).
    pub cuda_available: bool,
    pub cuda_device_count: u32,
}

pub fn backend_info() -> BackendInfo {
    // Runtime dlopen probe — `tch::Cuda` would need the loader initialized.
    let hw = crate::runtime::detect();
    let gpu_count =
        hw.cuda.as_ref().map_or(0, |v| v.len()) + hw.rocm.as_ref().map_or(0, |v| v.len());
    BackendInfo {
        vulkan_compiled: cfg!(feature = "burn"),
        libtorch_compiled: cfg!(feature = "tch"),
        #[cfg(feature = "tch")]
        libtorch_version: Some(crate::tch::LIBTORCH_VERSION.to_string()),
        #[cfg(not(feature = "tch"))]
        libtorch_version: None,
        cuda_available: hw.supports_gpu(),
        cuda_device_count: gpu_count as u32,
    }
}

/// Pick an engine for a model based on the requested backend and the compiled
/// features. `Auto` prefers the `tch` (libtorch) engine when a CUDA/ROCm GPU is
/// present, falling back to Vulkan. `data_dir` is where the runtime libtorch is
/// downloaded/cached (see `crate::runtime`).
pub fn engine_for_model(
    model: &ModelRef,
    backend: EngineBackend,
    data_dir: &Path,
) -> Result<Box<dyn InferenceEngine>> {
    let _ = model;
    let _ = data_dir; // only used under `feature = "tch"`
    match backend {
        EngineBackend::Vulkan => {
            #[cfg(feature = "burn")]
            {
                log::info!("engine: burn-Vulkan ({})", model.id);
                Ok(Box::new(crate::burn::BurnEngine::new()))
            }
            #[cfg(not(feature = "burn"))]
            {
                Err(crate::Error::new(
                    "vulkan backend not compiled (enable the `burn` feature)",
                ))
            }
        }
        EngineBackend::LibTorch => {
            #[cfg(feature = "tch")]
            {
                log::info!("engine: libtorch ({})", model.id);
                Ok(Box::new(crate::tch::TchEngine::runtime(data_dir)?))
            }
            #[cfg(not(feature = "tch"))]
            {
                Err(crate::Error::new(
                    "libtorch backend not compiled (enable the `tch` feature)",
                ))
            }
        }
        EngineBackend::Auto => {
            #[cfg(feature = "tch")]
            match crate::tch::TchEngine::runtime(data_dir) {
                Ok(engine) => {
                    log::info!("engine: auto -> libtorch ({})", model.id);
                    return Ok(Box::new(engine));
                }
                Err(e) => log::warn!(
                    "engine: auto -> libtorch unavailable ({}): {e}; falling back to burn-Vulkan",
                    model.id
                ),
            }
            #[cfg(feature = "burn")]
            {
                log::info!("engine: auto -> burn-Vulkan ({})", model.id);
                Ok(Box::new(crate::burn::BurnEngine::new()))
            }
            #[cfg(not(feature = "burn"))]
            {
                return Err(crate::Error::new(
                    "no engine available (libtorch needs a CUDA/ROCm device and `burn` is not compiled)",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(h: usize, w: usize) -> Tensor {
        let mut data = vec![0f32; 3 * h * w];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i % 251) as f32 / 251.0;
        }
        Tensor::new(vec![1, 3, h, w], data)
    }

    struct IdentityEngine;

    impl InferenceEngine for IdentityEngine {
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { tiles: true }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            Ok(input.clone())
        }
    }

    struct ScaleEngine {
        factor: usize,
    }

    impl InferenceEngine for ScaleEngine {
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { tiles: true }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            let h = input.shape[2];
            let w = input.shape[3];
            Ok(crate::bilinear(input, h * self.factor, w * self.factor))
        }
    }

    struct NoTilesEngine;

    impl InferenceEngine for NoTilesEngine {
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { tiles: false }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            Ok(input.clone())
        }
    }

    struct CountingEngine {
        calls: usize,
    }

    impl InferenceEngine for CountingEngine {
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { tiles: true }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            self.calls += 1;
            Ok(input.clone())
        }
    }

    #[test]
    fn tiled_identity_reconstructs() {
        let t = input(16, 16);
        let mut engine = IdentityEngine;
        let opts = InferOptions { tile_size: Some(8) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.shape, t.shape);
        assert_eq!(out.data, t.data);
    }

    #[test]
    fn tiled_scaled_dims() {
        let t = input(16, 16);
        let mut engine = ScaleEngine { factor: 2 };
        let opts = InferOptions { tile_size: Some(8) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.shape, vec![1, 3, 32, 32]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn small_input_skips_tiling() {
        let t = input(8, 8);
        let mut engine = ScaleEngine { factor: 2 };
        let opts = InferOptions {
            tile_size: Some(16),
        };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.shape, vec![1, 3, 16, 16]);
    }

    #[test]
    fn engine_without_tiling_goes_whole() {
        let t = input(16, 16);
        let mut engine = NoTilesEngine;
        let opts = InferOptions { tile_size: Some(8) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.data, t.data);
    }

    #[test]
    fn hd_input_runs_single_full_frame_pass() {
        // 720p > tile size but ≤ full-HD threshold → exactly one engine call.
        let t = input(720, 1280);
        let mut engine = CountingEngine { calls: 0 };
        let opts = InferOptions {
            tile_size: Some(512),
        };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(engine.calls, 1);
        assert_eq!(out.shape, t.shape);
    }

    #[test]
    fn large_input_batches_tiles_in_chunks() {
        // 1080p input wider than full-HD threshold → tiled, but batched in
        // chunks of 4 (15 tiles → 4 engine calls instead of 15).
        let t = input(1080, 2048);
        let mut engine = CountingEngine { calls: 0 };
        let opts = InferOptions {
            tile_size: Some(512),
        };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert!(
            engine.calls < 8,
            "expected few batched calls, got {}",
            engine.calls
        );
        assert_eq!(out.shape, vec![1, 3, 1080, 2048]);
    }
}
