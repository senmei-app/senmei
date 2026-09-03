//! Fused RGB8 inference path: GPU tile accumulation with feather blending,
//! single readback as packed rgb24 bytes. Shared by burn (Vulkan f16) and
//! tch (libtorch f32) engines.
#![cfg(any(feature = "burn", feature = "tch"))]

use super::core::Model;
use super::{Rgb8Batch, Rgb8Frames};
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{f16, Tensor as BurnTensor, TensorData};
use burn::tensor::{
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Element conversion traits
// ---------------------------------------------------------------------------

/// Convert a backend float element to a rounded `u8` (`(x+0.5) as u8`).
/// `burn` uses `f16`, `tch` `f32` — both route through their own `to_f32`.
pub(crate) trait ElemToU8 {
    fn to_u8(self) -> u8;
}
impl ElemToU8 for f32 {
    fn to_u8(self) -> u8 {
        (self + 0.5) as u8
    }
}
impl ElemToU8 for f16 {
    fn to_u8(self) -> u8 {
        (self.to_f32() + 0.5) as u8
    }
}

/// Convert an f32 into the backend's float element (f16 on burn/tch-f16, f32
/// on tch-f32) without materializing a full f32 copy of the buffer.
pub(crate) trait ElemFromF32 {
    fn from_f32(v: f32) -> Self;
}
impl ElemFromF32 for f32 {
    fn from_f32(v: f32) -> Self {
        v
    }
}
impl ElemFromF32 for f16 {
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}

/// The element capability the fused RGB8 path needs end-to-end: cast the f32
/// CPU buffer to the backend element on upload and back to `u8` on readback.
pub(crate) trait Rgb8Elem: ElemToU8 + ElemFromF32 + Copy {}
impl<T: ElemToU8 + ElemFromF32 + Copy> Rgb8Elem for T {}

// ---------------------------------------------------------------------------
// Tensor conversion helpers
// ---------------------------------------------------------------------------

/// Edge-replicate pad `ph×pw` writing the backend's float element directly —
/// no padded f32 buffer, no `to_burn` clone, no separate f32→f16 convert.
pub(super) fn pad_to_f16<B: Backend>(input: &Tensor, ph: usize, pw: usize) -> Vec<B::FloatElem>
where
    B::FloatElem: ElemFromF32,
{
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let mut data = Vec::with_capacity(c * ph * pw);
    data.resize(c * ph * pw, B::FloatElem::from_f32(0.0));
    for ci in 0..c {
        for yy in 0..ph {
            let sy = yy.min(h - 1);
            let row = &input.data[(ci * h + sy) * w..(ci * h + sy) * w + w];
            let dst = &mut data[(ci * ph + yy) * pw..(ci * ph + yy) * pw + pw];
            for (d, s) in dst.iter_mut().zip(row.iter()) {
                *d = B::FloatElem::from_f32(*s);
            }
            let last = B::FloatElem::from_f32(row[w - 1]);
            for d in dst.iter_mut().skip(w) {
                *d = last;
            }
        }
    }
    data
}

pub(super) fn pad_to_burn<B: Backend>(
    input: &Tensor,
    ph: usize,
    pw: usize,
    device: &B::Device,
) -> Result<BurnTensor<B, 4>>
where
    B::FloatElem: ElemFromF32,
{
    let [n, c, h, w] = [
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ];
    debug_assert!(ph >= h && pw >= w, "pad_to_burn only grows the canvas");
    Ok(BurnTensor::<B, 4>::from_data(
        TensorData::new(pad_to_f16::<B>(input, ph, pw), [n, c, ph, pw]),
        device,
    ))
}

// ---------------------------------------------------------------------------
// VRAM guard
// ---------------------------------------------------------------------------

/// Effective fused-path peak ceiling: the crash-safe cap (just under the
/// observed ~3.2 GB single-allocation OOM at 1080p×4) or half the GPU's total
/// VRAM when that is tighter.
pub(super) fn fused_peak_limit(total_vram: Option<u64>) -> u64 {
    const FUSED_PEAK_CEILING: u64 = (2 * 1024 + 512) * 1024 * 1024; // 2.5 GiB
    match total_vram {
        Some(t) => (t.saturating_mul(50) / 100).min(FUSED_PEAK_CEILING),
        None => FUSED_PEAK_CEILING,
    }
}

/// Fused-path VRAM guard: reject when the estimate exceeds this fraction of
/// the currently free VRAM (DRM sysfs).
const VRAM_THRESHOLD_PCT: u64 = 85;

/// Fused RGB8 path peak-allocation estimate (bytes), used by the VRAM guard.
pub(super) fn fused_peak_allocation(n: usize, out_h: usize, out_w: usize, c: usize) -> u64 {
    let canvas = (n * out_h * out_w * c * 2) as u64; // accs (f16)
    let readback = (n * out_h * out_w * 3 * 4) as u64; // packed rgb24 f32
    (canvas + readback) * 4
}

// ---------------------------------------------------------------------------
// Deferred readback
// ---------------------------------------------------------------------------

/// Deferred readback of one fused batch: blocks on the GPU→CPU transfer, then
/// converts to packed rgb24 on the CPU.
pub struct BurnRgb8Batch<B: Backend> {
    pub(super) accs: Vec<BurnTensor<B, 4>>,
    pub(super) out_w_t: usize,
    pub(super) out_h_t: usize,
}

impl<B: Backend> Rgb8Batch for BurnRgb8Batch<B>
where
    B::FloatElem: ElemToU8,
{
    fn resolve(self: Box<Self>) -> Result<Vec<(Vec<u8>, u32, u32)>> {
        let mut result = Vec::with_capacity(self.accs.len());
        for acc in self.accs {
            let [_, c, _, _] = acc.dims();
            let avg = acc
                .slice([0..1, 0..c, 0..self.out_h_t, 0..self.out_w_t])
                .permute([0, 2, 3, 1])
                * 255.0;
            let data: Vec<B::FloatElem> = match avg.into_data().to_vec::<B::FloatElem>() {
                Ok(v) => v,
                Err(e) => return Err(Error::new(e.to_string())),
            };
            // Parallel elem→u8: split across cores so the main thread
            // re-queues the GPU sooner.
            let mut bytes = vec![0u8; data.len()];
            std::thread::scope(|s| {
                let nt = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4)
                    .min(16);
                let chunk = data.len().div_ceil(nt);
                for (d, o) in data.chunks(chunk).zip(bytes.chunks_mut(chunk)) {
                    s.spawn(move || {
                        for (dd, oo) in d.iter().zip(o.iter_mut()) {
                            *oo = ElemToU8::to_u8(*dd);
                        }
                    });
                }
            });
            result.push((bytes, self.out_w_t as u32, self.out_h_t as u32));
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tiled fused RGB8 path (burn + tch)
// ---------------------------------------------------------------------------

/// Bit-identical to `n` separate `infer_rgb8` calls (batch dim is independent).
pub fn infer_rgb8_batch<B: Backend>(
    model: &Model<B>,
    inputs: &[Tensor],
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<Rgb8Frames>>
where
    B::FloatElem: Rgb8Elem,
{
    let batch = match infer_rgb8_batch_prepare(model, inputs, native_scale, scale, device) {
        Some(Ok(b)) => b,
        Some(Err(e)) => return Some(Err(e)),
        None => return None,
    };
    Some(Box::new(batch).resolve())
}

pub fn infer_rgb8<B: Backend>(
    model: &Model<B>,
    input: &Tensor,
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<(Vec<u8>, u32, u32)>>
where
    B::FloatElem: Rgb8Elem,
{
    let batch = infer_rgb8_batch(
        model,
        std::slice::from_ref(input),
        native_scale,
        scale,
        device,
    )?;
    Some(batch.map(|mut v| v.pop().expect("batch is non-empty")))
}

/// Deferred readback — caller can queue the next forward before blocking
/// on this one (readback pipelining).
pub fn infer_rgb8_batch_prepare<B: Backend>(
    model: &Model<B>,
    inputs: &[Tensor],
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<BurnRgb8Batch<B>>>
where
    B::FloatElem: Rgb8Elem,
{
    if inputs.is_empty() {
        return Some(Err(Error::new("empty batch")));
    }
    let n = inputs.len();
    let [_, c, h, w] = [
        inputs[0].shape[0],
        inputs[0].shape[1],
        inputs[0].shape[2],
        inputs[0].shape[3],
    ];
    if inputs[0].shape.len() != 4 {
        return Some(Err(Error::new("expected NCHW input")));
    }
    for inp in inputs {
        if inp.shape.len() != 4 || inp.shape[1] != c || inp.shape[2] != h || inp.shape[3] != w {
            return Some(Err(Error::new("batch inputs must share NCHW dims")));
        }
    }
    let tile = crate::current_tile_size();
    let overlap = tile / 4;
    let step = tile - overlap;
    let num_y = (h.saturating_sub(tile)).div_ceil(step) + 1;
    let num_x = (w.saturating_sub(tile)).div_ceil(step) + 1;
    let ph = (num_y - 1) * step + tile;
    let pw = (num_x - 1) * step + tile;
    let scale_f = scale as f32;
    let out_h = (ph as f32 * scale_f).round() as usize;
    let out_w = (pw as f32 * scale_f).round() as usize;
    let resample = scale != native_scale;
    let native_f = native_scale as f32;
    let acc_scale = if resample { native_f } else { scale_f };
    let acc_h = (ph as f32 * acc_scale).round() as usize;
    let acc_w = (pw as f32 * acc_scale).round() as usize;
    // VRAM guard — reject with a clear error before the big canvas/readback
    // allocation instead of silently dropping to the slow CPU path.
    let expected = fused_peak_allocation(n, out_h, out_w, c);
    let limit = fused_peak_limit(crate::vram_total_bytes());
    if expected > limit {
        return Some(Err(Error::new(format!(
            "fused RGB8 path needs ~{} MB peak (limit ~{} MB) — use a lower \
             scale (e.g. x2) or smaller resolution",
            expected / (1024 * 1024),
            limit / (1024 * 1024),
        ))));
    }
    if let Some(free) = crate::vram_available_bytes() {
        if expected > free.saturating_mul(VRAM_THRESHOLD_PCT) / 100 {
            return Some(Err(Error::new(format!(
                "fused RGB8 path needs ~{} MB VRAM ({} MB free) — close GPU apps or \
                 lower scale/resolution",
                expected / (1024 * 1024),
                free / (1024 * 1024),
            ))));
        }
    }
    // Single pad+upload per frame — tile regions are sliced on-device.
    let mut gpu_frames = Vec::with_capacity(inputs.len());
    for inp in inputs {
        match pad_to_burn::<B>(inp, ph, pw, device) {
            Ok(t) => gpu_frames.push(t),
            Err(e) => return Some(Err(e)),
        }
    }
    // Tile grid shared by all frames (y-outer/x-inner).
    let mut grid = Vec::new();
    let mut ty = 0;
    while ty + tile <= ph {
        let mut tx = 0;
        while tx + tile <= pw {
            grid.push((tx, ty));
            tx += step;
        }
        ty += step;
    }
    let ov = (overlap as f32 * acc_scale).round() as usize;
    // Feather ramp (partition of unity): a tile edge bordering a neighbour
    // is weighted ~0 → 1 across the overlap.
    let feather = |n: usize, low: bool, high: bool| -> Vec<f32> {
        let mut w = vec![1.0f32; n];
        let o = ov.min(n);
        if low {
            for k in 0..o {
                w[k] = (k as f32 + 1.0) / (ov as f32 + 1.0);
            }
        }
        if high {
            for k in 0..o {
                w[n - 1 - k] = (k as f32 + 1.0) / (ov as f32 + 1.0);
            }
        }
        w
    };
    // Accumulate tiles into one weighted canvas per frame on the device.
    let mut accs: Vec<BurnTensor<B, 4>> = (0..n)
        .map(|_| BurnTensor::<B, 4>::zeros([1, c, acc_h, acc_w], device))
        .collect();
    // Cache feather masks: ≤9 device tensors per batch instead of rebuild per tile.
    let mut masks: HashMap<((bool, bool), (bool, bool)), BurnTensor<B, 4>> = HashMap::new();
    for (x, y) in grid {
        let batch = if n == 1 {
            gpu_frames[0]
                .clone()
                .slice([0..1, 0..c, y..y + tile, x..x + tile])
        } else {
            let parts: Vec<_> = gpu_frames
                .iter()
                .map(|f| f.clone().slice([0..1, 0..c, y..y + tile, x..x + tile]))
                .collect();
            BurnTensor::cat(parts, 0)
        };
        let out = match model.forward(batch) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let [_, _, oh, ow] = out.dims();
        let sx = (x as f32 * acc_scale).round() as usize;
        let sy = (y as f32 * acc_scale).round() as usize;
        let out = out.clamp(0.0, 1.0);
        let key = ((y > 0, y + tile < ph), (x > 0, x + tile < pw));
        let wmask = match masks.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let wy = feather(oh, (key.0).0, (key.0).1);
                let wx = feather(ow, (key.1).0, (key.1).1);
                let mut wv = Vec::with_capacity(oh * ow);
                for &wyy in &wy {
                    for &wxx in &wx {
                        wv.push(wyy * wxx);
                    }
                }
                e.insert(BurnTensor::<B, 4>::from_data(
                    TensorData::new(wv, [1, 1, oh, ow]).convert::<B::FloatElem>(),
                    device,
                ))
                .clone()
            }
        };
        let placeholder = BurnTensor::<B, 4>::zeros([1, 1, 1, 1], device);
        for f in 0..n {
            let one = out.clone().slice([f..f + 1, 0..c, 0..oh, 0..ow]);
            let region = [0..1, 0..c, sy..sy + oh, sx..sx + ow];
            let sum = {
                let cur = accs[f].clone().slice(region.clone());
                cur + one.mul(wmask.clone())
            };
            let prev = std::mem::replace(&mut accs[f], placeholder.clone());
            accs[f] = prev.slice_assign(region, sum);
        }
    }
    // One-shot re-sample of the native-scale canvas to the requested scale.
    if resample {
        let opts = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
        for acc in accs.iter_mut() {
            *acc = interpolate(acc.clone(), [out_h, out_w], opts.clone());
        }
    }
    let out_h_t = (h as f32 * scale_f).round() as usize;
    let out_w_t = (w as f32 * scale_f).round() as usize;
    Some(Ok(BurnRgb8Batch {
        accs,
        out_w_t,
        out_h_t,
    }))
}

// ---------------------------------------------------------------------------
// Full-frame fused RGB8 path (tch engine only)
// ---------------------------------------------------------------------------

/// Fused RGB8 full-frame path (tch engine): one forward over the whole frame
/// (even-padded), GPU RGB8 pack, single readback — no 640px tile grid.
#[cfg(feature = "tch")]
pub fn infer_rgb8_full_frame<B: Backend>(
    model: &Model<B>,
    input: &Tensor,
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<(Vec<u8>, u32, u32)>>
where
    B::FloatElem: Rgb8Elem,
{
    let batch = infer_rgb8_full_frame_batch(
        model,
        std::slice::from_ref(input),
        native_scale,
        scale,
        device,
    )?;
    Some(batch.map(|mut v| v.pop().expect("batch is non-empty")))
}

/// Synchronous full-frame batch (tch engine): resolves the deferred readback
/// immediately (blocking).
#[cfg(feature = "tch")]
pub fn infer_rgb8_full_frame_batch<B: Backend>(
    model: &Model<B>,
    inputs: &[Tensor],
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<Rgb8Frames>>
where
    B::FloatElem: Rgb8Elem,
{
    let batch =
        match infer_rgb8_full_frame_batch_prepare(model, inputs, native_scale, scale, device) {
            Some(Ok(b)) => b,
            Some(Err(e)) => return Some(Err(e)),
            None => return None,
        };
    Some(Box::new(batch).resolve())
}

/// VRAM guard returns `None` (not an error) so the caller falls back to tiled.
#[cfg(feature = "tch")]
pub fn infer_rgb8_full_frame_batch_prepare<B: Backend>(
    model: &Model<B>,
    inputs: &[Tensor],
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<BurnRgb8Batch<B>>>
where
    B::FloatElem: Rgb8Elem,
{
    if inputs.is_empty() {
        return Some(Err(Error::new("empty batch")));
    }
    let n = inputs.len();
    let [_, c, h, w] = [
        inputs[0].shape[0],
        inputs[0].shape[1],
        inputs[0].shape[2],
        inputs[0].shape[3],
    ];
    if inputs[0].shape.len() != 4 {
        return Some(Err(Error::new("expected NCHW input")));
    }
    for inp in inputs {
        if inp.shape.len() != 4 || inp.shape[1] != c || inp.shape[2] != h || inp.shape[3] != w {
            return Some(Err(Error::new("batch inputs must share NCHW dims")));
        }
    }
    // Even-pad only (pixel-shuffle ×2 needs even H/W).
    let ph = h + (h & 1);
    let pw = w + (w & 1);
    let scale_f = scale as f32;
    let out_h = (ph as f32 * scale_f).round() as usize;
    let out_w = (pw as f32 * scale_f).round() as usize;
    let resample = scale != native_scale;
    // Same VRAM guard as the tiled path, but reject with `None` so the tch
    // engine falls back to tiled fused instead of erroring.
    let expected = fused_peak_allocation(n, out_h, out_w, c);
    let limit = fused_peak_limit(crate::vram_total_bytes());
    if expected > limit {
        return None;
    }
    if let Some(free) = crate::vram_available_bytes() {
        if expected > free.saturating_mul(VRAM_THRESHOLD_PCT) / 100 {
            return None;
        }
    }
    let mut gpu_frames = Vec::with_capacity(n);
    for inp in inputs {
        match pad_to_burn::<B>(inp, ph, pw, device) {
            Ok(t) => gpu_frames.push(t),
            Err(e) => return Some(Err(e)),
        }
    }
    // Per-frame forwards, not one batched forward: a batched full-frame conv
    // blows up MIOpen's workspace and is slower than per-frame anyway.
    let mut accs = Vec::with_capacity(n);
    for f in &gpu_frames {
        let out = match model.forward(f.clone()) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let out = out.clamp(0.0, 1.0);
        let out = if resample {
            let opts = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
            interpolate(out, [out_h, out_w], opts)
        } else {
            out
        };
        accs.push(out);
    }
    Some(Ok(BurnRgb8Batch {
        accs,
        out_w_t: (w as f32 * scale_f).round() as usize,
        out_h_t: (h as f32 * scale_f).round() as usize,
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::{fused_peak_allocation, fused_peak_limit, pad_to_f16};
    use burn::tensor::f16;

    #[test]
    fn fused_vram_guard_window() {
        const LIMIT: u64 = (2 * 1024 + 512) * 1024 * 1024;
        assert!(fused_peak_allocation(1, 5120, 8320, 3) > LIMIT);
        assert!(fused_peak_allocation(1, 4480, 8320, 3) <= LIMIT);
        assert!(fused_peak_allocation(1, 4480, 6400, 3) <= LIMIT);
        assert!(fused_peak_allocation(1, 2880, 5120, 3) <= LIMIT);
        assert!(fused_peak_allocation(1, 2240, 4160, 3) <= LIMIT);
        assert_eq!(fused_peak_limit(None), LIMIT);
        assert_eq!(fused_peak_limit(Some(16 * 1024 * 1024 * 1024)), LIMIT);
        assert_eq!(
            fused_peak_limit(Some(4 * 1024 * 1024 * 1024)),
            2 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn feather_is_partition_of_unity() {
        let tile = 16usize;
        let overlap = tile / 4;
        let step = tile - overlap;
        let h = 40usize;
        let w = 52usize;
        let num_y = (h.saturating_sub(tile)).div_ceil(step) + 1;
        let num_x = (w.saturating_sub(tile)).div_ceil(step) + 1;
        let ph = (num_y - 1) * step + tile;
        let pw = (num_x - 1) * step + tile;
        let ov = overlap;
        let feather = |n: usize, low: bool, high: bool| -> Vec<f32> {
            let mut w = vec![1.0f32; n];
            let o = ov.min(n);
            if low {
                for k in 0..o {
                    w[k] = (k as f32 + 1.0) / (ov as f32 + 1.0);
                }
            }
            if high {
                for k in 0..o {
                    w[n - 1 - k] = (k as f32 + 1.0) / (ov as f32 + 1.0);
                }
            }
            w
        };
        let mut cov = vec![0f32; ph * pw];
        let mut ty = 0;
        while ty + tile <= ph {
            let mut tx = 0;
            while tx + tile <= pw {
                let wy = feather(tile, ty > 0, ty + tile < ph);
                let wx = feather(tile, tx > 0, tx + tile < pw);
                for yy in 0..tile {
                    for xx in 0..tile {
                        cov[(ty + yy) * pw + tx + xx] += wy[yy] * wx[xx];
                    }
                }
                tx += step;
            }
            ty += step;
        }
        for (i, v) in cov.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "coverage at pixel {i} is {v}, not 1"
            );
        }
    }

    #[test]
    fn pad_to_f16_matches_pad_to() {
        use crate::tensor::Tensor;
        let h = 5usize;
        let w = 7usize;
        let data: Vec<f32> = (0..3 * h * w).map(|i| i as f32 * 0.5).collect();
        let t = Tensor::new(vec![1, 3, h, w], data);
        for (ph, pw) in [(5usize, 7usize), (8, 8), (12, 9)] {
            let fused = pad_to_f16::<crate::BurnBackend<f16>>(&t, ph, pw);
            let reference = crate::pad_to(&t, ph, pw).data;
            assert_eq!(fused.len(), reference.len(), "len at {ph}x{pw}");
            for (a, b) in fused.iter().zip(reference.iter()) {
                assert_eq!(f32::from(*a), *b, "mismatch at {ph}x{pw}");
            }
        }
    }
}
