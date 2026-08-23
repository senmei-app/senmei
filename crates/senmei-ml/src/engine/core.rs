//! Backend-generic engine core, shared by the burn (Vulkan f16) and tch
//! (libtorch f32) engines. Holds the arch `Model<B>` enum plus the load/infer
//! logic; only the element cast (`B::FloatElem`) and `B::Device` differ, both
//! passed in by the engines.
#![cfg(any(feature = "burn", feature = "tch"))]

use crate::arch::{
    Dncnn, Drunet, Ffdnet, IfrNet, NafNet, RealPlk, RrdbNet, RifeNet, SafmnNet, Scunet, Span,
    SrvggNet, UpCunet2x, UpCunet2xFast,
};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use super::Rgb8Batch;
use burn::tensor::backend::Backend;
use burn::tensor::{f16, Tensor as BurnTensor, TensorData};
#[cfg(feature = "burn")]
use burn::tensor::{module::interpolate, ops::{InterpolateMode, InterpolateOptions}};
use burn_store::{BurnpackStore, ModuleSnapshot};

/// The loaded arch, generic over the backend (`BurnBackend<f16>` or
/// `LibTorch<f32>`).
pub enum Model<B: Backend> {
    UpCunet2x(UpCunet2x<B>),
    UpCunet2xFast(UpCunet2xFast<B>),
    RrdbNet(RrdbNet<B>),
    SrvggNet(SrvggNet<B>),
    RifeNet(RifeNet<B>),
    IfrNet(IfrNet<B>),
    Drunet(Drunet<B>),
    Dncnn(Dncnn<B>),
    Ffdnet(Ffdnet<B>),
    Scunet(Scunet<B>),
    NafNet(NafNet<B>),
    RealPlk(RealPlk<B>),
    Span(Span<B>),
    SafmnNet(SafmnNet<B>),
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: BurnTensor<B, 4>) -> Result<BurnTensor<B, 4>> {
        match self {
            Model::UpCunet2x(m) => Ok(m.forward(x)),
            Model::UpCunet2xFast(m) => Ok(m.forward(x)),
            Model::RrdbNet(m) => Ok(m.forward(x)),
            Model::SrvggNet(m) => Ok(m.forward(x)),
            Model::RealPlk(m) => Ok(m.forward(x)),
            Model::Drunet(m) => Ok(m.forward(x)),
            Model::Dncnn(m) => Ok(m.forward(x)),
            Model::Scunet(m) => Ok(m.forward(x)),
            Model::NafNet(m) => Ok(m.forward(x)),
            Model::Span(m) => Ok(m.forward(x)),
            Model::SafmnNet(m) => Ok(m.forward(x)),
            Model::RifeNet(_) | Model::IfrNet(_) | Model::Ffdnet(_) => {
                Err(Error::new("no single-input forward"))
            }
        }
    }

    pub fn interp(
        &self,
        a: BurnTensor<B, 4>,
        b: BurnTensor<B, 4>,
        t: BurnTensor<B, 4>,
    ) -> Result<BurnTensor<B, 4>> {
        match self {
            Model::RifeNet(m) => Ok(m.forward(a, b, t)),
            Model::IfrNet(m) => Ok(m.forward(a, b, t)),
            _ => Err(Error::new("model has no frame interpolation")),
        }
    }

    pub fn is_rife(&self) -> bool {
        matches!(self, Model::RifeNet(_))
    }
}

/// Build the arch on `device` from a burnpack `store` (f16 weights). The
/// 13-branch dispatch is identical for both engines; only the store's
/// from-adapter differs (tch converts f16→f32 at load).
pub fn load_arch<B: Backend>(
    model: &ModelRef,
    store: &mut BurnpackStore,
    device: &B::Device,
) -> Result<Model<B>> {
    match model.arch.as_str() {
        "upcunet2x" => {
            let mut m = UpCunet2x::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::UpCunet2x(m))
        }
        "upcunet2x-fast" | "fallin-cugan" => {
            // Fallin (renarchi CUGAN retrain) is an `UpCunet2x_fast` with
            // the same 38px reflect pad — only the weights differ.
            let mut m = UpCunet2xFast::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::UpCunet2xFast(m))
        }
        "realesrgan" => {
            let mut m = RrdbNet::new(
                model.scale as usize,
                model.num_block as usize,
                model.shuffle as usize,
                device,
            );
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::RrdbNet(m))
        }
        "srvgg" => {
            // Registered SRVGGNetCompact models: 64 features, body conv count
            // from the registry (16 animevideo-xs, 32 general-x4v3).
            let mut m = SrvggNet::new(64, model.num_conv as usize, model.scale as usize, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::SrvggNet(m))
        }
        "ifrnet" => {
            let mut m = IfrNet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::IfrNet(m))
        }
        "drunet" => {
            let mut m = Drunet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Drunet(m))
        }
        "dncnn" => {
            let mut m = Dncnn::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Dncnn(m))
        }
        "ffdnet" => {
            let mut m = Ffdnet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Ffdnet(m))
        }
        "scunet" => {
            let mut m = Scunet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Scunet(m))
        }
        "nafnet" => {
            let mut m = NafNet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::NafNet(m))
        }
        "real-plksr" => {
            let mut m = RealPlk::new(model.scale as usize, model.layer_norm, model.dysample, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::RealPlk(m))
        }
        "span" => {
            let mut m = Span::new(
                model.feature_channels as usize,
                model.scale as usize,
                device,
            );
            m.set_no_norm(model.no_norm);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            m.pad_k96(device);
            Ok(Model::Span(m))
        }
        "safmn" => {
            // SAFMN-L Real (registered models are fixed): dim 128 / 16 blocks
            // / ffn_scale 2.0; only the scale differs between x2 and x4.
            let mut m = SafmnNet::new(128, 16, 2.0, model.scale as usize, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::SafmnNet(m))
        }
        other => Err(Error::new(format!("unsupported arch: {other}"))),
    }
}

/// Copy an f32 `Tensor` (NCHW) onto the backend, cast to `B::FloatElem`.
fn to_burn<B: Backend>(input: &Tensor, device: &B::Device) -> Result<BurnTensor<B, 4>> {
    if input.shape.len() != 4 {
        return Err(Error::new("expected NCHW input"));
    }
    let [n, c, h, w] = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]];
    Ok(BurnTensor::<B, 4>::from_data(
        TensorData::new(input.data.clone(), [n, c, h, w]).convert::<B::FloatElem>(),
        device,
    ))
}

/// Read a backend tensor back as an f32 `Tensor` (NCHW).
fn to_tensor<B: Backend>(out: BurnTensor<B, 4>, shape: [usize; 4]) -> Result<Tensor> {
    let data = out
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| Error::new(e.to_string()))?;
    Ok(Tensor::new(shape.to_vec(), data))
}

/// Models whose single-input `forward` takes a 3-channel RGB tensor (used to
/// pick warmup inputs; DRUNet wants 4ch, FFDNet/RIFE/IFRNet have no
/// single-input forward at all).
pub fn single_input_rgb<B: Backend>(model: &Model<B>) -> bool {
    !matches!(
        model,
        Model::Drunet(_) | Model::Ffdnet(_) | Model::RifeNet(_) | Model::IfrNet(_)
    )
}

pub fn infer<B: Backend>(model: &Model<B>, input: &Tensor, device: &B::Device) -> Result<Tensor> {
    let x = to_burn::<B>(input, device)?;
    let out = model.forward(x)?;
    let [_, _, oh, ow] = out.dims();
    to_tensor(out, [input.shape[0], input.shape[1], oh, ow])
}

/// Fused RGB8 path: hands back packed rgb24 bytes. The model runs on the
/// GPU (autotuned, f16) in 640px tiles (avoids the full-frame im2col OOM,
/// see docs/upstream-issues.md §2; 640 beats 512 and 768 — docs/benchmarks.md).
/// Tiles are accumulated into one canvas on the device (overlap averaging)
/// and read back as a single packed frame — one readback instead of one u8
/// readback per tile plus a CPU stitch. `native_scale` is the model's own
/// upscale factor; a requested `scale` above/below it is applied on the GPU
/// (bilinear re-sample of each tile output) so the fused path works for e.g.
/// x2 models rendered at x4. Burn-only: the tch engine relies on the trait
/// default (`None`).
#[cfg(feature = "burn")]
pub fn infer_rgb8<B: Backend>(
    model: &Model<B>,
    input: &Tensor,
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<(Vec<u8>, u32, u32)>> {
    let batch =
        infer_rgb8_batch(model, std::slice::from_ref(input), native_scale, scale, device)?;
    Some(batch.map(|mut v| v.pop().unwrap()))
}

/// Fused multi-frame RGB8 path: `n` same-shaped NCHW inputs, each handed back
/// as packed rgb24 bytes. Tiles still run one tile position at a time (larger
/// batched matmuls are pathologically slower on this backend —
/// docs/benchmarks.md), but each forward carries all `n` frames' tile for that
/// position in the batch dim — fewer launches/readbacks, the per-tile feather
/// mask is computed once and shared. Output is bit-identical to `n` separate
/// `infer_rgb8` calls (batch dim is independent in every conv).
#[cfg(feature = "burn")]
pub fn infer_rgb8_batch<B: Backend>(
    model: &Model<B>,
    inputs: &[Tensor],
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<Vec<(Vec<u8>, u32, u32)>>> {
    let batch = match infer_rgb8_batch_prepare(model, inputs, native_scale, scale, device) {
        Some(Ok(b)) => b,
        Some(Err(e)) => return Some(Err(e)),
        None => return None,
    };
    Some(Box::new(batch).resolve())
}

/// Forward + GPU canvas accumulation for one batch; the readback is deferred
/// to [`BurnRgb8Batch::resolve`] so the caller can queue the next forward
/// before blocking on this one (readback pipelining).
#[cfg(feature = "burn")]
pub fn infer_rgb8_batch_prepare<B: Backend>(
    model: &Model<B>,
    inputs: &[Tensor],
    native_scale: u32,
    scale: u32,
    device: &B::Device,
) -> Option<Result<BurnRgb8Batch<B>>> {
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
        if inp.shape.len() != 4
            || inp.shape[1] != c
            || inp.shape[2] != h
            || inp.shape[3] != w
        {
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
    // 8K/oversize pre-check (Koharu max_pixels): reject before the tile grid
    // is built — at 8K that's 144+ tiles + pad allocations, all wasted if the
    // fused path can't fit the window below.
    let scale_f = scale as f32;
    let out_h = (ph as f32 * scale_f).round() as usize;
    let out_w = (pw as f32 * scale_f).round() as usize;
    // VRAM guard — no CPU fallback: reject with a clear error before the big
    // canvas/readback allocation instead of silently dropping to the slow CPU
    // path or hitting the OOM crash (which loses the wgpu device handle). The
    // fused path's peak allocation scales with the output canvas; a ~3.2 GB
    // single allocation was observed at 1080p×4 (estimate ~2.8 GiB), independent
    // of tile size and autotune level (wgpu/burn internal buffer), so the
    // ceiling stays just under that crash zone. It adapts to the system: half
    // the GPU's total VRAM on smaller cards (tighter cap), plus the current
    // free-VRAM budget read from DRM sysfs.
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
        if expected > free.saturating_mul(85) / 100 {
            return Some(Err(Error::new(format!(
                "fused RGB8 path needs ~{} MB VRAM ({} MB free) — close GPU apps or \
                 lower scale/resolution",
                expected / (1024 * 1024),
                free / (1024 * 1024),
            ))));
        }
    }
    // Pad + tile each frame once; all frames share the tile grid.
    let resample = scale != native_scale;
    let frames: Vec<Vec<(usize, usize, Tensor)>> = inputs
        .iter()
        .map(|inp| crate::uniform_tile(&crate::pad_to(inp, ph, pw), tile, step))
        .collect();
    let ntiles = frames[0].len();
    let ov = (overlap as f32 * scale_f).round() as usize;
    // Feather ramp (partition of unity): a tile edge bordering a neighbour
    // is weighted ~0 → 1 across the overlap, so the model's 1-2px border
    // lines vanish at the seams; the canvas border keeps full weight
    // (single coverage, nothing to blend with).
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
    // Accumulate tiles into one weighted canvas per frame on the device. We
    // sum rather than replace so the overlap is truly averaged: slice_assign
    // alone leaves the next tile's edge line visible at every seam. The
    // intermediate view is scoped so the backend can write in place instead
    // of copy-on-write. A single readback per frame at the end avoids the
    // burn-fusion ordering panic (docs/burn-bugs.md Bug 1).
    let mut accs: Vec<BurnTensor<B, 4>> = (0..n)
        .map(|_| BurnTensor::<B, 4>::zeros([1, c, out_h, out_w], device))
        .collect();
    let mut covs: Vec<BurnTensor<B, 4>> = (0..n)
        .map(|_| BurnTensor::<B, 4>::zeros([1, 1, out_h, out_w], device))
        .collect();
    for k in 0..ntiles {
        let (x, y, _) = frames[0][k];
        let mut data = Vec::with_capacity(n * c * tile * tile);
        for f in &frames {
            data.extend_from_slice(&f[k].2.data);
        }
        let batch = BurnTensor::<B, 4>::from_data(
            TensorData::new(data, [n, c, tile, tile]).convert::<B::FloatElem>(),
            device,
        );
        let out = match model.forward(batch) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        // Re-sample the model's native-scale tile output to the requested
        // scale on the GPU (e.g. x2 model at x4), so the canvas placement
        // and feather mask below match `scale` exactly.
        let out = if resample {
            let oh = (tile as f32 * scale_f).round() as usize;
            let ow = (tile as f32 * scale_f).round() as usize;
            interpolate(
                out,
                [oh, ow],
                InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
            )
        } else {
            out
        };
        let [_, _, oh, ow] = out.dims();
        let sx = (x as f32 * scale_f).round() as usize;
        let sy = (y as f32 * scale_f).round() as usize;
        // Clamp before accumulating: out-of-range values (>1.0 at hard
        // edges, e.g. burnt-in subtitles) would wrap on the u8 cast below.
        let out = out.clamp(0.0, 1.0);
        let wy = feather(oh, y > 0, y + tile < ph);
        let wx = feather(ow, x > 0, x + tile < pw);
        let mut wv = Vec::with_capacity(oh * ow);
        for yy in 0..oh {
            let wyy = wy[yy];
            for xx in 0..ow {
                wv.push(wyy * wx[xx]);
            }
        }
        let wmask = BurnTensor::<B, 4>::from_data(
            TensorData::new(wv, [1, 1, oh, ow]).convert::<B::FloatElem>(),
            device,
        );
        // slice_assign consumes the canvas, so swap a tiny placeholder out of
        // the Vec and put the updated canvas back (avoids a full-canvas COW).
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
            let cregion = [0..1, 0..1, sy..sy + oh, sx..sx + ow];
            let csum = {
                let ccur = covs[f].clone().slice(cregion.clone());
                ccur + wmask.clone()
            };
            let prev = std::mem::replace(&mut covs[f], placeholder.clone());
            covs[f] = prev.slice_assign(cregion, csum);
        }
    }
    let out_h_t = (h as f32 * scale_f).round() as usize;
    let out_w_t = (w as f32 * scale_f).round() as usize;
    Some(Ok(BurnRgb8Batch {
        accs,
        covs,
        out_w,
        out_w_t,
        out_h_t,
    }))
}

/// Effective fused-path peak ceiling: the crash-safe cap (just under the
/// observed ~3.2 GB single-allocation OOM at 1080p×4, a wgpu/burn-internal
/// buffer) or half the GPU's total VRAM when that is tighter — adapts down on
/// smaller cards.
#[cfg(feature = "burn")]
fn fused_peak_limit(total_vram: Option<u64>) -> u64 {
    const FUSED_PEAK_CEILING: u64 = (2 * 1024 + 512) * 1024 * 1024; // 2.5 GiB
    match total_vram {
        Some(t) => (t.saturating_mul(50) / 100).min(FUSED_PEAK_CEILING),
        None => FUSED_PEAK_CEILING,
    }
}

/// Fused RGB8 path peak-allocation estimate (bytes), used by the VRAM guard.
/// Canvas (f16 accs+covs) + f32 readback, ×4 for ops/COW/autotune overhead —
/// a ~3.2 GB single allocation was observed at 1080p×4 on RADV, independent of
/// tile size and autotune level.
#[cfg(feature = "burn")]
fn fused_peak_allocation(n: usize, out_h: usize, out_w: usize, c: usize) -> u64 {
    let canvas = (n * out_h * out_w * (c + 1) * 2) as u64; // accs + covs (f16)
    let readback = (n * out_h * out_w * 3 * 4) as u64; // packed rgb24 f32
    (canvas + readback) * 4
}

/// Deferred readback of one fused batch: blocks on the GPU→CPU transfer, then
/// converts to packed rgb24 on the CPU (same steps `infer_rgb8_batch` used).
#[cfg(feature = "burn")]
pub struct BurnRgb8Batch<B: Backend> {
    accs: Vec<BurnTensor<B, 4>>,
    covs: Vec<BurnTensor<B, 4>>,
    out_w: usize,
    out_w_t: usize,
    out_h_t: usize,
}

#[cfg(feature = "burn")]
impl<B: Backend> Rgb8Batch for BurnRgb8Batch<B> {
    fn resolve(self: Box<Self>) -> Result<Vec<(Vec<u8>, u32, u32)>> {
        let mut result = Vec::with_capacity(self.accs.len());
        for (acc, cov) in self.accs.into_iter().zip(self.covs) {
            let avg = (acc / cov).permute([0, 2, 3, 1]) * 255.0;
            // Read back in f16 and cast to u8 in plain Rust: cubecl-wgpu already
            // pools the staging buffers, so the remaining per-frame cost is the
            // CPU-side convert — reading f16 directly skips the full f32 copy
            // (a `convert::<u8>()` readback would trip the burn-fusion ordering
            // panic, docs/burn-bugs.md Bug 1). The fused path only runs on
            // `BurnBackend<f16>`, so the element type is always f16 here.
            let data: Vec<f16> = match avg.into_data().to_vec::<f16>() {
                Ok(v) => v,
                Err(e) => return Err(Error::new(e.to_string())),
            };
            let mut bytes = Vec::with_capacity(data.len());
            for v in data {
                bytes.push((v.to_f32() + 0.5) as u8);
            }
            let cropped = crate::crop_rgb24(&bytes, self.out_w, self.out_h_t, self.out_w_t);
            result.push((cropped, self.out_w_t as u32, self.out_h_t as u32));
        }
        Ok(result)
    }
}

pub fn infer_interp<B: Backend>(
    model: &Model<B>,
    a: &Tensor,
    b: &Tensor,
    t: f32,
    device: &B::Device,
) -> Option<Result<Tensor>> {
    if !matches!(model, Model::RifeNet(_) | Model::IfrNet(_)) {
        return None; // not an interpolation model → caller falls back
    }
    let [n, c, h, w] = [a.shape[0], a.shape[1], a.shape[2], a.shape[3]];
    let a_t = match to_burn::<B>(a, device) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    let b_t = match to_burn::<B>(b, device) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    // The flow estimators run on a downscaled grid (RIFE 1/32, IFRNet 1/16
    // via its pyramid), so pad to a multiple and crop back (like the refs).
    let pad = if model.is_rife() { 32 } else { 16 };
    let pad_h = (h + pad - 1) / pad * pad;
    let pad_w = (w + pad - 1) / pad * pad;
    let pad = |x: BurnTensor<B, 4>| {
        let mut x = x;
        if pad_h > h {
            let z = BurnTensor::<B, 4>::zeros([n, c, pad_h - h, w], device);
            x = BurnTensor::cat(vec![x, z], 2);
        }
        if pad_w > w {
            let z = BurnTensor::<B, 4>::zeros([n, c, pad_h, pad_w - w], device);
            x = BurnTensor::cat(vec![x, z], 3);
        }
        x
    };
    let a_t = pad(a_t);
    let b_t = pad(b_t);
    // ncnn broadcasts the scalar timestep over the (padded) spatial grid.
    let t_t = BurnTensor::<B, 4>::ones([n, 1, pad_h, pad_w], device) * t;
    let out = match model.interp(a_t, b_t, t_t) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    let out = out.slice([0..n, 0..c, 0..h, 0..w]);
    Some(to_tensor(out, [n, c, h, w]))
}

/// DRUNet denoise: appends a constant noise-level map (sigma in [0,1]) to
/// the 3-channel input, pads the spatial dims to multiples of 8 (the UNet
/// downsamples 3× stride-2), runs the model, and crops back. FFDNet gets σ
/// directly, DnCNN/SCUNet are blind. Other models return `None`.
pub fn infer_denoise<B: Backend>(
    model: &Model<B>,
    input: &Tensor,
    sigma: f32,
    device: &B::Device,
) -> Option<Result<Tensor>> {
    let is_drunet = matches!(model, Model::Drunet(_));
    if !matches!(
        model,
        Model::Drunet(_) | Model::Dncnn(_) | Model::Ffdnet(_) | Model::Scunet(_)
    ) {
        return None;
    }
    if input.shape.len() != 4 || input.shape[1] != 3 {
        return Some(Err(Error::new("expected 3-channel NCHW input")));
    }
    let [n, _c, h, w] = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]];
    let rgb = match to_burn::<B>(input, device) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    // FFDNet takes the noise level internally; DnCNN/SCUNet are blind
    // (3ch in, no sigma map); DRUNet gets a constant sigma map + 8-aligned
    // spatial dims (3× stride-2 downsample) — pad and crop.
    if let Model::Ffdnet(m) = model {
        let out = m.forward(rgb, sigma);
        return Some(to_tensor(out, [n, 3, h, w]));
    }
    if !is_drunet {
        let out = match model.forward(rgb) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        return Some(to_tensor(out, [n, 3, h, w]));
    }
    let sigma_map = BurnTensor::<B, 4>::ones([n, 1, h, w], device) * sigma;
    let x = BurnTensor::cat(vec![rgb, sigma_map], 1);
    let pad_h = (h + 7) / 8 * 8;
    let pad_w = (w + 7) / 8 * 8;
    let mut x = x;
    if pad_h > h {
        let z = BurnTensor::<B, 4>::zeros([n, 4, pad_h - h, w], device);
        x = BurnTensor::cat(vec![x, z], 2);
    }
    if pad_w > w {
        let z = BurnTensor::<B, 4>::zeros([n, 4, pad_h, pad_w - w], device);
        x = BurnTensor::cat(vec![x, z], 3);
    }
    let out = match model.forward(x) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    let out = out.slice([0..n, 0..3, 0..h, 0..w]);
    Some(to_tensor(out, [n, 3, h, w]))
}

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::{fused_peak_allocation, fused_peak_limit};

    /// The VRAM guard's ceiling: 1080p×4 is over it (rejected — matches the
    /// observed ~3.2 GB single-allocation OOM on RADV); the common x4/x2 fused
    /// renders stay under it, and the cap scales down with total VRAM.
    #[test]
    fn fused_vram_guard_window() {
        const LIMIT: u64 = (2 * 1024 + 512) * 1024 * 1024;
        assert!(fused_peak_allocation(1, 4480, 8320, 3) > LIMIT); // 1080p×4 crash zone
        assert!(fused_peak_allocation(1, 4480, 6400, 3) <= LIMIT); // SD/720p×4 (~2.2 GiB)
        assert!(fused_peak_allocation(1, 2880, 5120, 3) <= LIMIT); // 720p×4
        assert!(fused_peak_allocation(1, 2240, 4160, 3) <= LIMIT); // 1080p×2
        // Adaptive: crash cap on big cards, half of total VRAM on small ones.
        assert_eq!(fused_peak_limit(None), LIMIT);
        assert_eq!(fused_peak_limit(Some(16 * 1024 * 1024 * 1024)), LIMIT);
        assert_eq!(fused_peak_limit(Some(4 * 1024 * 1024 * 1024)), 2 * 1024 * 1024 * 1024);
    }
}
