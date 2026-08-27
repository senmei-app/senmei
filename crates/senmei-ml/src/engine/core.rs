//! Backend-generic engine core, shared by the burn (Vulkan f16) and tch
//! (libtorch f32) engines. Holds the arch `Model<B>` enum plus the load/infer
//! logic; only the element cast (`B::FloatElem`) and `B::Device` differ, both
//! passed in by the engines.
#![cfg(any(feature = "burn", feature = "tch"))]

use super::{Rgb8Batch, Rgb8Frames};
use crate::arch::{
    DisNet, Dncnn, Drunet, Ffdnet, IfrNet, NafNet, ParagonSrNet, RealPlk, RifeNet, RrdbNet,
    SafmnNet, Scunet, Span, SrvggNet, UpCunet2x, UpCunet2xFast,
};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{f16, Tensor as BurnTensor, TensorData};
#[cfg(any(feature = "burn", feature = "tch"))]
use burn::tensor::{
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use burn_store::{BurnpackStore, ModuleSnapshot};
use std::collections::HashMap;

/// The loaded arch, generic over the backend (`BurnBackend<f16>` or
/// `LibTorch<f32>`).
pub enum Model<B: Backend> {
    UpCunet2x(UpCunet2x<B>),
    UpCunet2xFast(UpCunet2xFast<B>),
    RrdbNet(RrdbNet<B>),
    SrvggNet(SrvggNet<B>),
    RifeNet(Box<RifeNet<B>>),
    IfrNet(IfrNet<B>),
    Drunet(Drunet<B>),
    Dncnn(Dncnn<B>),
    Ffdnet(Ffdnet<B>),
    Scunet(Scunet<B>),
    NafNet(NafNet<B>),
    RealPlk(RealPlk<B>),
    Span(Span<B>),
    SafmnNet(SafmnNet<B>),
    ParagonSrNet(ParagonSrNet<B>),
    DisNet(DisNet<B>),
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
            Model::ParagonSrNet(m) => Ok(m.forward(x)),
            Model::DisNet(m) => Ok(m.forward(x)),
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
        "dis" => {
            // Registered DIS models: 32 features, body blocks from the
            // registry (8 DIS_Fast, 12 DIS_Balanced).
            let mut m = DisNet::new(32, model.num_block as usize, model.scale as usize, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::DisNet(m))
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
            let mut m = RealPlk::new(
                model.scale as usize,
                model.layer_norm,
                model.dysample,
                device,
            );
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
        "paragonsr" => {
            // ParagonSR Nano (registered model is fixed): num_feat 24 / 3
            // residual groups × 2 blocks / ffn_expansion 1.5.
            let mut m = ParagonSrNet::new(model.scale as usize, 24, 3, 2, 1.5, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::ParagonSrNet(m))
        }
        other => Err(Error::new(format!("unsupported arch: {other}"))),
    }
}

/// Copy an f32 `Tensor` (NCHW) onto the backend, cast to `B::FloatElem`.
fn to_burn<B: Backend>(input: &Tensor, device: &B::Device) -> Result<BurnTensor<B, 4>> {
    if input.shape.len() != 4 {
        return Err(Error::new("expected NCHW input"));
    }
    let [n, c, h, w] = [
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ];
    Ok(BurnTensor::<B, 4>::from_data(
        TensorData::new(input.data.clone(), [n, c, h, w]).convert::<B::FloatElem>(),
        device,
    ))
}

/// Edge-replicate pad `ph×pw` writing the backend's float element directly —
/// no padded f32 buffer, no `to_burn` clone, no separate f32→f16 convert (the
/// fused RGB8 path's per-frame staging was three full-frame allocations). Same
/// layout as `pad_to` + cast, so on-device tile slicing is unchanged.
fn pad_to_f16<B: Backend>(input: &Tensor, ph: usize, pw: usize) -> Vec<B::FloatElem>
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

/// `pad_to_f16` + device upload in one call.
fn pad_to_burn<B: Backend>(
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
#[cfg(feature = "burn")]
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
/// x2 models rendered at x4. Shared by both engines: burn (Vulkan f16) and
/// tch (libtorch f32) call it with their own backend.
#[cfg(any(feature = "burn", feature = "tch"))]
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
    Some(batch.map(|mut v| v.pop().unwrap()))
}

/// Fused multi-frame RGB8 path: `n` same-shaped NCHW inputs, each handed back
/// as packed rgb24 bytes. Tiles still run one tile position at a time (larger
/// batched matmuls are pathologically slower on this backend —
/// docs/benchmarks.md), but each forward carries all `n` frames' tile for that
/// position in the batch dim — fewer launches/readbacks, the per-tile feather
/// mask is computed once and shared. Output is bit-identical to `n` separate
/// `infer_rgb8` calls (batch dim is independent in every conv).
#[cfg(any(feature = "burn", feature = "tch"))]
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

/// Forward + GPU canvas accumulation for one batch; the readback is deferred
/// to [`BurnRgb8Batch::resolve`] so the caller can queue the next forward
/// before blocking on this one (readback pipelining).
#[cfg(any(feature = "burn", feature = "tch"))]
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
    // 8K/oversize pre-check (Koharu max_pixels): reject before the tile grid
    // is built — at 8K that's 144+ tiles + pad allocations, all wasted if the
    // fused path can't fit the window below.
    let scale_f = scale as f32;
    let out_h = (ph as f32 * scale_f).round() as usize;
    let out_w = (pw as f32 * scale_f).round() as usize;
    // A requested scale ≠ the model's native scale (e.g. a 2× model at 4×):
    // accumulate at the native scale and re-sample once at the end. Per-tile
    // re-sampling plus a requested-scale canvas dominated GPU memory traffic
    // (measured ~72% memory activity) — the native canvas is 4× smaller here.
    let resample = scale != native_scale;
    let native_f = native_scale as f32;
    let acc_scale = if resample { native_f } else { scale_f };
    let acc_h = (ph as f32 * acc_scale).round() as usize;
    let acc_w = (pw as f32 * acc_scale).round() as usize;
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
    // Pad each frame once (CPU, edge-replicate) and upload it once as f16;
    // tile regions are then sliced on the device per forward. This removes
    // the per-tile CPU gather + f32→f16 convert + PCIe upload that kept the
    // GPU idle between tile forwards (measured ~50% GPU busy with two CPU
    // cores pegged on the render worker).
    let mut gpu_frames = Vec::with_capacity(inputs.len());
    for inp in inputs {
        // Fused pad + f16 cast + upload: writes the padded buffer in the
        // backend's float element directly, skipping `pad_to`'s padded f32
        // buffer, `to_burn`'s clone and the separate f32→f16 convert pass.
        match pad_to_burn::<B>(inp, ph, pw, device) {
            Ok(t) => gpu_frames.push(t),
            Err(e) => return Some(Err(e)),
        }
    }
    // Tile grid shared by all frames (y-outer/x-inner — same order as the CPU
    // `uniform_tile`, so the accumulation order is unchanged).
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
    // feather weights are a partition of unity, so the coverage canvas
    // (`acc / cov`) is ≡1 and drops out — we accumulate the weighted sum
    // directly: one less canvas channel, one full-tile add + slice_assign per
    // tile, and half the readback. The intermediate view is scoped so the
    // backend can write in place instead of copy-on-write. A single readback
    // per frame at the end avoids the burn-fusion ordering panic
    // (docs/burn-bugs.md Bug 1).
    let mut accs: Vec<BurnTensor<B, 4>> = (0..n)
        .map(|_| BurnTensor::<B, 4>::zeros([1, c, acc_h, acc_w], device))
        .collect();
    // Feather masks depend only on the tile's border class (≤3 per axis), so
    // cache them: ≤9 device tensors per batch instead of a rebuild + upload
    // per tile (40 tiles @1080p → ~4× less CPU churn and PCIe).
    let mut masks: HashMap<((bool, bool), (bool, bool)), BurnTensor<B, 4>> = HashMap::new();
    for (x, y) in grid {
        // Slice the tile region on-device (rows are contiguous, row stride pw;
        // `clone` is a cheap handle copy in burn, `slice` shares the device
        // buffer). n=1 slices directly; n>1 stacks the frames' slices.
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
        // Place the native-scale tile in the accumulation canvas (which is at
        // the model's native scale when a requested scale differs); the single
        // final re-sample to the requested scale happens after the tile loop.
        let sx = (x as f32 * acc_scale).round() as usize;
        let sy = (y as f32 * acc_scale).round() as usize;
        // Clamp before accumulating: out-of-range values (>1.0 at hard
        // edges, e.g. burnt-in subtitles) would wrap on the u8 cast below.
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

/// Effective fused-path peak ceiling: the crash-safe cap (just under the
/// observed ~3.2 GB single-allocation OOM at 1080p×4, a wgpu/burn-internal
/// buffer) or half the GPU's total VRAM when that is tighter — adapts down on
/// smaller cards.
#[cfg(any(feature = "burn", feature = "tch"))]
fn fused_peak_limit(total_vram: Option<u64>) -> u64 {
    const FUSED_PEAK_CEILING: u64 = (2 * 1024 + 512) * 1024 * 1024; // 2.5 GiB
    match total_vram {
        Some(t) => (t.saturating_mul(50) / 100).min(FUSED_PEAK_CEILING),
        None => FUSED_PEAK_CEILING,
    }
}

/// Fused RGB8 path peak-allocation estimate (bytes), used by the VRAM guard.
/// Canvas + readback, ×4 for ops/COW/autotune overhead — a ~3.2 GB single
/// allocation was observed at 1080p×4 on RADV, independent of tile size and
/// autotune level.
#[cfg(any(feature = "burn", feature = "tch"))]
fn fused_peak_allocation(n: usize, out_h: usize, out_w: usize, c: usize) -> u64 {
    // accs only — the coverage canvas was dropped (feather weights are a
    // partition of unity), halving the readback and removing one canvas
    // channel.
    let canvas = (n * out_h * out_w * c * 2) as u64; // accs (f16)
    let readback = (n * out_h * out_w * 3 * 4) as u64; // packed rgb24 f32
    (canvas + readback) * 4
}

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
/// `half::f16` has no `From<f32>`, so like `ElemToU8` this routes each
/// supported element through its own constructor (same rounding as burn's
/// `TensorData::convert::<f16>`, which uses `f16::from_f32`).
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
#[cfg(any(feature = "burn", feature = "tch"))]
pub(crate) trait Rgb8Elem: ElemToU8 + ElemFromF32 + Copy {}
#[cfg(any(feature = "burn", feature = "tch"))]
impl<T: ElemToU8 + ElemFromF32 + Copy> Rgb8Elem for T {}

/// Deferred readback of one fused batch: blocks on the GPU→CPU transfer, then
/// converts to packed rgb24 on the CPU (same steps `infer_rgb8_batch` used).
#[cfg(any(feature = "burn", feature = "tch"))]
pub struct BurnRgb8Batch<B: Backend> {
    accs: Vec<BurnTensor<B, 4>>,
    out_w_t: usize,
    out_h_t: usize,
}

#[cfg(any(feature = "burn", feature = "tch"))]
impl<B: Backend> Rgb8Batch for BurnRgb8Batch<B>
where
    B::FloatElem: ElemToU8,
{
    fn resolve(self: Box<Self>) -> Result<Vec<(Vec<u8>, u32, u32)>> {
        let mut result = Vec::with_capacity(self.accs.len());
        for acc in self.accs {
            // The feather weights are a partition of unity, so the coverage
            // canvas is ≡1 and the `acc / cov` division is dropped.
            //
            // Crop on the GPU before the readback: the edge-replicate pad sits
            // on the bottom/right, so the top-left `out_h_t × out_w_t` is the
            // real frame. Reading back only that region skips the CPU
            // `crop_rgb24` pass and shrinks the PCIe transfer (the padding is
            // dropped, not copied to the host; the `* 255` below materializes
            // a contiguous buffer, so the readback is not strided).
            let [_, c, _, _] = acc.dims();
            let avg = acc
                .slice([0..1, 0..c, 0..self.out_h_t, 0..self.out_w_t])
                .permute([0, 2, 3, 1])
                * 255.0;
            // Read back in the backend's own float elem and cast to u8 in
            // plain Rust: cubecl-wgpu already pools the staging buffers, so
            // the remaining per-frame cost is the CPU-side convert — reading
            // the native elem directly skips the full f32 copy (a
            // `convert::<u8>()` readback would trip the burn-fusion ordering
            // panic, docs/burn-bugs.md Bug 1). burn uses f16, tch f32.
            let data: Vec<B::FloatElem> = match avg.into_data().to_vec::<B::FloatElem>() {
                Ok(v) => v,
                Err(e) => return Err(Error::new(e.to_string())),
            };
            // Parallel elem→u8: the convert is the main-thread stall after the
            // GPU readback (with pipeline_depth it overlaps the next forward,
            // but a 12 M-element convert still delays the next submit). Split
            // across cores so the main thread re-queues the GPU sooner.
            let mut bytes = vec![0u8; data.len()];
            std::thread::scope(|s| {
                let nt = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4)
                    .min(16);
                let chunk = data.len().div_ceil(nt);
                // chunks/chunks_mut yield disjoint slices, so each spawned
                // closure owns a non-overlapping (data, out) pair.
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
    let pad_h = h.div_ceil(pad) * pad;
    let pad_w = w.div_ceil(pad) * pad;
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
    let [n, _c, h, w] = [
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ];
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
    let pad_h = h.div_ceil(8) * 8;
    let pad_w = w.div_ceil(8) * 8;
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
    use super::{fused_peak_allocation, fused_peak_limit, pad_to_f16};
    use burn::tensor::f16;

    /// The VRAM guard's ceiling: a ~4K-class padded canvas is over it
    /// (rejected); the common x4/x2 fused renders stay under it, and the cap
    /// scales down with total VRAM. Dropping the coverage canvas (partition
    /// of unity) brought the 1080p×4 estimate below the ceiling.
    #[test]
    fn fused_vram_guard_window() {
        const LIMIT: u64 = (2 * 1024 + 512) * 1024 * 1024;
        assert!(fused_peak_allocation(1, 5120, 8320, 3) > LIMIT); // ~4K-class padded crash zone
        assert!(fused_peak_allocation(1, 4480, 8320, 3) <= LIMIT); // 1080p×4, now under the cap
        assert!(fused_peak_allocation(1, 4480, 6400, 3) <= LIMIT); // SD/720p×4 (~2.1 GiB)
        assert!(fused_peak_allocation(1, 2880, 5120, 3) <= LIMIT); // 720p×4
        assert!(fused_peak_allocation(1, 2240, 4160, 3) <= LIMIT); // 1080p×2
                                                                   // Adaptive: crash cap on big cards, half of total VRAM on small ones.
        assert_eq!(fused_peak_limit(None), LIMIT);
        assert_eq!(fused_peak_limit(Some(16 * 1024 * 1024 * 1024)), LIMIT);
        assert_eq!(
            fused_peak_limit(Some(4 * 1024 * 1024 * 1024)),
            2 * 1024 * 1024 * 1024
        );
    }

    /// The feather weights form a partition of unity: every canvas pixel's
    /// total weight across covering tiles is exactly 1.0, which is the
    /// invariant the fused path relies on to drop the coverage canvas.
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
            assert!((v - 1.0).abs() < 1e-5, "coverage at pixel {i} is {v}, not 1");
        }
    }

    /// The fused f16 upload is layout-identical to `pad_to` + cast — on-device
    /// tile slicing must not change (bit-exact for f16, which uses `from_f32`).
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
