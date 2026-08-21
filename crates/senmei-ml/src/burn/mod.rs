//! burn inference engine (Vulkan, Metal on macOS).
//!
//! Runs clean re-implementations of the adopted SR archs on the GPU backend
//! (`Vulkan<f16>` everywhere, `Metal<f16>` on macOS). Weights are loaded from a
//! pre-converted f16 burnpack (`.bpk`) — `PytorchStore` cannot cast f32→f16 at
//! load, so the app consumes the converted format (see `rust-sr-bench`'s
//! `convert-f16` for the one-time conversion). The arch is chosen from `ModelRef::arch`.

mod dncnn;
mod drunet;
mod ffdnet;
mod ifrnet;
mod nafnet;
mod real_plksr;
mod scunet;
mod span;

use crate::arch::{RrdbNet, RifeNet, UpCunet2x, UpCunet2xFast};
use crate::engine::{EngineCaps, InferOptions, InferenceEngine};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::BurnBackend;
use crate::{Error, Result};
use burn::module::ParamId;
use burn::tensor::backend::Backend;
use burn::tensor::{f16, Tensor as BurnTensor, TensorData};
use burn_store::{
    BurnpackStore, HalfPrecisionAdapter, KeyRemapper, ModuleSnapshot, PytorchStore, TensorSnapshot,
};
use burn_wgpu::WgpuDevice;
use std::path::Path;

use dncnn::Dncnn;
use drunet::Drunet;
use ffdnet::Ffdnet;
use scunet::Scunet;
use ifrnet::IfrNet;
use nafnet::NafNet;
use real_plksr::RealPlk;
use span::Span;

pub struct BurnEngine {
    model: Option<Model>,
    device: WgpuDevice,
    scale: u32,
}

enum Model {
    UpCunet2x(UpCunet2x<BurnBackend<f16>>),
    UpCunet2xFast(UpCunet2xFast<BurnBackend<f16>>),
    RrdbNet(RrdbNet<BurnBackend<f16>>),
    RifeNet(RifeNet<BurnBackend<f16>>),
    IfrNet(IfrNet<BurnBackend<f16>>),
    Drunet(Drunet<BurnBackend<f16>>),
    Dncnn(Dncnn<BurnBackend<f16>>),
    Ffdnet(Ffdnet<BurnBackend<f16>>),
    Scunet(Scunet<BurnBackend<f16>>),
    NafNet(NafNet<BurnBackend<f16>>),
    RealPlk(RealPlk<BurnBackend<f16>>),
    Span(Span<BurnBackend<f16>>),
}

impl Model {
    fn forward(
        &self,
        x: BurnTensor<BurnBackend<f16>, 4>,
    ) -> Result<BurnTensor<BurnBackend<f16>, 4>> {
        match self {
            Model::UpCunet2x(m) => Ok(m.forward(x)),
            Model::UpCunet2xFast(m) => Ok(m.forward(x)),
            Model::RrdbNet(m) => Ok(m.forward(x)),
            Model::RealPlk(m) => Ok(m.forward(x)),
            Model::Drunet(m) => Ok(m.forward(x)),
            Model::Dncnn(m) => Ok(m.forward(x)),
            Model::Scunet(m) => Ok(m.forward(x)),
            Model::NafNet(m) => Ok(m.forward(x)),
            Model::Span(m) => Ok(m.forward(x)),
            Model::RifeNet(_) | Model::IfrNet(_) | Model::Ffdnet(_) => {
                Err(Error::new("no single-input forward"))
            }
        }
    }

    fn interp(
        &self,
        a: BurnTensor<BurnBackend<f16>, 4>,
        b: BurnTensor<BurnBackend<f16>, 4>,
        t: BurnTensor<BurnBackend<f16>, 4>,
    ) -> Result<BurnTensor<BurnBackend<f16>, 4>> {
        match self {
            Model::RifeNet(m) => Ok(m.forward(a, b, t)),
            Model::IfrNet(m) => Ok(m.forward(a, b, t)),
            _ => Err(Error::new("model has no frame interpolation")),
        }
    }
}

impl BurnEngine {
    pub fn new() -> Self {
        Self {
            model: None,
            device: WgpuDevice::DiscreteGpu(0),
            scale: 1,
        }
    }

    /// RIFE loads from the raw ncnn `flownet.bin` (fp16 weights), not a burnpack.
    fn load_rife(&self, path: &Path) -> Result<Model> {
        let bytes = std::fs::read(path).map_err(|e| Error::new(e.to_string()))?;
        let mut m = RifeNet::new(&self.device);
        m.load_from_ncnn(&bytes, &self.device).map_err(Error::new)?;
        Ok(Model::RifeNet(m))
    }

    fn load_arch(&self, model: &ModelRef, store: &mut BurnpackStore) -> Result<Model> {
        match model.arch.as_str() {
            "upcunet2x" => {
                let mut m = UpCunet2x::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::UpCunet2x(m))
            }
            "upcunet2x-fast" | "fallin-cugan" => {
                // Fallin (renarchi CUGAN retrain) is an `UpCunet2x_fast` with
                // the same 38px reflect pad — only the weights differ.
                let mut m = UpCunet2xFast::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::UpCunet2xFast(m))
            }
            "realesrgan" => {
                let mut m =
                    RrdbNet::new(model.scale as usize, model.num_block as usize, &self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::RrdbNet(m))
            }
            "ifrnet" => {
                let mut m = IfrNet::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::IfrNet(m))
            }
            "drunet" => {
                let mut m = Drunet::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::Drunet(m))
            }
            "dncnn" => {
                let mut m = Dncnn::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::Dncnn(m))
            }
            "ffdnet" => {
                let mut m = Ffdnet::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::Ffdnet(m))
            }
            "scunet" => {
                let mut m = Scunet::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::Scunet(m))
            }
            "nafnet" => {
                let mut m = NafNet::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::NafNet(m))
            }
            "real-plksr" => {
                let mut m = RealPlk::new(model.scale as usize, model.layer_norm, &self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::RealPlk(m))
            }
            "span" => {
                let mut m = Span::new(
                    model.feature_channels as usize,
                    model.scale as usize,
                    &self.device,
                );
                m.set_no_norm(model.no_norm);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::Span(m))
            }
            other => Err(Error::new(format!("unsupported arch: {other}"))),
        }
    }
}

impl InferenceEngine for BurnEngine {
    fn capabilities(&self) -> EngineCaps {
        EngineCaps { tiles: true }
    }

    fn load(&mut self, model: &ModelRef) -> Result<()> {
        self.model = Some(match model.arch.as_str() {
            "rife425" | "rife46" => self.load_rife(&model.path)?,
            _ => {
                let mut store = BurnpackStore::from_file(&model.path);
                self.load_arch(model, &mut store)?
            }
        });
        self.scale = model.scale;
        Ok(())
    }

    fn infer(&mut self, input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::new("model not loaded"))?;
        if input.shape.len() != 4 {
            return Err(Error::new("expected NCHW input"));
        }
        let n = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];

        let data = TensorData::new(input.data.clone(), [n, c, h, w]).convert::<f16>();
        let x = BurnTensor::<BurnBackend<f16>, 4>::from_data(data, &self.device);
        let out = model.forward(x)?;
        let [_, _, oh, ow] = out.dims();
        let data = out
            .into_data()
            .convert::<f32>()
            .to_vec()
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(Tensor::new(vec![n, c, oh, ow], data))
    }

    /// Fused RGB8 path: hands back packed rgb24 bytes. The model runs on the
    /// GPU (autotuned, f16) in 640px tiles (avoids the full-frame im2col OOM,
    /// see docs/upstream-issues.md §2; 640 beats 512 and 768 — docs/benchmarks.md).
    /// Tiles are accumulated into one f16 canvas on the GPU (overlap averaging)
    /// and read back as a single packed frame — one readback instead of one u8
    /// readback per tile plus a CPU stitch.
    /// Only used when the requested scale matches the model.
    fn infer_rgb8(&mut self, input: &Tensor, scale: u32) -> Option<Result<(Vec<u8>, u32, u32)>> {
        if self.scale != scale {
            return None;
        }
        let model = self.model.as_ref()?;
        if input.shape.len() != 4 {
            return Some(Err(Error::new("expected NCHW input")));
        }
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        let tile = crate::current_tile_size();
        let overlap = tile / 4;
        let step = tile - overlap;
        let num_y = (h.saturating_sub(tile)).div_ceil(step) + 1;
        let num_x = (w.saturating_sub(tile)).div_ceil(step) + 1;
        let ph = (num_y - 1) * step + tile;
        let pw = (num_x - 1) * step + tile;
        let padded = crate::pad_to(input, ph, pw);
        let tiles = crate::uniform_tile(&padded, tile, step);
        let device = &self.device;

        let scale_f = self.scale as f32;
        let out_h = (ph as f32 * scale_f).round() as usize;
        let out_w = (pw as f32 * scale_f).round() as usize;
        // Accumulate tiles into one f16 canvas (overlap averaging) on the GPU.
        // Tiles run one-by-one: larger batched matmuls are pathologically
        // slower on this backend (see docs/benchmarks.md tile-size note).
        let mut acc = BurnTensor::<BurnBackend<f16>, 4>::zeros([1, c, out_h, out_w], device);
        let mut cov = BurnTensor::<BurnBackend<f16>, 4>::zeros([1, 1, out_h, out_w], device);
        for (x, y, t) in &tiles {
            let data = TensorData::new(t.data.clone(), [1, c, tile, tile]).convert::<f16>();
            let xt = BurnTensor::<BurnBackend<f16>, 4>::from_data(data, device);
            let out = match model.forward(xt) {
                Ok(o) => o,
                Err(e) => return Some(Err(e)),
            };
            let [_, _, oh, ow] = out.dims();
            let sx = (*x as f32 * scale_f).round() as usize;
            let sy = (*y as f32 * scale_f).round() as usize;
            // Clamp before accumulating: out-of-range values (>1.0 at hard
            // edges, e.g. burnt-in subtitles) would wrap on the u8 cast below.
            let out = out.clamp(0.0, 1.0);
            acc = acc.slice_assign([0..1, 0..c, sy..sy + oh, sx..sx + ow], out);
            let ones = BurnTensor::<BurnBackend<f16>, 4>::ones([1, 1, oh, ow], device);
            cov = cov.slice_assign([0..1, 0..1, sy..sy + oh, sx..sx + ow], ones);
        }
        let avg = (acc / cov).permute([0, 2, 3, 1]) * 255.0;
        let bytes: Vec<u8> = match avg.cast(burn::tensor::IntDType::U8).into_data().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        let out_h_t = (h as f32 * scale_f).round() as usize;
        let out_w_t = (w as f32 * scale_f).round() as usize;
        let cropped = crate::crop_rgb24(&bytes, out_w, out_h_t, out_w_t);
        Some(Ok((cropped, out_w_t as u32, out_h_t as u32)))
    }

    fn infer_interp(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        t: f32,
        _opts: &InferOptions,
    ) -> Option<Result<Tensor>> {
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return Some(Err(Error::new("model not loaded"))),
        };
        if !matches!(model, Model::RifeNet(_) | Model::IfrNet(_)) {
            return None; // not an interpolation model → caller falls back
        }
        // The flow estimators work on a downscaled grid (RIFE 1/32, IFRNet
        // 1/16 via the 4-level pyramid), so the input is padded to a multiple
        // and cropped back to the original dims (same as the references).
        let pad = if matches!(model, Model::RifeNet(_)) { 32 } else { 16 };
        let [n, c, h, w] = [a.shape[0], a.shape[1], a.shape[2], a.shape[3]];
        let a_t = BurnTensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(a.data.clone(), [n, c, h, w]).convert::<f16>(),
            &self.device,
        );
        let b_t = BurnTensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(b.data.clone(), [n, c, h, w]).convert::<f16>(),
            &self.device,
        );
        // RIFE's internal flow estimation runs at 1/32 scale, so the reference
        // (rife-ncnn-vulkan) pads the input to multiples of 32. Do the same and
        // crop the output back to the original dims.
        let pad_h = (h + pad - 1) / pad * pad;
        let pad_w = (w + pad - 1) / pad * pad;
        let pad = |x: BurnTensor<BurnBackend<f16>, 4>| {
            let mut x = x;
            if pad_h > h {
                let z =
                    BurnTensor::<BurnBackend<f16>, 4>::zeros([n, c, pad_h - h, w], &self.device);
                x = BurnTensor::cat(vec![x, z], 2);
            }
            if pad_w > w {
                let z = BurnTensor::<BurnBackend<f16>, 4>::zeros(
                    [n, c, pad_h, pad_w - w],
                    &self.device,
                );
                x = BurnTensor::cat(vec![x, z], 3);
            }
            x
        };
        let a_t = pad(a_t);
        let b_t = pad(b_t);
        // ncnn broadcasts the scalar timestep over the (padded) spatial grid.
        let t_t = BurnTensor::<BurnBackend<f16>, 4>::ones([n, 1, pad_h, pad_w], &self.device) * t;
        let out = match model.interp(a_t, b_t, t_t) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let out = out.slice([0..n, 0..c, 0..h, 0..w]);
        let data = match out.into_data().convert::<f32>().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        Some(Ok(Tensor::new(vec![n, c, h, w], data)))
    }

    /// DRUNet denoise: appends a constant noise-level map (sigma in [0,1]) to
    /// the 3-channel input, pads the spatial dims to multiples of 8 (the UNet
    /// downsamples 3× stride-2), runs the model, and crops back. Other models
    /// return `None` so the caller falls back to the CPU reference.
    fn infer_denoise(
        &mut self,
        input: &Tensor,
        sigma: f32,
        _opts: &InferOptions,
    ) -> Option<Result<Tensor>> {
        let model = self.model.as_ref()?;
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
        let [n, c, h, w] = [
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        ];
        let device = &self.device;
        let rgb = BurnTensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(input.data.clone(), [n, c, h, w]).convert::<f16>(),
            device,
        );
        // DnCNN is blind (3ch in, no sigma map) and all stride-1 convs — run it
        // directly. FFDNet takes the noise level internally (σ). DRUNet gets a
        // constant noise-level map and needs 8-aligned spatial dims (3× stride-2
        // downsample): pad + crop.
        if let Model::Ffdnet(m) = model {
            let out = m.forward(rgb, sigma);
            let data = match out.into_data().convert::<f32>().to_vec() {
                Ok(v) => v,
                Err(e) => return Some(Err(Error::new(e.to_string()))),
            };
            return Some(Ok(Tensor::new(vec![n, 3, h, w], data)));
        }
        if !is_drunet {
            let out = match model.forward(rgb) {
                Ok(o) => o,
                Err(e) => return Some(Err(e)),
            };
            let data = match out.into_data().convert::<f32>().to_vec() {
                Ok(v) => v,
                Err(e) => return Some(Err(Error::new(e.to_string()))),
            };
            return Some(Ok(Tensor::new(vec![n, 3, h, w], data)));
        }
        let sigma_map = BurnTensor::<BurnBackend<f16>, 4>::ones([n, 1, h, w], device) * sigma;
        let x = BurnTensor::cat(vec![rgb, sigma_map], 1);
        // UNetRes needs multiples of 8 (3× stride-2 downsample); pad + crop.
        let pad_h = (h + 7) / 8 * 8;
        let pad_w = (w + 7) / 8 * 8;
        let mut x = x;
        if pad_h > h {
            let z = BurnTensor::<BurnBackend<f16>, 4>::zeros([n, 4, pad_h - h, w], device);
            x = BurnTensor::cat(vec![x, z], 2);
        }
        if pad_w > w {
            let z = BurnTensor::<BurnBackend<f16>, 4>::zeros([n, 4, pad_h, pad_w - w], device);
            x = BurnTensor::cat(vec![x, z], 3);
        }
        let out = match model.forward(x) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        }
        .slice([0..n, 0..3, 0..h, 0..w]);
        let data = match out.into_data().convert::<f32>().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        Some(Ok(Tensor::new(vec![n, 3, h, w], data)))
    }
}

/// One-time `.pth` → f16 `.bpk` conversion for an arch (maintainer step).
/// Loads the f32 state dict on the Vulkan backend (upcunet key remap), then
/// saves through `HalfPrecisionAdapter` so `BurnEngine` can load it as f16.
pub fn convert_pth_to_bpk(
    arch: &str,
    pth_path: &Path,
    bpk_path: &Path,
    scale: u32,
    num_block: u32,
    layer_norm: bool,
) -> Result<()> {
    let device = WgpuDevice::DiscreteGpu(0);
    let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(HalfPrecisionAdapter::new());
    match arch {
        "upcunet2x" | "upcunet2x-fast" | "fallin-cugan" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"\.conv\.0\.", ".conv.")
                .with_key_remapping(r"\.conv\.2\.", ".conv2.");
            match arch {
                "upcunet2x" => {
                    let mut m = UpCunet2x::<BurnBackend>::new(&device);
                    m.load_from(&mut store)
                        .map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save)
                        .map_err(|e| Error::new(e.to_string()))?;
                }
                _ => {
                    // upcunet2x-fast and fallin-cugan share the module layout.
                    let mut m = UpCunet2xFast::<BurnBackend>::new(&device);
                    m.load_from(&mut store)
                        .map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save)
                        .map_err(|e| Error::new(e.to_string()))?;
                }
            }
        }
        "realesrgan" => {
            // Also handles BSRGAN (KAIR): same RRDBNet, but its keys use the
            // older BasicSR naming (`RRDB_trunk.{i}.RDB{j}.conv{k}`, `trunk_conv`,
            // `upconv1/2`, `HRconv`); the rules only match those, so standard
            // Real-ESRGAN pths (`body.{i}.rdb{j}.conv{k}`, `conv_body`,
            // `conv_up1/2`, `conv_hr`) pass through unchanged.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(
                    r"^RRDB_trunk\.(\d+)\.RDB(\d+)\.conv(\d+)\.",
                    "body.$1.rdb$2.conv$3.",
                )
                .with_key_remapping(r"^params_ema\.", "")
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"^trunk_conv\.", "conv_body.")
                .with_key_remapping(r"^upconv1\.", "conv_up1.")
                .with_key_remapping(r"^upconv2\.", "conv_up2.")
                .with_key_remapping(r"^HRconv\.", "conv_hr.");
            let mut m = RrdbNet::<BurnBackend>::new(scale as usize, num_block as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "ifrnet" => {
            // Torch Sequential/ResBlock keys (pyramid1.0.0, convblock.1.conv1.0,
            // …) are mapped onto the burn field paths (p1.c0.conv, cb1.c1.conv,
            // …) with capture-group rules; strips a DataParallel `module.` prefix.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^module\.", "")
                .with_key_remapping(r"encoder\.pyramid(\d)\.(\d)\.0\.", "encoder.p$1.c$2.conv.")
                .with_key_remapping(r"encoder\.pyramid(\d)\.(\d)\.1\.", "encoder.p$1.c$2.prelu.")
                .with_key_remapping(r"decoder(\d)\.convblock\.0\.0\.", "decoder$1.cb0.conv.")
                .with_key_remapping(r"decoder(\d)\.convblock\.0\.1\.", "decoder$1.cb0.prelu.")
                .with_key_remapping(
                    r"decoder(\d)\.convblock\.1\.conv([1-4])\.0\.",
                    "decoder$1.cb1.c$2.conv.",
                )
                .with_key_remapping(
                    r"decoder(\d)\.convblock\.1\.conv([1-4])\.1\.",
                    "decoder$1.cb1.c$2.prelu.",
                )
                .with_key_remapping(r"decoder(\d)\.convblock\.1\.conv5\.", "decoder$1.cb1.c5.")
                .with_key_remapping(r"decoder(\d)\.convblock\.1\.prelu\.", "decoder$1.cb1.pl.")
                .with_key_remapping(r"decoder(\d)\.convblock\.2\.", "decoder$1.cb2.");
            let mut m = IfrNet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "drunet" => {
            // Torch Sequential ResBlock keys (m_down1.0.res.0/.res.2, the
            // index-4 stride-conv m_down1.4, and the index-0 deconv m_up3.0)
            // are mapped onto the burn field paths (b0.c1/b0.c2, down, up)
            // with capture-group rules.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^module\.", "")
                .with_key_remapping(r"m_down(\d)\.(\d)\.res\.0\.", "m_down$1.b$2.c1.")
                .with_key_remapping(r"m_down(\d)\.(\d)\.res\.2\.", "m_down$1.b$2.c2.")
                .with_key_remapping(r"m_down(\d)\.4\.", "m_down$1.down.")
                .with_key_remapping(r"m_body\.(\d)\.res\.0\.", "m_body.b$1.c1.")
                .with_key_remapping(r"m_body\.(\d)\.res\.2\.", "m_body.b$1.c2.")
                .with_key_remapping(r"m_up(\d)\.(\d)\.res\.0\.", "m_up$1.b$2.c1.")
                .with_key_remapping(r"m_up(\d)\.(\d)\.res\.2\.", "m_up$1.b$2.c2.")
                .with_key_remapping(r"m_up(\d)\.0\.", "m_up$1.up.");
            let mut m = Drunet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "dncnn" => {
            // Torch `model.{2i}.weight/bias` (ReLU sits at odd `{2i+1}` slots,
            // no params) map onto the burn `c{2i}` field names 1:1.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^model\.(\d+)\.", "c$1.");
            let mut m = Dncnn::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "ffdnet" => {
            // Same `model.{2i}` layout as DnCNN (ReLU at odd slots).
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^model\.(\d+)\.", "c$1.");
            let mut m = Ffdnet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "scunet" => {
            // Torch `m_{head,down,body,up,tail}` Sequential keys map onto the
            // burn field paths: head/tail are `m_head.0.`/`m_tail.0.`; down
            // levels keep block indices 0-3 and the index-4 stride conv maps
            // to `_down`; up levels map the index-0 deconv to `_up`. MLP/conv
            // blocks are torch Sequentials (`.mlp.0`/`.mlp.2`,
            // `.conv_block.0`/`.conv_block.2`) and LayerNorm weight/bias are
            // burn `gamma`/`beta`.
            //
            // The `relative_position_params` bare-tensor param lives in the
            // custom `Wmsa` module, which is not in the default half-precision
            // set — add it so the f16 bpk stores it as F16 (otherwise the f16
            // model loads it F32 and the attention add fails DTypeMismatch).
            let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(
                HalfPrecisionAdapter::new().with_module("Wmsa"),
            );
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^module\.", "")
                .with_key_remapping(r"^m_head\.0\.", "m_head.")
                .with_key_remapping(r"^m_tail\.0\.", "m_tail.")
                .with_key_remapping(r"^m_down(\d)\.4\.", "m_down${1}_down.")
                .with_key_remapping(r"^m_up(\d)\.0\.", "m_up${1}_up.")
                .with_key_remapping(r"\.trans_block\.mlp\.0\.", ".trans_block.mlp0.")
                .with_key_remapping(r"\.trans_block\.mlp\.2\.", ".trans_block.mlp2.")
                .with_key_remapping(r"\.conv_block\.0\.", ".conv_block.c0.")
                .with_key_remapping(r"\.conv_block\.2\.", ".conv_block.c2.")
                .with_key_remapping(r"\.ln([12])\.weight", ".ln$1.gamma")
                .with_key_remapping(r"\.ln([12])\.bias", ".ln$1.beta");
            let mut m = Scunet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "nafnet" => {
            // Torch NAFBlock keys (encoders.0.0.conv1, sca.1, middle_blks.0,
            // ups.0.0, downs.0) map onto the burn field paths
            // (encoders.0.blocks.0.conv1, sca_conv, middle.0, ups.0.conv,
            // downs.0) with capture-group rules. The checkpoint wraps the
            // state dict under `params`. The custom `NafBlock`/`LayerNorm2d`
            // structs hold `beta`/`gamma`/norm params that aren't in the
            // default half-precision set, so add them for the f16 conversion.
            let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(
                HalfPrecisionAdapter::new()
                    .with_module("NafBlock")
                    .with_module("LayerNorm2d"),
            );
            let mut store = PytorchStore::from_file(pth_path)
                .with_top_level_key("params")
                .with_key_remapping(r"^encoders\.(\d+)\.(\d+)\.", "encoders.$1.blocks.$2.")
                .with_key_remapping(r"^decoders\.(\d+)\.(\d+)\.", "decoders.$1.blocks.$2.")
                .with_key_remapping(r"^middle_blks\.(\d+)\.", "middle.$1.")
                .with_key_remapping(r"^ups\.(\d+)\.0\.", "ups.$1.conv.")
                .with_key_remapping(r"sca\.1\.", "sca_conv.");
            let mut m = NafNet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "real-plksr" => {
            // Remap the torch `feats.{i}` / `to_img.` keys onto the module
            // record paths (`head`/`blocks`/`tail`, and `offset`/`scope`/
            // `end_conv`). The channel_mixer/attn are torch `nn.Sequential`,
            // so their sub-convs are indexed (`channel_mixer.0`/`.2`,
            // `attn.f.0`) rather than named. LayerNorm blocks add
            // `feats.{i}.layer_norm.{weight,bias}` (record
            // `blocks.{i-1}.layer_norm.{weight,bias}`) and drop the GroupNorm.
            //
            // Some pths (4x-alchemy) wrap the state dict under `params`, others
            // (2xPublic) are flat — the reader recurses nested dicts by default,
            // so `^params\.` → "" handles both (no-op on flat files).
            //
            // NOTE: the pth must have contiguous tensors — burn-store's reader
            // ignores strides (docs/upstream-issues.md §4), so a channels-last
            // state dict (e.g. the raw `4x_Alchemy.pth`) loads scrambled.
            // Preprocess with `{k: v.contiguous() for k, v in sd.items()}`.
            let store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"^feats\.0\.", "head.")
                .with_key_remapping(r"^feats\.30\.", "tail.")
                .with_key_remapping(r"^to_img\.", "")
                .with_key_remapping(r"\.channel_mixer\.0\.", ".channel_mixer.conv1.")
                .with_key_remapping(r"\.channel_mixer\.2\.", ".channel_mixer.conv2.")
                .with_key_remapping(r"\.attn\.f\.0\.", ".attn.f.");
            let store = (1..=28usize).fold(store, |s, i| {
                s.with_key_remapping(format!(r"^feats\.{i}\."), format!("blocks.{}.", i - 1))
            });
            let mut store = store;
            let mut m = RealPlk::<BurnBackend>::new(scale as usize, layer_norm, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "span" => {
            // Phhofm is flat; TNTwise wraps in `params` (stripped). Stale
            // `eval_conv.*` and `no_norm` are ignored by `load_from`. The 5th
            // CLI arg (num_block slot) is the feature-channel count: 48 for
            // the Phhofm 2× family, 64 for TNTwise ModernSpanimation V1/V1.5.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"\.conv\.0\.", ".conv0.")
                .with_key_remapping(r"\.conv\.1\.", ".conv1.")
                .with_key_remapping(r"\.conv\.2\.", ".conv2.")
                .with_key_remapping(r"^upsampler\.0\.", "upsampler.");
            let mut m = Span::<BurnBackend>::new(num_block as usize, scale as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
}

/// One-time ONNX → f16 `.bpk` conversion (maintainer + `download_model`).
///
/// Reads only the `initializer` tensors via the built-in protobuf reader (no
/// ONNX Runtime); the names already match the module state dict apart from the
/// torch `.conv.0` / `.conv.2` quirk, which is remapped here. Weights are
/// decoded to f32 and saved through `HalfPrecisionAdapter` like the `.pth` path.
pub fn convert_onnx_to_bpk(
    arch: &str,
    onnx_path: &Path,
    bpk_path: &Path,
    scale: u32,
    num_block: u32,
) -> Result<()> {
    let bytes = std::fs::read(onnx_path)?;
    let tensors = crate::onnx::read_initializers(&bytes).map_err(Error::new)?;
    let mut snapshots = Vec::with_capacity(tensors.len());
    for t in tensors {
        let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
        let data = onnx_data_to_f32(&t)?;
        let mut s = TensorSnapshot::from_data(
            TensorData::new(data, shape),
            t.name.split('.').map(str::to_string).collect(),
            Vec::new(),
            ParamId::new(),
        );
        s.container_stack = None;
        s.tensor_id = None;
        snapshots.push(s);
    }
    let remapper = KeyRemapper::from_patterns(vec![
        (r"\.conv\.0\.", ".conv."),
        (r"\.conv\.2\.", ".conv2."),
    ])
    .map_err(|e| Error::new(e.to_string()))?;
    let (snapshots, _) = remapper.remap(snapshots);

    let device = WgpuDevice::DiscreteGpu(0);
    let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(HalfPrecisionAdapter::new());
    match arch {
        "upcunet2x" => {
            let mut m = UpCunet2x::<BurnBackend>::new(&device);
            apply_and_save(&mut m, snapshots, &mut save)?;
        }
        "upcunet2x-fast" | "fallin-cugan" => {
            let mut m = UpCunet2xFast::<BurnBackend>::new(&device);
            apply_and_save(&mut m, snapshots, &mut save)?;
        }
        "realesrgan" => {
            let mut m = RrdbNet::<BurnBackend>::new(scale as usize, num_block as usize, &device);
            apply_and_save(&mut m, snapshots, &mut save)?;
        }
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
}

fn apply_and_save<B, M>(
    m: &mut M,
    snapshots: Vec<TensorSnapshot>,
    save: &mut BurnpackStore,
) -> Result<()>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let result = m.apply(snapshots, None, None, true);
    if !result.missing.is_empty() {
        return Err(Error::new(format!("missing tensors:\n{result}")));
    }
    m.save_into(save).map_err(|e| Error::new(e.to_string()))?;
    Ok(())
}

fn onnx_data_to_f32(t: &crate::onnx::OnnxTensor) -> Result<Vec<f32>> {
    let n = t.dims.iter().map(|&d| d as usize).product::<usize>();
    let mut out = Vec::with_capacity(n);
    match t.dtype {
        1 => {
            for c in t.data.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        10 => {
            for c in t.data.chunks_exact(2) {
                out.push(f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32());
            }
        }
        11 => {
            for c in t.data.chunks_exact(8) {
                out.push(
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
                );
            }
        }
        6 => {
            for c in t.data.chunks_exact(4) {
                out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
        }
        7 => {
            for c in t.data.chunks_exact(8) {
                out.push(
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
                );
            }
        }
        other => {
            return Err(Error::new(format!(
                "unsupported ONNX dtype {other} for {}",
                t.name
            )))
        }
    }
    if out.len() != n {
        return Err(Error::new(format!("data length mismatch for {}", t.name)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan + models/up2x-no-denoise.pth.f16.bpk (via senmei-ml-convert)"]
    fn burn_engine_loads_and_infers_up2x() {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let bpk = dir.join("up2x-no-denoise.pth.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing .bpk, skipping");
            return;
        }
        let mut registry = crate::model::Registry::new();
        registry.load_dir(&dir).unwrap();
        let mref = registry.resolve("real-cugan-x2", &dir).unwrap();
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();
        let input = Tensor::new(vec![1, 3, 32, 32], vec![0.5f32; 3 * 32 * 32]);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    /// End-to-end Fallin check: engine output must match the ONNX reference
    /// (onnx2torch) on a deterministic 256x256 input. Dumps the raw f32 output
    /// to /tmp/fallin/rust_out.bin for comparison against ref256.npy.
    #[test]
    #[ignore = "requires Vulkan + a converted Fallin .bpk (senmei-ml-convert)"]
    fn fallin_inference_matches_onnx_reference() {
        let bpk = std::path::Path::new("/tmp/fallin/fallin_soft.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "fallin-soft".into(),
            arch: "fallin-cugan".into(),
            scale: 2,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();
        let h = 256;
        let w = 256;
        let n = 1 * 3 * h * w;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as u64 * 2654435761) % 10000) as f32 / 10000.0)
            .collect();
        let input = Tensor::new(vec![1, 3, h, w], data);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, h * 2, w * 2]);
        assert!(out.data.iter().all(|v| v.is_finite()));
        let mut bytes = Vec::with_capacity(out.data.len() * 4);
        for v in &out.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write("/tmp/fallin/rust_out.bin", bytes).unwrap();
        eprintln!("wrote /tmp/fallin/rust_out.bin");
    }

    /// End-to-end RealPLKSR (4x-alchemy) check against the torch reference on
    /// the deterministic 64x64 input (`/tmp/alchemy_in.f32` → `alchemy_ref.f32`,
    /// generated by `plksr_ref.py`). f16 vs f32 so a loose tolerance is used.
    #[test]
    #[ignore = "requires Vulkan + a converted 4x_Alchemy.f16.bpk (senmei-ml-convert)"]
    fn real_plksr_4x_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/4x_Alchemy.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "4x-alchemy".into(),
            arch: "real-plksr".into(),
            scale: 4,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/alchemy_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(input.len(), 3 * 64 * 64);
        let input = Tensor::new(vec![1, 3, 64, 64], input);

        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 256, 256]);

        let ref_data = std::fs::read("/tmp/alchemy_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut nan_count = 0usize;
        for (a, b) in out.data.iter().zip(&reference) {
            if !a.is_finite() || !b.is_finite() {
                nan_count += 1;
                continue;
            }
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!(
            "real_plksr_4x mae={mae:.5} max_abs={max_abs:.5} nan={nan_count}/{}",
            out.data.len()
        );
        assert_eq!(nan_count, 0, "output contains non-finite values");
        assert!(max_abs < 0.05, "max abs diff {max_abs} exceeds 0.05");
    }

    /// End-to-end RealPLKSR 1x (deh264, no DySample) check against the torch
    /// reference on the deterministic 64x64 input. Isolates the `feats` chain
    /// from the DySample tail.
    #[test]
    #[ignore = "requires Vulkan + a converted 1xDeH264_realplksr.f16.bpk (senmei-ml-convert)"]
    fn real_plksr_1x_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/deh264.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "real-plksr-deh264".into(),
            arch: "real-plksr".into(),
            scale: 1,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/deh264_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input = Tensor::new(vec![1, 3, 64, 64], input);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);

        let ref_data = std::fs::read("/tmp/deh264_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        for (a, b) in out.data.iter().zip(&reference) {
            assert!(a.is_finite(), "output contains non-finite value");
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!("real_plksr_1x mae={mae:.5} max_abs={max_abs:.5}");
        assert!(max_abs < 0.05, "max abs diff {max_abs} exceeds 0.05");
    }

    /// End-to-end RealPLKSR 1x (dejpg, no DySample) check against the torch
    /// reference — same arch/scale as deh264, different weights.
    #[test]
    #[ignore = "requires Vulkan + a converted 1xDeJPG_realplksr_otf.f16.bpk (senmei-ml-convert)"]
    fn real_plksr_dejpg_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/dejpg.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "real-plksr-dejpg".into(),
            arch: "real-plksr".into(),
            scale: 1,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/dejpg_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input = Tensor::new(vec![1, 3, 64, 64], input);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);

        let ref_data = std::fs::read("/tmp/dejpg_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        for (a, b) in out.data.iter().zip(&reference) {
            assert!(a.is_finite(), "output contains non-finite value");
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!("real_plksr_dejpg mae={mae:.5} max_abs={max_abs:.5}");
        assert!(max_abs < 0.05, "max abs diff {max_abs} exceeds 0.05");
    }

    /// End-to-end RealPLKSR 2× (2xPublic, DySample + channel LayerNorm) check
    /// against the ONNX reference on the deterministic 256x256 input
    /// (`/tmp/realplksr2x_in.f32` → `realplksr2x_public_ref.f32`, onnxruntime
    /// CPU on the official fp32 ONNX — the DySample tail is static-sized, so
    /// the input is fixed at 256²). Exercises the LayerNorm-at-block-start
    /// path; f16 vs f32 → loose tolerance.
    #[test]
    #[ignore = "requires Vulkan + a converted 2xpublic_ln.f16.bpk (senmei-ml-convert)"]
    fn real_plksr_2x_public_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/2xpublic_ln.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "real-plksr-2x-public".into(),
            arch: "real-plksr".into(),
            scale: 2,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: true,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/realplksr2x_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(input.len(), 3 * 256 * 256);
        let input = Tensor::new(vec![1, 3, 256, 256], input);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 512, 512]);

        let ref_data = std::fs::read("/tmp/realplksr2x_public_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut nan_count = 0usize;
        for (a, b) in out.data.iter().zip(&reference) {
            if !a.is_finite() {
                nan_count += 1;
                continue;
            }
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!(
            "real_plksr_2x_public mae={mae:.5} max_abs={max_abs:.5} nan={nan_count}/{}",
            out.data.len()
        );
        assert_eq!(nan_count, 0, "output contains non-finite values");
        assert!(max_abs < 0.05, "max abs diff {max_abs} exceeds 0.05");
    }

    #[test]
    #[ignore = "requires Vulkan + models/flownet.bin; needs RUST_MIN_STACK=33554432"]
    fn rife_loads_weights_and_interpolates() {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let bin = dir.join("flownet.bin");
        if !bin.exists() {
            eprintln!("missing flownet.bin, skipping");
            return;
        }
        let mut registry = crate::model::Registry::new();
        registry.load_dir(&dir).unwrap();
        let mref = registry.resolve("rife-v4.6", &dir).unwrap();
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let h = 64;
        let w = 64;
        // Two gradient frames with a clean 8px horizontal shift (content moves
        // right, edges clamp) so optical flow is unambiguous.
        let mut a = vec![0f32; 3 * h * w];
        let mut b = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    a[(c * h + y) * w + x] = x as f32 / (w - 1) as f32;
                    b[(c * h + y) * w + x] = ((x as i32 - 8).max(0) as f32) / (w - 1) as f32;
                }
            }
        }
        let a = Tensor::new(vec![1, 3, h, w], a);
        let b = Tensor::new(vec![1, 3, h, w], b);
        let opts = InferOptions { tile_size: None };
        let mid = engine
            .infer_interp(&a, &b, 0.5, &opts)
            .expect("engine should handle RIFE")
            .unwrap();
        assert_eq!(mid.shape, vec![1, 3, h, w]);
        assert!(mid.data.iter().all(|v| v.is_finite()));

        // The reference short-circuits t=0/t=1 (copies the input), so exact
        // endpoints aren't a network property. What must hold is that the
        // result is flow-based (not the linear blend) and consistent.
        let blend = crate::interpolate::blend(&a, &b, 0.5);
        let diff = mean_abs_diff(&mid.data, &blend.data);
        assert!(
            diff > 0.002,
            "output matches the linear blend, engine not used: {diff}"
        );

        // Symmetry: interpolating (a,b) at t must equal interpolating (b,a) at
        // 1-t — the same in-between frame. Any swapped/negated flow wiring
        // breaks this badly.
        let t1 = engine.infer_interp(&a, &b, 0.25, &opts).unwrap().unwrap();
        let t2 = engine.infer_interp(&b, &a, 0.75, &opts).unwrap().unwrap();
        let sym = mean_abs_diff(&t1.data, &t2.data);
        assert!(sym < 0.02, "interp is not symmetric: {sym}");

        // Directionality: t=0.05 stays near a, t=0.95 near b (fp16, so loose).
        let lo = engine.infer_interp(&a, &b, 0.05, &opts).unwrap().unwrap();
        let hi = engine.infer_interp(&a, &b, 0.95, &opts).unwrap().unwrap();
        assert!(
            mean_abs_diff(&lo.data, &a.data) < mean_abs_diff(&lo.data, &b.data),
            "t=0.05 drifted to b"
        );
        assert!(
            mean_abs_diff(&hi.data, &b.data) < mean_abs_diff(&hi.data, &a.data),
            "t=0.95 drifted to a"
        );
    }

    fn mean_abs_diff(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.len() as f32
    }

    #[test]
    fn tensor_data_f16_to_vec_is_available() {
        // Exercises the CPU-side API we need for a fused f16->RGB8 output path
        // (no GPU involved): f16 TensorData must hand back raw f16 values.
        let data =
            burn::tensor::TensorData::new(vec![0.5f32, 1.0, 0.0, 0.25], [4]).convert::<f16>();
        let v: Vec<f16> = data.to_vec().unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v[0].to_f32(), 0.5);
        assert_eq!(v[2].to_f32(), 0.0);
    }

    /// The tiled-fused RGB8 path must be reliable over many frames (the
    /// full-frame variant OOM'd autotune — docs/upstream-issues.md §2) and match
    /// the reference (`infer` + NCHW→rgb24 interleave).
    #[test]
    #[ignore = "requires Vulkan + fallin-soft bpk"]
    fn infer_rgb8_tiled_is_reliable_and_correct() {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let mut registry = crate::model::Registry::new();
        registry.load_dir(&dir).unwrap();
        let mref = registry.resolve("fallin-soft", &dir).unwrap();
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        // Correctness: single 640 tile, so tiled == full pass -> exact match.
        let (h, w): (usize, usize) = (640, 640);
        let input = Tensor::new(vec![1, 3, h, w], vec![0.5f32; 1 * 3 * h * w]);
        let out = engine
            .infer(
                &input,
                &InferOptions {
                    tile_size: Some(640),
                },
            )
            .unwrap();
        let (_, _, oh, ow) = (out.shape[0], out.shape[1], out.shape[2], out.shape[3]);
        let hw = oh * ow;
        let mut ref_bytes = Vec::with_capacity(3 * hw);
        for p in 0..hw {
            ref_bytes.push((out.data[p].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            ref_bytes.push((out.data[hw + p].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            ref_bytes.push((out.data[2 * hw + p].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        let (bytes, rw, rh) = engine.infer_rgb8(&input, 2).unwrap().unwrap();
        assert_eq!((rw, rh), (ow as u32, oh as u32));
        // GPU fused path computes in fp16, the reference in f32: allow ±2
        // rounding noise (layout must be identical, values ~identical).
        let max_diff = bytes
            .iter()
            .zip(&ref_bytes)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        let mean_diff = bytes
            .iter()
            .zip(&ref_bytes)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .sum::<i32>() as f32
            / bytes.len() as f32;
        assert!(
            max_diff <= 2,
            "max diff {max_diff}, mean diff {mean_diff:.2}"
        );

        // Reliability: 48 frames of 1080p (4x2 tiles, real stitch path).
        let (h, w): (usize, usize) = (1080, 1920);
        let input = Tensor::new(vec![1, 3, h, w], vec![0.5f32; 1 * 3 * h * w]);
        for _ in 0..48 {
            let (bytes, rw, rh) = engine.infer_rgb8(&input, 2).unwrap().unwrap();
            assert_eq!(rw, (w * 2) as u32);
            assert_eq!(rh, (h * 2) as u32);
            assert_eq!(bytes.len(), (rw * rh * 3) as usize);
        }
    }

    /// The DRUNet denoise path appends the sigma map, pads to multiples of 8,
    /// and crops back — output must be 3ch at the input size. Uses a height
    /// that is *not* a multiple of 8 to exercise the pad/crop.
    #[test]
    #[ignore = "requires Vulkan + drunet bpk; needs RUST_MIN_STACK=33554432"]
    fn infer_denoise_drunet_pads_and_crops() {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let mut registry = crate::model::Registry::new();
        registry.load_dir(&dir).unwrap();
        let mref = registry.resolve("drunet-color", &dir).unwrap();
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let (h, w): (usize, usize) = (66, 64); // not a multiple of 8
        let input = Tensor::new(vec![1, 3, h, w], vec![0.5f32; 1 * 3 * h * w]);
        let out = engine
            .infer_denoise(
                &input,
                0.05,
                &InferOptions {
                    tile_size: Some(640),
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, h, w]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    /// End-to-end DnCNN (color blind) check against the spandrel reference on
    /// the deterministic 64x64 input (`/tmp/dncnn_in.f32` → `dncnn_ref.f32`).
    /// The blind model takes 3ch (no sigma map); f16 vs f32 → loose tolerance.
    #[test]
    #[ignore = "requires Vulkan + a converted dncnn_color_blind.f16.bpk (senmei-ml-convert)"]
    fn dncnn_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/dncnn.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "dncnn-color".into(),
            arch: "dncnn".into(),
            scale: 1,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/dncnn_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input = Tensor::new(vec![1, 3, 64, 64], input);
        let out = engine
            .infer_denoise(
                &input,
                0.1,
                &InferOptions {
                    tile_size: None,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);

        let ref_data = std::fs::read("/tmp/dncnn_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        for (a, b) in out.data.iter().zip(&reference) {
            assert!(a.is_finite(), "output contains non-finite value");
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!("dncnn mae={mae:.5} max_abs={max_abs:.5}");
        assert!(max_abs < 0.05, "max abs diff {max_abs} exceeds 0.05");
    }

    /// End-to-end FFDNet (color) check against the KAIR reconstruction on the
    /// deterministic 64x64 input (`/tmp/ffdnet_in.f32` → `ffdnet_ref.f32`, both
    /// with σ=0.1). The burn port must match the torch even-pad + pixel-
    /// unshuffle + noise-map pipeline; f16 vs f32 → loose tolerance.
    #[test]
    #[ignore = "requires Vulkan + a converted ffdnet_color.f16.bpk (senmei-ml-convert)"]
    fn ffdnet_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/ffdnet.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "ffdnet-color".into(),
            arch: "ffdnet".into(),
            scale: 1,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/ffdnet_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input = Tensor::new(vec![1, 3, 64, 64], input);
        let out = engine
            .infer_denoise(
                &input,
                0.1,
                &InferOptions {
                    tile_size: None,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);

        let ref_data = std::fs::read("/tmp/ffdnet_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut max_abs = 0.0f64;
        for (a, b) in out.data.iter().zip(&reference) {
            assert!(a.is_finite(), "output contains non-finite value");
            max_abs = max_abs.max((*a - *b).abs() as f64);
        }
        assert!(max_abs < 0.05, "max abs diff {max_abs} exceeds 0.05");
    }

    /// End-to-end SCUNet (color, σ=15, config [4,4,4,4,4,4,4]) check against
    /// the torch reconstruction on the deterministic 64x64 input
    /// (`/tmp/scunet_in.f32` → `scunet_ref.f32`). Exercises the Swin W/SW-MSA
    /// window partition, relative-position bias and the shift mask; f16 vs f32
    /// → loose tolerance.
    #[test]
    #[ignore = "requires Vulkan + a converted scunet_color_15.f16.bpk (senmei-ml-convert)"]
    fn scunet_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/scunet.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "scunet-color".into(),
            arch: "scunet".into(),
            scale: 1,
            num_block: 4,
            feature_channels: 64,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/scunet_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input = Tensor::new(vec![1, 3, 64, 64], input);
        let out = engine
            .infer_denoise(
                &input,
                0.1,
                &InferOptions {
                    tile_size: None,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);

        let ref_data = std::fs::read("/tmp/scunet_ref.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        for (a, b) in out.data.iter().zip(&reference) {
            assert!(a.is_finite(), "output contains non-finite value");
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!("scunet mae={mae:.5} max_abs={max_abs:.5}");
        // 28 Swin blocks in f16 → edge-pixel noise is higher than the conv-only
        // denoisers; 0.06 still separates a correct port (mae ≈ 0.002, isolated
        // boundary pixels) from a logic bug (mae > 0.005, widespread).
        assert!(max_abs < 0.06, "max abs diff {max_abs} exceeds 0.06");
    }

    /// SCUNet on a non-multiple input (66×50 → internal replication pad to
    /// 128×64, crop back) — regression for the pad off-by-one that produced a
    /// 385px row instead of 384 on 640×360 (only hit in real renders, the 64×64
    /// reference never padded). `/tmp/scunet_in_66x50.f32` → ref.
    #[test]
    #[ignore = "requires Vulkan + a converted scunet_color_15.f16.bpk (senmei-ml-convert)"]
    fn scunet_matches_torch_reference_nonaligned() {
        let bpk = std::path::Path::new("/tmp/scunet.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "scunet-color".into(),
            arch: "scunet".into(),
            scale: 1,
            num_block: 4,
            feature_channels: 64,
            no_norm: false,
            layer_norm: false,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/scunet_in_66x50.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input = Tensor::new(vec![1, 3, 66, 50], input);
        let out = engine
            .infer_denoise(
                &input,
                0.1,
                &InferOptions {
                    tile_size: None,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 66, 50]);

        let ref_data = std::fs::read("/tmp/scunet_ref_66x50.f32").unwrap();
        let reference: Vec<f32> = ref_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(reference.len(), out.data.len());

        let mut mae = 0.0f64;
        let mut max_abs = 0.0f64;
        for (a, b) in out.data.iter().zip(&reference) {
            assert!(a.is_finite(), "output contains non-finite value");
            let d = (*a - *b).abs() as f64;
            mae += d;
            max_abs = max_abs.max(d);
        }
        mae /= out.data.len() as f64;
        eprintln!("scunet 66x50 mae={mae:.5} max_abs={max_abs:.5}");
        assert!(max_abs < 0.06, "max abs diff {max_abs} exceeds 0.06");
    }

    /// NAFNet via the generic `infer` path (what the Deblur step uses): scale-1
    /// 3ch→3ch, pads internally to multiples of 16. Height not a multiple of 16
    /// exercises the internal pad/crop.
    #[test]
    #[ignore = "requires Vulkan + nafnet bpk; needs RUST_MIN_STACK=33554432"]
    fn infer_nafnet_deblurs_via_generic_infer() {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let mut registry = crate::model::Registry::new();
        registry.load_dir(&dir).unwrap();
        let mref = registry.resolve("nafnet-gopro-width32", &dir).unwrap();
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let (h, w): (usize, usize) = (66, 64); // not a multiple of 16
                                               // Smooth gradient (fp16-safe; constant/flat inputs overflow the model's
                                               // deepest activations — see docs/upstream-issues.md §6).
        let data: Vec<f32> = (0..3 * h * w)
            .map(|i| ((i % w) as f32 / (w - 1) as f32) * 0.5 + 0.25)
            .collect();
        let input = Tensor::new(vec![1, 3, h, w], data);
        let out = engine
            .infer(
                &input,
                &InferOptions {
                    tile_size: Some(640),
                },
            )
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, h, w]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }
}
