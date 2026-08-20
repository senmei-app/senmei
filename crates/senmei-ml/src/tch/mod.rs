//! Optional libtorch backend (`burn-tch`) for high-performance local runs.
//!
//! Runs the shared `crate::arch` re-implementations on `LibTorch<f32>`. The
//! device is picked from a small portable enum so the same code path covers
//! CPU, CUDA (→ ROCm libtorch on AMD) and MPS (Apple silicon). On a stock
//! `download-libtorch` build only `Cpu` is available; CUDA/ROCm needs a
//! matching system libtorch (e.g. a ROCm build for the Radeon path).

use crate::arch::{RrdbNet, RifeNet, UpCunet2x, UpCunet2xFast};
use crate::engine::{EngineCaps, InferOptions, InferenceEngine};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::{Tensor as BurnTensor, TensorData};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use burn_tch::{LibTorch, LibTorchDevice};
use std::path::Path;

type B = LibTorch<f32>;

/// Portable device selection for the libtorch backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TchDevice {
    /// Best available device, resolved at construction.
    Auto,
    Cpu,
    /// CUDA device index; maps to ROCm on an AMD libtorch build.
    Cuda(usize),
    /// Apple silicon Metal Performance Shaders.
    Mps,
}

impl TchDevice {
    /// Pick the best available device: CUDA/ROCm if present, else MPS on Apple
    /// silicon, else CPU. The MPS branch is a compile-time heuristic — a real
    /// libtorch MPS check would need a runtime probe on `tch::Device`.
    pub fn automatic() -> Self {
        if tch::Cuda::is_available() {
            TchDevice::Cuda(0)
        } else if cfg!(target_os = "macos") {
            TchDevice::Mps
        } else {
            TchDevice::Cpu
        }
    }
}

impl From<TchDevice> for LibTorchDevice {
    fn from(d: TchDevice) -> Self {
        match d {
            TchDevice::Auto => TchDevice::automatic().into(),
            TchDevice::Cpu => LibTorchDevice::Cpu,
            TchDevice::Cuda(i) => LibTorchDevice::Cuda(i),
            TchDevice::Mps => LibTorchDevice::Mps,
        }
    }
}

enum Model {
    UpCunet2x(UpCunet2x<B>),
    UpCunet2xFast(UpCunet2xFast<B>),
    RrdbNet(RrdbNet<B>),
    RifeNet(RifeNet<B>),
}

impl Model {
    fn forward(&self, x: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
        match self {
            Model::UpCunet2x(m) => m.forward(x),
            Model::UpCunet2xFast(m) => m.forward(x),
            Model::RrdbNet(m) => m.forward(x),
            Model::RifeNet(_) => panic!("RifeNet has no single-input forward"),
        }
    }

    fn interp(
        &self,
        a: BurnTensor<B, 4>,
        b: BurnTensor<B, 4>,
        t: BurnTensor<B, 4>,
    ) -> BurnTensor<B, 4> {
        match self {
            Model::RifeNet(m) => m.forward(a, b, t),
            _ => panic!("model has no frame interpolation"),
        }
    }
}

pub struct TchEngine {
    model: Option<Model>,
    device: LibTorchDevice,
    scale: u32,
}

impl TchEngine {
    pub fn new(device: TchDevice) -> Self {
        Self {
            model: None,
            device: device.into(),
            scale: 1,
        }
    }

    pub fn default_cpu() -> Self {
        Self::new(TchDevice::Cpu)
    }

    /// RIFE loads from the raw ncnn `flownet.bin` (fp16 weights), like the
    /// burn engine — `load_from_ncnn` is backend-generic.
    fn load_rife(&self, path: &Path) -> Result<Model> {
        let bytes = std::fs::read(path).map_err(|e| Error::new(e.to_string()))?;
        let mut m = RifeNet::new(&self.device);
        m.load_from_ncnn(&bytes, &self.device).map_err(Error::new)?;
        Ok(Model::RifeNet(m))
    }

    /// Burnpacks are saved f16; the `from` adapter casts back to the f32
    /// params of `LibTorch<f32>`.
    fn load_arch(&self, model: &ModelRef, store: &mut BurnpackStore) -> Result<Model> {
        match model.arch.as_str() {
            "upcunet2x" => {
                let mut m = UpCunet2x::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::UpCunet2x(m))
            }
            "upcunet2x-fast" | "fallin-cugan" => {
                let mut m = UpCunet2xFast::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::UpCunet2xFast(m))
            }
            "realesrgan" => {
                let mut m = RrdbNet::new(model.scale as usize, model.num_block as usize, &self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::RrdbNet(m))
            }
            other => Err(Error::new(format!("unsupported arch: {other}"))),
        }
    }
}

impl InferenceEngine for TchEngine {
    fn capabilities(&self) -> EngineCaps {
        EngineCaps { tiles: true }
    }

    fn load(&mut self, model: &ModelRef) -> Result<()> {
        self.model = Some(match model.arch.as_str() {
            "rife425" | "rife46" => self.load_rife(&model.path)?,
            _ => {
                let mut store = BurnpackStore::from_file(&model.path)
                    .with_from_adapter(HalfPrecisionAdapter::new());
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
        let [n, c, h, w] = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]];
        let x = BurnTensor::<B, 4>::from_data(
            TensorData::new(input.data.clone(), [n, c, h, w]),
            &self.device,
        );
        let out = model.forward(x);
        let [_, _, oh, ow] = out.dims();
        let data = out.into_data().to_vec().map_err(|e| Error::new(e.to_string()))?;
        Ok(Tensor::new(vec![n, c, oh, ow], data))
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
        if !matches!(model, Model::RifeNet(_)) {
            return None; // not an interpolation model → caller falls back
        }
        let [n, c, h, w] = [a.shape[0], a.shape[1], a.shape[2], a.shape[3]];
        let a_t = BurnTensor::<B, 4>::from_data(
            TensorData::new(a.data.clone(), [n, c, h, w]),
            &self.device,
        );
        let b_t = BurnTensor::<B, 4>::from_data(
            TensorData::new(b.data.clone(), [n, c, h, w]),
            &self.device,
        );
        // RIFE's internal flow estimation runs at 1/32 scale, so the reference
        // (rife-ncnn-vulkan) pads the input to multiples of 32. Do the same and
        // crop the output back to the original dims.
        let pad_h = (h + 31) / 32 * 32;
        let pad_w = (w + 31) / 32 * 32;
        let pad = |x: BurnTensor<B, 4>| {
            let mut x = x;
            if pad_h > h {
                let z = BurnTensor::<B, 4>::zeros([n, c, pad_h - h, w], &self.device);
                x = BurnTensor::cat(vec![x, z], 2);
            }
            if pad_w > w {
                let z = BurnTensor::<B, 4>::zeros([n, c, pad_h, pad_w - w], &self.device);
                x = BurnTensor::cat(vec![x, z], 3);
            }
            x
        };
        let a_t = pad(a_t);
        let b_t = pad(b_t);
        // ncnn broadcasts the scalar timestep over the (padded) spatial grid.
        let t_t = BurnTensor::<B, 4>::ones([n, 1, pad_h, pad_w], &self.device) * t;
        let out = model.interp(a_t, b_t, t_t);
        let out = out.slice([0..n, 0..c, 0..h, 0..w]);
        let data = match out.into_data().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        Some(Ok(Tensor::new(vec![n, c, h, w], data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concept validation: a random-init UpCunet2xFast is saved as an f16
    /// burnpack on CPU, loaded through `TchEngine` (with the f16→f32 adapter)
    /// and inferred. Proves the burn-store → LibTorch plumbing end to end
    /// without needing committed weights. Requires the libtorch `download`
    /// feature (CPU build).
    #[test]
    fn tch_engine_roundtrips_bpk_on_cpu() {
        let device = LibTorchDevice::Cpu;
        let tmp = std::env::temp_dir().join("senmei_tch_concept.bpk");
        let _ = std::fs::remove_file(&tmp);

        let m = UpCunet2xFast::<B>::new(&device);
        let mut save =
            BurnpackStore::from_file(&tmp).with_to_adapter(HalfPrecisionAdapter::new());
        m.save_into(&mut save).unwrap();

        let mref = ModelRef {
            id: "concept".into(),
            arch: "fallin-cugan".into(),
            scale: 2,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            path: tmp.clone(),
        };
        let mut engine = TchEngine::new(TchDevice::Cpu);
        engine.load(&mref).unwrap();
        let input = Tensor::new(vec![1, 3, 64, 64], vec![0.5f32; 3 * 64 * 64]);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 128, 128]);
        assert!(out.data.iter().all(|v| v.is_finite()));

        let _ = std::fs::remove_file(&tmp);
    }

    /// Same roundtrip as the CPU test, but on `Cuda(0)` (→ ROCm libtorch with
    /// the AMD runtime). Skips when no CUDA/ROCm device is present; run with
    /// `LIBTORCH` pointing at a ROCm libtorch to exercise the GPU path.
    #[test]
    fn tch_engine_roundtrips_bpk_on_rocm() {
        if !tch::Cuda::is_available() {
            eprintln!("no CUDA/ROCm libtorch, skipping");
            return;
        }
        let device = LibTorchDevice::Cuda(0);
        let tmp = std::env::temp_dir().join("senmei_tch_rocm_concept.bpk");
        let _ = std::fs::remove_file(&tmp);

        let m = UpCunet2xFast::<B>::new(&device);
        let mut save =
            BurnpackStore::from_file(&tmp).with_to_adapter(HalfPrecisionAdapter::new());
        m.save_into(&mut save).unwrap();

        let mref = ModelRef {
            id: "concept".into(),
            arch: "fallin-cugan".into(),
            scale: 2,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            path: tmp.clone(),
        };
        let mut engine = TchEngine::new(TchDevice::Cuda(0));
        engine.load(&mref).unwrap();
        let input = Tensor::new(vec![1, 3, 128, 128], vec![0.5f32; 3 * 128 * 128]);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 256, 256]);
        assert!(out.data.iter().all(|v| v.is_finite()));

        let _ = std::fs::remove_file(&tmp);
    }
}
