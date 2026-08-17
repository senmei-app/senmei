//! burn (Vulkan) inference engine.
//!
//! Runs clean re-implementations of the adopted SR archs on the `Vulkan<f16>`
//! backend. Weights are loaded from a pre-converted f16 burnpack (`.bpk`) —
//! `PytorchStore` cannot cast f32→f16 at load, so the app consumes the
//! converted format (see `rust-sr-bench`'s `convert-f16` for the one-time
//! conversion). The arch is chosen from `ModelRef::arch`.

mod realesrgan;
mod upcunet;

use crate::engine::{Backend, EngineCaps, InferOptions, InferenceEngine};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::{Tensor as BurnTensor, TensorData, f16};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PytorchStore};
use burn_wgpu::{Vulkan, WgpuDevice};
use std::path::Path;

use realesrgan::RrdbNet;
use upcunet::{UpCunet2x, UpCunet2xFast};

pub struct BurnEngine {
    model: Option<Model>,
    device: WgpuDevice,
}

enum Model {
    UpCunet2x(UpCunet2x<Vulkan<f16>>),
    UpCunet2xFast(UpCunet2xFast<Vulkan<f16>>),
    RrdbNet(RrdbNet<Vulkan<f16>>),
}

impl Model {
    fn forward(&self, x: BurnTensor<Vulkan<f16>, 4>) -> BurnTensor<Vulkan<f16>, 4> {
        match self {
            Model::UpCunet2x(m) => m.forward(x),
            Model::UpCunet2xFast(m) => m.forward(x),
            Model::RrdbNet(m) => m.forward(x),
        }
    }
}

impl BurnEngine {
    pub fn new() -> Self {
        Self { model: None, device: WgpuDevice::DiscreteGpu(0) }
    }

    fn load_arch(&self, model: &ModelRef, store: &mut BurnpackStore) -> Result<Model> {
        match model.arch.as_str() {
            "upcunet2x" => {
                let mut m = UpCunet2x::new(&self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::UpCunet2x(m))
            }
            "upcunet2x-fast" => {
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

impl InferenceEngine for BurnEngine {
    fn name(&self) -> &'static str {
        "burn-vulkan"
    }

    fn capabilities(&self) -> EngineCaps {
        EngineCaps { backend: Backend::Vulkan, half: true, tiles: true }
    }

    fn load(&mut self, model: &ModelRef) -> Result<()> {
        let mut store = BurnpackStore::from_file(&model.path);
        self.model = Some(self.load_arch(model, &mut store)?);
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
        let x = BurnTensor::<Vulkan<f16>, 4>::from_data(data, &self.device);
        let out = model.forward(x);
        let [_, _, oh, ow] = out.dims();
        let data = out
            .into_data()
            .convert::<f32>()
            .to_vec()
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(Tensor::new(vec![n, c, oh, ow], data))
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
) -> Result<()> {
    let device = WgpuDevice::DiscreteGpu(0);
    let mut save =
        BurnpackStore::from_file(bpk_path).with_to_adapter(HalfPrecisionAdapter::new());
    match arch {
        "upcunet2x" | "upcunet2x-fast" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"\.conv\.0\.", ".conv.")
                .with_key_remapping(r"\.conv\.2\.", ".conv2.");
            match arch {
                "upcunet2x" => {
                    let mut m = UpCunet2x::<Vulkan>::new(&device);
                    m.load_from(&mut store).map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save).map_err(|e| Error::new(e.to_string()))?;
                }
                _ => {
                    let mut m = UpCunet2xFast::<Vulkan>::new(&device);
                    m.load_from(&mut store).map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save).map_err(|e| Error::new(e.to_string()))?;
                }
            }
        }
        "realesrgan" => {
            let mut store = PytorchStore::from_file(pth_path);
            let mut m = RrdbNet::<Vulkan>::new(scale as usize, num_block as usize, &device);
            m.load_from(&mut store).map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save).map_err(|e| Error::new(e.to_string()))?;
        }
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
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
            .infer(&input, &InferOptions { half: true, tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }
}
