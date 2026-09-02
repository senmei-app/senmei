//! burn inference engine (Vulkan, Metal on macOS).
//!
//! Thin wrapper over the backend-generic `engine::core`: holds the loaded
//! `Model<BurnBackend<f16>>` and delegates load/infer there. Weights are f16
//! burnpacks (`.bpk`); RIFE loads the raw ncnn `flownet.bin`.

use crate::arch::RifeNet;
use crate::engine::{core, EngineCaps, InferOptions, InferenceEngine, Model, Rgb8Batch};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::BurnBackend;
use crate::{gpu_index, Error, Result};
use burn::tensor::f16;
use burn_store::BurnpackStore;
use burn_wgpu::WgpuDevice;
use std::path::Path;

pub struct BurnEngine {
    model: Option<Model<BurnBackend<f16>>>,
    device: WgpuDevice,
    scale: u32,
}

impl Default for BurnEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BurnEngine {
    pub fn new() -> Self {
        Self {
            model: None,
            // Multi-GPU: the configured discrete-GPU index (0 = first).
            device: WgpuDevice::DiscreteGpu(gpu_index() as usize),
            scale: 1,
        }
    }

    /// RIFE loads from the raw ncnn `flownet.bin` (fp16 weights), not a burnpack.
    fn load_rife(&self, path: &Path) -> Result<Model<BurnBackend<f16>>> {
        let bytes = std::fs::read(path).map_err(|e| Error::new(e.to_string()))?;
        let mut m = RifeNet::new(&self.device);
        m.load_from_ncnn(&bytes, &self.device).map_err(Error::new)?;
        Ok(Model::RifeNet(Box::new(m)))
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
                core::load_arch(model, &mut store, &self.device)?
            }
        });
        self.scale = model.scale;
        // Pre-tune the autotune cache at the real tile shapes so the first
        // render frame doesn't stutter. Best-effort: models without a
        // 3-channel single-input forward (DRUNet, RIFE, …) are skipped.
        self.warmup();
        Ok(())
    }

    fn warmup(&mut self) {
        let Some(model) = self.model.as_ref() else {
            return;
        };
        if !core::single_input_rgb(model) {
            return;
        }
        let tile = crate::current_tile_size();
        let x = burn::tensor::Tensor::<BurnBackend<f16>, 4>::from_data(
            burn::tensor::TensorData::new(vec![0.0f32; 3 * tile * tile], [1, 3, tile, tile])
                .convert::<f16>(),
            &self.device,
        );
        // into_data forces the kernels (and their autotune) to actually run.
        let _ = model.forward(x).map(|o| o.into_data());
    }

    fn native_scale(&self) -> u32 {
        self.scale
    }

    fn infer(&mut self, input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::new("model not loaded"))?;
        core::infer(model, input, &self.device)
    }

    /// Fused RGB8 (GPU re-sample when the requested scale ≠ model scale — the
    /// tiling/overlap/readback lives in `engine::core::infer_rgb8`).
    fn infer_rgb8(&mut self, input: &Tensor, scale: u32) -> Option<Result<(Vec<u8>, u32, u32)>> {
        let model = self.model.as_ref()?;
        core::infer_rgb8(model, input, self.scale, scale, &self.device)
    }

    /// Fused multi-frame RGB8 with a deferred readback, so the caller can
    /// queue the next forward before blocking on this batch's transfer.
    fn infer_rgb8_submit(
        &mut self,
        inputs: &[Tensor],
        scale: u32,
    ) -> Option<Result<Box<dyn Rgb8Batch>>> {
        let model = self.model.as_ref()?;
        core::infer_rgb8_batch_prepare(model, inputs, self.scale, scale, &self.device)
            .map(|r| r.map(|b| Box::new(b) as Box<dyn Rgb8Batch>))
    }

    /// Fused multi-frame RGB8 (synchronous — resolves the submit immediately).
    fn infer_rgb8_batch(
        &mut self,
        inputs: &[Tensor],
        scale: u32,
    ) -> Option<Result<Vec<(Vec<u8>, u32, u32)>>> {
        self.infer_rgb8_submit(inputs, scale)
            .map(|r| r.and_then(|b| b.resolve()))
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
        core::infer_interp(model, a, b, t, &self.device)
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
        core::infer_denoise(model, input, sigma, &self.device)
    }
}

#[cfg(test)]
mod tests;
