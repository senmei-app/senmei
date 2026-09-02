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

    /// GPU-only init check before the heavy model loads: run one tiny compute
    /// on the configured device. A missing/broken Vulkan driver used to
    /// surface as a wgpu panic mid-load; this returns a clear error instead.
    /// No CPU/software fallback — if this fails the backend simply isn't there.
    pub fn new_checked() -> Result<Self> {
        let device = WgpuDevice::DiscreteGpu(gpu_index() as usize);
        // A failing probe would still print its panic via the default hook;
        // silence it so only the mapped error below reaches the user.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let x = burn::tensor::Tensor::<BurnBackend<f16>, 2>::from_data(
                burn::tensor::TensorData::new(vec![1.0f32, 2.0f32], [1, 2]).convert::<f16>(),
                &device,
            );
            let _ = (x.clone() + x).into_data();
        }));
        std::panic::set_hook(prev_hook);
        probe.map_err(|p| {
            let text = if let Some(s) = p.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown error".to_string()
            };
            Error::new(format!(
                "Vulkan unavailable at GPU index {} ({text}); install/update the Vulkan \
                 driver or choose another GPU index — no CPU fallback",
                gpu_index()
            ))
        })?;
        Ok(Self::new())
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
mod tests {
    use super::*;
    use burn::tensor::f16;

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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: true,
            dysample: true,
            shuffle: 1,
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
        // max_abs is dominated by the DySample extremes (the f32 reference
        // itself reaches ±0.27 on this input), so gate on mae; the DySample
        // grid-sample in f16 lands around mae 0.018 (spandrel f32 = 0.00015).
        assert_eq!(nan_count, 0, "output contains non-finite values");
        assert!(mae < 0.05, "mae {mae} exceeds 0.05");
    }

    /// End-to-end RealPLKSR 4× (4xNomosWebPhoto, GroupNorm + pixel-shuffle
    /// tail, no DySample) check against the ONNX reference on the
    /// deterministic 256x256 input (`/tmp/nomoswebphoto_in.f32` →
    /// `/tmp/nomoswebphoto_ref.f32`, onnxruntime CPU on the official fp32
    /// ONNX). Exercises the `dysample=false` pixel-shuffle path; f16 vs f32 →
    /// loose tolerance.
    #[test]
    #[ignore = "requires Vulkan + a converted nomoswebphoto.f16.bpk (senmei-ml-convert)"]
    fn real_plksr_4x_nomoswebphoto_matches_torch_reference() {
        let bpk = std::path::Path::new("/tmp/nomoswebphoto.f16.bpk");
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "4x-nomoswebphoto-realplksr".into(),
            arch: "real-plksr".into(),
            scale: 4,
            num_block: 4,
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: false,
            shuffle: 1,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let in_data = std::fs::read("/tmp/nomoswebphoto_in.f32").unwrap();
        let input: Vec<f32> = in_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(input.len(), 3 * 256 * 256);
        let input = Tensor::new(vec![1, 3, 256, 256], input);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 1024, 1024]);

        let ref_data = std::fs::read("/tmp/nomoswebphoto_ref.f32").unwrap();
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
            "real_plksr_4x_nomoswebphoto mae={mae:.5} max_abs={max_abs:.5} nan={nan_count}/{}",
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

        // Scale mismatch: the x2 model at requested x4 must stay on the fused
        // path (GPU bilinear re-sample of each tile) and produce 4x output.
        let (h, w): (usize, usize) = (540, 960); // 2x1 tiles @640 — tiled path
        let input = Tensor::new(vec![1, 3, h, w], vec![0.5f32; 1 * 3 * h * w]);
        for _ in 0..4 {
            let (bytes, rw, rh) = engine.infer_rgb8(&input, 4).unwrap().unwrap();
            assert_eq!(rw, (w * 4) as u32);
            assert_eq!(rh, (h * 4) as u32);
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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            .infer_denoise(&input, 0.1, &InferOptions { tile_size: None })
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
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            .infer_denoise(&input, 0.1, &InferOptions { tile_size: None })
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
            num_conv: 16,
            feature_channels: 64,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            .infer_denoise(&input, 0.1, &InferOptions { tile_size: None })
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
            num_conv: 16,
            feature_channels: 64,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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
            .infer_denoise(&input, 0.1, &InferOptions { tile_size: None })
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

    /// Regression: `infer_denoise_tiled` runs U-Net denoisers full-frame — a
    /// >1080p (but <4K) input must produce the same result as the plain
    /// full-frame `infer_denoise`. Tiling SCUNet/DRUNet is invalid (window
    /// attention + ÷8 pyramid → mae ≈ 0.13 vs full-frame, ghost copies on
    /// moving content); a shared 4K cap keeps this from regressing.
    #[test]
    #[ignore = "diagnostic: needs Vulkan + scunet bpk (senmei-ml-convert)"]
    fn scunet_tiled_ghosts_at_tile_seams() {
        let bpk = std::path::Path::new(
            "/home/mzach/.local/share/senmei/models/scunet_color_15.pth.f16.bpk",
        );
        if !bpk.exists() {
            eprintln!("missing bpk, skipping");
            return;
        }
        let mref = ModelRef {
            id: "scunet-color".into(),
            arch: "scunet".into(),
            scale: 1,
            num_block: 4,
            num_conv: 16,
            feature_channels: 64,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
            path: bpk.to_path_buf(),
        };
        let mut engine = BurnEngine::new();
        engine.load(&mref).unwrap();

        let (h, w): (usize, usize) = (1500, 1400); // 2.1 MP: above 1080p, below 4K
        let mut data = Vec::with_capacity(3 * h * w);
        for y in 0..h {
            for x in 0..w {
                let bar = if (x / 180) % 2 == 0 { 0.85f32 } else { 0.25 };
                let g = (y as f32 / h as f32) * 0.3 + 0.15;
                let n = (((x * 31 + y * 17) % 97) as f32 / 97.0) * 0.06;
                for c in 0..3 {
                    data.push((bar * (1.0 - 0.12 * c as f32) + g + n).clamp(0.0, 1.0));
                }
            }
        }
        let input = Tensor::new(vec![1, 3, h, w], data);
        let full = engine
            .infer_denoise(&input, 0.1, &InferOptions { tile_size: None })
            .unwrap()
            .unwrap();
        let tiled = crate::infer_denoise_tiled(
            &mut engine,
            &input,
            0.1,
            &InferOptions {
                tile_size: Some(512),
            },
        )
        .unwrap();
        assert_eq!(full.shape, tiled.shape);

        let mut mae = 0f64;
        for (a, b) in full.data.iter().zip(&tiled.data) {
            assert!(a.is_finite(), "output contains non-finite value");
            mae += (*a - *b).abs() as f64;
        }
        mae /= full.data.len() as f64;
        eprintln!("scunet denoise full-frame vs infer_denoise_tiled: mae={mae:.5}");
        // Both go through the same full-frame path now; any tiling regression
        // shows up as a large diff (the broken tiled path measured mae ≈ 0.13).
        assert!(
            mae < 0.001,
            "denoise tiled path diverged from full-frame (mae {mae:.5})"
        );
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
