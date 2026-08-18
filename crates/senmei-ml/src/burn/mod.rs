//! burn (Vulkan) inference engine.
//!
//! Runs clean re-implementations of the adopted SR archs on the `Vulkan<f16>`
//! backend. Weights are loaded from a pre-converted f16 burnpack (`.bpk`) —
//! `PytorchStore` cannot cast f32→f16 at load, so the app consumes the
//! converted format (see `rust-sr-bench`'s `convert-f16` for the one-time
//! conversion). The arch is chosen from `ModelRef::arch`.

mod realesrgan;
mod rife;
mod upcunet;
mod warp;

use crate::engine::{EngineCaps, InferOptions, InferenceEngine};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::{Tensor as BurnTensor, TensorData, f16};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, PytorchStore};
use burn_wgpu::{Vulkan, WgpuDevice};
use std::path::Path;

use realesrgan::RrdbNet;
use rife::RifeNet;
use upcunet::{UpCunet2x, UpCunet2xFast};

pub struct BurnEngine {
    model: Option<Model>,
    device: WgpuDevice,
    scale: u32,
}

enum Model {
    UpCunet2x(UpCunet2x<Vulkan<f16>>),
    UpCunet2xFast(UpCunet2xFast<Vulkan<f16>>),
    RrdbNet(RrdbNet<Vulkan<f16>>),
    RifeNet(RifeNet<Vulkan<f16>>),
}

impl Model {
    fn forward(&self, x: BurnTensor<Vulkan<f16>, 4>) -> BurnTensor<Vulkan<f16>, 4> {
        match self {
            Model::UpCunet2x(m) => m.forward(x),
            Model::UpCunet2xFast(m) => m.forward(x),
            Model::RrdbNet(m) => m.forward(x),
            Model::RifeNet(_) => panic!("RifeNet has no single-input forward"),
        }
    }

    fn interp(
        &self,
        a: BurnTensor<Vulkan<f16>, 4>,
        b: BurnTensor<Vulkan<f16>, 4>,
        t: BurnTensor<Vulkan<f16>, 4>,
    ) -> BurnTensor<Vulkan<f16>, 4> {
        match self {
            Model::RifeNet(m) => m.forward(a, b, t),
            _ => panic!("model has no frame interpolation"),
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
        let a_t = BurnTensor::<Vulkan<f16>, 4>::from_data(
            TensorData::new(a.data.clone(), [n, c, h, w]).convert::<f16>(),
            &self.device,
        );
        let b_t = BurnTensor::<Vulkan<f16>, 4>::from_data(
            TensorData::new(b.data.clone(), [n, c, h, w]).convert::<f16>(),
            &self.device,
        );
        // RIFE's internal flow estimation runs at 1/32 scale, so the reference
        // (rife-ncnn-vulkan) pads the input to multiples of 32. Do the same and
        // crop the output back to the original dims.
        let pad_h = (h + 31) / 32 * 32;
        let pad_w = (w + 31) / 32 * 32;
        let pad = |x: BurnTensor<Vulkan<f16>, 4>| {
            let mut x = x;
            if pad_h > h {
                let z = BurnTensor::<Vulkan<f16>, 4>::zeros([n, c, pad_h - h, w], &self.device);
                x = BurnTensor::cat(vec![x, z], 2);
            }
            if pad_w > w {
                let z = BurnTensor::<Vulkan<f16>, 4>::zeros([n, c, pad_h, pad_w - w], &self.device);
                x = BurnTensor::cat(vec![x, z], 3);
            }
            x
        };
        let a_t = pad(a_t);
        let b_t = pad(b_t);
        // ncnn broadcasts the scalar timestep over the (padded) spatial grid.
        let t_t = BurnTensor::<Vulkan<f16>, 4>::ones([n, 1, pad_h, pad_w], &self.device) * t;
        let out = model.interp(a_t, b_t, t_t);
        let out = out.slice([0..n, 0..c, 0..h, 0..w]);
        let data = match out.into_data().convert::<f32>().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        Some(Ok(Tensor::new(vec![n, c, h, w], data)))
    }

    /// Fused output path: keeps everything on the GPU — transposes NCHW→NHWC,
    /// scales to 0..255 and casts to U8 on-device, then downloads the packed
    /// RGB bytes once (24.8 MB instead of a ~100 MB f32 round-trip).
    /// Only used when the requested scale matches the model.
    fn infer_rgb8(&mut self, input: &Tensor, scale: u32) -> Option<Result<(Vec<u8>, u32, u32)>> {
        if self.scale != scale {
            return None;
        }
        let model = self.model.as_ref()?;
        if input.shape.len() != 4 {
            return Some(Err(Error::new("expected NCHW input")));
        }
        let n = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];

        let data = TensorData::new(input.data.clone(), [n, c, h, w]).convert::<f16>();
        let x = BurnTensor::<Vulkan<f16>, 4>::from_data(data, &self.device);
        let out = model.forward(x);
        let [_, _, oh, ow] = out.dims();

        // NCHW -> NHWC, clamp to 0..1, then round to 0..255 and cast to U8 on
        // the GPU. Without the clamp, out-of-range values (>1.0 at hard edges,
        // e.g. burnt-in subtitles) wrap on the U8 cast -> neon color artifacts.
        let nhwc = out.permute([0, 2, 3, 1]);
        let rgb_f = (nhwc.clamp(0.0, 1.0) * 255.0) + 0.5;
        let rgb_u8 = rgb_f.cast(burn::tensor::IntDType::U8);
        let bytes: Vec<u8> = match rgb_u8.into_data().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        Some(Ok((bytes, ow as u32, oh as u32)))
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
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);
        assert!(out.data.iter().all(|v| v.is_finite()));
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
        assert!(diff > 0.002, "output matches the linear blend, engine not used: {diff}");

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
        assert!(mean_abs_diff(&lo.data, &a.data) < mean_abs_diff(&lo.data, &b.data), "t=0.05 drifted to b");
        assert!(mean_abs_diff(&hi.data, &b.data) < mean_abs_diff(&hi.data, &a.data), "t=0.95 drifted to a");
    }

    fn mean_abs_diff(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.len() as f32
    }

    #[test]
    fn tensor_data_f16_to_vec_is_available() {
        // Exercises the CPU-side API we need for a fused f16->RGB8 output path
        // (no GPU involved): f16 TensorData must hand back raw f16 values.
        let data = burn::tensor::TensorData::new(
            vec![0.5f32, 1.0, 0.0, 0.25],
            [4],
        )
        .convert::<f16>();
        let v: Vec<f16> = data.to_vec().unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v[0].to_f32(), 0.5);
        assert_eq!(v[2].to_f32(), 0.0);
    }
}
