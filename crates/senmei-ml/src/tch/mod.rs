//! Optional libtorch backend (`burn-tch`) for high-performance local runs.
//!
//! Runs the shared `crate::arch` re-implementations on `LibTorch<f32>`. The
//! libtorch runtime is resolved on demand (CUDA/ROCm only — see
//! `crate::runtime`) and dlopen'd via `torch_sys::loader`; no CPU libtorch,
//! CPU stays on the burn-Vulkan engine.

use crate::arch::{
    Dncnn, Drunet, Ffdnet, IfrNet, NafNet, RealPlk, RrdbNet, RifeNet, Scunet, Span, SrvggNet,
    UpCunet2x, UpCunet2xFast,
};
use crate::engine::{EngineCaps, InferOptions, InferenceEngine};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::{Tensor as BurnTensor, TensorData};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use burn_tch::{LibTorch, LibTorchDevice};
use std::path::Path;
use std::sync::OnceLock;

type B = LibTorch<f32>;

/// libtorch release with ROCm-7 builds that the runtime downloads. Must stay
/// in sync with `crate::runtime::torch` (and the torch-sys headers used to
/// build the wrapper).
pub const LIBTORCH_VERSION: &str = "2.11.0";

/// Device for the libtorch backend. CPU is intentionally absent — the
/// burn-Vulkan engine owns the CPU path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TchDevice {
    /// CUDA device index; maps to ROCm on an AMD libtorch build.
    Cuda(usize),
    /// Apple silicon Metal Performance Shaders.
    Mps,
}

impl From<TchDevice> for LibTorchDevice {
    fn from(d: TchDevice) -> Self {
        match d {
            TchDevice::Cuda(i) => LibTorchDevice::Cuda(i),
            TchDevice::Mps => LibTorchDevice::Mps,
        }
    }
}

/// Resolved + dlopen'd libtorch install, cached per process (idempotent).
static RUNTIME_LIBTORCH: OnceLock<
    std::result::Result<Option<crate::runtime::TorchInstall>, String>,
> = OnceLock::new();

/// Handles for the preloaded ROCm/HIP runtime libs, kept alive for the process
/// lifetime (dropping a `Library` unloads it).
static PRELOADED: std::sync::Mutex<Vec<libloading::os::unix::Library>> =
    std::sync::Mutex::new(Vec::new());

/// Preload every shared lib in `dir` with RTLD_GLOBAL so the ROCm runtime is in
/// the global scope when `torch_sys::loader::init` dlopens `libtorch_hip`.
/// Failures (optional deps, non-loadable files) are skipped.
fn preload_runtime_libs(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut libs = match PRELOADED.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_so = path.extension().and_then(|e| e.to_str()) == Some("so")
            || path.file_name().is_some_and(|n| n.to_string_lossy().ends_with(".so"));
        if !is_so || !path.is_file() {
            continue;
        }
        unsafe {
            if let Ok(lib) = libloading::os::unix::Library::open(
                Some(&path),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
            ) {
                libs.push(lib);
            }
        }
    }
}

/// Resolve (download on first use) and dlopen a CUDA/ROCm libtorch, once per
/// process. `Ok(None)` when no GPU is present — the caller (burn) owns CPU.
fn ensure_loaded(data_dir: &Path) -> Result<()> {
    let install = RUNTIME_LIBTORCH.get_or_init(|| {
        let hw = crate::runtime::detect();
        let resolved = crate::runtime::resolve(data_dir, &hw);
        if let Ok(Some(inst)) = &resolved {
            // Preload the system ROCm runtime for any ROCm build so the
            // `libamdhip64.so.7` SONAME resolves to the installed version. A
            // downloaded libtorch may bundle its own HIP runtime, but mixing a
            // bundled older HIP with the system HSA runtime crashes on load —
            // the preloaded system lib shadows the bundled one (RTLD_GLOBAL).
            if matches!(
                inst.variant,
                crate::runtime::TorchVariant::Rocm(_)
            ) {
                if let Some(dir) = hw.rocm_runtime_dir.as_deref() {
                    preload_runtime_libs(dir);
                }
            }
            if let Err(e) = torch_sys::loader::init(&inst.lib_dir) {
                return Err(e);
            }
        }
        resolved
    });
    match install {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(Error::new("no CUDA/ROCm device — CPU stays on burn-Vulkan")),
        Err(e) => Err(Error::new(e.clone())),
    }
}

enum Model {
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
}

impl Model {
    fn forward(&self, x: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
        match self {
            Model::UpCunet2x(m) => m.forward(x),
            Model::UpCunet2xFast(m) => m.forward(x),
            Model::RrdbNet(m) => m.forward(x),
            Model::SrvggNet(m) => m.forward(x),
            Model::Drunet(m) => m.forward(x),
            Model::Dncnn(m) => m.forward(x),
            Model::Scunet(m) => m.forward(x),
            Model::NafNet(m) => m.forward(x),
            Model::RealPlk(m) => m.forward(x),
            Model::Span(m) => m.forward(x),
            Model::RifeNet(_) | Model::IfrNet(_) | Model::Ffdnet(_) => {
                panic!("model has no single-input forward")
            }
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
            Model::IfrNet(m) => m.forward(a, b, t),
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

    /// Resolve + dlopen a CUDA/ROCm libtorch at runtime and build the engine on
    /// it. Errors when no GPU is present (CPU stays on burn-Vulkan).
    pub fn runtime(data_dir: &Path) -> Result<Self> {
        ensure_loaded(data_dir)?;
        Ok(Self::new(TchDevice::Cuda(0)))
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
            "srvgg" => {
                let mut m = SrvggNet::new(64, 16, model.scale as usize, &self.device);
                m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
                Ok(Model::SrvggNet(m))
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
                let mut m = RealPlk::new(
                    model.scale as usize,
                    model.layer_norm,
                    model.dysample,
                    &self.device,
                );
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
                m.pad_k96(&self.device);
                Ok(Model::Span(m))
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
        if !matches!(model, Model::RifeNet(_) | Model::IfrNet(_)) {
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
        // The flow estimators run on a downscaled grid (RIFE 1/32, IFRNet 1/16
        // via its pyramid), so pad to a multiple and crop back (like the refs).
        let pad = if matches!(model, Model::RifeNet(_)) { 32 } else { 16 };
        let pad_h = (h + pad - 1) / pad * pad;
        let pad_w = (w + pad - 1) / pad * pad;
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

    /// DRUNet denoise: appends a constant noise-level map (sigma in [0,1]) to
    /// the 3-channel input, pads the spatial dims to multiples of 8 (the UNet
    /// downsamples 3× stride-2), runs the model, and crops back. FFDNet gets σ
    /// directly, DnCNN/SCUNet are blind. Other models return `None`.
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
        let rgb = BurnTensor::<B, 4>::from_data(
            TensorData::new(input.data.clone(), [n, c, h, w]),
            device,
        );
        // FFDNet takes the noise level internally; DnCNN/SCUNet are blind
        // (3ch in, no sigma map); DRUNet gets a constant sigma map + 8-aligned
        // spatial dims (3× stride-2 downsample) — pad and crop.
        if let Model::Ffdnet(m) = model {
            let out = m.forward(rgb, sigma);
            let data = match out.into_data().to_vec() {
                Ok(v) => v,
                Err(e) => return Some(Err(Error::new(e.to_string()))),
            };
            return Some(Ok(Tensor::new(vec![n, 3, h, w], data)));
        }
        if !is_drunet {
            let out = model.forward(rgb);
            let data = match out.into_data().to_vec() {
                Ok(v) => v,
                Err(e) => return Some(Err(Error::new(e.to_string()))),
            };
            return Some(Ok(Tensor::new(vec![n, 3, h, w], data)));
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
        let out = model.forward(x).slice([0..n, 0..3, 0..h, 0..w]);
        let data = match out.into_data().to_vec() {
            Ok(v) => v,
            Err(e) => return Some(Err(Error::new(e.to_string()))),
        };
        Some(Ok(Tensor::new(vec![n, 3, h, w], data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Roundtrip on the runtime-resolved GPU (CUDA/ROCm): a random-init
    /// UpCunet2xFast is saved as an f16 burnpack on the device, loaded through
    /// `TchEngine` (f16→f32 adapter) and inferred — proves the burn-store →
    /// LibTorch plumbing over the dlopen path. `#[ignore]` because resolving
    /// libtorch downloads it (~2 GB) on first use; skips without a GPU.
    #[test]
    #[ignore]
    fn tch_engine_roundtrips_bpk_on_gpu() {
        if !crate::runtime::detect().supports_gpu() {
            eprintln!("no CUDA/ROCm device, skipping");
            return;
        }
        let data_dir = std::env::temp_dir().join("senmei_tch_test_data");
        // Init the runtime loader before any torch_sys/tch call.
        eprintln!("[tch_test] runtime...");
        let mut engine = TchEngine::runtime(&data_dir).unwrap();
        eprintln!("[tch_test] runtime ok, model init...");
        let tmp = std::env::temp_dir().join("senmei_tch_gpu_concept.bpk");
        let _ = std::fs::remove_file(&tmp);

        let device = LibTorchDevice::Cuda(0);
        let m = UpCunet2xFast::<B>::new(&device);
        let mut save =
            BurnpackStore::from_file(&tmp).with_to_adapter(HalfPrecisionAdapter::new());
        eprintln!("[tch_test] save_into...");
        m.save_into(&mut save).unwrap();
        eprintln!("[tch_test] saved, load...");

        let mref = ModelRef {
            id: "concept".into(),
            arch: "fallin-cugan".into(),
            scale: 2,
            num_block: 4,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            path: tmp.clone(),
        };
        engine.load(&mref).unwrap();
        eprintln!("[tch_test] loaded, infer...");
        let input = Tensor::new(vec![1, 3, 128, 128], vec![0.5f32; 3 * 128 * 128]);
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        eprintln!("[tch_test] inferred {:?}", out.shape);
        assert_eq!(out.shape, vec![1, 3, 256, 256]);
        assert!(out.data.iter().all(|v| v.is_finite()));

        let _ = std::fs::remove_file(&tmp);
    }
}
