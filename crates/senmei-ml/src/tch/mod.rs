//! Optional libtorch backend (`burn-tch`) for high-performance local runs.
//!
//! Runs the shared `crate::arch` re-implementations on `LibTorch<f32>`. The
//! libtorch runtime is resolved on demand (CUDA/ROCm only — see
//! `crate::runtime`) and dlopen'd via `torch_sys::loader`; no CPU libtorch,
//! CPU stays on the burn-Vulkan engine.

use crate::arch::RifeNet;
use crate::engine::{core, EngineCaps, InferOptions, InferenceEngine, Model};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn_store::{BurnpackStore, HalfPrecisionAdapter};
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
/// lifetime (dropping a `Library` unloads it). Unix-only: the dlopen
/// RTLD_GLOBAL preload is a Unix mechanism (Windows loads libtorch via
/// LoadLibrary in `torch_sys::loader`).
#[cfg(unix)]
static PRELOADED: std::sync::Mutex<Vec<libloading::os::unix::Library>> =
    std::sync::Mutex::new(Vec::new());

/// Preload every shared lib in `dir` with RTLD_GLOBAL so the ROCm runtime is in
/// the global scope when `torch_sys::loader::init` dlopens `libtorch_hip`.
/// Failures (optional deps, non-loadable files) are skipped.
#[cfg(unix)]
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
                #[cfg(unix)]
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

pub struct TchEngine {
    model: Option<Model<B>>,
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
    fn load_rife(&self, path: &Path) -> Result<Model<B>> {
        let bytes = std::fs::read(path).map_err(|e| Error::new(e.to_string()))?;
        let mut m = RifeNet::new(&self.device);
        m.load_from_ncnn(&bytes, &self.device).map_err(Error::new)?;
        Ok(Model::RifeNet(m))
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
                core::load_arch(model, &mut store, &self.device)?
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
        core::infer(model, input, &self.device)
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
    /// downsamples 3× stride-2), runs the model, and crops back. FFDNet gets σ
    /// directly, DnCNN/SCUNet are blind. Other models return `None`.
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
    use crate::arch::UpCunet2xFast;
    use burn_store::ModuleSnapshot;

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
