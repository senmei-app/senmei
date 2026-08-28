//! Optional libtorch backend (`burn-tch`) for high-performance local runs.
//!
//! Runs the shared `crate::arch` re-implementations on `LibTorch<f16>`. The
//! libtorch runtime is resolved on demand (CUDA/ROCm only — see
//! `crate::runtime`) and dlopen'd via `torch_sys::loader`; no CPU libtorch,
//! CPU stays on the burn-Vulkan engine.

use crate::arch::RifeNet;
use crate::engine::{core, EngineCaps, InferOptions, InferenceEngine, Model, Rgb8Batch};
use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::f16;
use burn_store::BurnpackStore;
use burn_tch::{LibTorch, LibTorchDevice};
use std::path::Path;
use std::sync::OnceLock;

type B = LibTorch<f16>;

/// libtorch release with ROCm-7 builds that the runtime downloads. Must stay
/// in sync with `crate::runtime::torch` (and the torch-sys headers used to
/// build the wrapper).
pub const LIBTORCH_VERSION: &str = "2.12.0";

/// A/B switch: `SENMEI_TCH_TILED=1` skips the full-frame fused RGB8 path and
/// always uses the 640px-tiled one (re-measures the pre-full-frame behavior).
/// Honored in all builds — the pipeline benches drive it through a dependency,
/// so a `#[cfg(test)]` gate would dead-code it here; the `warn!` keeps it from
/// being silently active in production.
static TCH_TILED: OnceLock<bool> = OnceLock::new();
fn tch_tiled() -> bool {
    *TCH_TILED.get_or_init(|| {
        let on = std::env::var("SENMEI_TCH_TILED").as_deref() == Ok("1");
        if on {
            log::warn!("SENMEI_TCH_TILED=1: forcing the tiled path (benchmark A/B)");
        }
        on
    })
}

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
#[cfg(windows)]
static PRELOADED: std::sync::Mutex<Vec<libloading::Library>> = std::sync::Mutex::new(Vec::new());

/// Preload the downloaded per-GPU ROCm SDK libs (Koharu's ordered list) with
/// RTLD_LAZY|GLOBAL so the versioned SONAMEs (`libMIOpen.so.1`, …) that the
/// pytorch libtorch zip lacks resolve at dlopen time.
fn preload_sdk(root: &Path) {
    for rel in crate::runtime::rocm::preload_libs() {
        let path = root.join(rel);
        if !path.is_file() {
            continue;
        }
        unsafe {
            #[cfg(unix)]
            let opened = libloading::os::unix::Library::open(
                Some(&path),
                libloading::os::unix::RTLD_LAZY | libloading::os::unix::RTLD_GLOBAL,
            );
            #[cfg(windows)]
            let opened = libloading::Library::new(&path);
            if let Ok(lib) = opened {
                if let Ok(mut g) = PRELOADED.lock() {
                    g.push(lib);
                }
            }
        }
    }
}

/// Probe that Half + Float tensors can be created through the loaded wrapper
/// **with the right dtype**. A broken wrapper/runtime ABI (wrapper compiled
/// against other headers than the runtime) surfaces as a wrong dtype — e.g.
/// Half comes back as Int16 or `torch::zeros` throws "incoherent element sizes
/// in bytes" — so only checking `is_ok()` gives false positives. The caller
/// falls back to burn-Vulkan when this fails, instead of corrupting memory
/// mid-render.
fn probe_tensor_ok() -> bool {
    std::panic::catch_unwind(|| {
        let half = tch::Tensor::f_from_data_size(&[0u8; 4], &[2], tch::Kind::Half);
        let float = tch::Tensor::f_from_data_size(&[0u8; 4], &[2], tch::Kind::Float);
        half.is_ok_and(|t| t.kind() == tch::Kind::Half)
            && float.is_ok_and(|t| t.kind() == tch::Kind::Float)
    })
    .unwrap_or(false)
}

/// Resolve (download on first use) and dlopen a CUDA/ROCm libtorch, once per
/// process. `Ok(None)` when no GPU is present — the caller (burn) owns CPU.
/// ROCm builds preload the per-GPU ROCm SDK (Koharu-style) so the versioned
/// SONAMEs libtorch dlopens resolve; the tensor probe (dtype-correct) guards
/// against a wrapper/runtime ABI mismatch → clean fallback to burn-Vulkan.
fn ensure_loaded(data_dir: &Path) -> Result<()> {
    let install = RUNTIME_LIBTORCH.get_or_init(|| {
        let hw = crate::runtime::detect();
        let resolved = crate::runtime::resolve(data_dir, &hw);
        if let Ok(Some(inst)) = &resolved {
            if matches!(inst.variant, crate::runtime::TorchVariant::Rocm(_)) {
                // Preload the pinned per-GPU ROCm SDK (Koharu-style). Never
                // touch the system ROCm: a system HIP/HSA mixed with the SDK's
                // copies (both RTLD_GLOBAL) gives two HIP runtimes and crashes
                // at the first kernel launch (`hip::StatC::getStatFunc`). The
                // SDK's versioned SONAMEs (`libMIOpen.so.1`, …) that libtorch
                // dlopens resolve via the preload.
                if let Some(target) = hw.rocm_target.as_deref() {
                    match crate::runtime::rocm::download(data_dir, target) {
                        Ok(root) => preload_sdk(&root),
                        Err(e) => log::warn!("rocm sdk download failed for {target}: {e}"),
                    }
                }
                if let Err(e) = torch_sys::loader::init(&inst.lib_dir) {
                    log::warn!("libtorch dlopen failed: {e}");
                    return Err(e);
                }
                if !probe_tensor_ok() {
                    // Wrapper/runtime ABI mismatch — most often a local
                    // `LIBTORCH` opt-in (SENMEI_LIBTORCH_ENV) pointing at a
                    // different torch version than the wrapper was built for.
                    return Err(
                        "libtorch tensor probe failed (wrapper/runtime ABI mismatch; set \
                         SENMEI_LIBTORCH_ENV to use a local LIBTORCH install)"
                            .into(),
                    );
                }
                return Ok(Some(inst.clone()));
            }
            if let Err(e) = torch_sys::loader::init(&inst.lib_dir) {
                log::warn!("libtorch dlopen failed: {e}");
                return Err(e);
            }
            // Same ABI guard as the ROCm branch: a mismatched wrapper/runtime
            // (e.g. a local LIBTORCH opt-in that predates the wrapper headers)
            // must fail here instead of corrupting memory mid-render.
            if !probe_tensor_ok() {
                return Err(
                    "libtorch tensor probe failed (wrapper/runtime ABI mismatch; set \
                     SENMEI_LIBTORCH_ENV to use a local LIBTORCH install)"
                        .into(),
                );
            }
        } else if let Err(e) = &resolved {
            log::warn!("libtorch resolve failed: {e}");
        }
        resolved
    });
    match install {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            log::info!("no CUDA/ROCm device — CPU stays on burn-Vulkan");
            Err(Error::new("no CUDA/ROCm device — CPU stays on burn-Vulkan"))
        }
        Err(e) => {
            log::error!("libtorch backend unavailable: {e}");
            Err(Error::new(e.clone()))
        }
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
        Ok(Model::RifeNet(Box::new(m)))
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
                // f16 backend: the f16 .bpk weights load as-is, no f16→f32
                // adapter (burn's Vulkan path does the same).
                let mut store = BurnpackStore::from_file(&model.path);
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

    fn native_scale(&self) -> u32 {
        self.scale
    }

    /// Fused RGB8 (GPU re-sample when the requested scale ≠ model scale).
    /// Runs full-frame first — the 640px tile grid is pure overhead on tch
    /// (59 vs 34 ms @640×360, 453 vs 384 @1080p) — and falls back to the
    /// shared tiled fused path only when the full-frame VRAM guard rejects
    /// (8K/oversize).
    fn infer_rgb8(&mut self, input: &Tensor, scale: u32) -> Option<Result<(Vec<u8>, u32, u32)>> {
        let model = self.model.as_ref()?;
        if !tch_tiled() {
            if let Some(Ok(v)) =
                core::infer_rgb8_full_frame(model, input, self.scale, scale, &self.device)
            {
                return Some(Ok(v));
            }
        }
        // Fused rejection (full-frame guard or a forward error): fall back to
        // the tiled fused path, whose own rejection lands on `infer_tiled`.
        match core::infer_rgb8(model, input, self.scale, scale, &self.device) {
            Some(Err(_)) => None,
            other => other,
        }
    }

    /// Fused multi-frame RGB8 with a deferred readback, so the caller can
    /// queue the next forward before blocking on this batch's transfer.
    fn infer_rgb8_submit(
        &mut self,
        inputs: &[Tensor],
        scale: u32,
    ) -> Option<Result<Box<dyn Rgb8Batch>>> {
        let model = self.model.as_ref()?;
        if !tch_tiled() {
            if let Some(Ok(b)) = core::infer_rgb8_full_frame_batch_prepare(
                model,
                inputs,
                self.scale,
                scale,
                &self.device,
            ) {
                return Some(Ok(Box::new(b) as Box<dyn Rgb8Batch>));
            }
        }
        // Full-frame guard/error: fall back to the tiled fused path.
        match core::infer_rgb8_batch_prepare(model, inputs, self.scale, scale, &self.device) {
            Some(Ok(b)) => Some(Ok(Box::new(b) as Box<dyn Rgb8Batch>)),
            // Fused path can't handle this input (VRAM guard): fall back to
            // the tiled path rather than surfacing a hard error.
            Some(Err(_)) | None => None,
        }
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
    /// `TchEngine` (f16 weights, no adapter) and inferred — proves the
    /// burn-store → LibTorch plumbing over the dlopen path. `#[ignore]`
    /// because resolving libtorch downloads it (~2 GB) on first use; skips
    /// without a GPU.
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
        let mut save = BurnpackStore::from_file(&tmp);
        eprintln!("[tch_test] save_into...");
        m.save_into(&mut save).unwrap();
        eprintln!("[tch_test] saved, load...");

        let mref = ModelRef {
            id: "concept".into(),
            arch: "fallin-cugan".into(),
            scale: 2,
            num_block: 4,
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
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

    /// The tch engine's full-frame fused RGB8 path must match the plain
    /// full-frame `infer` output (both run the model once over the whole
    /// frame). `infer_rgb8` rounds on GPU f16, `infer`+convert on CPU f32, so
    /// allow ±1 LSB. `#[ignore]` like the roundtrip test (resolves libtorch
    /// on first use).
    #[test]
    #[ignore]
    fn tch_full_frame_rgb8_matches_infer() {
        if !crate::runtime::detect().supports_gpu() {
            eprintln!("no CUDA/ROCm device, skipping");
            return;
        }
        let data_dir = std::env::temp_dir().join("senmei_tch_test_data");
        let mut engine = TchEngine::runtime(&data_dir).unwrap();
        let tmp = std::env::temp_dir().join("senmei_tch_fframe.bpk");
        let _ = std::fs::remove_file(&tmp);

        let device = LibTorchDevice::Cuda(0);
        let m = UpCunet2xFast::<B>::new(&device);
        let mut save = BurnpackStore::from_file(&tmp);
        m.save_into(&mut save).unwrap();
        let mref = ModelRef {
            id: "concept".into(),
            arch: "fallin-cugan".into(),
            scale: 2,
            num_block: 4,
            num_conv: 16,
            feature_channels: 48,
            no_norm: false,
            layer_norm: false,
            dysample: true,
            shuffle: 1,
            path: tmp.clone(),
        };
        engine.load(&mref).unwrap();

        let (h, w) = (128u32, 128u32);
        let input = Tensor::new(
            vec![1, 3, h as usize, w as usize],
            vec![0.5f32; 3 * 128 * 128],
        );
        let (bytes, oh, ow) = engine
            .infer_rgb8(&input, 2)
            .expect("full-frame fused")
            .expect("infer_rgb8");
        assert_eq!((oh, ow), (h * 2, w * 2));
        let out = engine
            .infer(&input, &InferOptions { tile_size: None })
            .unwrap();
        assert_eq!(out.shape, vec![1, 3, (h * 2) as usize, (w * 2) as usize]);

        // `infer_rgb8` returns packed HWC bytes; `infer` an f32 NCHW tensor.
        let oho = out.shape[2] as usize;
        let owo = out.shape[3] as usize;
        let hw = oho * owo;
        let mut max_diff = 0i64;
        for y in 0..oho {
            for x in 0..owo {
                for ch in 0..3usize {
                    let v = out.data[ch * hw + y * owo + x];
                    let expect = ((v * 255.0 + 0.5).floor().clamp(0.0, 255.0)) as i64;
                    let got = bytes[(y * owo + x) * 3 + ch] as i64;
                    max_diff = max_diff.max((got - expect).abs());
                }
            }
        }
        assert!(
            max_diff <= 1,
            "full-frame RGB8 diverged from infer by {max_diff} LSB"
        );

        let _ = std::fs::remove_file(&tmp);
    }
}
