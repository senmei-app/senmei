mod engine;
mod interpolate;
mod model;
#[cfg(feature = "burn")]
mod onnx;
mod resize;
mod runtime;
mod tensor;
mod tiling;

// Shared backend-generic archs, used by both engines.
#[cfg(any(feature = "burn", feature = "tch"))]
mod arch;

use std::sync::atomic::{AtomicU32, Ordering};

/// Fused RGB8 tile-size override (app settings); 0 = unset.
static TILE_SIZE: AtomicU32 = AtomicU32::new(0);

/// Override the fused RGB8 tile size (px). Falls back to `SENMEI_TILE`, then 640.
pub fn set_tile_size(n: u32) {
    TILE_SIZE.store(n, Ordering::Relaxed);
}

/// Discrete-GPU index for inference (app settings); 0 = first discrete GPU.
static GPU_INDEX: AtomicU32 = AtomicU32::new(0);

/// Select the discrete-GPU index used by the burn engine (0-based).
pub fn set_gpu_index(n: u32) {
    GPU_INDEX.store(n, Ordering::Relaxed);
}

/// Discrete-GPU index for inference (0 = first discrete GPU).
#[cfg(feature = "burn")]
pub(crate) fn gpu_index() -> u32 {
    GPU_INDEX.load(Ordering::Relaxed)
}

/// Read the fused RGB8 tile size (both engines tile internally).
#[cfg(any(feature = "burn", feature = "tch"))]
pub(crate) fn current_tile_size() -> usize {
    let n = TILE_SIZE.load(Ordering::Relaxed);
    if n > 0 {
        n as usize
    } else {
        std::env::var("SENMEI_TILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(640)
    }
}

#[cfg(feature = "burn")]
mod burn;
#[cfg(feature = "burn")]
mod convert;

#[cfg(feature = "tch")]
mod tch;

/// GPU backend: Vulkan everywhere, Metal on macOS (MoltenVK needs the SDK).
#[cfg(feature = "burn")]
#[cfg(target_os = "macos")]
pub(crate) use burn_wgpu::Metal as BurnBackend;
#[cfg(feature = "burn")]
#[cfg(not(target_os = "macos"))]
pub(crate) use burn_wgpu::Vulkan as BurnBackend;

#[cfg(feature = "burn")]
pub use burn::BurnEngine;
#[cfg(feature = "burn")]
pub use convert::{convert_onnx_to_bpk, convert_pth_to_bpk, convert_safetensors_to_bpk, ConvertOptions};
pub use engine::{
    backend_info, engine_for_model, infer_denoise_tiled, infer_tiled, BackendInfo, EngineBackend,
    EngineCaps, InferOptions, InferenceEngine, Rgb8Batch,
};
pub use interpolate::{blend, is_scene_cut, mean_abs_diff};
pub use model::{ModelKind, ModelMetadata, ModelRef, Registry};
pub use resize::bilinear;
pub use runtime::{
    detect, pick_device, pick_variant, resolve, vram_available_bytes, vram_total_bytes, Hardware,
    TorchInstall, TorchVariant,
};
#[cfg(feature = "tch")]
pub use tch::{TchDevice, TchEngine};
pub use tensor::Tensor;
pub use tiling::{crop, pad_to, stitch, uniform_tile};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }
}
