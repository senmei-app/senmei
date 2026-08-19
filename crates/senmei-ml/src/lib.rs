mod engine;
mod interpolate;
mod model;
mod onnx;
mod resize;
mod tensor;
mod tiling;

use std::sync::atomic::{AtomicU32, Ordering};

/// Fused RGB8 tile-size override (app settings); 0 = unset.
static TILE_SIZE: AtomicU32 = AtomicU32::new(0);

/// Override the fused RGB8 tile size (px). Falls back to `SENMEI_TILE`, then 640.
pub fn set_tile_size(n: u32) {
    TILE_SIZE.store(n, Ordering::Relaxed);
}

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

/// GPU backend: Vulkan everywhere, Metal on macOS (MoltenVK needs the SDK).
#[cfg(feature = "burn")]
#[cfg(target_os = "macos")]
pub(crate) use burn_wgpu::Metal as BurnBackend;
#[cfg(feature = "burn")]
#[cfg(not(target_os = "macos"))]
pub(crate) use burn_wgpu::Vulkan as BurnBackend;

#[cfg(feature = "burn")]
pub use burn::{convert_onnx_to_bpk, convert_pth_to_bpk, BurnEngine};
pub use engine::{
    engine_for_model, infer_denoise_tiled, infer_tiled, EngineCaps, InferOptions, InferenceEngine,
};
pub use interpolate::{blend, is_scene_cut, mean_abs_diff};
pub use model::{ModelKind, ModelMetadata, ModelRef, Registry};
pub use resize::bilinear;
pub use tensor::Tensor;
pub use tiling::{crop, crop_rgb24, pad_to, stitch, uniform_tile};

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
