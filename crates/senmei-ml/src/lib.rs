mod engine;
mod interpolate;
mod model;
mod onnx;
mod resize;
mod tensor;
mod tiling;

#[cfg(feature = "burn")]
mod burn;

pub use engine::{engine_for_model, infer_tiled, EngineCaps, InferOptions, InferenceEngine};
#[cfg(feature = "burn")]
pub use burn::{BurnEngine, convert_onnx_to_bpk, convert_pth_to_bpk};
pub use interpolate::{blend, is_scene_cut, mean_abs_diff};
pub use model::{ModelKind, ModelMetadata, ModelRef, Registry};
pub use resize::bilinear;
pub use tensor::Tensor;
pub use tiling::{crop, crop_rgb24, pad_to, stitch, stitch_rgb24, uniform_tile};

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
