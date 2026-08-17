mod engine;
mod interpolate;
mod model;
mod resize;
mod tensor;
mod tiling;

#[cfg(feature = "burn")]
mod burn;

pub use engine::{engine_for_model, infer_tiled, Backend, EngineCaps, InferOptions, InferenceEngine};
#[cfg(feature = "burn")]
pub use burn::{BurnEngine, convert_pth_to_bpk};
pub use interpolate::{blend, is_scene_cut, mean_abs_diff};
pub use model::{ModelKind, ModelMetadata, ModelRef, Registry};
pub use resize::bilinear;
pub use tensor::Tensor;
pub use tiling::{crop, pad_to, stitch, tile, uniform_tile};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn unimplemented(what: &'static str) -> Self {
        Self::Unimplemented(what)
    }

    pub fn new(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }
}
