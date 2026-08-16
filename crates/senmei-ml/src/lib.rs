mod engine;
mod model;
mod resize;
mod tensor;
mod tiling;

pub use engine::{engine_for_model, infer_tiled, Backend, EngineCaps, InferOptions, InferenceEngine, NcnnEngine, TorchEngine};
pub use model::{ModelKind, ModelMetadata, ModelRef, Registry};
pub use resize::bilinear;
pub use tensor::Tensor;
pub use tiling::{stitch, tile};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),
    #[error("{0}")]
    Message(String),
    #[cfg(feature = "torch")]
    #[error(transparent)]
    Torch(#[from] tch::TchError),
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
