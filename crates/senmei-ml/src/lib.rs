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

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Error(String);

impl Error {
    pub fn unimplemented(what: &'static str) -> Self {
        Self(format!("not implemented: {what}"))
    }

    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self(err.to_string())
    }
}
