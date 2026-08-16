mod engine;
mod model;
mod tensor;

pub use engine::{Backend, EngineCaps, InferOptions, InferenceEngine, NcnnEngine, TorchEngine};
pub use model::{ModelKind, ModelMetadata, ModelRef, Registry};
pub use tensor::Tensor;

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Error(String);

impl Error {
    pub fn unimplemented(what: &'static str) -> Self {
        Self(format!("not implemented: {what}"))
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
