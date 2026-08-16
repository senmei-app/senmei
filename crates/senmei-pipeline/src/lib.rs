mod pipeline;
mod step;

pub use pipeline::{Pipeline, Progress};
pub use step::{Passthrough, Step, Upscale};

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(String);

impl Error {
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

impl From<senmei_media::Error> for Error {
    fn from(err: senmei_media::Error) -> Self {
        Self(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self(err.to_string())
    }
}
