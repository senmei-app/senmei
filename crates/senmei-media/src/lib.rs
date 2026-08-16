mod decoder;
mod encoder;
mod ffmpeg;
mod frame;
mod probe;

pub use decoder::Decoder;
pub use encoder::Encoder;
pub use ffmpeg::{download, probe as probe_ffmpeg, resolve, FfmpegInfo};
pub use frame::Frame;
pub use probe::{probe, VideoInfo};

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Error(String);

impl Error {
    pub fn command_failed(msg: String) -> Self {
        Self(msg)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self(err.to_string())
    }
}
