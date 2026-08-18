mod decoder;
mod downloader;
mod encoder;
mod ffmpeg;
mod frame;
mod preview;
mod preview_stream;
mod probe;
mod process;

pub use decoder::Decoder;
pub use downloader::download_to_temp;
pub use encoder::Encoder;
pub use ffmpeg::{download, probe as probe_ffmpeg, resolve, FfmpegInfo};
pub use frame::Frame;
pub use preview::{encode_png, extract_frame};
pub use preview_stream::PreviewCache;
pub use probe::{probe, VideoInfo};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Command(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
}
