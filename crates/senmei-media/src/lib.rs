mod content;
mod decoder;
mod downloader;
mod encoder;
mod ffmpeg;
mod frame;
mod preview;
mod preview_stream;
mod preview_worker;
mod probe;
mod process;
mod videos;

pub use content::is_anime;
pub use decoder::{Decoder, Tonemap};
pub use downloader::download_to_temp;
pub use downloader::{
    extract_binary, extract_zip, extract_zip_prefix, fetch, sha256_hex, sha256_hex_str,
    verify_checksum,
};
pub use encoder::Encoder;
pub use ffmpeg::{download, ffprobe_next_to, probe as probe_ffmpeg, resolve, FfmpegInfo};
pub use frame::Frame;
pub use preview::{encode_png, stream_pcm, PcmPipe};
pub use preview_stream::{PreviewCache, PREVIEW_MAX_DIM};
pub use preview_worker::PreviewWorker;
pub use probe::{probe, VideoInfo};
pub use videos::find_videos;

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
