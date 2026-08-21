mod frame;
mod interpolate;
mod pipeline;
mod step;

pub use frame::{frame_to_tensor, tensor_to_frame};
pub use interpolate::Interpolator;
pub use pipeline::{Pipeline, Progress, StepTiming};
pub use step::{Deblur, Dedup, Denoise, Filter, Passthrough, Resize, Step, Upscale};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Media(#[from] senmei_media::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }

    pub fn cancelled() -> Self {
        Self::Message("cancelled".into())
    }
}
