mod frame;
mod interpolate;
mod pipeline;
mod steps;

use std::sync::atomic::{AtomicUsize, Ordering};

pub use frame::{frame_to_tensor, tensor_to_frame};
pub use interpolate::Interpolator;
pub use pipeline::{Pipeline, Progress, StepTiming};
pub use steps::{Deblur, Dedup, Denoise, Filter, Passthrough, Resize, Step, Upscale};

/// How many batches the upscale step keeps in flight (readback pipelining).
/// Set per render via [`set_pipeline_depth`]; more depth overlaps the readback
/// with more GPU forwards, at the cost of VRAM and cancel latency. 2 is the
/// sweet spot (docs/benchmarks.md): depth 3 adds ~1% over depth 2.
const DEFAULT_PIPELINE_DEPTH: usize = 2;
static PIPELINE_DEPTH: AtomicUsize = AtomicUsize::new(DEFAULT_PIPELINE_DEPTH);

/// `0` (unset) uses the owning default; explicit depths are clamped to ≥1.
pub fn set_pipeline_depth(depth: usize) {
    let d = if depth == 0 {
        DEFAULT_PIPELINE_DEPTH
    } else {
        depth.max(1)
    };
    PIPELINE_DEPTH.store(d, Ordering::Relaxed);
}

pub fn pipeline_depth() -> usize {
    PIPELINE_DEPTH.load(Ordering::Relaxed)
}

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
