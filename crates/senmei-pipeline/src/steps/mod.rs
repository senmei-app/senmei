mod deblur;
mod dedup;
mod denoise;
mod filter;
mod resize;
mod upscale;

pub use deblur::Deblur;
pub use dedup::Dedup;
pub use denoise::Denoise;
pub use filter::Filter;
pub use resize::Resize;
pub use upscale::Upscale;

use senmei_media::Frame;

/// Default tile size handed to engines that advertise tiling support.
pub(crate) const TILE_SIZE: u32 = 512;

pub trait Step: Send {
    fn name(&self) -> &'static str;
    /// Transform a frame; return `false` to drop it from the output (dedup).
    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool>;

    /// Transform a batch in place; the default runs `process` per frame and
    /// drops the frames that return `false`. Steps with a fused multi-frame
    /// engine path override this (see `Upscale`).
    fn process_batch(&mut self, frames: &mut Vec<Frame>) -> crate::Result<()> {
        let mut i = 0;
        while i < frames.len() {
            if self.process(&mut frames[i])? {
                i += 1;
            } else {
                frames.remove(i);
            }
        }
        Ok(())
    }

    /// Emit any buffered state after the last batch (default no-op).
    fn flush(&mut self, _frames: &mut Vec<Frame>) -> crate::Result<()> {
        Ok(())
    }
}

/// Run `process` per frame, removing the frames the step drops. Shared by the
/// trait default and steps that fall back from a fused batch path.
pub(crate) fn process_individually(
    step: &mut dyn Step,
    frames: &mut Vec<Frame>,
) -> crate::Result<()> {
    let mut i = 0;
    while i < frames.len() {
        if step.process(&mut frames[i])? {
            i += 1;
        } else {
            frames.remove(i);
        }
    }
    Ok(())
}

pub struct Passthrough;

impl Step for Passthrough {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn process(&mut self, _frame: &mut Frame) -> crate::Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
