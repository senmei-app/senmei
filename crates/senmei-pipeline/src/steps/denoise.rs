use senmei_media::Frame;
use senmei_ml::{InferOptions, InferenceEngine};

use crate::frame::{frame_to_tensor, tensor_to_frame};
use crate::Step;

use super::TILE_SIZE;

/// Denoise: runs the frame through an ML denoiser engine (DRUNet) when one is
/// available, else falls back to a box blur of the planar RGB.
pub struct Denoise {
    radius: u32,
    sigma: f32,
    engine: Option<Box<dyn InferenceEngine>>,
}

impl Denoise {
    /// `radius`: box-blur radius for the CPU fallback; also the base ML noise
    /// level (`sigma = radius/20`). `engine`: ML denoiser (DRUNet) when the
    /// user selected a denoise model, else `None` → box blur.
    pub fn new(radius: u32, engine: Option<Box<dyn InferenceEngine>>) -> Self {
        Self {
            radius: radius.max(1),
            sigma: (radius as f32 * 0.05).clamp(0.0, 1.0),
            engine,
        }
    }
}

impl Step for Denoise {
    fn name(&self) -> &'static str {
        "denoise"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        if let Some(engine) = self.engine.as_mut() {
            let input = frame_to_tensor(frame);
            let opts = InferOptions {
                tile_size: Some(TILE_SIZE),
            };
            match senmei_ml::infer_denoise_tiled(engine.as_mut(), &input, self.sigma, &opts) {
                Ok(out) => {
                    *frame = tensor_to_frame(&out, frame.width, frame.height);
                    return Ok(true);
                }
                Err(e) => log::warn!("denoise engine failed, using box blur: {e}"),
            }
        }
        let r = self.radius as usize;
        let h = frame.height as usize;
        let w = frame.width as usize;
        let src = &frame.data;
        let mut out = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0u32; 3];
                let mut n = 0u32;
                for dy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    for dx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                        let p = (dy * w + dx) * 3;
                        sum[0] += src[p] as u32;
                        sum[1] += src[p + 1] as u32;
                        sum[2] += src[p + 2] as u32;
                        n += 1;
                    }
                }
                let p = (y * w + x) * 3;
                out[p] = (sum[0] / n.max(1)) as u8;
                out[p + 1] = (sum[1] / n.max(1)) as u8;
                out[p + 2] = (sum[2] / n.max(1)) as u8;
            }
        }
        frame.data = out;
        Ok(true)
    }
}
