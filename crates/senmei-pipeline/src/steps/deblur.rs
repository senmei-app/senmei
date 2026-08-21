use senmei_media::Frame;
use senmei_ml::{InferOptions, InferenceEngine};

use crate::frame::{frame_to_tensor, tensor_to_frame};
use crate::Step;

use super::TILE_SIZE;

/// Deblur: runs the frame through an ML deblur engine (NAFNet) when one is
/// available, else falls back to the unsharp-mask reference.
pub struct Deblur {
    amount: f32,
    engine: Option<Box<dyn InferenceEngine>>,
}

impl Deblur {
    /// `amount`: unsharp-mask strength for the CPU fallback. `engine`: ML
    /// deblur (NAFNet, scale 1, pads internally) when the user selected a
    /// deblur model, else `None` → unsharp mask.
    pub fn new(amount: f32, engine: Option<Box<dyn InferenceEngine>>) -> Self {
        Self {
            amount: amount.clamp(0.0, 2.0),
            engine,
        }
    }
}

impl Step for Deblur {
    fn name(&self) -> &'static str {
        "deblur"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        if let Some(engine) = self.engine.as_mut() {
            let input = frame_to_tensor(frame);
            let opts = InferOptions {
                tile_size: Some(TILE_SIZE),
            };
            match senmei_ml::infer_tiled(engine.as_mut(), &input, &opts) {
                Ok(out) => {
                    *frame = tensor_to_frame(&out, frame.width, frame.height);
                    return Ok(true);
                }
                Err(e) => log::warn!("deblur engine failed, using unsharp mask: {e}"),
            }
        }
        if self.amount <= 0.0 {
            return Ok(true);
        }
        let h = frame.height as usize;
        let w = frame.width as usize;
        let src = &frame.data;
        let mut blur = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0u32; 3];
                let mut n = 0u32;
                for dy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for dx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let p = (dy * w + dx) * 3;
                        sum[0] += src[p] as u32;
                        sum[1] += src[p + 1] as u32;
                        sum[2] += src[p + 2] as u32;
                        n += 1;
                    }
                }
                let p = (y * w + x) * 3;
                blur[p] = (sum[0] / n.max(1)) as u8;
                blur[p + 1] = (sum[1] / n.max(1)) as u8;
                blur[p + 2] = (sum[2] / n.max(1)) as u8;
            }
        }
        let a = self.amount;
        let mut out = vec![0u8; src.len()];
        for i in 0..src.len() {
            let v = src[i] as f32;
            let l = blur[i] as f32;
            out[i] = (v + a * (v - l)).clamp(0.0, 255.0) as u8;
        }
        frame.data = out;
        Ok(true)
    }
}
