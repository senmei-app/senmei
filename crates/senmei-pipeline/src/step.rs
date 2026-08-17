use senmei_media::Frame;
use senmei_ml::{InferenceEngine, InferOptions};

use crate::frame::{frame_to_tensor, tensor_to_frame};

pub trait Step: Send {
    fn name(&self) -> &'static str;
    fn process(&mut self, frame: &mut Frame) -> crate::Result<()>;
}

pub struct Passthrough;

impl Step for Passthrough {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn process(&mut self, _frame: &mut Frame) -> crate::Result<()> {
        Ok(())
    }
}

/// Default tile size handed to engines that advertise tiling support.
const TILE_SIZE: u32 = 512;

/// Upscale step: runs the input frame through an ML engine, or falls back
/// to a CPU bilinear scaler when no engine is available.
pub struct Upscale {
    scale: u32,
    engine: Option<Box<dyn InferenceEngine>>,
}

impl Upscale {
    pub fn new(scale: u32, engine: Option<Box<dyn InferenceEngine>>) -> Self {
        Self { scale: scale.max(1), engine }
    }
}

impl Step for Upscale {
    fn name(&self) -> &'static str {
        "upscale"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<()> {
        let input = frame_to_tensor(frame);
        // Fused GPU output path (f16 -> RGB8 directly) when the engine supports it.
        if let Some(engine) = self.engine.as_mut() {
            if let Some(res) = engine.infer_rgb8(&input, self.scale) {
                let (bytes, w, h) = res.map_err(|e| crate::Error::new(e.to_string()))?;
                *frame = Frame { width: w, height: h, data: bytes };
                return Ok(());
            }
        }
        let out = match self.engine.as_mut() {
            Some(engine) => {
                let opts = InferOptions { half: false, tile_size: Some(TILE_SIZE) };
                senmei_ml::infer_tiled(engine.as_mut(), &input, &opts)
                    .map_err(|e| crate::Error::new(e.to_string()))?
            }
            None => senmei_ml::bilinear(&input, frame.height as usize * self.scale as usize, frame.width as usize * self.scale as usize),
        };
        // An engine may upscale at a fixed factor (e.g. x4); enforce the
        // requested scale when the engine's output dims differ.
        let target_h = frame.height * self.scale;
        let target_w = frame.width * self.scale;
        let out = if out.shape[2] != target_h as usize || out.shape[3] != target_w as usize {
            senmei_ml::bilinear(&out, target_h as usize, target_w as usize)
        } else {
            out
        };
        let new_w = out.shape[3] as u32;
        let new_h = out.shape[2] as u32;
        *frame = tensor_to_frame(&out, new_w, new_h);
        Ok(())
    }
}

/// Resize step: bilinear-resamples the planar RGB frame by a scale factor.
pub struct Resize {
    factor: f32,
}

impl Resize {
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl Step for Resize {
    fn name(&self) -> &'static str {
        "resize"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<()> {
        let nw = ((frame.width as f32) * self.factor).round().max(1.0) as u32;
        let nh = ((frame.height as f32) * self.factor).round().max(1.0) as u32;
        resize_frame(frame, nw, nh)
    }
}

fn resize_frame(frame: &mut Frame, nw: u32, nh: u32) -> crate::Result<()> {
    let h = frame.height as usize;
    let w = frame.width as usize;
    let nw = nw as usize;
    let nh = nh as usize;
    if w == nw && h == nh {
        return Ok(());
    }

    let x_ratio = if nw > 1 { (w as f32 - 1.0) / (nw as f32 - 1.0) } else { 0.0 };
    let y_ratio = if nh > 1 { (h as f32 - 1.0) / (nh as f32 - 1.0) } else { 0.0 };

    let mut out = vec![0u8; 3 * nw * nh];
    for c in 0..3 {
        let src = &frame.data[c * w * h..(c + 1) * w * h];
        let dst = &mut out[c * nw * nh..(c + 1) * nw * nh];
        for ny in 0..nh {
            let sy = y_ratio * ny as f32;
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = sy - y0 as f32;
            for nx in 0..nw {
                let sx = x_ratio * nx as f32;
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = sx - x0 as f32;
                let a = src[y0 * w + x0] as f32;
                let b = src[y0 * w + x1] as f32;
                let c0 = src[y1 * w + x0] as f32;
                let d = src[y1 * w + x1] as f32;
                let top = a + (b - a) * fx;
                let bot = c0 + (d - c0) * fx;
                dst[ny * nw + nx] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    frame.width = nw as u32;
    frame.height = nh as u32;
    frame.data = out;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use senmei_ml::Tensor;

    #[test]
    fn upscale_reference_doubles_size() {
        let mut frame = Frame {
            width: 4,
            height: 4,
            data: vec![128u8; 3 * 4 * 4],
        };
        let mut step = Upscale::new(2, None);
        step.process(&mut frame).unwrap();
        assert_eq!(frame.width, 8);
        assert_eq!(frame.height, 8);
        assert_eq!(frame.data.len(), 3 * 8 * 8);
    }

    #[test]
    fn resize_doubles_and_shrinks() {
        let mut frame = Frame {
            width: 4,
            height: 4,
            data: vec![10u8; 3 * 4 * 4],
        };
        Resize::new(2.0).process(&mut frame).unwrap();
        assert_eq!((frame.width, frame.height), (8, 8));
        assert_eq!(frame.data.len(), 3 * 8 * 8);

        Resize::new(0.5).process(&mut frame).unwrap();
        assert_eq!((frame.width, frame.height), (4, 4));
        assert_eq!(frame.data.len(), 3 * 4 * 4);
    }

    #[test]
    fn resize_factor_one_is_noop() {
        let mut frame = Frame {
            width: 4,
            height: 4,
            data: vec![7u8; 3 * 4 * 4],
        };
        let before = frame.data.clone();
        Resize::new(1.0).process(&mut frame).unwrap();
        assert_eq!(frame.data, before);
    }

    #[test]
    fn resize_preserves_solid_color() {
        let mut frame = Frame {
            width: 2,
            height: 2,
            data: vec![0u8; 3 * 2 * 2],
        };
        frame.data[0] = 255; // top-left pixel red
        Resize::new(2.0).process(&mut frame).unwrap();
        assert_eq!(frame.data[0], 255); // top-left corner stays red
    }

    // 2x2 packed rgb24 frame with four distinct pixel colors.
    const PIXELS: [u8; 12] = [
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        255, 255, 255, // white
    ];

    #[test]
    fn frame_tensor_roundtrip_preserves_pixels() {
        let frame = Frame {
            width: 2,
            height: 2,
            data: PIXELS.to_vec(),
        };
        let t = frame_to_tensor(&frame);
        assert_eq!(t.shape, vec![1, 3, 2, 2]);
        let back = tensor_to_frame(&t, 2, 2);
        assert_eq!(back.data, PIXELS.to_vec());
    }

    #[test]
    fn upscale_x1_preserves_pixels() {
        let mut frame = Frame {
            width: 2,
            height: 2,
            data: PIXELS.to_vec(),
        };
        let mut step = Upscale::new(1, None);
        step.process(&mut frame).unwrap();
        assert_eq!((frame.width, frame.height), (2, 2));
        assert_eq!(frame.data, PIXELS.to_vec());
    }

    // Fake engine that always upscales 4x, regardless of the requested scale.
    struct QuadEngine;

    impl senmei_ml::InferenceEngine for QuadEngine {
        fn name(&self) -> &'static str {
            "quad-test"
        }
        fn capabilities(&self) -> senmei_ml::EngineCaps {
            senmei_ml::EngineCaps {
                backend: senmei_ml::Backend::Cpu,
                half: false,
                tiles: false,
            }
        }
        fn load(&mut self, _m: &senmei_ml::ModelRef) -> senmei_ml::Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &senmei_ml::InferOptions) -> senmei_ml::Result<Tensor> {
            let h = input.shape[2];
            let w = input.shape[3];
            Ok(senmei_ml::bilinear(input, h * 4, w * 4))
        }
    }

    #[test]
    fn engine_output_resized_to_requested_scale() {
        let mut frame = Frame {
            width: 4,
            height: 4,
            data: vec![128u8; 3 * 4 * 4],
        };
        let mut step = Upscale::new(2, Some(Box::new(QuadEngine)));
        step.process(&mut frame).unwrap();
        assert_eq!((frame.width, frame.height), (8, 8)); // 4x engine output forced back to 2x
        assert_eq!(frame.data.len(), 3 * 8 * 8);
    }
}
