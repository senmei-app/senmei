use senmei_media::Frame;
use senmei_ml::{InferOptions, InferenceEngine};

use crate::frame::{frame_to_tensor, tensor_to_frame};

pub trait Step: Send {
    fn name(&self) -> &'static str;
    /// Transform a frame; return `false` to drop it from the output (dedup).
    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool>;
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

/// Reference denoise: box blur of the luma-ish planar RGB. A cheap, tunable
/// stand-in until a real denoiser model is ported.
pub struct Denoise {
    radius: u32,
}

impl Denoise {
    pub fn new(radius: u32) -> Self {
        Self {
            radius: radius.max(1),
        }
    }
}

impl Step for Denoise {
    fn name(&self) -> &'static str {
        "denoise"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
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

/// Reference deblur: unsharp mask (sharpen), `amount` scales the high-pass.
pub struct Deblur {
    amount: f32,
}

impl Deblur {
    pub fn new(amount: f32) -> Self {
        Self {
            amount: amount.clamp(0.0, 2.0),
        }
    }
}

impl Step for Deblur {
    fn name(&self) -> &'static str {
        "deblur"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
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

/// Drop consecutive frames that are near-duplicates of the previous one
/// (mean pixel diff below `threshold` in [0,1]). Never drops more than
/// `max_consecutive` in a row, so a static scene keeps a usable frame rate
/// instead of collapsing to a single frame.
pub struct Dedup {
    threshold: f32,
    prev: Option<Frame>,
    max_consecutive: usize,
    consecutive: usize,
}

impl Dedup {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            prev: None,
            max_consecutive: 5,
            consecutive: 0,
        }
    }
}

impl Step for Dedup {
    fn name(&self) -> &'static str {
        "dedup"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        let dup = self.prev.as_ref().is_some_and(|prev| {
            prev.width == frame.width
                && prev.height == frame.height
                && mean_abs_diff(&prev.data, &frame.data) < self.threshold
        });
        if dup && self.consecutive < self.max_consecutive {
            self.consecutive += 1;
            return Ok(false);
        }
        self.consecutive = 0;
        self.prev = Some(frame.clone());
        Ok(true)
    }
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().max(1);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.abs_diff(*y) as u32) as f32)
        .sum::<f32>()
        / (n as f32 * 255.0)
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
        Self {
            scale: scale.max(1),
            engine,
        }
    }
}

impl Step for Upscale {
    fn name(&self) -> &'static str {
        "upscale"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        let input = frame_to_tensor(frame);
        // Fused tiled RGB8 output path (GPU conversion) when the engine
        // supports it; otherwise fall back to infer_tiled + tensor_to_frame.
        if let Some(engine) = self.engine.as_mut() {
            if let Some(res) = engine.infer_rgb8(&input, self.scale) {
                let (bytes, w, h) = res.map_err(|e| crate::Error::new(e.to_string()))?;
                *frame = Frame {
                    width: w,
                    height: h,
                    data: bytes,
                };
                return Ok(true);
            }
        }
        let out = match self.engine.as_mut() {
            Some(engine) => {
                let opts = InferOptions {
                    tile_size: Some(TILE_SIZE),
                };
                senmei_ml::infer_tiled(engine.as_mut(), &input, &opts)
                    .map_err(|e| crate::Error::new(e.to_string()))?
            }
            None => senmei_ml::bilinear(
                &input,
                frame.height as usize * self.scale as usize,
                frame.width as usize * self.scale as usize,
            ),
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
        Ok(true)
    }
}

/// Resize step: bilinear-resamples the packed rgb24 frame by a scale factor.
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

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        let nw = ((frame.width as f32) * self.factor).round().max(1.0) as u32;
        let nh = ((frame.height as f32) * self.factor).round().max(1.0) as u32;
        resize_frame(frame, nw, nh)?;
        Ok(true)
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

    let x_ratio = if nw > 1 {
        (w as f32 - 1.0) / (nw as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if nh > 1 {
        (h as f32 - 1.0) / (nh as f32 - 1.0)
    } else {
        0.0
    };

    let src = &frame.data;
    let mut out = vec![0u8; 3 * nw * nh];
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
            for c in 0..3 {
                let a = src[(y0 * w + x0) * 3 + c] as f32;
                let b = src[(y0 * w + x1) * 3 + c] as f32;
                let c0 = src[(y1 * w + x0) * 3 + c] as f32;
                let d = src[(y1 * w + x1) * 3 + c] as f32;
                let top = a + (b - a) * fx;
                let bot = c0 + (d - c0) * fx;
                out[(ny * nw + nx) * 3 + c] =
                    (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
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
        fn capabilities(&self) -> senmei_ml::EngineCaps {
            senmei_ml::EngineCaps { tiles: false }
        }
        fn load(&mut self, _m: &senmei_ml::ModelRef) -> senmei_ml::Result<()> {
            Ok(())
        }
        fn infer(
            &mut self,
            input: &Tensor,
            _o: &senmei_ml::InferOptions,
        ) -> senmei_ml::Result<Tensor> {
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

    #[test]
    fn denoise_smooths_noise() {
        let mut frame = Frame {
            width: 8,
            height: 8,
            data: vec![100u8; 3 * 8 * 8],
        };
        frame.data[0] = 255; // salt noise in the top-left pixel
        Denoise::new(1).process(&mut frame).unwrap();
        // The isolated bright pixel is pulled toward the surrounding value.
        assert!(frame.data[0] < 255 && frame.data[0] > 100);
        assert_eq!((frame.width, frame.height), (8, 8));
    }

    #[test]
    fn deblur_sharpens_edge() {
        // A vertical hard edge; unsharp masking must increase the contrast at it.
        let mut frame = Frame {
            width: 8,
            height: 1,
            data: vec![0u8; 3 * 8],
        };
        for x in 4..8 {
            for c in 0..3 {
                frame.data[x * 3 + c] = 200;
            }
        }
        Deblur::new(0.5).process(&mut frame).unwrap();
        // The bright edge pixel is pushed past its original value (overshoot).
        assert!(frame.data[4 * 3] > 200);
    }

    #[test]
    fn denoise_keeps_channels_separate() {
        // Pure red packed frame: a channel-independent denoise keeps G/B at 0.
        let mut frame = Frame {
            width: 8,
            height: 8,
            data: vec![0u8; 3 * 8 * 8],
        };
        for px in frame.data.chunks_exact_mut(3) {
            px[0] = 255;
        }
        Denoise::new(1).process(&mut frame).unwrap();
        assert_eq!(frame.data[1], 0, "G contaminated");
        assert_eq!(frame.data[2], 0, "B contaminated");
    }

    #[test]
    fn deblur_keeps_channels_separate() {
        let mut frame = Frame {
            width: 8,
            height: 8,
            data: vec![0u8; 3 * 8 * 8],
        };
        for px in frame.data.chunks_exact_mut(3) {
            px[0] = 255;
        }
        Deblur::new(0.5).process(&mut frame).unwrap();
        assert_eq!(frame.data[1], 0, "G contaminated");
        assert_eq!(frame.data[2], 0, "B contaminated");
    }

    #[test]
    fn resize_keeps_channels_separate() {
        // 2x2 packed frame with four distinct colors; resampling must keep each
        // pixel's channel triplet intact (top-left stays red).
        let pixels: [u8; 12] = [
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            255, 255, 255, // white
        ];
        let mut frame = Frame {
            width: 2,
            height: 2,
            data: pixels.to_vec(),
        };
        Resize::new(2.0).process(&mut frame).unwrap();
        assert_eq!(
            &frame.data[..3],
            &[255, 0, 0],
            "corner not red: {:?}",
            &frame.data[..12]
        );
    }

    #[test]
    fn dedup_drops_only_near_duplicates() {
        let mut step = Dedup::new(0.02);
        let a = Frame {
            width: 2,
            height: 2,
            data: vec![10u8; 12],
        };
        let b = Frame {
            width: 2,
            height: 2,
            data: vec![11u8; 12],
        }; // near-dup
        let c = Frame {
            width: 2,
            height: 2,
            data: vec![200u8; 12],
        }; // cut
        let d = Frame {
            width: 2,
            height: 2,
            data: vec![100u8; 12],
        }; // new cut

        let mut f = a.clone();
        assert!(step.process(&mut f).unwrap()); // first frame kept
        let mut f = b.clone();
        assert!(!step.process(&mut f).unwrap()); // near-dup dropped
        let mut f = c.clone();
        assert!(step.process(&mut f).unwrap()); // cut kept
        let mut f = c.clone();
        assert!(!step.process(&mut f).unwrap()); // identical to prev dropped
        let mut f = d.clone();
        assert!(step.process(&mut f).unwrap()); // new frame kept
    }

    #[test]
    fn dedup_never_collapses_static_run() {
        // 40 identical frames: dedup must still emit a frame every
        // `max_consecutive + 1` instead of collapsing to one.
        let mut step = Dedup::new(0.02);
        let mut kept = 0;
        for _ in 0..40 {
            let mut f = Frame {
                width: 2,
                height: 2,
                data: vec![10u8; 12],
            };
            if step.process(&mut f).unwrap() {
                kept += 1;
            }
        }
        assert_eq!(kept, 7); // frame 0 + force-kept every 6th
    }
}
