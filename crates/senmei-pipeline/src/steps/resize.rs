use senmei_media::Frame;

use crate::Step;

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
