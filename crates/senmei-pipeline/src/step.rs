use senmei_media::Frame;
use senmei_ml::{InferenceEngine, InferOptions, Tensor};

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

fn frame_to_tensor(frame: &Frame) -> Tensor {
    let h = frame.height as usize;
    let w = frame.width as usize;
    let mut data = vec![0f32; 3 * h * w];
    for i in 0..frame.data.len() {
        data[i] = frame.data[i] as f32 / 255.0;
    }
    Tensor::new(vec![1, 3, h, w], data)
}

fn tensor_to_frame(t: &Tensor, width: u32, height: u32) -> Frame {
    let h = t.shape[2];
    let w = t.shape[3];
    let mut data = vec![0u8; 3 * h * w];
    for i in 0..data.len() {
        let v = (t.data[i] * 255.0).round();
        data[i] = v.clamp(0.0, 255.0) as u8;
    }
    Frame {
        width,
        height,
        data,
    }
}

/// Default tile size handed to engines that advertise tiling support.
const TILE_SIZE: u32 = 256;

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
        let out = match self.engine.as_mut() {
            Some(engine) => {
                let opts = InferOptions { half: false, tile_size: Some(TILE_SIZE) };
                senmei_ml::infer_tiled(engine.as_mut(), &input, &opts)
                    .map_err(|e| crate::Error::new(e.to_string()))?
            }
            None => senmei_ml::bilinear(&input, frame.height as usize * self.scale as usize, frame.width as usize * self.scale as usize),
        };
        let new_w = out.shape[3] as u32;
        let new_h = out.shape[2] as u32;
        *frame = tensor_to_frame(&out, new_w, new_h);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
