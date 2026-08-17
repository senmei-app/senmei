use senmei_media::Frame;
use senmei_ml as ml;
use senmei_ml::{InferenceEngine, InferOptions};

use crate::frame::{frame_to_tensor, tensor_to_frame};
use crate::Result;

/// Mean absolute difference threshold (in [0,1]) above which two consecutive
/// frames are treated as a scene cut.
const SCENE_CUT_THRESHOLD: f32 = 0.25;

/// Interpolation runs at native resolution without tiling.
const INTERP_OPTS: InferOptions = InferOptions { half: true, tile_size: None };

/// Stateful frame interpolator: emits `factor - 1` intermediates between
/// consecutive frames. With an engine it uses `infer_interp` (e.g. RIFE);
/// without one it falls back to a linear blend, or duplicates across scene cuts.
pub struct Interpolator {
    factor: u32,
    prev: Option<Frame>,
    engine: Option<Box<dyn InferenceEngine>>,
}

impl Interpolator {
    pub fn new(factor: u32) -> Self {
        Self {
            factor: factor.max(2),
            prev: None,
            engine: None,
        }
    }

    pub fn with_engine(factor: u32, engine: Box<dyn InferenceEngine>) -> Self {
        Self {
            factor: factor.max(2),
            prev: None,
            engine: Some(engine),
        }
    }

    pub fn factor(&self) -> u32 {
        self.factor
    }

    /// Feed the next source frame; returns the frames to emit in order, with
    /// the current frame last. The first call emits just that frame.
    pub fn push(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        let mut out = Vec::with_capacity(self.factor as usize);
        if let Some(prev) = self.prev.take() {
            let a = frame_to_tensor(&prev);
            let b = frame_to_tensor(&frame);
            let n = self.factor - 1;
            let cut = ml::is_scene_cut(&a, &b, SCENE_CUT_THRESHOLD);
            for k in 1..=n {
                let t = k as f32 / (n + 1) as f32;
                let mid = match self.engine.as_mut() {
                    Some(engine) => match engine.infer_interp(&a, &b, t, &INTERP_OPTS) {
                        Some(res) => {
                            let out = res.map_err(|e| crate::Error::new(e.to_string()))?;
                            tensor_to_frame(&out, frame.width, frame.height)
                        }
                        None if cut => prev.clone(),
                        None => tensor_to_frame(&ml::blend(&a, &b, t), frame.width, frame.height),
                    },
                    None if cut => prev.clone(),
                    None => tensor_to_frame(&ml::blend(&a, &b, t), frame.width, frame.height),
                };
                out.push(mid);
            }
        }
        self.prev = Some(frame.clone());
        out.push(frame);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(value: u8) -> Frame {
        Frame {
            width: 2,
            height: 2,
            data: vec![value; 12],
        }
    }

    #[test]
    fn first_frame_emits_single() {
        let mut i = Interpolator::new(2);
        assert_eq!(i.push(gray(10)).unwrap().len(), 1);
    }

    #[test]
    fn factor_two_emits_blended_intermediate() {
        let mut i = Interpolator::new(2);
        let first = gray(0);
        i.push(first).unwrap();
        let out = i.push(gray(40)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].data, vec![20u8; 12]); // blend midpoint
        assert_eq!(out[1].data, vec![40u8; 12]); // current last
    }

    #[test]
    fn factor_three_emits_two_intermediates() {
        let mut i = Interpolator::new(3);
        i.push(gray(0)).unwrap();
        let out = i.push(gray(30)).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].data, vec![10u8; 12]); // t=1/3
        assert_eq!(out[1].data, vec![20u8; 12]); // t=2/3
        assert_eq!(out[2].data, vec![30u8; 12]);
    }

    #[test]
    fn scene_cut_duplicates_prev() {
        let mut i = Interpolator::new(2);
        let a = gray(0);
        i.push(a.clone()).unwrap();
        let b = gray(255);
        let out = i.push(b.clone()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].data, a.data); // no cross-fade across a cut
        assert_eq!(out[1].data, b.data);
    }
}
