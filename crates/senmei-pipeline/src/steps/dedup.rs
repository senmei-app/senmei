use senmei_media::Frame;

use crate::Step;

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
