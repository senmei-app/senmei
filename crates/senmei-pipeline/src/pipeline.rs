use std::path::Path;

use senmei_media::{Decoder, Encoder};

use crate::{Error, Interpolator, Result, Step};

#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub frames_processed: u64,
    pub total_frames: u64,
}

pub struct Pipeline {
    steps: Vec<Box<dyn Step>>,
    interpolator: Option<Interpolator>,
}

impl Pipeline {
    pub fn new(steps: Vec<Box<dyn Step>>) -> Self {
        Self {
            steps,
            interpolator: None,
        }
    }

    /// Enable frame interpolation (e.g. 24 → 48 fps) before the step chain.
    pub fn set_interpolator(&mut self, interpolator: Interpolator) {
        self.interpolator = Some(interpolator);
    }

    pub fn run(
        &mut self,
        ffmpeg: &Path,
        input: &Path,
        output: &Path,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<()> {
        log::info!("pipeline: decode/encode {input:?} -> {output:?}");
        let mut decoder = Decoder::open(ffmpeg, input)?;
        let factor = self.interpolator.as_ref().map(|i| i.factor()).unwrap_or(1) as u64;
        let total_frames = decoder.total_frames * factor;

        // Apply interpolation first, then the 1:1 steps; the encoder size and
        // fps must match the first emitted frame and the output frame rate.
        let first = match decoder.next_frame()? {
            Some(frame) => frame,
            None => return Err(Error::new("no frames decoded")),
        };
        let mut first_batch = self.emit(first)?;
        for step in &mut self.steps {
            for frame in &mut first_batch {
                step.process(frame)?;
            }
        }
        let mut encoder = Encoder::open(
            ffmpeg,
            output,
            first_batch[0].width,
            first_batch[0].height,
            decoder.fps * factor as f64,
        )?;

        let mut processed = 0u64;
        for frame in &first_batch {
            encoder.write_frame(frame)?;
            processed += 1;
            on_progress(Progress {
                frames_processed: processed,
                total_frames,
            });
        }

        while let Some(frame) = decoder.next_frame()? {
            let mut batch = self.emit(frame)?;
            for step in &mut self.steps {
                for frame in &mut batch {
                    step.process(frame)?;
                }
            }
            for frame in &batch {
                encoder.write_frame(frame)?;
                processed += 1;
                on_progress(Progress {
                    frames_processed: processed,
                    total_frames,
                });
            }
        }
        encoder.finish()?;

        Ok(())
    }

    fn emit(&mut self, frame: senmei_media::Frame) -> Result<Vec<senmei_media::Frame>> {
        match self.interpolator.as_mut() {
            Some(interpolator) => interpolator.push(frame),
            None => Ok(vec![frame]),
        }
    }
}

