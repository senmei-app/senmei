use std::path::Path;

use senmei_media::{Decoder, Encoder};

use crate::{Error, Result, Step};

#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub frames_processed: u64,
    pub total_frames: u64,
}

pub struct Pipeline {
    steps: Vec<Box<dyn Step>>,
}

impl Pipeline {
    pub fn new(steps: Vec<Box<dyn Step>>) -> Self {
        Self { steps }
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
        let total_frames = decoder.total_frames;

        // Steps may resize frames, so the encoder size must match the first
        // processed frame, not the decoded source.
        let mut first = match decoder.next_frame()? {
            Some(frame) => frame,
            None => return Err(Error::new("no frames decoded")),
        };
        for step in &mut self.steps {
            step.process(&mut first)?;
        }
        let mut encoder = Encoder::open(ffmpeg, output, first.width, first.height, decoder.fps)?;
        encoder.write_frame(&first)?;
        let mut processed = 1u64;
        on_progress(Progress {
            frames_processed: processed,
            total_frames,
        });

        while let Some(mut frame) = decoder.next_frame()? {
            for step in &mut self.steps {
                step.process(&mut frame)?;
            }
            encoder.write_frame(&frame)?;
            processed += 1;
            on_progress(Progress {
                frames_processed: processed,
                total_frames,
            });
        }
        encoder.finish()?;

        Ok(())
    }
}
