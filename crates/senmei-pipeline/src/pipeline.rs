use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    cancel: Arc<AtomicBool>,
}

impl Pipeline {
    pub fn new(steps: Vec<Box<dyn Step>>) -> Self {
        Self {
            steps,
            interpolator: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Install a cancellation flag; `run` aborts between frames once it is set.
    pub fn set_cancel(&mut self, cancel: Arc<AtomicBool>) {
        self.cancel = cancel;
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
        mut on_progress: impl FnMut(Progress) + Send + 'static,
    ) -> Result<()> {
        log::info!("pipeline: decode/encode {input:?} -> {output:?}");
        let mut decoder = Decoder::open(ffmpeg, input)?;
        let factor = self.interpolator.as_ref().map(|i| i.factor()).unwrap_or(1) as u64;
        let total_frames = decoder.total_frames * factor;
        let fps = decoder.fps * factor as f64;

        // First frame fixes the encoder dimensions.
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
        let (w, h) = (first_batch[0].width, first_batch[0].height);

        // 3-stage pipeline: decode (thread) -> process (main) -> encode (thread).
        // The CPU-side decode/encode runs while the GPU is busy on the current
        // frame, hiding those costs behind the inference.
        let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<senmei_media::Frame>(2);
        let (out_tx, out_rx) = std::sync::mpsc::sync_channel::<senmei_media::Frame>(2);

        let encoder = Encoder::open(ffmpeg, output, w, h, fps)?;
        let enc_handle = std::thread::spawn(move || -> Result<()> {
            let mut enc = encoder;
            let mut processed = 0u64;
            while let Ok(frame) = out_rx.recv() {
                enc.write_frame(&frame)?;
                processed += 1;
                on_progress(Progress {
                    frames_processed: processed,
                    total_frames,
                });
            }
            enc.finish().map_err(Error::from)
        });

        let dec_handle = std::thread::spawn(move || -> Result<()> {
            let mut dec = decoder;
            while let Some(frame) = dec.next_frame()? {
                if raw_tx.send(frame).is_err() {
                    break;
                }
            }
            Ok(())
        });

        let mut main_err: Option<Error> = None;
        for frame in first_batch {
            if out_tx.send(frame).is_err() {
                main_err = Some(Error::new("encode channel closed"));
                break;
            }
        }
        while main_err.is_none() {
            match raw_rx.recv() {
                Ok(frame) => {
                    if self.cancel.load(Ordering::Relaxed) {
                        main_err = Some(Error::cancelled());
                        break;
                    }
                    let mut batch = match self.emit(frame) {
                        Ok(b) => b,
                        Err(e) => {
                            main_err = Some(e);
                            break;
                        }
                    };
                    let mut failed = false;
                    for step in &mut self.steps {
                        for frame in &mut batch {
                            if let Err(e) = step.process(frame) {
                                main_err = Some(e);
                                failed = true;
                                break;
                            }
                        }
                        if failed {
                            break;
                        }
                    }
                    if failed {
                        break;
                    }
                    for frame in batch {
                        if out_tx.send(frame).is_err() {
                            main_err = Some(Error::new("encode channel closed"));
                            failed = true;
                            break;
                        }
                    }
                }
                Err(_) => break, // decoder finished
            }
        }

        drop(out_tx);
        drop(raw_rx); // unblock the decode thread if we bailed early
        let dec_res =
            dec_handle.join().unwrap_or_else(|_| Err(Error::new("decode thread panicked")));
        let enc_res =
            enc_handle.join().unwrap_or_else(|_| Err(Error::new("encode thread panicked")));

        if let Some(e) = main_err {
            return Err(e);
        }
        dec_res?;
        enc_res
    }

    fn emit(&mut self, frame: senmei_media::Frame) -> Result<Vec<senmei_media::Frame>> {
        match self.interpolator.as_mut() {
            Some(interpolator) => interpolator.push(frame),
            None => Ok(vec![frame]),
        }
    }
}

