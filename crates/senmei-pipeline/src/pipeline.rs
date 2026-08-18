use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    pause: Arc<AtomicBool>,
    encoder_args: Vec<String>,
    tonemap: senmei_media::Tonemap,
    range: Option<(u64, Option<u64>)>,
}

impl Pipeline {
    pub fn new(steps: Vec<Box<dyn Step>>) -> Self {
        Self {
            steps,
            interpolator: None,
            cancel: Arc::new(AtomicBool::new(false)),
            pause: Arc::new(AtomicBool::new(false)),
            encoder_args: Vec::new(),
            tonemap: senmei_media::Tonemap::Auto,
            range: None,
        }
    }

    /// Render only a time range (start ms, end ms); `None` end = to the end.
    pub fn set_range(&mut self, start_ms: u64, end_ms: Option<u64>) {
        self.range = Some((start_ms, end_ms));
    }

    /// Extra ffmpeg arguments appended to the output encode command.
    pub fn set_encoder_args(&mut self, args: Vec<String>) {
        self.encoder_args = args;
    }

    /// HDR→SDR tonemapping policy for the decode stage.
    pub fn set_tonemap(&mut self, tonemap: senmei_media::Tonemap) {
        self.tonemap = tonemap;
    }

    /// Install a cancellation flag; `run` aborts between frames once it is set.
    pub fn set_cancel(&mut self, cancel: Arc<AtomicBool>) {
        self.cancel = cancel;
    }

    /// Install a pause flag; `run` waits between frames while it is set.
    pub fn set_pause(&mut self, pause: Arc<AtomicBool>) {
        self.pause = pause;
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
        let (start_ms, end_ms) = self.range.unwrap_or((0, None));
        let mut decoder = Decoder::open_with_range(ffmpeg, input, start_ms, end_ms, self.tonemap)?;
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
            first_batch = run_step(step.as_mut(), first_batch)?;
        }
        let (w, h) = (first_batch[0].width, first_batch[0].height);

        // 3-stage pipeline: decode (thread) -> process (main) -> encode (thread).
        // The CPU-side decode/encode runs while the GPU is busy on the current
        // frame, hiding those costs behind the inference.
        let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<senmei_media::Frame>(2);
        let (out_tx, out_rx) = std::sync::mpsc::sync_channel::<senmei_media::Frame>(2);

        let encoder = Encoder::open(
            ffmpeg,
            input,
            output,
            w,
            h,
            fps,
            start_ms,
            &self.encoder_args,
        )?;
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
                    // Wait while paused; the bounded channels keep the decode/encode
                    // threads blocked, so no frames accumulate.
                    while self.pause.load(Ordering::Relaxed) && !self.cancel.load(Ordering::Relaxed)
                    {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
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
                        let next = std::mem::take(&mut batch);
                        match run_step(step.as_mut(), next) {
                            Ok(kept) => batch = kept,
                            Err(e) => {
                                main_err = Some(e);
                                failed = true;
                                break;
                            }
                        }
                    }
                    if failed {
                        break;
                    }
                    for frame in batch {
                        if out_tx.send(frame).is_err() {
                            main_err = Some(Error::new("encode channel closed"));
                            break;
                        }
                    }
                }
                Err(_) => break, // decoder finished
            }
        }

        drop(out_tx);
        drop(raw_rx); // unblock the decode thread if we bailed early
        let dec_res = dec_handle
            .join()
            .unwrap_or_else(|_| Err(Error::new("decode thread panicked")));
        let enc_res = enc_handle
            .join()
            .unwrap_or_else(|_| Err(Error::new("encode thread panicked")));

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

/// Run a batch through one step, keeping only frames the step doesn't drop.
fn run_step(
    step: &mut dyn Step,
    batch: Vec<senmei_media::Frame>,
) -> Result<Vec<senmei_media::Frame>> {
    let mut kept = Vec::with_capacity(batch.len());
    for mut frame in batch {
        if step.process(&mut frame)? {
            kept.push(frame);
        }
    }
    Ok(kept)
}
