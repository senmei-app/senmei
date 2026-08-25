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

/// Accumulated per-step timing for the FPS benchmark report.
#[derive(Debug, Clone)]
pub struct StepTiming {
    pub name: String,
    pub frames: u64,
    pub total: std::time::Duration,
}

/// Frames accumulated before a batched step pass. 1 = off: multi-frame
/// batching regresses on RDNA4/Vulkan (larger batched matmuls are
/// pathologically slower — docs/benchmarks.md, `bench_upscale_batch`), so the
/// fused multi-frame path is not exercised on the shipped backend.
const BATCH_SIZE: usize = 1;

pub struct Pipeline {
    steps: Vec<Box<dyn Step>>,
    timings: Vec<StepTiming>,
    interpolator: Option<Interpolator>,
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    encoder_args: Vec<String>,
    tonemap: senmei_media::Tonemap,
    range: Option<(u64, Option<u64>)>,
}

impl Pipeline {
    pub fn new(steps: Vec<Box<dyn Step>>) -> Self {
        let timings = steps
            .iter()
            .map(|s| StepTiming {
                name: s.name().to_string(),
                frames: 0,
                total: std::time::Duration::ZERO,
            })
            .collect();
        Self {
            steps,
            timings,
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
        let mut decoder =
            Decoder::open_with_range(ffmpeg, input, start_ms, end_ms, self.tonemap, None)?;
        let factor = self.interpolator.as_ref().map(|i| i.factor()).unwrap_or(1) as u64;
        // The interpolator emits 1 frame for the first input and `factor` for
        // each following one, so the output count is `1 + (N-1)*factor`, not
        // `N*factor` — the latter makes progress cap below 100%.
        let total_frames = decoder.total_frames.saturating_sub(1) * factor + 1;
        let fps = decoder.fps * factor as f64;

        // First frame fixes the encoder dimensions.
        let first = match decoder.next_frame()? {
            Some(frame) => frame,
            None => return Err(Error::new("no frames decoded")),
        };
        let mut first_batch = self.emit(first)?;
        run_steps_batch(&mut self.timings, &mut self.steps, &mut first_batch)?;
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
            end_ms.map(|e| e.saturating_sub(start_ms)),
            &self.encoder_args,
        )?;
        let enc_cancel = self.cancel.clone();
        let enc_handle = std::thread::spawn(move || -> Result<()> {
            let mut enc = encoder;
            let mut processed = 0u64;
            let mut enc_n = 0u64;
            let mut enc_acc = std::time::Duration::ZERO;
            while let Ok(frame) = out_rx.recv() {
                // Abort ffmpeg on cancel instead of draining the channel and
                // finalizing: the normal `finish` muxes the whole file and
                // holds the pipeline (and its GPU engine) until it returns.
                if enc_cancel.load(Ordering::Relaxed) {
                    enc.abort();
                    return Ok(());
                }
                let t0 = std::time::Instant::now();
                enc.write_frame(&frame)?;
                enc_acc += t0.elapsed();
                enc_n += 1;
                processed += 1;
                on_progress(Progress {
                    frames_processed: processed,
                    total_frames,
                });
                if enc_n % 60 == 0 {
                    log::info!(
                        "pipeline: encode {:.1} ms/frame",
                        enc_acc.as_secs_f64() * 1000.0 / enc_n as f64
                    );
                }
            }
            if enc_cancel.load(Ordering::Relaxed) {
                enc.abort();
                return Ok(());
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
        let mut proc_n = 0u64;
        let mut proc_acc = std::time::Duration::ZERO;
        let mut pending: Vec<senmei_media::Frame> = Vec::with_capacity(BATCH_SIZE);
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
                    let t0 = std::time::Instant::now();
                    pending.extend(match self.emit(frame) {
                        Ok(b) => b,
                        Err(e) => {
                            main_err = Some(e);
                            break;
                        }
                    });
                    // Emit happens per frame; with BATCH_SIZE=1 the step chain
                    // runs per frame and the upscale step handles pipelining.
                    if pending.len() < BATCH_SIZE {
                        continue;
                    }
                    if let Err(e) =
                        run_steps_batch(&mut self.timings, &mut self.steps, &mut pending)
                    {
                        main_err = Some(e);
                        break;
                    }
                    proc_n += pending.len() as u64;
                    proc_acc += t0.elapsed();
                    if proc_n % 60 == 0 {
                        log::info!(
                            "pipeline: process {:.1} ms/frame ({:.1} fps)",
                            proc_acc.as_secs_f64() * 1000.0 / proc_n as f64,
                            proc_n as f64 / proc_acc.as_secs_f64()
                        );
                    }
                    for frame in pending.drain(..) {
                        if out_tx.send(frame).is_err() {
                            main_err = Some(Error::new("encode channel closed"));
                            break;
                        }
                    }
                }
                Err(_) => {
                    // Decoder finished: push the trailing partial batch through
                    // the steps, then let stateful steps flush their tail.
                    if !pending.is_empty() {
                        if let Err(e) =
                            run_steps_batch(&mut self.timings, &mut self.steps, &mut pending)
                        {
                            main_err = Some(e);
                            break;
                        }
                        for frame in pending.drain(..) {
                            if out_tx.send(frame).is_err() {
                                main_err = Some(Error::new("encode channel closed"));
                                break;
                            }
                        }
                        if main_err.is_some() {
                            break;
                        }
                    }
                    let mut tail = Vec::new();
                    for (i, step) in self.steps.iter_mut().enumerate() {
                        if let Err(e) = run_flush(&mut self.timings[i], step.as_mut(), &mut tail) {
                            main_err = Some(e);
                            break;
                        }
                    }
                    if main_err.is_some() {
                        break;
                    }
                    for frame in tail {
                        if out_tx.send(frame).is_err() {
                            main_err = Some(Error::new("encode channel closed"));
                            break;
                        }
                    }
                    break;
                }
            }
        }

        if main_err
            .as_ref()
            .is_some_and(|e| e.to_string() == "cancelled")
        {
            log::info!("pipeline: cancelled");
        }

        drop(out_tx);
        drop(raw_rx); // unblock the decode thread if we bailed early
        let dec_res = dec_handle
            .join()
            .unwrap_or_else(|_| Err(Error::new("decode thread panicked")));
        let enc_res = enc_handle
            .join()
            .unwrap_or_else(|_| Err(Error::new("encode thread panicked")));

        // "encode channel closed" only means the encode thread exited first —
        // its join result carries the real cause (ffmpeg stderr). Cancellation
        // and step errors from the main loop win.
        match main_err {
            Some(e) if e.to_string() != "encode channel closed" => return Err(e),
            _ => {}
        }
        match enc_res {
            Err(e) => return Err(e),
            Ok(()) => {}
        }
        if let Err(e) = dec_res {
            return Err(e);
        }
        for t in &self.timings {
            if t.frames == 0 {
                continue;
            }
            let ms = t.total.as_secs_f64() * 1000.0 / t.frames as f64;
            let fps = t.frames as f64 / t.total.as_secs_f64();
            log::info!(
                "pipeline: step {} — {ms:.2} ms/frame ({fps:.1} fps)",
                t.name
            );
        }
        Ok(())
    }

    /// Per-step timing accumulated over the run (empty until `run`).
    pub fn step_timings(&self) -> &[StepTiming] {
        &self.timings
    }

    fn emit(&mut self, frame: senmei_media::Frame) -> Result<Vec<senmei_media::Frame>> {
        match self.interpolator.as_mut() {
            Some(interpolator) => interpolator.push(frame),
            None => Ok(vec![frame]),
        }
    }
}

/// Run a pending batch through every step in order; frames dropped by a step
/// are removed in place. Accumulate per-step time for the FPS report.
fn run_steps_batch(
    timings: &mut [StepTiming],
    steps: &mut [Box<dyn Step>],
    batch: &mut Vec<senmei_media::Frame>,
) -> Result<()> {
    for (timing, step) in timings.iter_mut().zip(steps.iter_mut()) {
        let t0 = std::time::Instant::now();
        let n = batch.len() as u64;
        step.process_batch(batch)?;
        timing.frames += n;
        timing.total += t0.elapsed();
    }
    Ok(())
}

/// Emit a step's buffered tail state (default no-op).
fn run_flush(
    timing: &mut StepTiming,
    step: &mut dyn Step,
    frames: &mut Vec<senmei_media::Frame>,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    step.flush(frames)?;
    timing.total += t0.elapsed();
    Ok(())
}
