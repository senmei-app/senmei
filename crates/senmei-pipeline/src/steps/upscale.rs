use std::collections::VecDeque;

use senmei_media::Frame;
use senmei_ml::{InferOptions, InferenceEngine, Rgb8Batch};

use crate::frame::{frame_to_tensor, tensor_to_frame};
use crate::Step;

use super::{process_individually, TILE_SIZE};

/// Upscale step: runs the input frame through an ML engine, or falls back
/// to a CPU bilinear scaler when no engine is available.
///
/// Readback pipelining: when the engine offers `infer_rgb8_submit`, batches
/// are deferred by the configured [`crate::pipeline_depth`] — the next batch's
/// GPU work is queued before the oldest readback resolves, so the GPU stays
/// busy during the transfer. The first batch (which fixes the encoder dims)
/// resolves synchronously; the trailing deferred batches come out on `flush`.
pub struct Upscale {
    scale: u32,
    engine: Option<Box<dyn InferenceEngine>>,
    /// Deferred batches whose readbacks are still pending (oldest first).
    inflight: VecDeque<Box<dyn Rgb8Batch>>,
    depth: usize,
    started: bool,
}

impl Upscale {
    pub fn new(scale: u32, engine: Option<Box<dyn InferenceEngine>>) -> Self {
        Self {
            scale: scale.max(1),
            engine,
            inflight: VecDeque::new(),
            depth: crate::pipeline_depth(),
            started: false,
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

    fn process_batch(&mut self, frames: &mut Vec<Frame>) -> crate::Result<()> {
        let Some(engine) = self.engine.as_mut() else {
            return process_individually(self, frames);
        };
        let inputs: Vec<_> = frames.iter().map(frame_to_tensor).collect();
        let handle = match engine.infer_rgb8_submit(&inputs, self.scale) {
            Some(Ok(h)) => h,
            Some(Err(e)) => return Err(crate::Error::new(e.to_string())),
            // Engine without a deferred path (e.g. tch): synchronous batch.
            None => return self.process_batch_sync(frames),
        };
        if !self.started {
            // First batch (fixes the encoder dims): resolve synchronously so
            // the pipeline sees the output size; nothing deferred yet.
            self.started = true;
            *frames = resolve_frames(handle)?;
            return Ok(());
        }
        // Steady state: keep up to `depth` batches in flight. When the queue
        // is full, resolve the oldest — the GPU is busy with the batch we
        // just submitted, so its readback overlaps the next forward.
        self.inflight.push_back(handle);
        *frames = if self.inflight.len() > self.depth {
            let oldest = self.inflight.pop_front().unwrap();
            resolve_frames(oldest)?
        } else {
            Vec::new()
        };
        Ok(())
    }

    fn flush(&mut self, frames: &mut Vec<Frame>) -> crate::Result<()> {
        while let Some(h) = self.inflight.pop_front() {
            frames.extend(resolve_frames(h)?);
        }
        Ok(())
    }
}

impl Upscale {
    /// Synchronous batch path for engines without `infer_rgb8_submit`: fused
    /// `infer_rgb8_batch` when inputs share dims, else per-frame.
    fn process_batch_sync(&mut self, frames: &mut Vec<Frame>) -> crate::Result<()> {
        let batchable = frames.len() > 1
            && self.engine.is_some()
            && frames
                .windows(2)
                .all(|w| w[0].width == w[1].width && w[0].height == w[1].height);
        if !batchable {
            return process_individually(self, frames);
        }
        let engine = self.engine.as_mut().expect("checked above");
        let inputs: Vec<_> = frames.iter().map(frame_to_tensor).collect();
        // Scale mismatch or an engine without the batch path returns `None`,
        // which falls back to the per-frame path (incl. bilinear re-scale).
        if let Some(res) = engine.infer_rgb8_batch(&inputs, self.scale) {
            let outs = res.map_err(|e| crate::Error::new(e.to_string()))?;
            for (frame, (bytes, w, h)) in frames.iter_mut().zip(outs) {
                *frame = Frame {
                    width: w,
                    height: h,
                    data: bytes,
                };
            }
            return Ok(());
        }
        process_individually(self, frames)
    }
}

/// Turn a resolved readback's packed rgb24 batches into frames.
fn resolve_frames(h: Box<dyn Rgb8Batch>) -> crate::Result<Vec<Frame>> {
    let outs = h.resolve().map_err(|e| crate::Error::new(e.to_string()))?;
    Ok(outs
        .into_iter()
        .map(|(data, width, height)| Frame {
            width,
            height,
            data,
        })
        .collect())
}
