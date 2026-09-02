//! Step chain tests.

use super::*;
use crate::frame::{frame_to_tensor, tensor_to_frame};
use senmei_ml::Tensor;

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

#[test]
fn resize_doubles_and_shrinks() {
    let mut frame = Frame {
        width: 4,
        height: 4,
        data: vec![10u8; 3 * 4 * 4],
    };
    Resize::new(2.0).process(&mut frame).unwrap();
    assert_eq!((frame.width, frame.height), (8, 8));
    assert_eq!(frame.data.len(), 3 * 8 * 8);

    Resize::new(0.5).process(&mut frame).unwrap();
    assert_eq!((frame.width, frame.height), (4, 4));
    assert_eq!(frame.data.len(), 3 * 4 * 4);
}

#[test]
fn resize_factor_one_is_noop() {
    let mut frame = Frame {
        width: 4,
        height: 4,
        data: vec![7u8; 3 * 4 * 4],
    };
    let before = frame.data.clone();
    Resize::new(1.0).process(&mut frame).unwrap();
    assert_eq!(frame.data, before);
}

#[test]
fn resize_preserves_solid_color() {
    let mut frame = Frame {
        width: 2,
        height: 2,
        data: vec![0u8; 3 * 2 * 2],
    };
    frame.data[0] = 255; // top-left pixel red
    Resize::new(2.0).process(&mut frame).unwrap();
    assert_eq!(frame.data[0], 255); // top-left corner stays red
}

// 2x2 packed rgb24 frame with four distinct pixel colors.
const PIXELS: [u8; 12] = [
    255, 0, 0, // red
    0, 255, 0, // green
    0, 0, 255, // blue
    255, 255, 255, // white
];

#[test]
fn frame_tensor_roundtrip_preserves_pixels() {
    let frame = Frame {
        width: 2,
        height: 2,
        data: PIXELS.to_vec(),
    };
    let t = frame_to_tensor(&frame);
    assert_eq!(t.shape, vec![1, 3, 2, 2]);
    let back = tensor_to_frame(&t, 2, 2);
    assert_eq!(back.data, PIXELS.to_vec());
}

#[test]
fn upscale_x1_preserves_pixels() {
    let mut frame = Frame {
        width: 2,
        height: 2,
        data: PIXELS.to_vec(),
    };
    let mut step = Upscale::new(1, None);
    step.process(&mut frame).unwrap();
    assert_eq!((frame.width, frame.height), (2, 2));
    assert_eq!(frame.data, PIXELS.to_vec());
}

// Fake engine that always upscales 4x, regardless of the requested scale.
struct QuadEngine;

impl senmei_ml::InferenceEngine for QuadEngine {
    fn capabilities(&self) -> senmei_ml::EngineCaps {
        senmei_ml::EngineCaps { tiles: false }
    }
    fn load(&mut self, _m: &senmei_ml::ModelRef) -> senmei_ml::Result<()> {
        Ok(())
    }
    fn infer(&mut self, input: &Tensor, _o: &senmei_ml::InferOptions) -> senmei_ml::Result<Tensor> {
        let h = input.shape[2];
        let w = input.shape[3];
        Ok(senmei_ml::bilinear(input, h * 4, w * 4))
    }
}

#[test]
fn engine_output_resized_to_requested_scale() {
    let mut frame = Frame {
        width: 4,
        height: 4,
        data: vec![128u8; 3 * 4 * 4],
    };
    let mut step = Upscale::new(2, Some(Box::new(QuadEngine)));
    step.process(&mut frame).unwrap();
    assert_eq!((frame.width, frame.height), (8, 8)); // 4x engine output forced back to 2x
    assert_eq!(frame.data.len(), 3 * 8 * 8);
}

// Fake engine that records `infer_rgb8_batch` calls and bilinearly doubles
// each frame (requested scale is honored per frame).
struct BatchEngine {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl BatchEngine {
    fn new() -> (Self, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            Self {
                calls: calls.clone(),
            },
            calls,
        )
    }
}

impl senmei_ml::InferenceEngine for BatchEngine {
    fn capabilities(&self) -> senmei_ml::EngineCaps {
        senmei_ml::EngineCaps { tiles: false }
    }
    fn load(&mut self, _m: &senmei_ml::ModelRef) -> senmei_ml::Result<()> {
        Ok(())
    }
    fn infer(&mut self, input: &Tensor, _o: &senmei_ml::InferOptions) -> senmei_ml::Result<Tensor> {
        let h = input.shape[2];
        let w = input.shape[3];
        Ok(senmei_ml::bilinear(input, h * 2, w * 2))
    }
    fn native_scale(&self) -> u32 {
        1
    }
    fn infer_rgb8_batch(
        &mut self,
        inputs: &[Tensor],
        scale: u32,
    ) -> Option<senmei_ml::Result<Vec<(Vec<u8>, u32, u32)>>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let outs = inputs
            .iter()
            .map(|t| {
                let (_, _, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
                let out = senmei_ml::bilinear(t, h * scale as usize, w * scale as usize);
                let mut bytes = vec![0u8; 3 * h * w * scale as usize * scale as usize];
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        for c in 0..3 {
                            let v = (out.data[c * h * w + i] * 255.0).round() as u8;
                            for dy in 0..scale as usize {
                                for dx in 0..scale as usize {
                                    let oy = y * scale as usize + dy;
                                    let ox = x * scale as usize + dx;
                                    bytes[(oy * w * scale as usize + ox) * 3 + c] = v;
                                }
                            }
                        }
                    }
                }
                (
                    bytes,
                    (w * scale as usize) as u32,
                    (h * scale as usize) as u32,
                )
            })
            .collect();
        Some(Ok(outs))
    }
}

#[test]
fn process_batch_uses_batch_engine() {
    let (engine, calls) = BatchEngine::new();
    let mut step = Upscale::new(2, Some(Box::new(engine)));
    let mut frames = vec![
        Frame {
            width: 4,
            height: 4,
            data: vec![10u8; 3 * 4 * 4],
        },
        Frame {
            width: 4,
            height: 4,
            data: vec![20u8; 3 * 4 * 4],
        },
        Frame {
            width: 4,
            height: 4,
            data: vec![30u8; 3 * 4 * 4],
        },
    ];
    step.process_batch(&mut frames).unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(frames.len(), 3);
    for f in &frames {
        assert_eq!((f.width, f.height), (8, 8));
        assert_eq!(f.data.len(), 3 * 8 * 8);
    }
}

#[test]
fn process_batch_mixed_sizes_falls_back() {
    let (engine, calls) = BatchEngine::new();
    let mut step = Upscale::new(2, Some(Box::new(engine)));
    let mut frames = vec![
        Frame {
            width: 4,
            height: 4,
            data: vec![10u8; 3 * 4 * 4],
        },
        Frame {
            width: 6,
            height: 4,
            data: vec![20u8; 3 * 4 * 6],
        },
    ];
    step.process_batch(&mut frames).unwrap();
    // Unequal dims must not hit the batch API; per-frame path still scales.
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!((frames[0].width, frames[0].height), (8, 8));
    assert_eq!((frames[1].width, frames[1].height), (12, 8));
}

// Fake engine whose `infer_rgb8` re-samples to the requested scale on the
// "GPU" (like the real fused path now does for a scale mismatch): native
// 2x model, any requested scale honored.
struct ResampleEngine;

impl senmei_ml::InferenceEngine for ResampleEngine {
    fn capabilities(&self) -> senmei_ml::EngineCaps {
        senmei_ml::EngineCaps { tiles: false }
    }
    fn load(&mut self, _m: &senmei_ml::ModelRef) -> senmei_ml::Result<()> {
        Ok(())
    }
    fn infer(&mut self, input: &Tensor, _o: &senmei_ml::InferOptions) -> senmei_ml::Result<Tensor> {
        let h = input.shape[2];
        let w = input.shape[3];
        Ok(senmei_ml::bilinear(input, h * 2, w * 2))
    }
    fn native_scale(&self) -> u32 {
        2
    }
    fn infer_rgb8(
        &mut self,
        input: &Tensor,
        scale: u32,
    ) -> Option<senmei_ml::Result<(Vec<u8>, u32, u32)>> {
        let (_, _, h, w) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let out = senmei_ml::bilinear(input, h * scale as usize, w * scale as usize);
        let mut bytes = vec![0u8; 3 * h * w * scale as usize * scale as usize];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                for c in 0..3 {
                    let v = (out.data[c * h * w + i] * 255.0).round() as u8;
                    for dy in 0..scale as usize {
                        for dx in 0..scale as usize {
                            let oy = y * scale as usize + dy;
                            let ox = x * scale as usize + dx;
                            bytes[(oy * w * scale as usize + ox) * 3 + c] = v;
                        }
                    }
                }
            }
        }
        Some(Ok((
            bytes,
            (w * scale as usize) as u32,
            (h * scale as usize) as u32,
        )))
    }
}

#[test]
fn upscale_fused_path_handles_scale_mismatch() {
    // x2-model engine, requested x4: the fused `infer_rgb8` re-samples and
    // must win directly — no `infer_tiled` + CPU re-scale fallback.
    let mut frame = Frame {
        width: 4,
        height: 4,
        data: vec![128u8; 3 * 4 * 4],
    };
    let mut step = Upscale::new(4, Some(Box::new(ResampleEngine)));
    step.process(&mut frame).unwrap();
    assert_eq!((frame.width, frame.height), (16, 16));
    assert_eq!(frame.data.len(), 3 * 16 * 16);
}

// Fake handle that yields one packed frame per submitted input on resolve.
struct FakeBatch(usize);

impl senmei_ml::Rgb8Batch for FakeBatch {
    fn resolve(self: Box<Self>) -> senmei_ml::Result<Vec<(Vec<u8>, u32, u32)>> {
        Ok((0..self.0)
            .map(|_| (vec![7u8; 3 * 4 * 4], 4u32, 4u32))
            .collect())
    }
}

// Fake engine with the deferred-readback path (`infer_rgb8_submit`).
struct PipelinedEngine;

impl senmei_ml::InferenceEngine for PipelinedEngine {
    fn capabilities(&self) -> senmei_ml::EngineCaps {
        senmei_ml::EngineCaps { tiles: false }
    }
    fn load(&mut self, _m: &senmei_ml::ModelRef) -> senmei_ml::Result<()> {
        Ok(())
    }
    fn infer(&mut self, input: &Tensor, _o: &senmei_ml::InferOptions) -> senmei_ml::Result<Tensor> {
        let h = input.shape[2];
        let w = input.shape[3];
        Ok(senmei_ml::bilinear(input, h * 2, w * 2))
    }
    fn native_scale(&self) -> u32 {
        1
    }
    fn infer_rgb8_submit(
        &mut self,
        inputs: &[Tensor],
        _scale: u32,
    ) -> Option<senmei_ml::Result<Box<dyn senmei_ml::Rgb8Batch>>> {
        Some(Ok(Box::new(FakeBatch(inputs.len()))))
    }
}

#[test]
fn upscale_process_batch_defers_then_flushes() {
    // The deferred-path handoff (1 in-flight, next submit resolves it)
    // assumes pipeline_depth = 1; pin it explicitly so the test is
    // independent of the global default (2).
    crate::set_pipeline_depth(1);
    let mut step = Upscale::new(2, Some(Box::new(PipelinedEngine)));
    let mk = |v: u8| Frame {
        width: 2,
        height: 2,
        data: vec![v; 12],
    };

    // First batch: resolves synchronously (fixes the encoder dims).
    let mut b1 = vec![mk(1)];
    step.process_batch(&mut b1).unwrap();
    assert_eq!(b1.len(), 1);

    // Second batch: deferred — held in-flight, nothing ready yet.
    let mut b2 = vec![mk(2)];
    step.process_batch(&mut b2).unwrap();
    assert!(b2.is_empty());

    // Third batch resolves the second; flush resolves the third.
    let mut b3 = vec![mk(3)];
    step.process_batch(&mut b3).unwrap();
    assert_eq!(b3.len(), 1);
    let mut tail = Vec::new();
    step.flush(&mut tail).unwrap();
    assert_eq!(tail.len(), 1);
    crate::set_pipeline_depth(0);
}

#[test]
fn upscale_process_batch_empty_is_noop() {
    // The pipeline drains a trailing empty batch at decoder EOF — the
    // deferred path must not submit zero inputs (would error "empty batch").
    let mut step = Upscale::new(2, Some(Box::new(PipelinedEngine)));
    let mut frames = Vec::new();
    step.process_batch(&mut frames).unwrap();
    assert!(frames.is_empty());
    // Still usable afterwards: the no-op must not set `started`.
    let mut b = vec![Frame {
        width: 2,
        height: 2,
        data: vec![1; 12],
    }];
    step.process_batch(&mut b).unwrap();
    assert_eq!(b.len(), 1);
}

// Fake step that drops frames whose first byte is the marker 42.
struct DropMarked;

impl Step for DropMarked {
    fn name(&self) -> &'static str {
        "drop-marked"
    }
    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        Ok(frame.data[0] != 42)
    }
}

#[test]
fn process_batch_default_drops_frames() {
    let mut step = DropMarked;
    let mut frames = vec![
        Frame {
            width: 1,
            height: 1,
            data: vec![1],
        },
        Frame {
            width: 1,
            height: 1,
            data: vec![42],
        },
        Frame {
            width: 1,
            height: 1,
            data: vec![3],
        },
        Frame {
            width: 1,
            height: 1,
            data: vec![42],
        },
    ];
    step.process_batch(&mut frames).unwrap();
    assert_eq!(frames.len(), 2);
    assert_ne!(frames[0].data[0], 42);
    assert_ne!(frames[1].data[0], 42);
}

#[test]
fn flush_default_is_noop() {
    let mut step = DropMarked;
    let mut frames = vec![Frame {
        width: 1,
        height: 1,
        data: vec![42],
    }];
    step.flush(&mut frames).unwrap();
    assert_eq!(frames.len(), 1);
}

#[test]
fn denoise_smooths_noise() {
    let mut frame = Frame {
        width: 8,
        height: 8,
        data: vec![100u8; 3 * 8 * 8],
    };
    frame.data[0] = 255; // salt noise in the top-left pixel
    Denoise::new(1, None).process(&mut frame).unwrap();
    // The isolated bright pixel is pulled toward the surrounding value.
    assert!(frame.data[0] < 255 && frame.data[0] > 100);
    assert_eq!((frame.width, frame.height), (8, 8));
}

#[test]
fn deblur_sharpens_edge() {
    // A vertical hard edge; unsharp masking must increase the contrast at it.
    let mut frame = Frame {
        width: 8,
        height: 1,
        data: vec![0u8; 3 * 8],
    };
    for x in 4..8 {
        for c in 0..3 {
            frame.data[x * 3 + c] = 200;
        }
    }
    Deblur::new(0.5, None).process(&mut frame).unwrap();
    // The bright edge pixel is pushed past its original value (overshoot).
    assert!(frame.data[4 * 3] > 200);
}

#[test]
fn denoise_keeps_channels_separate() {
    // Pure red packed frame: a channel-independent denoise keeps G/B at 0.
    let mut frame = Frame {
        width: 8,
        height: 8,
        data: vec![0u8; 3 * 8 * 8],
    };
    for px in frame.data.chunks_exact_mut(3) {
        px[0] = 255;
    }
    Denoise::new(1, None).process(&mut frame).unwrap();
    assert_eq!(frame.data[1], 0, "G contaminated");
    assert_eq!(frame.data[2], 0, "B contaminated");
}

#[test]
fn deblur_keeps_channels_separate() {
    let mut frame = Frame {
        width: 8,
        height: 8,
        data: vec![0u8; 3 * 8 * 8],
    };
    for px in frame.data.chunks_exact_mut(3) {
        px[0] = 255;
    }
    Deblur::new(0.5, None).process(&mut frame).unwrap();
    assert_eq!(frame.data[1], 0, "G contaminated");
    assert_eq!(frame.data[2], 0, "B contaminated");
}

#[test]
fn resize_keeps_channels_separate() {
    // 2x2 packed frame with four distinct colors; resampling must keep each
    // pixel's channel triplet intact (top-left stays red).
    let pixels: [u8; 12] = [
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        255, 255, 255, // white
    ];
    let mut frame = Frame {
        width: 2,
        height: 2,
        data: pixels.to_vec(),
    };
    Resize::new(2.0).process(&mut frame).unwrap();
    assert_eq!(
        &frame.data[..3],
        &[255, 0, 0],
        "corner not red: {:?}",
        &frame.data[..12]
    );
}

#[test]
fn dedup_drops_only_near_duplicates() {
    let mut step = Dedup::new(0.02);
    let a = Frame {
        width: 2,
        height: 2,
        data: vec![10u8; 12],
    };
    let b = Frame {
        width: 2,
        height: 2,
        data: vec![11u8; 12],
    }; // near-dup
    let c = Frame {
        width: 2,
        height: 2,
        data: vec![200u8; 12],
    }; // cut
    let d = Frame {
        width: 2,
        height: 2,
        data: vec![100u8; 12],
    }; // new cut

    let mut f = a.clone();
    assert!(step.process(&mut f).unwrap()); // first frame kept
    let mut f = b.clone();
    assert!(!step.process(&mut f).unwrap()); // near-dup dropped
    let mut f = c.clone();
    assert!(step.process(&mut f).unwrap()); // cut kept
    let mut f = c.clone();
    assert!(!step.process(&mut f).unwrap()); // identical to prev dropped
    let mut f = d.clone();
    assert!(step.process(&mut f).unwrap()); // new frame kept
}

#[test]
fn dedup_never_collapses_static_run() {
    // 40 identical frames: dedup must still emit a frame every
    // `max_consecutive + 1` instead of collapsing to one.
    let mut step = Dedup::new(0.02);
    let mut kept = 0;
    for _ in 0..40 {
        let mut f = Frame {
            width: 2,
            height: 2,
            data: vec![10u8; 12],
        };
        if step.process(&mut f).unwrap() {
            kept += 1;
        }
    }
    assert_eq!(kept, 7); // frame 0 + force-kept every 6th
}

fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn filter_negates_frame() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }
    let mut frame = Frame {
        width: 4,
        height: 4,
        data: vec![50u8; 3 * 4 * 4],
    };
    Filter::new("negate", "ffmpeg").process(&mut frame).unwrap();
    assert_eq!((frame.width, frame.height), (4, 4));
    assert!(frame.data.iter().all(|&b| b == 205)); // 255 - 50
}

#[test]
fn filter_rejects_size_change() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }
    let mut frame = Frame {
        width: 4,
        height: 4,
        data: vec![50u8; 3 * 4 * 4],
    };
    let err = Filter::new("scale=2:2", "ffmpeg")
        .process(&mut frame)
        .unwrap_err();
    assert!(
        format!("{err}").contains("frame-preserving"),
        "expected size-guard error, got: {err}"
    );
}
