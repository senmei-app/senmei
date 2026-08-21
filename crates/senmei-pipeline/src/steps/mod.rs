mod deblur;
mod dedup;
mod denoise;
mod filter;
mod resize;
mod upscale;

pub use deblur::Deblur;
pub use dedup::Dedup;
pub use denoise::Denoise;
pub use filter::Filter;
pub use resize::Resize;
pub use upscale::Upscale;

use senmei_media::Frame;

/// Default tile size handed to engines that advertise tiling support.
pub(crate) const TILE_SIZE: u32 = 512;

pub trait Step: Send {
    fn name(&self) -> &'static str;
    /// Transform a frame; return `false` to drop it from the output (dedup).
    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool>;
}

pub struct Passthrough;

impl Step for Passthrough {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn process(&mut self, _frame: &mut Frame) -> crate::Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
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
        fn infer(
            &mut self,
            input: &Tensor,
            _o: &senmei_ml::InferOptions,
        ) -> senmei_ml::Result<Tensor> {
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
}
