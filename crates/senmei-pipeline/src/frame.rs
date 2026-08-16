use senmei_media::Frame;
use senmei_ml::Tensor;

/// FFmpeg frames are packed `rgb24`; convert to planar NCHW (de-interleave).
pub fn frame_to_tensor(frame: &Frame) -> Tensor {
    let h = frame.height as usize;
    let w = frame.width as usize;
    let hw = h * w;
    let mut data = vec![0f32; 3 * hw];
    for p in 0..hw {
        let src = p * 3;
        data[p] = frame.data[src] as f32 / 255.0;
        data[hw + p] = frame.data[src + 1] as f32 / 255.0;
        data[2 * hw + p] = frame.data[src + 2] as f32 / 255.0;
    }
    Tensor::new(vec![1, 3, h, w], data)
}

/// Planar NCHW back to packed `rgb24` (interleave).
pub fn tensor_to_frame(t: &Tensor, width: u32, height: u32) -> Frame {
    let h = t.shape[2];
    let w = t.shape[3];
    let hw = h * w;
    let mut data = vec![0u8; 3 * hw];
    for p in 0..hw {
        let dst = p * 3;
        data[dst] = (t.data[p] * 255.0).round().clamp(0.0, 255.0) as u8;
        data[dst + 1] = (t.data[hw + p] * 255.0).round().clamp(0.0, 255.0) as u8;
        data[dst + 2] = (t.data[2 * hw + p] * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    Frame {
        width,
        height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIXELS: [u8; 12] = [
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        255, 255, 255, // white
    ];

    #[test]
    fn roundtrip_preserves_pixels() {
        let frame = Frame {
            width: 2,
            height: 2,
            data: PIXELS.to_vec(),
        };
        let t = frame_to_tensor(&frame);
        assert_eq!(t.shape, vec![1, 3, 2, 2]);
        assert_eq!(tensor_to_frame(&t, 2, 2).data, PIXELS.to_vec());
    }
}
