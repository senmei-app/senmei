use senmei_media::Frame;
use senmei_ml::Tensor;

/// Precomputed `x / 255.0` for every byte (avoids a division per pixel).
const DIV255: [f32; 256] = {
    let mut t = [0.0; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as f32 / 255.0;
        i += 1;
    }
    t
};

/// FFmpeg frames are packed `rgb24`; convert to planar NCHW (de-interleave).
pub fn frame_to_tensor(frame: &Frame) -> Tensor {
    let h = frame.height as usize;
    let w = frame.width as usize;
    let hw = h * w;
    let src = &frame.data;
    let mut data = vec![0f32; 3 * hw];
    let (r, rest) = data.split_at_mut(hw);
    let (g, b) = rest.split_at_mut(hw);
    for (p, px) in src.chunks_exact(3).enumerate() {
        r[p] = DIV255[px[0] as usize];
        g[p] = DIV255[px[1] as usize];
        b[p] = DIV255[px[2] as usize];
    }
    Tensor::new(vec![1, 3, h, w], data)
}

/// Planar NCHW back to packed `rgb24` (interleave).
///
/// `f32 -> u8` casts saturate in Rust, so `x*255 + 0.5` rounds and clamps in
/// one autovectorizable pass (no `round()`/`clamp()` per element).
pub fn tensor_to_frame(t: &Tensor, width: u32, height: u32) -> Frame {
    let h = t.shape[2];
    let w = t.shape[3];
    let hw = h * w;
    let r = &t.data[..hw];
    let g = &t.data[hw..2 * hw];
    let b = &t.data[2 * hw..3 * hw];
    let mut data = vec![0u8; 3 * hw];
    for (dst, ((&r, &g), &b)) in data
        .chunks_exact_mut(3)
        .zip(r.iter().zip(g.iter()).zip(b.iter()))
    {
        dst[0] = (r * 255.0 + 0.5) as u8;
        dst[1] = (g * 255.0 + 0.5) as u8;
        dst[2] = (b * 255.0 + 0.5) as u8;
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
