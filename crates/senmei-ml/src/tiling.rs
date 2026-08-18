use crate::Tensor;

/// Place tiles back into an NCHW canvas, averaging overlapping regions.
pub fn stitch(
    tiles: &[(usize, usize, Tensor)],
    out_h: usize,
    out_w: usize,
    c: usize,
) -> Tensor {
    let mut acc = vec![0f32; c * out_h * out_w];
    let mut count = vec![0f32; c * out_h * out_w];
    for (x, y, t) in tiles {
        let th = t.shape[2];
        let tw = t.shape[3];
        for ci in 0..c {
            for yy in 0..th {
                for xx in 0..tw {
                    let idx = (ci * out_h + (y + yy)) * out_w + (x + xx);
                    acc[idx] += t.data[(ci * th + yy) * tw + xx];
                    count[idx] += 1.0;
                }
            }
        }
    }
    for i in 0..acc.len() {
        acc[i] /= count[i].max(1.0);
    }
    Tensor::new(vec![1, c, out_h, out_w], acc)
}

/// Tile a canvas with only full `tile × tile` tiles at `step` spacing, stopping
/// before a partial tile would be produced. The caller is expected to have padded
/// the canvas so the grid ends exactly (all tiles uniform — required for batching).
pub fn uniform_tile(t: &Tensor, tile: usize, step: usize) -> Vec<(usize, usize, Tensor)> {
    let c = t.shape[1];
    let h = t.shape[2];
    let w = t.shape[3];
    let mut tiles = Vec::new();
    let mut y = 0;
    while y + tile <= h {
        let mut x = 0;
        while x + tile <= w {
            let mut data = vec![0f32; c * tile * tile];
            for ci in 0..c {
                for yy in 0..tile {
                    let src = ((ci * h) + (y + yy)) * w + x;
                    let dst = (ci * tile + yy) * tile;
                    data[dst..dst + tile].copy_from_slice(&t.data[src..src + tile]);
                }
            }
            tiles.push((x, y, Tensor::new(vec![1, c, tile, tile], data)));
            x += step;
        }
        y += step;
    }
    tiles
}

/// Zero-free edge-replicated padding to `ph × pw` (used to make tile grids
/// uniform so a whole frame can be inferred as one batch).
pub fn pad_to(t: &Tensor, ph: usize, pw: usize) -> Tensor {
    let c = t.shape[1];
    let h = t.shape[2];
    let w = t.shape[3];
    if h == ph && w == pw {
        return t.clone();
    }
    let mut data = vec![0f32; c * ph * pw];
    for ci in 0..c {
        for yy in 0..ph {
            let sy = yy.min(h - 1);
            for xx in 0..pw {
                let sx = xx.min(w - 1);
                data[(ci * ph + yy) * pw + xx] = t.data[(ci * h + sy) * w + sx];
            }
        }
    }
    Tensor::new(vec![1, c, ph, pw], data)
}

/// Crop a canvas back to `oh × ow` (drops padded rows/cols).
pub fn crop(t: &Tensor, oh: usize, ow: usize) -> Tensor {
    let c = t.shape[1];
    let h = t.shape[2];
    let w = t.shape[3];
    if h == oh && w == ow {
        return t.clone();
    }
    let mut data = Vec::with_capacity(c * oh * ow);
    for ci in 0..c {
        for yy in 0..oh {
            let src = (ci * h + yy) * w;
            data.extend_from_slice(&t.data[src..src + ow]);
        }
    }
    Tensor::new(vec![1, c, oh, ow], data)
}

/// Place packed rgb24 tiles back into a canvas, averaging overlapping regions.
/// Tiles are `(x, y, rgb24_bytes, tile_h, tile_w)` with top-left output coords.
/// Only used by the burn engine's fused `infer_rgb8` (feature-gated).
#[allow(dead_code)]
pub fn stitch_rgb24(
    tiles: &[(usize, usize, Vec<u8>, usize, usize)],
    out_h: usize,
    out_w: usize,
) -> Vec<u8> {
    let mut acc = vec![0u64; 3 * out_h * out_w];
    let mut count = vec![0u32; out_h * out_w];
    for (x, y, bytes, th, tw) in tiles {
        for yy in 0..*th {
            for xx in 0..*tw {
                let src = (yy * tw + xx) * 3;
                let dst = ((y + yy) * out_w + (x + xx)) * 3;
                acc[dst] += bytes[src] as u64;
                acc[dst + 1] += bytes[src + 1] as u64;
                acc[dst + 2] += bytes[src + 2] as u64;
                count[(y + yy) * out_w + (x + xx)] += 1;
            }
        }
    }
    let mut out = vec![0u8; 3 * out_h * out_w];
    for i in 0..out_h * out_w {
        let c = count[i].max(1) as u64;
        out[i * 3] = ((acc[i * 3] + c / 2) / c) as u8;
        out[i * 3 + 1] = ((acc[i * 3 + 1] + c / 2) / c) as u8;
        out[i * 3 + 2] = ((acc[i * 3 + 2] + c / 2) / c) as u8;
    }
    out
}

/// Crop a packed rgb24 canvas to its top-left `oh × ow` region.
/// Only used by the burn engine's fused `infer_rgb8` (feature-gated).
#[allow(dead_code)]
pub fn crop_rgb24(src: &[u8], src_w: usize, oh: usize, ow: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(3 * oh * ow);
    for yy in 0..oh {
        out.extend_from_slice(&src[yy * src_w * 3..yy * src_w * 3 + ow * 3]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(h: usize, w: usize) -> Tensor {
        let mut data = vec![0f32; 3 * h * w];
        for i in 0..data.len() {
            data[i] = i as f32;
        }
        Tensor::new(vec![1, 3, h, w], data)
    }

    #[test]
    fn pad_crop_roundtrip() {
        let t = tensor(6, 6);
        let p = pad_to(&t, 8, 8);
        assert_eq!(p.shape, vec![1, 3, 8, 8]);
        let c = crop(&p, 6, 6);
        assert_eq!(c.shape, vec![1, 3, 6, 6]);
        assert_eq!(c.data, t.data);
    }

    #[test]
    fn pad_edge_replicates() {
        let t = tensor(2, 2);
        let p = pad_to(&t, 3, 3);
        // bottom-right corner replicates the last pixel of channel 2.
        assert_eq!(p.data[2 * 9 + 8], t.data[2 * 4 + 3]);
    }
}
