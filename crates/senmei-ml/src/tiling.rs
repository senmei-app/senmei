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
