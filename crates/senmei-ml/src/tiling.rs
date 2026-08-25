use crate::Tensor;

/// Place tiles back into an NCHW canvas, feather-averaging overlapping
/// regions. A tile edge bordering a neighbour is weighted ~0 → 1 across the
/// overlap so per-tile border artifacts vanish at the seams (same ramp as
/// the fused `infer_rgb8` stitch). `overlap` is the output-space overlap.
pub fn stitch(
    tiles: &[(usize, usize, Tensor)],
    out_h: usize,
    out_w: usize,
    c: usize,
    overlap: (usize, usize),
) -> Tensor {
    let feather = |n: usize, low: bool, high: bool, ov: usize| -> Vec<f32> {
        let mut w = vec![1.0f32; n];
        let o = ov.min(n);
        if low {
            for k in 0..o {
                w[k] = (k as f32 + 1.0) / (ov as f32 + 1.0);
            }
        }
        if high {
            for k in 0..o {
                w[n - 1 - k] = (k as f32 + 1.0) / (ov as f32 + 1.0);
            }
        }
        w
    };
    let (ov_h, ov_w) = overlap;
    let mut acc = vec![0f32; c * out_h * out_w];
    let mut cov = vec![0f32; out_h * out_w];
    for (x, y, t) in tiles {
        let th = t.shape[2];
        let tw = t.shape[3];
        let wy = feather(th, *y > 0, *y + th < out_h, ov_h);
        let wx = feather(tw, *x > 0, *x + tw < out_w, ov_w);
        for ci in 0..c {
            for yy in 0..th {
                let w = wy[yy];
                let srow = &t.data[(ci * th + yy) * tw..(ci * th + yy) * tw + tw];
                let arow = &mut acc[(ci * out_h + (y + yy)) * out_w + x..][..tw];
                for xx in 0..tw {
                    arow[xx] += srow[xx] * (w * wx[xx]);
                }
            }
        }
        for yy in 0..th {
            let w = wy[yy];
            let crow = &mut cov[(y + yy) * out_w + x..][..tw];
            for xx in 0..tw {
                crow[xx] += w * wx[xx];
            }
        }
    }
    for i in 0..acc.len() {
        let yy = (i / out_w) % out_h;
        let xx = i % out_w;
        acc[i] /= cov[yy * out_w + xx].max(f32::EPSILON);
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

/// Crop a packed rgb24 canvas to its top-left `oh × ow` region.
/// Used by both engines' fused `infer_rgb8` (feature-gated).
#[cfg(any(feature = "burn", feature = "tch"))]
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

    #[test]
    fn stitch_feather_preserves_constant_and_single() {
        // Constant tiles → constant canvas: the feather weights form a
        // partition of unity, so no brightness drift at the seams.
        let const_tile = || Tensor::new(vec![1, 1, 8, 8], vec![124.0f32; 64]);
        let tiles = vec![
            (0usize, 0usize, const_tile()),
            (4, 0, const_tile()),
            (0, 4, const_tile()),
            (4, 4, const_tile()),
        ];
        let s = stitch(&tiles, 12, 12, 1, (4, 4));
        for v in &s.data {
            assert!((v - 124.0).abs() < 1e-3, "brightness leaked: {v}");
        }
        // Single tile → passthrough (no neighbour to blend with).
        let t = Tensor::new(vec![1, 1, 8, 8], (0..64).map(|i| i as f32).collect());
        let s2 = stitch(&[(0, 0, t)], 8, 8, 1, (2, 2));
        for (i, v) in s2.data.iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
    }

    #[test]
    fn stitch_feather_hides_dark_tile_edge() {
        // Two tiles side by side; the left tile's right edge is dark (model
        // border artifact). Feathering blends it out — the seam value must be
        // much closer to the core value than a hard box average would allow.
        let mut a = vec![124.0f32; 64];
        for i in (7..64).step_by(8) {
            a[i] = 83.0; // rightmost column dark
        }
        let ta = Tensor::new(vec![1, 1, 8, 8], a);
        let tb = Tensor::new(vec![1, 1, 8, 8], vec![124.0f32; 64]);
        let s = stitch(&[(0, 0, ta), (6, 0, tb)], 14, 14, 1, (2, 2));
        // Seam point x=7, y=4: blend of 124 (weight 2/3) + 83 (weight 1/3).
        let seam = s.data[4 * 14 + 7];
        assert!(seam > 110.0, "edge artifact still dominant: {seam}");
        // A point away from the seam stays at the core value.
        let core = s.data[4 * 14 + 3];
        assert!((core - 124.0).abs() < 1e-3, "core drifted: {core}");
    }
}
