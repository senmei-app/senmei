use crate::Tensor;

/// Split an NCHW tensor (N=1) into overlapping tiles of `tile` size.
/// Returns `(x, y, tile)` in input coordinates.
pub fn tile(t: &Tensor, tile: usize, overlap: usize) -> Vec<(usize, usize, Tensor)> {
    let c = t.shape[1];
    let h = t.shape[2];
    let w = t.shape[3];
    let step = tile.saturating_sub(overlap).max(1);
    let mut tiles = Vec::new();
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let th = (y + tile).min(h) - y;
            let tw = (x + tile).min(w) - x;
            let mut data = vec![0f32; c * th * tw];
            for ci in 0..c {
                for yy in 0..th {
                    let src = ((ci * h) + (y + yy)) * w + x;
                    let dst = (ci * th + yy) * tw;
                    data[dst..dst + tw].copy_from_slice(&t.data[src..src + tw]);
                }
            }
            tiles.push((x, y, Tensor::new(vec![1, c, th, tw], data)));
            x += step;
        }
        y += step;
    }
    tiles
}

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
    fn tile_stitch_reconstructs() {
        let t = tensor(8, 8);
        let tiles = tile(&t, 4, 0);
        assert_eq!(tiles.len(), 4);
        let out = stitch(&tiles, 8, 8, 3);
        assert_eq!(out.shape, vec![1, 3, 8, 8]);
        assert_eq!(out.data, t.data);
    }

    #[test]
    fn overlap_averages() {
        let t = tensor(6, 6);
        let tiles = tile(&t, 4, 2);
        let out = stitch(&tiles, 6, 6, 3);
        assert_eq!(out.shape, vec![1, 3, 6, 6]);
        for i in 0..out.data.len() {
            assert!((out.data[i] - t.data[i]).abs() < 1e-5);
        }
    }
}
