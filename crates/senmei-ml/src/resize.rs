use crate::Tensor;

/// Reference bilinear upscale of an NCHW tensor (N=1) to `new_h`/`new_w`.
/// Pure CPU implementation, used as the fallback scaler until ML models land.
pub fn bilinear(t: &Tensor, new_h: usize, new_w: usize) -> Tensor {
    let c = t.shape[1];
    let h = t.shape[2];
    let w = t.shape[3];
    let x_ratio = if new_w > 1 { (w as f32 - 1.0) / (new_w as f32 - 1.0) } else { 0.0 };
    let y_ratio = if new_h > 1 { (h as f32 - 1.0) / (new_h as f32 - 1.0) } else { 0.0 };
    let mut data = vec![0f32; c * new_h * new_w];
    for ci in 0..c {
        for ny in 0..new_h {
            let sy = y_ratio * ny as f32;
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = sy - y0 as f32;
            for nx in 0..new_w {
                let sx = x_ratio * nx as f32;
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = sx - x0 as f32;
                let a = t.data[(ci * h + y0) * w + x0];
                let b = t.data[(ci * h + y0) * w + x1];
                let c0 = t.data[(ci * h + y1) * w + x0];
                let d = t.data[(ci * h + y1) * w + x1];
                let top = a + (b - a) * fx;
                let bot = c0 + (d - c0) * fx;
                data[(ci * new_h + ny) * new_w + nx] = top + (bot - top) * fy;
            }
        }
    }
    Tensor::new(vec![1, c, new_h, new_w], data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upscales_dimensions() {
        let mut data = vec![0f32; 3 * 2 * 2];
        data[0] = 255.0;
        let t = Tensor::new(vec![1, 3, 2, 2], data);
        let out = bilinear(&t, 4, 4);
        assert_eq!(out.shape, vec![1, 3, 4, 4]);
        assert_eq!(out.data[0], 255.0);
    }
}
