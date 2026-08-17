//! Bilinear grid sampling (`torch.nn.functional.grid_sample`) for burn.
//!
//! Mirrors the semantics RIFE relies on: `align_corners=True`,
//! `padding_mode='border'`. The warp is built from `gather` over the H/W
//! dims with per-corner index tensors; the four samples are combined with
//! the bilinear weights.

use burn::tensor::backend::Backend;
use burn::tensor::{IntDType, Tensor};

/// Sample `input` at the normalized grid coordinates `grid` (in [-1,1], xy).
/// `grid` must have the same spatial size as `input`.
#[allow(dead_code)] // used by the RIFE interpolation arch (M3)
pub fn grid_sample<B: Backend>(input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = input.dims();
    let [gh, gw] = [grid.dims()[1], grid.dims()[2]];
    debug_assert_eq!((gh, gw), (h, w), "grid_sample grid must match input spatial size");

    // grid: [N,H,W,2] (x, y in [-1,1]) -> pixel coords, align_corners=True.
    let gx = grid.clone().slice([0..n, 0..gh, 0..gw, 0..1]).reshape([n, gh, gw]).unsqueeze_dim(1);
    let gy = grid.slice([0..n, 0..gh, 0..gw, 1..2]).reshape([n, gh, gw]).unsqueeze_dim(1);
    let x = (gx.clone() + 1.0) / 2.0 * ((w - 1) as f64);
    let y = (gy.clone() + 1.0) / 2.0 * ((h - 1) as f64);

    let x0f = x.clone().floor();
    let y0f = y.clone().floor();
    let wx1 = x.clone() - x0f.clone();
    let wy1 = y.clone() - y0f.clone();
    let wx0 = Tensor::ones_like(&wx1) - wx1.clone();
    let wy0 = Tensor::ones_like(&wy1) - wy1.clone();

    let x0 = x0f.clone().cast(IntDType::I32).clamp(0, (w - 1) as i64);
    let x1 = (x0f + 1.0).cast(IntDType::I32).clamp(0, (w - 1) as i64);
    let y0 = y0f.clone().cast(IntDType::I32).clamp(0, (h - 1) as i64);
    let y1 = (y0f + 1.0).cast(IntDType::I32).clamp(0, (h - 1) as i64);

    // Sample each corner with a single gather over a flattened spatial axis
    // (input [N*C, H*W], flat index y*W + x). Two chained dim gathers would
    // re-pair the indices wrongly: the second gather shifts which row index
    // lands on each column.
    let input_flat = input.reshape([n * c, h * w]);
    let corner = |yo: Tensor<B, 4, burn::tensor::Int>, xo: Tensor<B, 4, burn::tensor::Int>| {
        let flat = (yo * (w as i64) + xo)
            .reshape([n, 1, h * w])
            .repeat_dim(1, c)
            .reshape([n * c, h * w]);
        input_flat.clone().gather(1, flat).reshape([n, c, h, w])
    };

    let i00 = corner(y0.clone(), x0.clone());
    let i01 = corner(y0, x1.clone());
    let i10 = corner(y1.clone(), x0);
    let i11 = corner(y1, x1);

    wx0.clone() * wy0.clone() * i00 + wx1.clone() * wy0 * i01 + wx0 * wy1.clone() * i10 + wx1 * wy1 * i11
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor as BurnTensor, TensorData};

    /// CPU reference mirroring torch grid_sample (align_corners=True, border).
    fn ref_grid_sample(input: &[f32], n: usize, c: usize, h: usize, w: usize, grid: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; n * c * h * w];
        for ni in 0..n {
            for oy in 0..h {
                for ox in 0..w {
                    let gi = ((ni * h + oy) * w + ox) * 2;
                    let x = (grid[gi] + 1.0) / 2.0 * (w as f32 - 1.0);
                    let y = (grid[gi + 1] + 1.0) / 2.0 * (h as f32 - 1.0);
                    let (x0, y0) = (x.floor(), y.floor());
                    let (x1, y1) = (x0 + 1.0, y0 + 1.0);
                    let (wx1, wy1) = (x - x0, y - y0);
                    let (wx0, wy0) = (1.0 - wx1, 1.0 - wy1);
                    let (ix0, ix1) = (x0.clamp(0.0, (w - 1) as f32) as usize, x1.clamp(0.0, (w - 1) as f32) as usize);
                    let (iy0, iy1) = (y0.clamp(0.0, (h - 1) as f32) as usize, y1.clamp(0.0, (h - 1) as f32) as usize);
                    for ci in 0..c {
                        let src = |iy: usize, ix: usize| input[((ni * c + ci) * h + iy) * w + ix];
                        let v = wx0 * wy0 * src(iy0, ix0)
                            + wx1 * wy0 * src(iy0, ix1)
                            + wx0 * wy1 * src(iy1, ix0)
                            + wx1 * wy1 * src(iy1, ix1);
                        out[((ni * c + ci) * h + oy) * w + ox] = v;
                    }
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "requires Vulkan"]
    fn gather_corners_select_expected_pixels() {
        use burn::tensor::Int;
        use burn_wgpu::{Vulkan, WgpuDevice};
        let device = WgpuDevice::DiscreteGpu(0);
        let input = BurnTensor::<Vulkan<f32>, 4>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0], [1, 1, 2, 2]),
            &device,
        );
        // all indices constant: y0=0, y1=1, x0=0, x1=1
        let idx = |v: i32| TensorData::new(vec![v; 4], [1, 1, 2, 2]);
        let y0 = BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(0), &device);
        let y1 = BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(1), &device);
        let x0 = BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(0), &device);
        let x1 = BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(1), &device);

        let check = |name: &str, v: &[f32], expected: f32| {
            assert!(v.iter().all(|&x| (x - expected).abs() < 1e-6), "{name} = {v:?}");
        };
        let i00 = input.clone().gather(2, y0.clone()).gather(3, x0.clone());
        let i01 = input.clone().gather(2, y0).gather(3, x1.clone());
        let i10 = input.clone().gather(2, y1.clone()).gather(3, x0);
        let i11 = input.clone().gather(2, y1).gather(3, x1);
        check("i00", &i00.into_data().to_vec().unwrap(), 0.0);
        check("i01", &i01.into_data().to_vec().unwrap(), 1.0);
        check("i10", &i10.into_data().to_vec().unwrap(), 2.0);
        check("i11", &i11.into_data().to_vec().unwrap(), 3.0);

        // same gathers, but with indices repeated across a channel dim (as in
        // grid_sample, which uses repeat_dim instead of expand for gather).
        let idx = |v: i32| TensorData::new(vec![v; 4], [1, 1, 2, 2]);
        let r = |t: BurnTensor<Vulkan<f32>, 4, Int>| t.repeat_dim(1, 3);
        let (y0, y1, x0, x1) = (
            BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(0), &device),
            BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(1), &device),
            BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(0), &device),
            BurnTensor::<Vulkan<f32>, 4, Int>::from_ints(idx(1), &device),
        );
        let i00 = input.clone().gather(2, r(y0.clone())).gather(3, r(x0.clone()));
        let i01 = input.clone().gather(2, r(y0)).gather(3, r(x1.clone()));
        let i10 = input.clone().gather(2, r(y1.clone())).gather(3, r(x0));
        let i11 = input.gather(2, r(y1)).gather(3, r(x1));
        let g00: Vec<f32> = i00.into_data().to_vec().unwrap();
        let g01: Vec<f32> = i01.into_data().to_vec().unwrap();
        let g10: Vec<f32> = i10.into_data().to_vec().unwrap();
        let g11: Vec<f32> = i11.into_data().to_vec().unwrap();
        assert_eq!(g00, vec![0.0; 12], "repeated i00 should be 0 everywhere");
        assert_eq!(g01, vec![1.0; 12], "repeated i01 should be 1 everywhere");
        assert_eq!(g10, vec![2.0; 12], "repeated i10 should be 2 everywhere");
        assert_eq!(g11, vec![3.0; 12], "repeated i11 should be 3 everywhere");
    }

    #[test]
    #[ignore = "requires Vulkan"]
    fn grid_sample_matches_reference() {
        use burn_wgpu::{Vulkan, WgpuDevice};
        let device = WgpuDevice::DiscreteGpu(0);
        let (n, c, h, w) = (1usize, 3usize, 6usize, 8usize);
        let input: Vec<f32> = (0..n * c * h * w).map(|i| ((i * 37) % 100) as f32 / 100.0).collect();
        // grid in [-1.2, 1.2] to exercise border clamping.
        let grid: Vec<f32> = (0..n * h * w)
            .map(|i| {
                let x = ((i * 13) % 49) as f32 / 20.0 - 1.2;
                let y = ((i * 7) % 47) as f32 / 19.0 - 1.2;
                [x, y]
            })
            .flatten()
            .collect();

        let x = BurnTensor::<Vulkan<f32>, 4>::from_data(TensorData::new(input.clone(), [n, c, h, w]), &device);
        let g = BurnTensor::<Vulkan<f32>, 4>::from_data(TensorData::new(grid.clone(), [n, h, w, 2]), &device);

        let out = grid_sample(x, g);
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        let want = ref_grid_sample(&input, n, c, h, w, &grid);
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-3f32, "mismatch at {i}: burn {a} vs ref {b}");
        }
    }
}
