//! Bilinear grid sampling (`torch.nn.functional.grid_sample`) for burn.
//!
//! Mirrors torch grid_sample with `padding_mode='border'` and a selectable
//! `align_corners` (RIFE uses `true`, RealPLKSR's DySample tail uses `false`).
//! The output spatial size follows the grid, so upsampling grids work. The
//! warp is built from `gather` over the flattened spatial axis with per-corner
//! index tensors; the four samples are combined with the bilinear weights.

use burn::tensor::backend::Backend;
use burn::tensor::{IntDType, Tensor};

/// Sample `input` at the normalized grid coordinates `grid` (in [-1,1], xy).
/// `align_corners=true` (torch default; RIFE).
pub fn grid_sample<B: Backend>(input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
    grid_sample_with(input, grid, true)
}

/// `grid_sample` with selectable `align_corners` (false = DySample).
pub fn grid_sample_with<B: Backend>(
    input: Tensor<B, 4>,
    grid: Tensor<B, 4>,
    align_corners: bool,
) -> Tensor<B, 4> {
    let [n, c, h, w] = input.dims();
    let [gh, gw] = [grid.dims()[1], grid.dims()[2]];

    // grid: [N,GH,GW,2] (x, y in [-1,1]) -> pixel coords, border-clamped.
    let gx = grid
        .clone()
        .slice([0..n, 0..gh, 0..gw, 0..1])
        .reshape([n, gh, gw])
        .unsqueeze_dim(1);
    let gy = grid
        .slice([0..n, 0..gh, 0..gw, 1..2])
        .reshape([n, gh, gw])
        .unsqueeze_dim(1);
    let (x, y) = if align_corners {
        (
            (gx + 1.0) / 2.0 * ((w - 1) as f64),
            (gy + 1.0) / 2.0 * ((h - 1) as f64),
        )
    } else {
        ((gx + 1.0) / 2.0 * (w as f64) - 0.5, (gy + 1.0) / 2.0 * (h as f64) - 0.5)
    };
    // border: clamp the coordinate to [0, size-1] before flooring (torch grid_sampler).
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);

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
            .reshape([n, 1, gh * gw])
            .repeat_dim(1, c)
            .reshape([n * c, gh * gw]);
        input_flat.clone().gather(1, flat).reshape([n, c, gh, gw])
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

    /// CPU reference mirroring torch grid_sample (border padding).
    fn ref_grid_sample(
        input: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        gh: usize,
        gw: usize,
        grid: &[f32],
        align_corners: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * c * gh * gw];
        let coord = |g: f32, size: usize| {
            let px = if align_corners {
                (g + 1.0) / 2.0 * (size as f32 - 1.0)
            } else {
                (g + 1.0) / 2.0 * size as f32 - 0.5
            };
            px.clamp(0.0, size as f32 - 1.0)
        };
        for ni in 0..n {
            for oy in 0..gh {
                for ox in 0..gw {
                    let gi = ((ni * gh + oy) * gw + ox) * 2;
                    let x = coord(grid[gi], w);
                    let y = coord(grid[gi + 1], h);
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
                        out[((ni * c + ci) * gh + oy) * gw + ox] = v;
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

        // The grid_sample gather pattern: flatten input to [N*C, H*W] and
        // gather along the flattened spatial axis with flat indices y*W + x.
        // (Chained dim gathers with a widened channel dim don't work — burn's
        // gather requires matching non-gather dims.)
        let input_flat = input.reshape([1, 4]); // [1*1, 2*2]
        let flat = |y: i32, x: i32| TensorData::new(vec![y * 2 + x; 4], [1, 4]);
        let idx = |y: i32, x: i32| BurnTensor::<Vulkan<f32>, 2, Int>::from_ints(flat(y, x), &device);
        let i00 = input_flat.clone().gather(1, idx(0, 0));
        let i01 = input_flat.clone().gather(1, idx(0, 1));
        let i10 = input_flat.clone().gather(1, idx(1, 0));
        let i11 = input_flat.gather(1, idx(1, 1));
        check("flat i00", &i00.into_data().to_vec().unwrap(), 0.0);
        check("flat i01", &i01.into_data().to_vec().unwrap(), 1.0);
        check("flat i10", &i10.into_data().to_vec().unwrap(), 2.0);
        check("flat i11", &i11.into_data().to_vec().unwrap(), 3.0);
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
        let want = ref_grid_sample(&input, n, c, h, w, h, w, &grid, true);
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-3f32, "mismatch at {i}: burn {a} vs ref {b}");
        }
    }

    #[test]
    #[ignore = "requires Vulkan"]
    fn grid_sample_align_false_upsamples() {
        use burn_wgpu::{Vulkan, WgpuDevice};
        let device = WgpuDevice::DiscreteGpu(0);
        let (n, c, h, w) = (1usize, 3usize, 6usize, 8usize);
        let (gh, gw) = (2 * h, 2 * w); // upsampled grid -> larger output
        let input: Vec<f32> = (0..n * c * h * w).map(|i| ((i * 37) % 100) as f32 / 100.0).collect();
        // grid in [-1.15, 1.15] (DySample coords may exceed [-1,1] slightly).
        let grid: Vec<f32> = (0..n * gh * gw)
            .map(|i| {
                let x = ((i * 13) % 61) as f32 / 26.0 - 1.15;
                let y = ((i * 7) % 59) as f32 / 25.0 - 1.15;
                [x, y]
            })
            .flatten()
            .collect();

        let x = BurnTensor::<Vulkan<f32>, 4>::from_data(TensorData::new(input.clone(), [n, c, h, w]), &device);
        let g = BurnTensor::<Vulkan<f32>, 4>::from_data(TensorData::new(grid.clone(), [n, gh, gw, 2]), &device);

        let out = grid_sample_with(x, g, false);
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        let want = ref_grid_sample(&input, n, c, h, w, gh, gw, &grid, false);
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-3f32, "mismatch at {i}: burn {a} vs ref {b}");
        }
    }
}
