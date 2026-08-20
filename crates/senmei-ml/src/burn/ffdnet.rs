//! FFDNet (cszn/KAIR, MIT) — clean burn port.
//!
//! Sigma-blind denoiser: replication-pad odd dims to even, pixel-unshuffle(2)
//! (RGB → 12ch at half res), concat a constant noise-level map (σ in [0,1]),
//! run 12 conv layers (nc=96) with ReLU between (no BatchNorm,
//! `act_mode="R"`), pixel-shuffle(2) back and crop to the input size.
//!
//! Field names carry the torch `model.{2i}` indices (ReLU sits at odd slots and
//! carries no params), so the converter only needs `^model\.(\d+)\.` → `c$1.`.
//! Odd input dims are replication-padded to even on the right/bottom first
//! (the unshuffle needs them), matching the KAIR `ReplicationPad2d`.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

fn conv3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

/// torch `pixel_unshuffle(x, 2)`: `[N, C, H, W] → [N, C*4, H/2, W/2]`.
fn pixel_unshuffle<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    x.reshape([b, c, h / 2, 2, w / 2, 2])
        .permute([0, 1, 3, 5, 2, 4])
        .reshape([b, c * 4, h / 2, w / 2])
}

/// torch `pixel_shuffle(x, 2)`: `[N, C, H, W] → [N, C/4, H*2, W*2]`.
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let c_out = c / 4;
    x.reshape([b, c_out, 2, 2, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, c_out, h * 2, w * 2])
}

/// Replication-pad odd spatial dims to even (bottom/right row/col) — matches
/// the KAIR `ReplicationPad2d` pre-unshuffle step. No-op on even dims.
fn to_even<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let mut x = x;
    if h % 2 == 1 {
        let r = x.clone().slice([0..b, 0..c, h - 1..h, 0..w]);
        x = Tensor::cat(vec![x, r], 2);
    }
    let [_, _, h2, w2] = x.dims();
    if w2 % 2 == 1 {
        let r = x.clone().slice([0..b, 0..c, 0..h2, w2 - 1..w2]);
        x = Tensor::cat(vec![x, r], 3);
    }
    x
}

#[derive(Module, Debug)]
pub struct Ffdnet<B: Backend> {
    c0: Conv2d<B>,
    c2: Conv2d<B>,
    c4: Conv2d<B>,
    c6: Conv2d<B>,
    c8: Conv2d<B>,
    c10: Conv2d<B>,
    c12: Conv2d<B>,
    c14: Conv2d<B>,
    c16: Conv2d<B>,
    c18: Conv2d<B>,
    c20: Conv2d<B>,
    c22: Conv2d<B>,
}

impl<B: Backend> Ffdnet<B> {
    pub fn new(device: &B::Device) -> Self {
        let c = |i: usize, o: usize| conv3(i, o, device);
        Self {
            c0: c(13, 96),
            c2: c(96, 96),
            c4: c(96, 96),
            c6: c(96, 96),
            c8: c(96, 96),
            c10: c(96, 96),
            c12: c(96, 96),
            c14: c(96, 96),
            c16: c(96, 96),
            c18: c(96, 96),
            c20: c(96, 96),
            c22: c(96, 12),
        }
    }

    /// `sigma` is the constant noise-level map value in [0,1] (matches the
    /// input range; FFDNet feeds σ/255 with 0-255 inputs).
    pub fn forward(&self, x: Tensor<B, 4>, sigma: f32) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let device = x.device();
        let mut y = pixel_unshuffle(to_even(x));
        let [_, _, hh, ww] = y.dims();
        let noise = Tensor::ones([b, 1, hh, ww], &device) * sigma;
        y = Tensor::cat(vec![y, noise], 1);
        let mut y = relu(self.c0.forward(y));
        y = relu(self.c2.forward(y));
        y = relu(self.c4.forward(y));
        y = relu(self.c6.forward(y));
        y = relu(self.c8.forward(y));
        y = relu(self.c10.forward(y));
        y = relu(self.c12.forward(y));
        y = relu(self.c14.forward(y));
        y = relu(self.c16.forward(y));
        y = relu(self.c18.forward(y));
        y = relu(self.c20.forward(y));
        let out = pixel_shuffle(self.c22.forward(y));
        out.slice([0..b, 0..c, 0..h, 0..w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::f16;
    use burn::tensor::TensorData;
    use burn_wgpu::WgpuDevice;

    // f16 (like the model tests) so this test doesn't poison the shared
    // autotune cache for the f16 backend; 0..15 is exact in f16.
    #[test]
    #[ignore = "requires Vulkan"]
    fn pixel_roundtrip_is_identity() {
        let device = WgpuDevice::DiscreteGpu(0);
        let data: Vec<f32> = (0..4 * 4).map(|i| i as f32).collect();
        let x = Tensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(data.clone(), [1, 1, 4, 4]).convert::<f16>(),
            &device,
        );
        let y = pixel_shuffle(pixel_unshuffle(x));
        let back: Vec<f32> = y.into_data().convert::<f32>().to_vec().unwrap();
        eprintln!("roundtrip: {back:?}");
        assert_eq!(back, data, "pixel unshuffle+shuffle not identity");
    }
}
