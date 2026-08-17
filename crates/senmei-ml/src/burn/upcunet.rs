//! Real-CUGAN `UpCunet2x` (upcunet_v3) re-implemented in burn.
//!
//! Clean re-implementation from the MIT bilibili reference (`upcunet_v3.py`),
//! numerically verified against torch in `~/github/rust-sr-bench`. Field names
//! follow the PyTorch state-dict keys; the only mismatch is the `nn.Sequential`
//! inside `UnetConv` (keys `conv.0` / `conv.2`), fixed at load via
//! `KeyRemapper` in the engine.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::tensor::activation::{leaky_relu, relu, sigmoid};
use burn::tensor::ops::PadMode;
use burn::tensor::{Tensor, backend::Backend};

fn conv2d<B: Backend>(in_c: usize, out_c: usize, k: usize, s: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [k, k])
        .with_stride([s, s])
        .with_padding(PaddingConfig2d::Valid)
        .init(device)
}

fn crop<B: Backend>(x: Tensor<B, 4>, k: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    x.slice([0..n, 0..c, k..h - k, k..w - k])
}

#[derive(Module, Debug)]
struct SeBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> SeBlock<B> {
    fn new(in_c: usize, device: &B::Device) -> Self {
        let conv1 = conv2d(in_c, in_c / 8, 1, 1, device);
        let conv2 = conv2d(in_c / 8, in_c, 1, 1, device);
        Self { conv1, conv2 }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x0 = x.clone().mean_dims(&[2, 3]);
        let x0 = sigmoid(self.conv2.forward(relu(self.conv1.forward(x0))));
        x * x0
    }
}

#[derive(Module, Debug)]
struct UnetConv<B: Backend> {
    conv: Conv2d<B>,
    conv2: Conv2d<B>,
    seblock: Option<SeBlock<B>>,
}

impl<B: Backend> UnetConv<B> {
    fn new(in_c: usize, mid_c: usize, out_c: usize, se: bool, device: &B::Device) -> Self {
        let conv = conv2d(in_c, mid_c, 3, 1, device);
        let conv2 = conv2d(mid_c, out_c, 3, 1, device);
        let seblock = se.then(|| SeBlock::new(out_c, device));
        Self { conv, conv2, seblock }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = leaky_relu(self.conv2.forward(leaky_relu(self.conv.forward(x), 0.1)), 0.1);
        match &self.seblock {
            Some(se) => se.forward(z),
            None => z,
        }
    }
}

#[derive(Module, Debug)]
struct Unet1<B: Backend> {
    conv1: UnetConv<B>,
    conv1_down: Conv2d<B>,
    conv2: UnetConv<B>,
    conv2_up: ConvTranspose2d<B>,
    conv3: Conv2d<B>,
    conv_bottom: ConvTranspose2d<B>,
}

impl<B: Backend> Unet1<B> {
    fn new(in_c: usize, out_c: usize, device: &B::Device) -> Self {
        let conv1 = UnetConv::new(in_c, 32, 64, false, device);
        let conv1_down = conv2d(64, 64, 2, 2, device);
        let conv2 = UnetConv::new(64, 128, 64, true, device);
        let conv2_up = ConvTranspose2dConfig::new([64, 64], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .init(device);
        let conv3 = conv2d(64, 64, 3, 1, device);
        let conv_bottom = ConvTranspose2dConfig::new([64, out_c], [4, 4])
            .with_stride([2, 2])
            .with_padding([3, 3])
            .with_padding_out([0, 0])
            .init(device);
        Self { conv1, conv1_down, conv2, conv2_up, conv3, conv_bottom }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.conv1.forward(x);
        let x2 = leaky_relu(self.conv1_down.forward(x1.clone()), 0.1);
        let x1 = crop(x1, 4);
        let x2 = self.conv2.forward(x2);
        let x2 = leaky_relu(self.conv2_up.forward(x2), 0.1);
        let x3 = leaky_relu(self.conv3.forward(x1 + x2), 0.1);
        self.conv_bottom.forward(x3)
    }
}

#[derive(Module, Debug)]
struct Unet2<B: Backend> {
    conv1: UnetConv<B>,
    conv1_down: Conv2d<B>,
    conv2: UnetConv<B>,
    conv2_down: Conv2d<B>,
    conv3: UnetConv<B>,
    conv3_up: ConvTranspose2d<B>,
    conv4: UnetConv<B>,
    conv4_up: ConvTranspose2d<B>,
    conv5: Conv2d<B>,
    conv_bottom: Conv2d<B>,
}

impl<B: Backend> Unet2<B> {
    fn new(in_c: usize, out_c: usize, device: &B::Device) -> Self {
        let conv1 = UnetConv::new(in_c, 32, 64, false, device);
        let conv1_down = conv2d(64, 64, 2, 2, device);
        let conv2 = UnetConv::new(64, 64, 128, true, device);
        let conv2_down = conv2d(128, 128, 2, 2, device);
        let conv3 = UnetConv::new(128, 256, 128, true, device);
        let conv3_up = ConvTranspose2dConfig::new([128, 128], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .init(device);
        let conv4 = UnetConv::new(128, 64, 64, true, device);
        let conv4_up = ConvTranspose2dConfig::new([64, 64], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .init(device);
        let conv5 = conv2d(64, 64, 3, 1, device);
        let conv_bottom = conv2d(64, out_c, 3, 1, device);
        Self { conv1, conv1_down, conv2, conv2_down, conv3, conv3_up, conv4, conv4_up, conv5, conv_bottom }
    }

    fn forward(&self, x: Tensor<B, 4>, alpha: f64) -> Tensor<B, 4> {
        let x1 = self.conv1.forward(x);
        let x2 = self.conv1_down.forward(x1.clone());
        let x1 = crop(x1, 16);
        let x2 = leaky_relu(x2, 0.1);
        let x2 = self.conv2.forward(x2);
        let x3 = self.conv2_down.forward(x2.clone());
        let x2 = crop(x2, 4);
        let x3 = leaky_relu(x3, 0.1);
        let x3 = self.conv3.forward(x3);
        let x3 = leaky_relu(self.conv3_up.forward(x3), 0.1);
        let x4 = self.conv4.forward(x2 + x3).mul_scalar(alpha);
        let x4 = leaky_relu(self.conv4_up.forward(x4), 0.1);
        let x5 = leaky_relu(self.conv5.forward(x1 + x4), 0.1);
        self.conv_bottom.forward(x5)
    }
}

#[derive(Module, Debug)]
pub struct UpCunet2x<B: Backend> {
    unet1: Unet1<B>,
    unet2: Unet2<B>,
}

impl<B: Backend> UpCunet2x<B> {
    pub fn new(device: &B::Device) -> Self {
        Self { unet1: Unet1::<B>::new(3, 3, device), unet2: Unet2::<B>::new(3, 3, device) }
    }

    /// Full-image forward (`tile_mode=0`, `alpha=1`): reflect-pad 18, UNet1
    /// (2×), UNet2 on its output, crop 20 from UNet1, sum, trim to `2·h0 × 2·w0`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, h0, w0] = x.dims();
        let ph = ((h0 - 1) / 2 + 1) * 2;
        let pw = ((w0 - 1) / 2 + 1) * 2;
        let x = x.pad((18, 18 + pw - w0, 18, 18 + ph - h0), PadMode::Reflect);
        let u1 = self.unet1.forward(x);
        let u2 = self.unet2.forward(u1.clone(), 1.0);
        let u1 = crop(u1, 20);
        let out = u1 + u2;
        out.slice([0..n, 0..c, 0..h0 * 2, 0..w0 * 2])
    }
}

/// torch `F.pixel_unshuffle` for downscale 2: `[N, C, H, W] -> [N, C*4, H/2, W/2]`.
fn pixel_unshuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let (oh, ow) = (h / r, w / r);
    x.reshape([n, c, oh, r, ow, r])
        .permute([0, 1, 3, 5, 2, 4])
        .reshape([n, c * r * r, oh, ow])
}

/// torch `nn.PixelShuffle` for upscale 2: `[N, C, H, W] -> [N, C/4, H*2, W*2]`.
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let oc = c / (r * r);
    x.reshape([n, oc, r, r, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([n, oc, h * r, w * r])
}

/// Nearest-neighbour 2x upsample (each pixel -> 2x2 block).
pub(super) fn nearest2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    x.unsqueeze_dim::<5>(3)
        .repeat_dim(3, 2)
        .unsqueeze_dim::<6>(5)
        .repeat_dim(5, 2)
        .reshape([n, c, h * 2, w * 2])
}

/// `UpCunet2x_fast` = ShuffleCugan: pixel-unshuffle input (3->12), plain-conv
/// bottom, `conv_final` 64->12, pixel-shuffle output + nearest-2x residual.
#[derive(Module, Debug)]
pub struct UpCunet2xFast<B: Backend> {
    unet1: Unet1<B>,
    unet2: Unet2<B>,
    conv_final: Conv2d<B>,
}

impl<B: Backend> UpCunet2xFast<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            unet1: Unet1::<B>::new(12, 64, device),
            unet2: Unet2::<B>::new(64, 64, device),
            conv_final: conv2d(64, 12, 3, 1, device),
        }
    }

    /// TAS `UpCunet2x_fast.forward`: reflect-pad 38, unshuffle 2x, UNet1 ->
    /// UNet2, crop 20, sum, conv_final, crop 1, shuffle 2x, trim to
    /// `2·h0 × 2·w0`, add nearest-2x residual.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, h0, w0] = x.dims();
        let x00 = x.clone();
        let ph = ((h0 - 1) / 2 + 1) * 2;
        let pw = ((w0 - 1) / 2 + 1) * 2;
        let x = x.pad((38, 38 + pw - w0, 38, 38 + ph - h0), PadMode::Reflect);
        let x = pixel_unshuffle(x, 2);
        let u1 = self.unet1.forward(x);
        let u2 = self.unet2.forward(u1.clone(), 1.0);
        let u1 = crop(u1, 20);
        let x = self.conv_final.forward(u1 + u2);
        let x = pixel_shuffle(crop(x, 1), 2);
        let x = x.slice([0..n, 0..c, 0..h0 * 2, 0..w0 * 2]);
        x + nearest2x(x00)
    }
}
