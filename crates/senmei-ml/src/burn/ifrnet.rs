//! IFRNet (ltkong218/IFRNet, MIT) — clean burn port of the base variant.
//!
//! U-Net with two shared 4-level encoders and four coarse-to-fine decoders.
//! Flow is refined with bilinear upsampling only (no GRU in the base model);
//! frames are fused with a learned mask plus residual. The `ResBlock` uses
//! "side channels" (the last `side` channels pass through their own convs).
//! Backward bilinear warp is `align_corners=true`, border padding — identical
//! to RIFE's warp, so the grid sampling is shared. Inputs are [0,1]; the model
//! mean-subtracts per batch, re-adds the mean in the fusion and clamps to [0,1].
//!
//! Weight loading (converter): torch keys like `encoder.pyramid1.0.0.weight` /
//! `decoder4.convblock.1.conv1.0.weight` are remapped onto these field paths
//! (`encoder.p1.c0.conv` / `decoder4.cb1.c1.conv`) with capture-group rules in
//! `convert_pth_to_bpk`.

use burn::module::{Module, Param, ParamId};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Int, Tensor, TensorData};
/// Bilinear resize by a scale factor (`F.interpolate`, align_corners=false).
fn interp<B: Backend>(x: Tensor<B, 4>, scale: f32) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    let oh = ((h as f32) * scale).round() as usize;
    let ow = ((w as f32) * scale).round() as usize;
    interpolate(
        x,
        [oh, ow],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    )
}

/// Backward bilinear warp by a 2-channel flow (align_corners=true, border) —
/// matches IFRNet's `utils.warp` (same normalization as RIFE).
fn warp<B: Backend>(img: Tensor<B, 4>, flow: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, _c, h, w] = img.dims();
    let fx = flow.clone().slice([0..n, 0..1, 0..h, 0..w]);
    let fy = flow.slice([0..n, 1..2, 0..h, 0..w]);

    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, &img.device())
        .float()
        .reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, &img.device())
        .float()
        .reshape([1, 1, h, 1]);

    let sx = (xs + fx) / ((w - 1) as f64 / 2.0) - 1.0;
    let sy = (ys + fy) / ((h - 1) as f64 / 2.0) - 1.0;
    let grid = Tensor::cat(vec![sx.permute([0, 2, 3, 1]), sy.permute([0, 2, 3, 1])], 3);
    super::warp::grid_sample(img, grid)
}

fn slice_c<B: Backend>(x: Tensor<B, 4>, s: usize, e: usize) -> Tensor<B, 4> {
    let [n, _c, h, w] = x.dims();
    x.slice([0..n, s..e, 0..h, 0..w])
}

fn conv2d<B: Backend>(in_c: usize, out_c: usize, stride: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

/// Per-channel PReLU (`nn.PReLU(channels)`): `max(0,x) + w[c]*min(0,x)`.
/// Not in burn 0.21, so implemented as a tiny module with a 1D weight param.
#[derive(Module, Debug)]
pub struct Prelu<B: Backend> {
    weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> Prelu<B> {
    pub fn new(num: usize, device: &B::Device) -> Self {
        let data = TensorData::new(vec![0.25f32; num], [num]);
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::<B, 1>::from_data(data, device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let c = x.dims()[1];
        let w = self.weight.val().reshape([1, c, 1, 1]);
        x.clone().clamp_min(0.0) + w * x.clamp_max(0.0)
    }
}

#[derive(Module, Debug)]
struct ConvRelu<B: Backend> {
    conv: Conv2d<B>,
    prelu: Prelu<B>,
}

impl<B: Backend> ConvRelu<B> {
    fn new(in_c: usize, out_c: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            conv: conv2d(in_c, out_c, stride, device),
            prelu: Prelu::new(out_c, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.prelu.forward(self.conv.forward(x))
    }
}

/// Side-channel residual block: the last `side` channels pass through their
/// own convs between the full-channel convs.
#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    c1: ConvRelu<B>,
    c2: ConvRelu<B>,
    c3: ConvRelu<B>,
    c4: ConvRelu<B>,
    c5: Conv2d<B>,
    pl: Prelu<B>,
    side: usize,
}

impl<B: Backend> ResBlock<B> {
    fn new(in_c: usize, side: usize, device: &B::Device) -> Self {
        Self {
            c1: ConvRelu::new(in_c, in_c, 1, device),
            c2: ConvRelu::new(side, side, 1, device),
            c3: ConvRelu::new(in_c, in_c, 1, device),
            c4: ConvRelu::new(side, side, 1, device),
            c5: conv2d(in_c, in_c, 1, device),
            pl: Prelu::new(in_c, device),
            side,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, h, w] = x.dims();
        let side = self.side;
        let r = [0..n, (c - side)..c, 0..h, 0..w];

        let out1 = self.c1.forward(x.clone());
        let out2 = out1
            .clone()
            .slice_assign(r.clone(), self.c2.forward(out1.slice(r.clone())));
        let out3 = self.c3.forward(out2);
        let out4 = out3
            .clone()
            .slice_assign(r.clone(), self.c4.forward(out3.slice(r.clone())));
        self.pl.forward(out4 + x)
    }
}

#[derive(Module, Debug)]
struct Pyramid<B: Backend> {
    c0: ConvRelu<B>,
    c1: ConvRelu<B>,
}

impl<B: Backend> Pyramid<B> {
    fn new(in_c: usize, mid: usize, device: &B::Device) -> Self {
        Self {
            c0: ConvRelu::new(in_c, mid, 2, device),
            c1: ConvRelu::new(mid, mid, 1, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.c1.forward(self.c0.forward(x))
    }
}

#[derive(Module, Debug)]
struct Encoder<B: Backend> {
    p1: Pyramid<B>,
    p2: Pyramid<B>,
    p3: Pyramid<B>,
    p4: Pyramid<B>,
}

impl<B: Backend> Encoder<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            p1: Pyramid::new(3, 32, device),
            p2: Pyramid::new(32, 48, device),
            p3: Pyramid::new(48, 72, device),
            p4: Pyramid::new(72, 96, device),
        }
    }

    fn forward(
        &self,
        img: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let f1 = self.p1.forward(img);
        let f2 = self.p2.forward(f1.clone());
        let f3 = self.p3.forward(f2.clone());
        let f4 = self.p4.forward(f3.clone());
        (f1, f2, f3, f4)
    }
}

/// One decoder stage: `convrelu → ResBlock → ConvTranspose2d(×2)` over the
/// concat of the fused feature + warped encoder features + flows.
#[derive(Module, Debug)]
struct Decoder<B: Backend> {
    cb0: ConvRelu<B>,
    cb1: ResBlock<B>,
    cb2: ConvTranspose2d<B>,
}

impl<B: Backend> Decoder<B> {
    fn new(in_c: usize, mid: usize, side: usize, out_c: usize, device: &B::Device) -> Self {
        Self {
            cb0: ConvRelu::new(in_c, mid, 1, device),
            cb1: ResBlock::new(mid, side, device),
            cb2: ConvTranspose2dConfig::new([mid, out_c], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
        }
    }

    fn forward(&self, f_in: Tensor<B, 4>) -> Tensor<B, 4> {
        self.cb2.forward(self.cb1.forward(self.cb0.forward(f_in)))
    }
}

#[derive(Module, Debug)]
pub struct IfrNet<B: Backend> {
    encoder: Encoder<B>,
    decoder4: Decoder<B>,
    decoder3: Decoder<B>,
    decoder2: Decoder<B>,
    decoder1: Decoder<B>,
}

impl<B: Backend> IfrNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: Encoder::new(device),
            decoder4: Decoder::new(193, 192, 32, 76, device),
            decoder3: Decoder::new(220, 216, 32, 52, device),
            decoder2: Decoder::new(148, 144, 32, 36, device),
            decoder1: Decoder::new(100, 96, 32, 8, device),
        }
    }

    /// Interpolate frame `a` -> `b` at time `t` (scalar tensor in [0,1]).
    pub fn forward(&self, a: Tensor<B, 4>, b: Tensor<B, 4>, t: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, _, _, _] = a.dims();
        let mean_ = Tensor::cat(vec![a.clone(), b.clone()], 2)
            .mean_dim(1)
            .mean_dim(2)
            .mean_dim(3); // [n,1,1,1]

        let a = a - mean_.clone();
        let b = b - mean_.clone();

        let (f0_1, f0_2, f0_3, f0_4) = self.encoder.forward(a.clone());
        let (f1_1, f1_2, f1_3, f1_4) = self.encoder.forward(b.clone());

        let [_, _, h4, w4] = f0_4.dims();
        let embt = t.slice([0..n, 0..1, 0..h4, 0..w4]);

        let out4 = self
            .decoder4
            .forward(Tensor::cat(vec![f0_4, f1_4, embt], 1));
        let up_flow0_4 = slice_c(out4.clone(), 0, 2);
        let up_flow1_4 = slice_c(out4.clone(), 2, 4);
        let ft_3_ = slice_c(out4, 4, 76);

        let out3 = self.decoder3.forward(Tensor::cat(
            vec![
                ft_3_,
                warp(f0_3, up_flow0_4.clone()),
                warp(f1_3, up_flow1_4.clone()),
                up_flow0_4.clone(),
                up_flow1_4.clone(),
            ],
            1,
        ));
        let up_flow0_3 = slice_c(out3.clone(), 0, 2) + interp(up_flow0_4, 2.0) * 2.0;
        let up_flow1_3 = slice_c(out3.clone(), 2, 4) + interp(up_flow1_4, 2.0) * 2.0;
        let ft_2_ = slice_c(out3, 4, 52);

        let out2 = self.decoder2.forward(Tensor::cat(
            vec![
                ft_2_,
                warp(f0_2, up_flow0_3.clone()),
                warp(f1_2, up_flow1_3.clone()),
                up_flow0_3.clone(),
                up_flow1_3.clone(),
            ],
            1,
        ));
        let up_flow0_2 = slice_c(out2.clone(), 0, 2) + interp(up_flow0_3, 2.0) * 2.0;
        let up_flow1_2 = slice_c(out2.clone(), 2, 4) + interp(up_flow1_3, 2.0) * 2.0;
        let ft_1_ = slice_c(out2, 4, 36);

        let out1 = self.decoder1.forward(Tensor::cat(
            vec![
                ft_1_,
                warp(f0_1, up_flow0_2.clone()),
                warp(f1_1, up_flow1_2.clone()),
                up_flow0_2.clone(),
                up_flow1_2.clone(),
            ],
            1,
        ));
        let up_flow0_1 = slice_c(out1.clone(), 0, 2) + interp(up_flow0_2, 2.0) * 2.0;
        let up_flow1_1 = slice_c(out1.clone(), 2, 4) + interp(up_flow1_2, 2.0) * 2.0;
        let up_mask_1 = sigmoid(slice_c(out1.clone(), 4, 5));
        let up_res_1 = slice_c(out1, 5, 8);

        let img0_warp = warp(a, up_flow0_1);
        let img1_warp = warp(b, up_flow1_1);
        let imgt_merge = up_mask_1.clone() * img0_warp
            + (Tensor::ones_like(&up_mask_1) - up_mask_1) * img1_warp
            + mean_;
        (imgt_merge + up_res_1).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_wgpu::{Vulkan, WgpuDevice};

    #[test]
    #[ignore = "requires Vulkan; needs RUST_MIN_STACK=33554432 (burn autotune stack overflow on RADV)"]
    fn ifrnet_output_shape_matches_input() {
        let device = WgpuDevice::DiscreteGpu(0);
        let m = IfrNet::<Vulkan>::new(&device);
        let [n, c, h, w] = [1, 3, 64, 64];
        let a = Tensor::<Vulkan, 4>::from_data(
            TensorData::new(vec![0.5f32; n * c * h * w], [n, c, h, w]),
            &device,
        );
        let b = Tensor::<Vulkan, 4>::from_data(
            TensorData::new(vec![0.6f32; n * c * h * w], [n, c, h, w]),
            &device,
        );
        let t = Tensor::<Vulkan, 4>::from_data(
            TensorData::new(vec![0.5f32; h * w], [1, 1, h, w]),
            &device,
        );
        let out = m.forward(a, b, t);
        assert_eq!(out.dims(), [n, c, h, w]);
    }
}
