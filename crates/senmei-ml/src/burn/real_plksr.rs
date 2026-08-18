//! RealPLKSR re-implementation (clean port from the spandrel MIT reference).
//!
//! Partial Large Kernel CNNs for Efficient Super-Resolution
//! (https://arxiv.org/abs/2404.11848). Used by `4x-alchemy` (DySample
//! upsampling) and the 1x decompress models (`real-plksr-deh264`/`dejpg`,
//! pixel-shuffle identity). All adopted models share dim=64 / 28 blocks /
//! kernel 17 / split 0.25 / EA / GroupNorm(4) and differ only in `scale`
//! (`ModelRef::scale`) and whether the `DySample` tail is present.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::norm::{GroupNorm, GroupNormConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::{mish, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use super::warp::grid_sample_with;

fn conv2d<B: Backend>(
    in_c: usize,
    out_c: usize,
    k: usize,
    p: usize,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [k, k])
        .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
        .init(device)
}

/// Doubled Convolutional Channel Mixer: `Conv → Mish → Conv`.
#[derive(Module, Debug)]
pub struct Dccm<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> Dccm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            conv1: conv2d(dim, dim * 2, 3, 1, device),
            conv2: conv2d(dim * 2, dim, 3, 1, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv2.forward(mish(self.conv1.forward(x)))
    }
}

/// Partial Large Kernel Conv: convolves only the first `pdim` channels.
#[derive(Module, Debug)]
pub struct PlkConv2d<B: Backend> {
    conv: Conv2d<B>,
    pdim: usize,
}

impl<B: Backend> PlkConv2d<B> {
    pub fn new(pdim: usize, kernel: usize, device: &B::Device) -> Self {
        Self {
            conv: conv2d(pdim, pdim, kernel, kernel / 2, device),
            pdim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let total = x.dims()[1];
        let parts = x.split_with_sizes(vec![self.pdim, total - self.pdim], 1);
        Tensor::cat(
            vec![self.conv.forward(parts[0].clone()), parts[1].clone()],
            1,
        )
    }
}

/// Element-wise Attention: `x * sigmoid(conv(x))`.
#[derive(Module, Debug)]
pub struct Ea<B: Backend> {
    f: Conv2d<B>,
}

impl<B: Backend> Ea<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            f: conv2d(dim, dim, 3, 1, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x.clone() * sigmoid(self.f.forward(x))
    }
}

/// One PLK block: channel mixer → partial large-kernel conv → attention →
/// refine → group norm, with a residual skip.
#[derive(Module, Debug)]
pub struct PlkBlock<B: Backend> {
    channel_mixer: Dccm<B>,
    lk: PlkConv2d<B>,
    attn: Ea<B>,
    refine: Conv2d<B>,
    norm: GroupNorm<B>,
}

impl<B: Backend> PlkBlock<B> {
    pub fn new(
        dim: usize,
        pdim: usize,
        kernel: usize,
        norm_groups: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            channel_mixer: Dccm::new(dim, device),
            lk: PlkConv2d::new(pdim, kernel, device),
            attn: Ea::new(dim, device),
            refine: conv2d(dim, dim, 1, 0, device),
            norm: GroupNormConfig {
                num_groups: norm_groups,
                num_channels: dim,
                epsilon: 1e-5,
                affine: true,
            }
            .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let skip = x.clone();
        let h = self.channel_mixer.forward(x);
        let h = self.lk.forward(h);
        let h = self.attn.forward(h);
        let h = self.refine.forward(h);
        let h = group_norm(h, &self.norm);
        h + skip
    }
}

/// Numerically stable GroupNorm for f16 backends.
///
/// burn's `GroupNorm` divides the sum by the group element count via
/// `div_scalar`, whose f16 reciprocal underflows to 0 for large groups
/// (per-group count ≥ 2^14, e.g. 16·64·64 = 65536), so the mean/var collapse
/// to 0 and the output blows up. `mean_dim` runs a scaled kernel instead and
/// stays accurate, so the mean and variance are computed with it.
fn group_norm<B: Backend>(x: Tensor<B, 4>, norm: &GroupNorm<B>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let groups = norm.num_groups;
    let per = (c / groups) * h * w; // elements per group
    let xg = x.reshape([b, groups, per]);
    let mean = xg.clone().mean_dim(2); // [B, groups]
    let xc = xg.clone() - mean;
    let var = xc.clone().square().mean_dim(2); // [B, groups]
    let inv_std = (var + norm.epsilon).sqrt().recip();
    let out = xc * inv_std;
    let out = out.reshape([b, c, h, w]);
    match (&norm.gamma, &norm.beta) {
        (Some(g), Some(beta)) => out
            .mul(g.val().reshape([1, c, 1, 1]))
            .add(beta.val().reshape([1, c, 1, 1])),
        _ => out,
    }
}

/// Full model: `head → blocks → tail → (+repeat_interleave) → DySample`.
///
/// Record keys are `head.*`, `blocks.{i}.*`, `tail.*` plus the DySample convs
/// `offset.*`/`scope.*`/`end_conv.*` (present only for scale != 1) — the
/// converter maps the torch `feats.{i}` / `to_img.` keys onto them. scale=1
/// models have no DySample (PixelShuffle(1) identity), so only `feats` keys.
#[derive(Module, Debug)]
pub struct RealPlk<B: Backend> {
    head: Conv2d<B>,
    blocks: Vec<PlkBlock<B>>,
    tail: Conv2d<B>,
    offset: Option<Conv2d<B>>,
    scope: Option<Conv2d<B>>,
    end_conv: Option<Conv2d<B>>,
    scale: usize,
}

impl<B: Backend> RealPlk<B> {
    pub fn new(scale: usize, device: &B::Device) -> Self {
        let dim = 64;
        let in_ch = 3 * scale * scale;
        let dysample = scale > 1;
        let groups = if scale % 2 != 0 { 3 } else { 4 };
        let out = 2 * groups * scale * scale;
        Self {
            head: conv2d(3, dim, 3, 1, device),
            blocks: (0..28)
                .map(|_| PlkBlock::new(dim, dim / 4, 17, 4, device))
                .collect(),
            tail: conv2d(dim, in_ch, 3, 1, device),
            offset: dysample.then(|| conv2d(in_ch, out, 1, 0, device)),
            scope: dysample.then(|| {
                Conv2dConfig::new([in_ch, out], [1, 1])
                    .with_bias(false)
                    .init(device)
            }),
            end_conv: dysample.then(|| conv2d(in_ch, 3, 1, 0, device)),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = self.head.forward(x.clone());
        for b in &self.blocks {
            h = b.forward(h);
        }
        let h = self.tail.forward(h);
        let s2 = self.scale * self.scale;
        let h = h + repeat_interleave(x, s2);
        let groups = if self.scale % 2 != 0 { 3 } else { 4 };
        match (&self.offset, &self.scope, &self.end_conv) {
            (Some(o), Some(s), Some(e)) => dysample_forward(h, o, s, e, self.scale, groups),
            _ => pixel_shuffle(h, self.scale), // scale=1 → identity
        }
    }
}

/// Repeat each channel `n` times consecutively (torch `repeat_interleave` on
/// the channel dim). Burn's `repeat`/`reshape` interleaves copies at the wrong
/// level (`[c0,c1,c0,c1]` instead of `[c0,c0,c1,c1]`), so build it explicitly.
fn repeat_interleave<B: Backend>(x: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    let [_, c, _, _] = x.dims();
    let mut parts = Vec::with_capacity(c * n);
    for ci in 0..c {
        let ch = x.clone().narrow(1, ci, 1); // [B, 1, H, W]
        for _ in 0..n {
            parts.push(ch.clone());
        }
    }
    Tensor::cat(parts, 1)
}

/// torch `pixel_shuffle(x, r)`: `[N, C*r², H, W] → [N, C, H*r, W*r]`.
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let c_out = c / (r * r);
    x.reshape([b, c_out, r, r, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, c_out, h * r, w * r])
}

/// The DySample upsampler (from the spandrel reference): predicts per-pixel
/// sampling offsets with two 1×1 convs, builds a normalized grid and bilinear
/// grid-samples the feature map (`align_corners=False`, border padding).
fn dysample_forward<B: Backend>(
    x: Tensor<B, 4>,
    offset: &Conv2d<B>,
    scope: &Conv2d<B>,
    end_conv: &Conv2d<B>,
    scale: usize,
    groups: usize,
) -> Tensor<B, 4> {
    let [b, in_ch, h, w] = x.dims();
    let device = x.device();
    let s2 = scale * scale;

    // init_pos: [1, 2*groups*scale², 1, 1] — the base sampling offsets.
    let init_pos = init_pos(scale, groups, &device);

    let off = offset.forward(x.clone()) * sigmoid(scope.forward(x.clone())) * 0.5 + init_pos;
    // off: [B, 2*groups*s2, H, W] → [B, 2, groups*s2, H, W]
    let off = off.reshape([b, 2, groups * s2, h, w]);

    // Base coordinates [1, 2, 1, H, W]: channel 0 = W (x), channel 1 = H (y).
    let coords = coords_grid(h, w, &device);
    // normalizer per coordinate channel: [1/W, 1/H].
    let norm = Tensor::<B, 1>::from_data(
        TensorData::new(vec![1.0 / w as f32, 1.0 / h as f32], [2]).convert::<B::FloatElem>(),
        &device,
    )
    .reshape([1, 2, 1, 1, 1]);

    let coords = (coords + off) * 2.0 * norm - 1.0;
    // coords: [B, 2, groups*s2, H, W] → [B, 2*groups*s2, H, W] → pixel_shuffle
    let coords = pixel_shuffle(coords.reshape([b, 2 * groups * s2, h, w]), scale);
    // [B, 2*groups, scaleH, scaleW] → [B, 2, groups, scaleH, scaleW]
    // → permute → [B, groups, scaleH, scaleW, 2] → flatten [B*groups, ..., 2]
    let coords = coords
        .reshape([b, 2, groups, h * scale, w * scale])
        .permute([0, 2, 3, 4, 1])
        .reshape([b * groups, h * scale, w * scale, 2]);

    let xg = x.reshape([b * groups, in_ch / groups, h, w]);
    let out = grid_sample_with(xg, coords, false);
    // [B*groups, in_ch/groups, scaleH, scaleW] → [B, in_ch, scaleH, scaleW]
    let out = out.reshape([b, in_ch, h * scale, w * scale]);
    end_conv.forward(out)
}

/// The DySample `init_pos` buffer, recomputed deterministically (not in the
/// state dict). Matches the reference `stack(meshgrid(h,h,ij)).transpose(1,2)
/// .repeat(1, groups, 1).reshape(1,-1,1,1)` exactly: the flatten index `i`
/// maps to coord = i/(g·n²), and the value is `h[i % n]` for coord 0 (y) and
/// `h[(i/n) % n]` for coord 1 (x). Verified element-wise against the pth.
fn init_pos<B: Backend>(scale: usize, groups: usize, device: &B::Device) -> Tensor<B, 4> {
    let n = scale;
    let h: Vec<f32> = (0..n)
        .map(|i| (i as f32 - (n as f32 - 1.0) / 2.0) / n as f32)
        .collect();
    let total = 2 * groups * n * n;
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        let coord = i / (groups * n * n);
        let rem = i % (groups * n * n);
        data.push(if coord == 0 {
            h[rem % n]
        } else {
            h[(rem / n) % n]
        });
    }
    let data = TensorData::new(data, [1, total, 1, 1]).convert::<B::FloatElem>();
    Tensor::<B, 4>::from_data(data, device)
}

/// Base sampling coordinates `[1, 2, 1, H, W]` with channel 0 = W and
/// channel 1 = H (each `coords + 0.5`), matching the reference meshgrid.
fn coords_grid<B: Backend>(h: usize, w: usize, device: &B::Device) -> Tensor<B, 5> {
    let cw: Vec<f32> = (0..w).map(|i| i as f32 + 0.5).collect();
    let ch: Vec<f32> = (0..h).map(|i| i as f32 + 0.5).collect();
    // channel 0: W coords repeated over H rows
    let mut data = Vec::with_capacity(2 * h * w);
    for coord in 0..2 {
        for y in 0..h {
            for x in 0..w {
                data.push(if coord == 0 { cw[x] } else { ch[y] });
            }
        }
    }
    let data = TensorData::new(data, [1, 2, 1, h, w]).convert::<B::FloatElem>();
    Tensor::<B, 5>::from_data(data, device)
}
