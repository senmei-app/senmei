//! SAFMN (Spatially-Adaptive Feature Modulation Network) re-implemented in
//! burn. Clean port from the Apache-2.0 `sunny2109/SAFMN` reference (spandrel
//! `SAFMN` arch). Used by the `SAFMN_L_Real_LSDIR_x2/x4-v2` checkpoints
//! (dim 128 / 16 blocks / ffn_scale 2.0 / SAFM n_levels 4).
//!
//! Body: `to_feat(3→dim, 3×3)` → 16× `AttBlock` (global residual) →
//! `to_img(dim→3·scale², 3×3)` + PixelShuffle(scale). Each `AttBlock` is a
//! two-stage residual: SAFM over LayerNorm'd input, then CCM over the second
//! LayerNorm. SAFM splits the channels into 4 groups, runs each through a
//! depthwise 3×3 (`mfr`), max-pools groups 1-3 to `h/2^i` before the conv and
//! nearest-upsamples back, concatenates, aggregates 1×1, GELU, and gates the
//! input (spatial-adaptive feature modulation).

use burn::module::{Module, Param, ParamId};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::module::{interpolate, max_pool2d};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Tensor, TensorData};

fn conv2d<B: Backend>(
    in_c: usize,
    out_c: usize,
    k: usize,
    p: usize,
    groups: usize,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [k, k])
        .with_padding(PaddingConfig2d::Explicit(p, p, p, p))
        .with_groups(groups)
        .init(device)
}

/// PixelShuffle: `[N, C·r², H, W] → [N, C, H·r, W·r]`. The permute matches
/// torch's `pixel_shuffle` (view [N,C,r,r,H,W] → permute (0,1,4,2,5,3)).
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let oc = c / (r * r);
    x.reshape([n, oc, r, r, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([n, oc, h * r, w * r])
}

/// Channel-wise LayerNorm (per spatial location, over C, channels_first),
/// affine per channel, eps 1e-6. Computes the variance in a `x/S`-scaled
/// domain (fp16-safe, same trick as NAFNet) and rescales.
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(c: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32; c], [c]), device),
            ),
            bias: Param::initialized(ParamId::new(), Tensor::<B, 1>::zeros([c], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>, eps: f32) -> Tensor<B, 4> {
        let [_, c, _, _] = x.dims();
        let s: f32 = 128.0;
        let mu = (x.clone() / s).mean_dim(1) * s; // [n,1,h,w]
        let d = x - mu;
        let ds = d.clone() / s;
        let m = (ds.clone() * ds).mean_dim(1); // var / s^2
        let inv = (m.clamp_min(1e-7) + eps / (s * s)).sqrt().recip() / s;
        (d * inv) * self.weight.val().reshape([1, c, 1, 1]) + self.bias.val().reshape([1, c, 1, 1])
    }
}

/// Conv → GELU → Conv channel mixer (`ccm.ccm` in torch).
#[derive(Module, Debug)]
pub struct Ccm<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> Ccm<B> {
    pub fn new(dim: usize, ffn_scale: f32, device: &B::Device) -> Self {
        let hidden = (dim as f32 * ffn_scale) as usize;
        Self {
            conv1: conv2d(dim, hidden, 3, 1, 1, device),
            conv2: conv2d(hidden, dim, 1, 0, 1, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv2.forward(gelu(self.conv1.forward(x)))
    }
}

/// Spatial-Adaptive Feature Modulation: 4 depthwise convs on channel groups,
/// groups 1-3 pooled to `h/2^i` before and nearest-upsampled back, gated by
/// the GELU-activated aggregate.
#[derive(Module, Debug)]
pub struct Safm<B: Backend> {
    mfr: Vec<Conv2d<B>>,
    aggr: Conv2d<B>,
    n_levels: usize,
}

impl<B: Backend> Safm<B> {
    pub fn new(dim: usize, n_levels: usize, device: &B::Device) -> Self {
        let chunk = dim / n_levels;
        Self {
            mfr: (0..n_levels)
                .map(|_| conv2d(chunk, chunk, 3, 1, chunk, device))
                .collect(),
            aggr: conv2d(dim, dim, 1, 0, 1, device),
            n_levels,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, c, h, w] = x.dims();
        let chunk = c / self.n_levels;
        let parts = x.clone().split_with_sizes(vec![chunk; self.n_levels], 1);
        let mut out = Vec::with_capacity(self.n_levels);
        for i in 0..self.n_levels {
            if i > 0 {
                let k = 2usize.pow(i as u32);
                // h/w are multiples of 8 (padded in `SafmnNet::forward`), so
                // kernel=stride=2^i is an exact adaptive max-pool to h/2^i.
                let pooled = max_pool2d(parts[i].clone(), [k, k], [k, k], [0, 0], [1, 1], false);
                let s = self.mfr[i].forward(pooled);
                out.push(interpolate(
                    s,
                    [h, w],
                    InterpolateOptions::new(InterpolateMode::Nearest),
                ));
            } else {
                out.push(self.mfr[i].forward(parts[i].clone()));
            }
        }
        let agg = self.aggr.forward(Tensor::cat(out, 1));
        gelu(agg) * x
    }
}

/// Two-stage residual block: SAFM over norm1, CCM over norm2.
#[derive(Module, Debug)]
pub struct AttBlock<B: Backend> {
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    safm: Safm<B>,
    ccm: Ccm<B>,
}

impl<B: Backend> AttBlock<B> {
    pub fn new(dim: usize, ffn_scale: f32, device: &B::Device) -> Self {
        Self {
            norm1: LayerNorm::new(dim, device),
            norm2: LayerNorm::new(dim, device),
            safm: Safm::new(dim, 4, device),
            ccm: Ccm::new(dim, ffn_scale, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.safm.forward(self.norm1.forward(x.clone(), 1e-6)) + x.clone();
        self.ccm.forward(self.norm2.forward(y.clone(), 1e-6)) + y
    }
}

/// SAFMN (dim 128 / 16 blocks / ffn_scale 2.0 for the registered Real models).
/// Input H/W are edge-padded to a multiple of 8 (SAFM's `h/2^i` pools need
/// exact division), the output is cropped back to `h·scale × w·scale`.
#[derive(Module, Debug)]
pub struct SafmnNet<B: Backend> {
    to_feat: Conv2d<B>,
    blocks: Vec<AttBlock<B>>,
    to_img_conv: Conv2d<B>,
    scale: usize,
}

impl<B: Backend> SafmnNet<B> {
    pub fn new(
        dim: usize,
        n_blocks: usize,
        ffn_scale: f32,
        scale: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            to_feat: conv2d(3, dim, 3, 1, 1, device),
            blocks: (0..n_blocks)
                .map(|_| AttBlock::new(dim, ffn_scale, device))
                .collect(),
            to_img_conv: conv2d(dim, 3 * scale * scale, 3, 1, 1, device),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, _, h, w] = x.dims();
        let (ph, pw) = (h.div_ceil(8) * 8, w.div_ceil(8) * 8);
        let mut y = x;
        if ph != h || pw != w {
            y = y.pad([(0, 0), (0, 0), (0, ph - h), (0, pw - w)], PadMode::Edge);
        }
        let f = self.to_feat.forward(y);
        let mut body = f.clone();
        for b in &self.blocks {
            body = b.forward(body);
        }
        let out = pixel_shuffle(self.to_img_conv.forward(body + f), self.scale);
        out.slice([0..n, 0..3, 0..h * self.scale, 0..w * self.scale])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    /// Numerical check of the SAFMN port against torch:
    /// `tools/safmn_verify.py` writes `x.bin`/`ref.bin` (f32, 32×32 → scaled).
    /// Converts the .pth to a .bpk with `senmei-ml-convert safmn ...` first.
    #[test]
    #[ignore = "needs GPU + converted safmn bpk + torch ref bins (tools/safmn_verify.py); needs RUST_MIN_STACK=33554432"]
    fn safmn_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir =
            std::env::var("SENMEI_SAFMN_VERIFY_DIR").unwrap_or_else(|_| "/tmp/safmn_verify".into());
        let scale = std::env::var("SENMEI_SAFMN_SCALE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        let read = |name: &str, n: usize| -> Vec<f32> {
            let data = std::fs::read(format!("{dir}/{name}")).expect("missing ref bin");
            assert_eq!(data.len(), n * 4, "bad {name} size");
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let [n, c, h, w] = [1usize, 3, 32, 32];
        let x_v = read("x.bin", n * c * h * w);
        let ref_v = read("ref.bin", n * c * h * scale * w * scale);

        let mut m = SafmnNet::<BurnBackend<f16>>::new(128, 16, 2.0, scale, &device);
        let mut store = BurnpackStore::from_file(format!("{dir}/safmn_x{scale}.f16.bpk"));
        let res = m.load_from(&mut store).unwrap();
        println!(
            "load: applied={} missing={} unused={}",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
        );
        for u in &res.unused {
            println!("  unused {u}");
        }
        assert!(res.missing.is_empty(), "missing tensors: {:?}", res.missing);

        let x = Tensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(x_v, [n, c, h, w]).convert::<f16>(),
            &device,
        );
        let out = m.forward(x);
        let out_v = out.into_data().convert::<f32>().to_vec().unwrap();
        let mae: f32 = out_v
            .iter()
            .zip(&ref_v)
            .map(|(a, b): (&f32, &f32)| (a - b).abs())
            .sum::<f32>()
            / ref_v.len() as f32;
        println!("mae vs torch = {mae}");
        // x4 accumulates more fp16 error over the larger output (0.0084 x2 vs
        // 0.027 x4 on the worst-case random input); real frames are lower.
        let tol = if scale >= 4 { 0.035 } else { 0.02 };
        assert!(mae < tol, "mae too high: {mae}");
    }

    /// Smoke check: dim 128 / 16 blocks upscales by `scale` and stays finite
    /// on a non-multiple-of-8 input (padding path is exercised).
    #[test]
    #[ignore = "needs GPU + converted safmn bpk (senmei-ml-convert safmn)"]
    fn safmn_forward_upscales() {
        let device = WgpuDevice::DiscreteGpu(0);
        let mut m = SafmnNet::<BurnBackend<f16>>::new(128, 16, 2.0, 2, &device);
        let mut store = BurnpackStore::from_file("/tmp/safmn/safmn_x2.f16.bpk");
        let res = m.load_from(&mut store).unwrap();
        assert!(res.missing.is_empty(), "missing tensors: {:?}", res.missing);

        let (h, w): (usize, usize) = (17, 25); // not a multiple of 8
        let x = Tensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(
                (0..3 * h * w)
                    .map(|i| ((i % w) as f32 / w as f32) * 0.5 + 0.25)
                    .collect(),
                [1, 3, h, w],
            )
            .convert::<f16>(),
            &device,
        );
        let out = m.forward(x);
        assert_eq!(out.dims(), [1, 3, h * 2, w * 2]);
        assert!(
            out.into_data()
                .to_vec::<f16>()
                .unwrap()
                .iter()
                .all(|v| v.to_f32().is_finite()),
            "non-finite output"
        );
    }
}
