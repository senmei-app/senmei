//! SPAN (Swift Parameter-free Attention Network) — clean burn port from the
//! Apache-2.0 BasicSR reference (hongyuanyu/SPAN). Load: TNTwise checkpoints
//! keep the Conv3XC training branch (stale fused `eval_conv` ignored). f16-safe
//! on real frames (overflow only on synthetic noise); bf16 broken on RADV.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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

/// Conv3XC: 1×1 → 3×3 → 1×1 plus a 1×1 skip (gain1 = 2).
#[derive(Module, Debug)]
pub struct Conv3Xc<B: Backend> {
    conv0: Conv2d<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    sk: Conv2d<B>,
}

impl<B: Backend> Conv3Xc<B> {
    pub fn new(c_in: usize, c_out: usize, device: &B::Device) -> Self {
        Self {
            conv0: conv2d(c_in, c_in * 2, 1, 0, device),
            conv1: conv2d(c_in * 2, c_out * 2, 3, 1, device),
            conv2: conv2d(c_out * 2, c_out, 1, 0, device),
            sk: conv2d(c_in, c_out, 1, 0, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv2.forward(self.conv1.forward(self.conv0.forward(x.clone())));
        out + self.sk.forward(x)
    }
}

/// SPAB: three Conv3XC with SiLU, plus `sigmoid(out3) - 0.5` gating.
#[derive(Module, Debug)]
pub struct Spab<B: Backend> {
    c1_r: Conv3Xc<B>,
    c2_r: Conv3Xc<B>,
    c3_r: Conv3Xc<B>,
}

impl<B: Backend> Spab<B> {
    pub fn new(ch: usize, device: &B::Device) -> Self {
        Self {
            c1_r: Conv3Xc::new(ch, ch, device),
            c2_r: Conv3Xc::new(ch, ch, device),
            c3_r: Conv3Xc::new(ch, ch, device),
        }
    }

    /// `(out, out1, att)`; `out1` (pre-SiLU) feeds the head concat.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let out1 = self.c1_r.forward(x.clone());
        let out1_act = silu(out1.clone());
        let out2 = self.c2_r.forward(out1_act);
        let out2_act = silu(out2);
        let out3 = self.c3_r.forward(out2_act);
        let att = sigmoid(out3.clone()).sub_scalar(0.5);
        let out = (out3 + x) * att.clone();
        (out, out1, att)
    }
}

/// torch `pixel_shuffle(x, r)`: `[N, C*r², H, W] → [N, C, H*r, W*r]`.
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let c_out = c / (r * r);
    x.reshape([b, c_out, r, r, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, c_out, h * r, w * r])
}

/// SPAN: head conv → 6 SPAB → tail conv → 4-way concat → pixel-shuffle head.
#[derive(Module, Debug)]
pub struct Span<B: Backend> {
    conv_1: Conv3Xc<B>,
    block_1: Spab<B>,
    block_2: Spab<B>,
    block_3: Spab<B>,
    block_4: Spab<B>,
    block_5: Spab<B>,
    block_6: Spab<B>,
    conv_2: Conv3Xc<B>,
    conv_cat: Conv2d<B>,
    upsampler: Conv2d<B>,
    scale: usize,
}

impl<B: Backend> Span<B> {
    pub fn new(ch: usize, scale: usize, device: &B::Device) -> Self {
        Self {
            conv_1: Conv3Xc::new(3, ch, device),
            block_1: Spab::new(ch, device),
            block_2: Spab::new(ch, device),
            block_3: Spab::new(ch, device),
            block_4: Spab::new(ch, device),
            block_5: Spab::new(ch, device),
            block_6: Spab::new(ch, device),
            conv_2: Conv3Xc::new(ch, ch, device),
            conv_cat: conv2d(ch * 4, ch, 1, 0, device),
            upsampler: conv2d(ch, 3 * scale * scale, 3, 1, device),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // (x - mean) * 255, mean (0.4488, 0.4371, 0.4040) — checkpoints carry no_norm=0.
        let mean = Tensor::<B, 1>::from_floats([0.4488, 0.4371, 0.4040], &x.device())
            .reshape([1, 3, 1, 1]);
        let x = (x - mean).mul_scalar(255.0);

        let feat = self.conv_1.forward(x);
        let (b1, _, _) = self.block_1.forward(feat.clone());
        let (b2, _, _) = self.block_2.forward(b1.clone());
        let (b3, _, _) = self.block_3.forward(b2);
        let (b4, _, _) = self.block_4.forward(b3);
        let (b5, _, _) = self.block_5.forward(b4);
        let (b6, b5_2, _) = self.block_6.forward(b5);
        let b6 = self.conv_2.forward(b6);
        let cat = Tensor::cat(vec![feat, b6, b1, b5_2], 1);
        let out = self.upsampler.forward(self.conv_cat.forward(cat));
        pixel_shuffle(out, self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    #[test]
    #[ignore = "needs Vulkan + /tmp/senmei_models/span_v2.f16.bpk + torch ref bins; needs RUST_MIN_STACK=33554432"]
    fn span_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir = "/tmp/senmei_models";
        let read = |name: &str, n: usize| -> Vec<f32> {
            let data = std::fs::read(format!("{dir}/{name}")).expect("missing ref bin");
            assert_eq!(data.len(), n * 4, "bad {name} size");
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let [n, c, h, w] = [1usize, 3, 64, 64];
        let x_v = read("span_in.bin", n * c * h * w);
        let ref_v = read("span_ref.bin", n * c * 4 * h * w);

        let mut m = Span::<BurnBackend<f16>>::new(48, 2, &device);
        let mut store = BurnpackStore::from_file("/tmp/senmei_models/span_v2.f16.bpk");
        let res = m.load_from(&mut store).unwrap();
        println!(
            "load: applied={} missing={} unused={}",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
        );
        for (p, _) in &res.missing {
            println!("  missing {p}");
        }

        let x = Tensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(x_v, [n, c, h, w]).convert::<f16>(),
            &device,
        );
        let finite = |t: &Tensor<BurnBackend<f16>, 4>, name: &str| {
            let v: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap();
            let (nans, infs) = v.iter().fold((0usize, 0usize), |(a, b), f| {
                (a + f.is_nan() as usize, b + f.is_infinite() as usize)
            });
            let mn = v.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!("{name}: nan={nans} inf={infs} min={mn:.3} max={mx:.3}");
        };

        let mean = Tensor::<BurnBackend<f16>, 1>::from_floats([0.4488, 0.4371, 0.4040], &device)
            .reshape([1, 3, 1, 1]);
        let xn = (x - mean).mul_scalar(255.0);
        finite(&xn, "norm");
        let feat = m.conv_1.forward(xn);
        finite(&feat, "conv_1");
        let (b1, _, _) = m.block_1.forward(feat.clone());
        finite(&b1, "block_1");
        let (b2, _, _) = m.block_2.forward(b1.clone());
        finite(&b2, "block_2");
        let (b3, _, _) = m.block_3.forward(b2);
        finite(&b3, "block_3");
        let (b4, _, _) = m.block_4.forward(b3);
        finite(&b4, "block_4");
        let (b5, _, _) = m.block_5.forward(b4);
        finite(&b5, "block_5");
        let (b6, b5_2, _) = m.block_6.forward(b5);
        finite(&b6, "block_6");
        let b6 = m.conv_2.forward(b6);
        finite(&b6, "conv_2");
        let cat = Tensor::cat(vec![feat, b6, b1, b5_2], 1);
        finite(&cat, "cat");
        let cc = m.conv_cat.forward(cat);
        finite(&cc, "conv_cat");
        let up = m.upsampler.forward(cc);
        finite(&up, "upsampler");
        let out = pixel_shuffle(up, 2);
        finite(&out, "pixel_shuffle");

        let out_v: Vec<f32> = out.into_data().convert::<f32>().to_vec().unwrap();
        let mae: f32 = out_v
            .iter()
            .zip(&ref_v)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / out_v.len() as f32;
        let maxe = out_v
            .iter()
            .zip(&ref_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("mae={mae:.6} max={maxe:.6}");
        assert!(mae < 5e-3, "mae too high: {mae}");
    }
}
