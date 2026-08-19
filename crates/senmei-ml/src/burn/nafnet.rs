//! NAFNet (megvii-research/NAFNet, Apache-2.0) — clean burn port of the
//! GoPro-width32 deblur variant.
//!
//! U-Net of `NAFBlock`s: LayerNorm2d → conv1(1×1, c→2c) → depthwise conv2(3×3)
//! → SimpleGate (channel split × multiply) → SCA (`x * conv1x1(avgpool(x))`,
//! no sigmoid) → conv3, residual scaled by `beta`; then FFN (norm2 → conv4 →
//! SimpleGate → conv5) scaled by `gamma`. Encoder downs are stride-2 convs,
//! decoder ups are Conv1×1 + PixelShuffle(2), and the output is `ending + inp`
//! (padded to multiples of 16, cropped back). Pure conv/gate graph — no
//! cross-channel slicing beyond the gate split, no fusion issue.
//!
//! Weight loading (converter): torch keys like `encoders.0.0.conv1.weight` /
//! `encoders.0.0.sca.1.weight` / `middle_blks.0.conv1.weight` /
//! `ups.0.0.weight` / `downs.0.weight` are remapped onto these field paths
//! (`encoders.0.blocks.0.conv1` / `encoders.0.blocks.0.sca_conv` /
//! `middle.0.conv1` / `ups.0.conv` / `downs.0`) with capture-group rules in
//! `convert_pth_to_bpk`.

use burn::module::{Module, Param, ParamId};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

fn conv1x1<B: Backend>(in_c: usize, out_c: usize, bias: bool, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [1, 1])
        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
        .with_bias(bias)
        .init(device)
}

fn conv3x3<B: Backend>(in_c: usize, out_c: usize, groups: usize, device: &B::Device) -> Conv2d<B> {
    // Depthwise (groups = out) or plain 3×3, torch `padding=1`.
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_groups(groups)
        .with_bias(true)
        .init(device)
}

/// Channel-wise LayerNorm2d (per spatial location, over C), affine per channel.
/// Computes the variance in a `x/S`-scaled domain (fp16-safe at the 512-channel
/// bottleneck where |x| can be ~175) and rescales.
#[derive(Module, Debug)]
struct LayerNorm2d<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerNorm2d<B> {
    fn new(c: usize, device: &B::Device) -> Self {
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
        let s: f32 = 128.0; // scale for the fp16-safe reduction (|x|~682 at the
                            // bottleneck; sums would overflow fp16 otherwise)
        let mu = (x.clone() / s).mean_dim(1) * s; // [n,1,h,w]
        let d = x - mu;
        let ds = d.clone() / s;
        let m = (ds.clone() * ds).mean_dim(1); // var / s^2
                                               // fp16 loses `eps/s^2` (6e-11 underflows), so clamp m to a small
                                               // representable floor; near-constant channels (m≈0 ⇒ d≈0) then get a
                                               // finite reciprocal and contribute ~0, matching torch's `var+eps`.
        let inv = (m.clamp_min(1e-7) + eps / (s * s)).sqrt().recip() / s; // 1/sqrt(var+eps)
        (d * inv) * self.weight.val().reshape([1, c, 1, 1]) + self.bias.val().reshape([1, c, 1, 1])
    }
}

/// `x.chunk(2, dim=1)` → elementwise multiply (channel gate).
fn simple_gate<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let half = c / 2;
    let a = x.clone().slice([0..n, 0..half, 0..h, 0..w]);
    let b = x.slice([0..n, half..c, 0..h, 0..w]);
    a * b
}

/// NAFBlock: gated conv block with scaled residuals (`beta`/`gamma`).
#[derive(Module, Debug)]
struct NafBlock<B: Backend> {
    conv1: Conv2d<B>,    // 1×1, c→2c
    conv2: Conv2d<B>,    // 3×3 depthwise
    conv3: Conv2d<B>,    // 1×1, c→c
    sca_conv: Conv2d<B>, // SCA: 1×1 on the avg-pooled channel stats
    conv4: Conv2d<B>,    // 1×1, c→2c
    conv5: Conv2d<B>,    // 1×1, c→c
    norm1: LayerNorm2d<B>,
    norm2: LayerNorm2d<B>,
    beta: Param<Tensor<B, 4>>, // [1,c,1,1]
    gamma: Param<Tensor<B, 4>>,
}

impl<B: Backend> NafBlock<B> {
    fn new(c: usize, device: &B::Device) -> Self {
        Self {
            conv1: conv1x1(c, 2 * c, true, device),
            conv2: conv3x3(2 * c, 2 * c, 2 * c, device),
            conv3: conv1x1(c, c, true, device),
            sca_conv: conv1x1(c, c, true, device),
            conv4: conv1x1(c, 2 * c, true, device),
            conv5: conv1x1(c, c, true, device),
            norm1: LayerNorm2d::new(c, device),
            norm2: LayerNorm2d::new(c, device),
            beta: Param::initialized(ParamId::new(), Tensor::<B, 4>::zeros([1, c, 1, 1], device)),
            gamma: Param::initialized(ParamId::new(), Tensor::<B, 4>::zeros([1, c, 1, 1], device)),
        }
    }

    fn forward(&self, inp: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.norm1.forward(inp.clone(), 1e-6);
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = simple_gate(x);
        // SCA avg-pool: mean over H×W; scale the reduction (fp16 sums over
        // ~4k spatial elements of large activations overflow).
        let pooled = (x.clone() / 64.0).mean_dim(2).mean_dim(3) * 64.0;
        let x = x.clone() * self.sca_conv.forward(pooled);
        let x = self.conv3.forward(x);
        let y = inp + x * self.beta.val();

        let x = self.conv4.forward(self.norm2.forward(y.clone(), 1e-6));
        let x = simple_gate(x);
        let x = self.conv5.forward(x);
        y + x * self.gamma.val()
    }
}

#[derive(Module, Debug)]
struct NafEnc<B: Backend> {
    blocks: Vec<NafBlock<B>>,
}

#[derive(Module, Debug)]
struct NafDec<B: Backend> {
    blocks: Vec<NafBlock<B>>,
}

impl<B: Backend> NafEnc<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.blocks.iter().fold(x, |acc, b| b.forward(acc))
    }
}

impl<B: Backend> NafDec<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.blocks.iter().fold(x, |acc, b| b.forward(acc))
    }
}

/// Decoder up: Conv1×1 (c→2c) + PixelShuffle(2) → c at 2× spatial.
#[derive(Module, Debug)]
struct NafUp<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> NafUp<B> {
    fn new(c: usize, device: &B::Device) -> Self {
        Self {
            conv: conv1x1(c, 2 * c, false, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, _c, h, w] = x.dims();
        let y = self.conv.forward(x);
        let [_, c4, _, _] = y.dims();
        let c = c4 / 4;
        // depth-to-space (PixelShuffle 2): [n, c, 2, 2, h, w] → [n, c, 2h, 2w]
        y.reshape([n, c, 2, 2, h, w])
            .permute([0, 1, 4, 2, 5, 3])
            .reshape([n, c, 2 * h, 2 * w])
    }
}

#[derive(Module, Debug)]
pub struct NafNet<B: Backend> {
    intro: Conv2d<B>,
    ending: Conv2d<B>,
    encoders: [NafEnc<B>; 4],
    downs: [Conv2d<B>; 4],
    middle: Vec<NafBlock<B>>,
    ups: [NafUp<B>; 4],
    decoders: [NafDec<B>; 4],
}

impl<B: Backend> NafNet<B> {
    /// NAFNet-GoPro-width32: 4 encoder levels (1/1/1/28 blocks) at
    /// 32/64/128/256, 1 middle block at 512, 4 decoders at 256/128/64/32.
    pub fn new(device: &B::Device) -> Self {
        Self {
            intro: conv3x3(3, 32, 1, device),
            ending: conv3x3(32, 3, 1, device),
            encoders: NafNet::enc([1, 1, 1, 28], [32, 64, 128, 256], device),
            downs: [
                conv2s2(32, 64, device),
                conv2s2(64, 128, device),
                conv2s2(128, 256, device),
                conv2s2(256, 512, device),
            ],
            middle: vec![NafBlock::new(512, device)],
            ups: [
                NafUp::new(512, device),
                NafUp::new(256, device),
                NafUp::new(128, device),
                NafUp::new(64, device),
            ],
            decoders: NafNet::dec([1, 1, 1, 1], [256, 128, 64, 32], device),
        }
    }

    fn enc(counts: [usize; 4], chans: [usize; 4], device: &B::Device) -> [NafEnc<B>; 4] {
        counts
            .into_iter()
            .zip(chans)
            .map(|(n, c)| NafEnc {
                blocks: (0..n).map(|_| NafBlock::new(c, device)).collect(),
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn dec(counts: [usize; 4], chans: [usize; 4], device: &B::Device) -> [NafDec<B>; 4] {
        counts
            .into_iter()
            .zip(chans)
            .map(|(n, c)| NafDec {
                blocks: (0..n).map(|_| NafBlock::new(c, device)).collect(),
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Deblur a 3-channel input (`[0,1]`); output is the residual-corrected
    /// estimate (`ending + inp`), not clamped. Spatial dims are padded to
    /// multiples of 16 and cropped back.
    pub fn forward(&self, inp: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, h, w] = inp.dims();
        let ph = (h + 15) / 16 * 16;
        let pw = (w + 15) / 16 * 16;
        let mut x = inp.clone();
        if ph > h {
            let z = Tensor::<B, 4>::zeros([n, c, ph - h, w], &inp.device());
            x = Tensor::cat(vec![x, z], 2);
        }
        if pw > w {
            let z = Tensor::<B, 4>::zeros([n, c, ph, pw - w], &inp.device());
            x = Tensor::cat(vec![x, z], 3);
        }
        let inp_padded = x.clone();

        x = self.intro.forward(x);
        let mut encs = Vec::with_capacity(4);
        for i in 0..4 {
            x = self.encoders[i].forward(x);
            encs.push(x.clone());
            x = self.downs[i].forward(x);
        }
        for b in &self.middle {
            x = b.forward(x);
        }
        for i in 0..4 {
            x = self.ups[i].forward(x) + encs[3 - i].clone();
            x = self.decoders[i].forward(x);
        }
        x = self.ending.forward(x) + inp_padded;
        x.slice([0..n, 0..c, 0..h, 0..w])
    }
}

/// Encoder downsample: `Conv2d(chan, 2chan, 2, 2)` (kernel 2, stride 2).
fn conv2s2<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [2, 2])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
        .with_bias(true)
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    #[test]
    #[ignore = "requires Vulkan; needs RUST_MIN_STACK=33554432 (burn autotune stack overflow on RADV)"]
    fn nafnet_output_shape_matches_input() {
        let device = WgpuDevice::DiscreteGpu(0);
        let m = NafNet::<BurnBackend>::new(&device);
        let [n, c, h, w] = [1, 3, 64, 66]; // h not a multiple of 16 → pad/crop
        let x = Tensor::<BurnBackend, 4>::from_data(
            TensorData::new(vec![0.5f32; n * c * h * w], [n, c, h, w]),
            &device,
        );
        let out = m.forward(x);
        assert_eq!(out.dims(), [n, c, h, w]);
    }

    /// Numerical check against the official torch model (tools/nafnet_verify.py
    /// writes `x.bin`/`ref.bin` as f32 little-endian). Loads the real f16
    /// burnpack, runs the same input, and asserts a small mean abs error.
    #[test]
    #[ignore = "needs Vulkan + models/NAFNet-GoPro-width32.pth.f16.bpk + torch ref bins (tools/nafnet_verify.py); needs RUST_MIN_STACK=33554432"]
    fn nafnet_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir = std::env::var("SENMEI_NAFNET_VERIFY_DIR")
            .unwrap_or_else(|_| "/tmp/nafnet_verify".into());
        let read = |name: &str, n: usize| -> Vec<f32> {
            let data = std::fs::read(format!("{dir}/{name}")).expect("missing ref bin");
            assert_eq!(data.len(), n * 4, "bad {name} size");
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let [n, c, h, w] = [1usize, 3, 64, 66];
        let x_v = read("x.bin", n * c * h * w);
        let ref_v = read("ref.bin", n * c * h * w);

        let mut m = NafNet::<BurnBackend<f16>>::new(&device);
        let mut store = BurnpackStore::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/NAFNet-GoPro-width32.pth.f16.bpk"
        ));
        let res = m.load_from(&mut store).unwrap();
        println!(
            "load: applied={} missing={} unused={}",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
        );
        for (p, c) in &res.missing {
            println!("  missing {p} ({c})");
        }
        for u in &res.unused {
            println!("  unused {u}");
        }

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
        assert!(mae < 0.01, "mae too large: {mae}");
    }
}
