//! DIS (Direct Image Supersampling) re-implemented in burn.
//! Clean port from the Apache-2.0 `Kim2091/DIS` (inference) reference — an
//! ultra-lightweight real-time SR arch: 32 feat / 4–12 `FastResBlock`s, PReLU
//! (no BN, no global norm → tileable + FP16-safe), PixelShuffle upsampler and
//! a bilinear global residual.

use super::srvgg::{pixel_shuffle, Prelu};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{backend::Backend, Tensor};

fn conv3x3<B: Backend>(in_c: usize, out_c: usize, bias: bool, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(bias)
        .init(device)
}

/// Ultra-fast residual block: `conv → PReLU → conv + residual`, no BN.
#[derive(Module, Debug)]
pub struct FastResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    act: Prelu<B>,
}

impl<B: Backend> FastResBlock<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: conv3x3(channels, channels, false, device),
            conv2: conv3x3(channels, channels, false, device),
            act: Prelu::new(channels, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv2.forward(self.act.forward(self.conv1.forward(x.clone()))) + x
    }
}

/// PixelShuffle upsampler (ESPCN style): `conv → PixelShuffle → PReLU`.
#[derive(Module, Debug)]
pub struct PixelShuffleUpsampler<B: Backend> {
    conv: Conv2d<B>,
    act: Prelu<B>,
    scale: usize,
}

impl<B: Backend> PixelShuffleUpsampler<B> {
    pub fn new(in_c: usize, out_c: usize, scale: usize, device: &B::Device) -> Self {
        Self {
            conv: conv3x3(in_c, out_c * scale * scale, true, device),
            act: Prelu::new(out_c, device),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.act.forward(pixel_shuffle(self.conv.forward(x), self.scale))
    }
}

/// `DIS`: `head → PReLU → body (FastResBlock×) → fusion + residual →
/// upsampler → tail + bilinear-residual`. scale 2/3 = one upsampler stage,
/// scale 4 = two ×2 stages, scale 1 = identity.
#[derive(Module, Debug)]
pub struct DisNet<B: Backend> {
    head: Conv2d<B>,
    head_act: Prelu<B>,
    body: Vec<FastResBlock<B>>,
    fusion: Conv2d<B>,
    upsampler: Vec<PixelShuffleUpsampler<B>>,
    tail: Conv2d<B>,
    scale: usize,
}

impl<B: Backend> DisNet<B> {
    pub fn new(num_features: usize, num_blocks: usize, scale: usize, device: &B::Device) -> Self {
        let mut upsampler = Vec::new();
        match scale {
            1 => {}
            2 | 3 => upsampler.push(PixelShuffleUpsampler::new(
                num_features, num_features, scale, device,
            )),
            4 => {
                upsampler.push(PixelShuffleUpsampler::new(num_features, num_features, 2, device));
                upsampler.push(PixelShuffleUpsampler::new(num_features, num_features, 2, device));
            }
            other => panic!("unsupported DIS scale {other}"),
        }
        Self {
            head: conv3x3(3, num_features, true, device),
            head_act: Prelu::new(num_features, device),
            body: (0..num_blocks)
                .map(|_| FastResBlock::new(num_features, device))
                .collect(),
            fusion: conv3x3(num_features, num_features, true, device),
            upsampler,
            tail: conv3x3(num_features, 3, true, device),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, h, w] = x.dims();
        let base = if self.scale == 1 {
            x.clone()
        } else {
            interpolate(
                x.clone(),
                [h * self.scale, w * self.scale],
                InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
            )
        };
        let feat = self.head_act.forward(self.head.forward(x));
        let mut out = feat.clone();
        for block in &self.body {
            out = block.forward(out);
        }
        out = self.fusion.forward(out) + feat;
        for up in &self.upsampler {
            out = up.forward(out);
        }
        self.tail.forward(out) + base
    }
}

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    /// Numerical check of the DIS port against torch:
    /// `tools/dis_verify.py` writes `x.bin`/`ref.bin` (f32, 32×32 → scaled).
    /// Converts the safetensors to a .bpk with `senmei-ml-convert dis` first.
    #[test]
    #[ignore = "needs GPU + converted dis bpk + torch ref bins (tools/dis_verify.py); needs RUST_MIN_STACK=33554432"]
    fn dis_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir =
            std::env::var("SENMEI_DIS_VERIFY_DIR").unwrap_or_else(|_| "/tmp/dis_verify".into());
        let scale = std::env::var("SENMEI_DIS_SCALE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        let num_blocks = std::env::var("SENMEI_DIS_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8);
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

        let mut m = DisNet::<BurnBackend<f16>>::new(32, num_blocks, scale, &device);
        let mut store = BurnpackStore::from_file(format!("{dir}/dis_x{scale}.f16.bpk"));
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
        assert!(mae < 0.02, "mae too high: {mae}");
    }

    /// Smoke check: 12 blocks (Balanced), scale 2 upscales and stays finite.
    #[test]
    #[ignore = "needs GPU + converted dis bpk (senmei-ml-convert dis)"]
    fn dis_forward_upscales() {
        let device = WgpuDevice::DiscreteGpu(0);
        let mut m = DisNet::<BurnBackend<f16>>::new(32, 12, 2, &device);
        let mut store = BurnpackStore::from_file("/tmp/dis_balanced_x2.f16.bpk");
        let res = m.load_from(&mut store).unwrap();
        assert!(res.missing.is_empty(), "missing tensors: {:?}", res.missing);

        let (h, w): (usize, usize) = (16, 24);
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
