//! Real-ESRGAN `SRVGGNetCompact` (animevideo-xs) re-implemented in burn.
//! Clean port from the BSD-3-Clause `xinntao/Real-ESRGAN` reference.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{backend::Backend, Tensor};

fn conv3x3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

/// Per-channel PReLU. SRVGG animevideo-xs reuses ONE activation for every body
/// layer (all `body.{odd}.weight` entries share the same value), but the
/// general-x4v3 checkpoint has a distinct PReLU per layer — so `SrvggNet` holds
/// one `Prelu` per mid conv (`num_conv + 1` total) and the converter remaps
/// each `body.{2k+1}.weight` to `prelu.{k}.weight`; shared checkpoints then
/// just fill every entry with the same value.
#[derive(Module, Debug)]
pub struct Prelu<B: Backend> {
    weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> Prelu<B> {
    pub fn new(num_parameters: usize, device: &B::Device) -> Self {
        Self {
            // torch initializes PReLU to 0.25.
            weight: Param::from_tensor(Tensor::ones([num_parameters], device).mul_scalar(0.25)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let c = self.weight.val().dims()[0];
        let w = self.weight.val().reshape([1, c, 1, 1]);
        x.clone().clamp_min(0.0) + w * x.clamp_max(0.0)
    }
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

/// `SRVGGNetCompact` (animevideo-xs / general-x4v3): `conv_first +
/// num_conv× (conv + prelu) + upscale conv (num_feat → 3·scale²)`, then one
/// PixelShuffle(scale). The checkpoints fold the upsampler into the body — the
/// last body conv is `64 → 12` (2×) / `48` (4×) and there are no
/// `upsampler.*`/`conv_last` keys. animevideo-xs shares one PReLU across all
/// layers; general-x4v3 has one per layer (handled by `Vec<Prelu>`).
#[derive(Module, Debug)]
pub struct SrvggNet<B: Backend> {
    /// The convs (torch `body.{0,2,4,…}` remapped onto this Vec); the last one
    /// is the upscale conv (`num_feat → 3·scale²`).
    body: Vec<Conv2d<B>>,
    /// One PReLU per mid conv (torch `body.{1,3,5,…}.weight`), all equal for
    /// the shared animevideo-xs checkpoints.
    prelu: Vec<Prelu<B>>,
    /// Upscale factor for the final PixelShuffle.
    scale: usize,
}

impl<B: Backend> SrvggNet<B> {
    pub fn new(num_feat: usize, num_conv: usize, scale: usize, device: &B::Device) -> Self {
        let mut body = Vec::with_capacity(num_conv + 2);
        body.push(conv3x3(3, num_feat, device));
        for _ in 0..num_conv {
            body.push(conv3x3(num_feat, num_feat, device));
        }
        body.push(conv3x3(num_feat, 3 * scale * scale, device));
        Self {
            body,
            prelu: (0..=num_conv).map(|_| Prelu::new(num_feat, device)).collect(),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, h, w] = x.dims();
        let mut y = x.clone();
        for (i, conv) in self.body.iter().enumerate() {
            y = conv.forward(y);
            if i + 1 < self.body.len() {
                y = self.prelu[i].forward(y);
            }
        }
        // SRVGG learns the RESIDUAL: the nearest-upsampled input is added to
        // the PixelShuffle output (spandrel `SRVGGNetCompact.forward`). Without
        // it the output is the near-black residual alone.
        let base = interpolate(
            x,
            [h * self.scale, w * self.scale],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );
        pixel_shuffle(y, self.scale) + base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    /// Numerical check of the SRVGGNetCompact port against torch:
    /// `tools/srvgg_verify.py` writes `x.bin`/`ref.bin` (f32, 32×32 → scaled).
    /// Converts the random .pth to a .bpk with `senmei-ml-convert srvgg` first.
    #[test]
    #[ignore = "needs GPU + converted srvgg bpk + torch ref bins (tools/srvgg_verify.py); needs RUST_MIN_STACK=33554432"]
    fn srvgg_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir = std::env::var("SENMEI_SRVGG_VERIFY_DIR")
            .unwrap_or_else(|_| "/tmp/srvgg_verify".into());
        let scale = std::env::var("SENMEI_SRVGG_SCALE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        let num_conv = std::env::var("SENMEI_SRVGG_NUM_CONV")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16);
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

        let mut m = SrvggNet::<BurnBackend<f16>>::new(64, num_conv, scale, &device);
        let mut store = BurnpackStore::from_file(format!("{dir}/srvgg_x{scale}.f16.bpk"));
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
        assert!(
            res.missing.is_empty(),
            "missing tensors: {:?}",
            res.missing
        );

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

    /// Smoke check of the animevideo-xs structure against the converted bpk:
    /// 18 body convs load with no missing and the forward upscales by `scale`
    /// (x2 → 64→12 body tail + PixelShuffle(2)). Catches the arch/checkpoint
    /// mismatch the key-contract test guards at the key level.
    #[test]
    #[ignore = "needs GPU + converted animevideo bpk (senmei-ml-convert srvgg)"]
    fn srvgg_forward_upscales() {
        let device = WgpuDevice::DiscreteGpu(0);
        let mut m = SrvggNet::<BurnBackend<f16>>::new(64, 16, 2, &device);
        let mut store = BurnpackStore::from_file("/tmp/animevideo_x2.f16.bpk");
        let res = m.load_from(&mut store).unwrap();
        assert!(
            res.missing.is_empty(),
            "missing tensors: {:?}",
            res.missing
        );

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
            out.into_data().to_vec::<f16>().unwrap().iter().all(|v| v.to_f32().is_finite()),
            "non-finite output"
        );
    }
}
