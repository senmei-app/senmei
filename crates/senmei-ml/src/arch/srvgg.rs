//! Real-ESRGAN `SRVGGNetCompact` (animevideo-xs) re-implemented in burn.
//! Clean port from the BSD-3-Clause `xinntao/Real-ESRGAN` reference.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::{backend::Backend, Tensor};

fn conv3x3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

/// Per-channel PReLU. SRVGG reuses ONE activation for every body layer, so all
/// PReLU entries in the state dict (`body.1.weight`, `body.3.weight`, …) share
/// the same weight — a single param is enough (the duplicates stay unused).
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

/// PixelShuffle: `[N, C·r², H, W] → [N, C, H·r, W·r]`.
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let oc = c / (r * r);
    x.reshape([n, oc, r, r, h, w])
        .permute([0, 1, 3, 5, 2, 4])
        .reshape([n, oc, h * r, w * r])
}

/// `SRVGGNetCompact`: conv_first + (num_conv−2)× (conv + prelu) + last conv,
/// then a PixelShuffle upsampler (1 step for 2×, 2 for 4×) and a final conv.
#[derive(Module, Debug)]
pub struct SrvggNet<B: Backend> {
    /// The convs (torch `body.{0,2,4,…}` remapped onto this Vec).
    body: Vec<Conv2d<B>>,
    /// The shared PReLU (torch `body.{1,3,5,…}.weight`, all identical).
    prelu: Prelu<B>,
    /// 1 conv for 2×, 2 convs for 4× (torch `upsampler.{0,2}`).
    upsampler: Vec<Conv2d<B>>,
    conv_last: Conv2d<B>,
}

impl<B: Backend> SrvggNet<B> {
    pub fn new(num_feat: usize, num_conv: usize, scale: usize, device: &B::Device) -> Self {
        let mut body = Vec::with_capacity(num_conv);
        body.push(conv3x3(3, num_feat, device));
        for _ in 1..num_conv {
            body.push(conv3x3(num_feat, num_feat, device));
        }
        let mut upsampler = Vec::new();
        upsampler.push(conv3x3(num_feat, num_feat * 4, device));
        if scale >= 4 {
            upsampler.push(conv3x3(num_feat, num_feat * 4, device));
        }
        Self {
            body,
            prelu: Prelu::new(num_feat, device),
            upsampler,
            conv_last: conv3x3(num_feat, 3, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut y = x;
        for (i, conv) in self.body.iter().enumerate() {
            y = conv.forward(y);
            if i + 1 < self.body.len() {
                y = self.prelu.forward(y);
            }
        }
        for conv in &self.upsampler {
            y = pixel_shuffle(conv.forward(y), 2);
        }
        self.conv_last.forward(y)
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

        let mut m = SrvggNet::<BurnBackend<f16>>::new(64, 16, scale, &device);
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
}
