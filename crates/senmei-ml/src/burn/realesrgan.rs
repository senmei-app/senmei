//! Real-ESRGAN `RRDBNet` re-implemented in burn.
//! Clean port from the BSD-3-Clause `xinntao/Real-ESRGAN` reference.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::leaky_relu;
use burn::tensor::{backend::Backend, Tensor};

use super::upcunet::nearest2x;

fn conv3x3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

/// Dense block with 5 convs; output `x5 * 0.2 + x` (residual scaling).
#[derive(Module, Debug)]
struct ResidualDenseBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
}

impl<B: Backend> ResidualDenseBlock<B> {
    fn new(num_feat: usize, num_grow_ch: usize, device: &B::Device) -> Self {
        Self {
            conv1: conv3x3(num_feat, num_grow_ch, device),
            conv2: conv3x3(num_feat + num_grow_ch, num_grow_ch, device),
            conv3: conv3x3(num_feat + 2 * num_grow_ch, num_grow_ch, device),
            conv4: conv3x3(num_feat + 3 * num_grow_ch, num_grow_ch, device),
            conv5: conv3x3(num_feat + 4 * num_grow_ch, num_feat, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x0 = x.clone();
        let x1 = leaky_relu(self.conv1.forward(x.clone()), 0.2);
        let x2 = leaky_relu(
            self.conv2
                .forward(Tensor::cat(vec![x.clone(), x1.clone()], 1)),
            0.2,
        );
        let x3 = leaky_relu(
            self.conv3
                .forward(Tensor::cat(vec![x.clone(), x1.clone(), x2.clone()], 1)),
            0.2,
        );
        let x4 = leaky_relu(
            self.conv4.forward(Tensor::cat(
                vec![x.clone(), x1.clone(), x2.clone(), x3.clone()],
                1,
            )),
            0.2,
        );
        let x5 = self.conv5.forward(Tensor::cat(vec![x, x1, x2, x3, x4], 1));
        x5.mul_scalar(0.2) + x0
    }
}

/// Three dense blocks; output `out * 0.2 + x`.
#[derive(Module, Debug)]
struct Rrdb<B: Backend> {
    rdb1: ResidualDenseBlock<B>,
    rdb2: ResidualDenseBlock<B>,
    rdb3: ResidualDenseBlock<B>,
}

impl<B: Backend> Rrdb<B> {
    fn new(num_feat: usize, num_grow_ch: usize, device: &B::Device) -> Self {
        Self {
            rdb1: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
            rdb2: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
            rdb3: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x0 = x.clone();
        let out = self.rdb1.forward(x.clone());
        let out = self.rdb2.forward(out);
        let out = self.rdb3.forward(out);
        out.mul_scalar(0.2) + x0
    }
}

/// `RRDBNet`: conv_first -> `num_block` RRDBs -> conv_body (residual add) ->
/// nearest 2x upsampling (`conv_up2` only for scale 4) -> conv_hr -> conv_last.
#[derive(Module, Debug)]
pub struct RrdbNet<B: Backend> {
    conv_first: Conv2d<B>,
    body: Vec<Rrdb<B>>,
    conv_body: Conv2d<B>,
    conv_up1: Conv2d<B>,
    conv_up2: Option<Conv2d<B>>,
    conv_hr: Conv2d<B>,
    conv_last: Conv2d<B>,
}

impl<B: Backend> RrdbNet<B> {
    pub fn new(scale: usize, num_block: usize, device: &B::Device) -> Self {
        let num_feat = 64;
        let num_grow_ch = 32;
        let body = (0..num_block)
            .map(|_| Rrdb::new(num_feat, num_grow_ch, device))
            .collect();
        Self {
            conv_first: conv3x3(3, num_feat, device),
            body,
            conv_body: conv3x3(num_feat, num_feat, device),
            conv_up1: conv3x3(num_feat, num_feat, device),
            conv_up2: (scale == 4).then(|| conv3x3(num_feat, num_feat, device)),
            conv_hr: conv3x3(num_feat, num_feat, device),
            conv_last: conv3x3(num_feat, 3, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let feat = self.conv_first.forward(x);
        let mut body = feat.clone();
        for rrdb in &self.body {
            body = rrdb.forward(body);
        }
        let feat = feat + self.conv_body.forward(body);
        let feat = leaky_relu(self.conv_up1.forward(nearest2x(feat)), 0.2);
        let feat = match &self.conv_up2 {
            Some(c) => leaky_relu(c.forward(nearest2x(feat)), 0.2),
            None => feat,
        };
        self.conv_last
            .forward(leaky_relu(self.conv_hr.forward(feat), 0.2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    /// Numerical check of the BSRGAN (RRDBNet 23, scale 4) port against the
    /// official weights: `tools/bsrgan_verify.py` writes `x.bin`/`ref.bin`
    /// (f32, 32×32 → 128×128). Loads the f16 burnpack, runs the same input, and
    /// asserts a small mean abs error.
    #[test]
    #[ignore = "needs GPU + models/BSRGAN.f16.bpk + torch ref bins (tools/bsrgan_verify.py); needs RUST_MIN_STACK=33554432"]
    fn bsrgan_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir = std::env::var("SENMEI_BSRGAN_VERIFY_DIR")
            .unwrap_or_else(|_| "/tmp/bsrgan_verify".into());
        let read = |name: &str, n: usize| -> Vec<f32> {
            let data = std::fs::read(format!("{dir}/{name}")).expect("missing ref bin");
            assert_eq!(data.len(), n * 4, "bad {name} size");
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let [n, c, h, w] = [1usize, 3, 32, 32];
        let x_v = read("x.bin", n * c * h * w);
        let ref_v = read("ref.bin", n * c * 128 * 128);

        let mut m = RrdbNet::<BurnBackend<f16>>::new(4, 23, &device);
        let mut store = BurnpackStore::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/BSRGAN.f16.bpk"
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
