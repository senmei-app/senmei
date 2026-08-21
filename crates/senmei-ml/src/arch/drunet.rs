//! DRUNet (DPIR, cszn/KAIR) — clean burn port of `UNetRes`.
//!
//! Residual U-Net denoiser: 3 stride-2 downsamples, 4 residual blocks per
//! level (Conv→ReLU→Conv + skip), 3 ConvTranspose2d upsamples, 4 residual
//! blocks per up level, single-conv head/tail. All convs are `bias=false`.
//! Input is `in_nc=4` (DPIR feeds `[RGB, constant noise-level map]`), output
//! `out_nc=3`. Pure conv graph — no channel slicing (unlike IFRNet's ResBlock),
//! so it does not hit the burn-fusion Bug 6.
//!
//! Weight loading (converter): torch keys like `m_down1.0.res.0.weight` /
//! `m_up3.1.res.2.weight` / `m_down1.4.weight` / `m_up3.0.weight` are remapped
//! onto these field paths (`m_down1.b0.c1` / `m_up3.b1.c2` / `m_down1.down` /
//! `m_up3.up`) with capture-group rules in `convert_pth_to_bpk`.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

fn conv3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    // Explicit(1,1,1,1) == torch `padding=1` (burn's `Same` pads asymmetrically
    // at stride>1; here stride is 1, but keep it explicit to be safe).
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
        .init(device)
}

fn conv2<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    // stride-conv downsample: kernel 2, stride 2, padding 0 (torch).
    Conv2dConfig::new([in_c, out_c], [2, 2])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
        .with_bias(false)
        .init(device)
}

fn deconv2<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> ConvTranspose2d<B> {
    // ConvTranspose2d kernel 2, stride 2, padding 0 (torch), bias=false.
    ConvTranspose2dConfig::new([in_c, out_c], [2, 2])
        .with_stride([2, 2])
        .with_padding([0, 0])
        .with_bias(false)
        .init(device)
}

/// `x + conv2(relu(conv1(x)))`.
#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    c1: Conv2d<B>,
    c2: Conv2d<B>,
}

impl<B: Backend> ResBlock<B> {
    fn new(ch: usize, device: &B::Device) -> Self {
        Self {
            c1: conv3(ch, ch, device),
            c2: conv3(ch, ch, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x.clone() + self.c2.forward(relu(self.c1.forward(x)))
    }
}

#[derive(Module, Debug)]
struct Down<B: Backend> {
    b0: ResBlock<B>,
    b1: ResBlock<B>,
    b2: ResBlock<B>,
    b3: ResBlock<B>,
    down: Conv2d<B>,
}

impl<B: Backend> Down<B> {
    fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self {
        Self {
            b0: ResBlock::new(in_ch, device),
            b1: ResBlock::new(in_ch, device),
            b2: ResBlock::new(in_ch, device),
            b3: ResBlock::new(in_ch, device),
            down: conv2(in_ch, out_ch, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self
            .b3
            .forward(self.b2.forward(self.b1.forward(self.b0.forward(x))));
        self.down.forward(x)
    }
}

#[derive(Module, Debug)]
struct Body<B: Backend> {
    b0: ResBlock<B>,
    b1: ResBlock<B>,
    b2: ResBlock<B>,
    b3: ResBlock<B>,
}

impl<B: Backend> Body<B> {
    fn new(ch: usize, device: &B::Device) -> Self {
        Self {
            b0: ResBlock::new(ch, device),
            b1: ResBlock::new(ch, device),
            b2: ResBlock::new(ch, device),
            b3: ResBlock::new(ch, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.b3
            .forward(self.b2.forward(self.b1.forward(self.b0.forward(x))))
    }
}

#[derive(Module, Debug)]
struct Up<B: Backend> {
    up: ConvTranspose2d<B>,
    // torch Sequential: index 0 = deconv, ResBlocks at 1..=4.
    b1: ResBlock<B>,
    b2: ResBlock<B>,
    b3: ResBlock<B>,
    b4: ResBlock<B>,
}

impl<B: Backend> Up<B> {
    fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self {
        Self {
            up: deconv2(in_ch, out_ch, device),
            b1: ResBlock::new(out_ch, device),
            b2: ResBlock::new(out_ch, device),
            b3: ResBlock::new(out_ch, device),
            b4: ResBlock::new(out_ch, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.up.forward(x);
        self.b4
            .forward(self.b3.forward(self.b2.forward(self.b1.forward(x))))
    }
}

#[derive(Module, Debug)]
pub struct Drunet<B: Backend> {
    m_head: Conv2d<B>,
    m_down1: Down<B>,
    m_down2: Down<B>,
    m_down3: Down<B>,
    m_body: Body<B>,
    m_up3: Up<B>,
    m_up2: Up<B>,
    m_up1: Up<B>,
    m_tail: Conv2d<B>,
}

impl<B: Backend> Drunet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            m_head: conv3(4, 64, device),
            m_down1: Down::new(64, 128, device),
            m_down2: Down::new(128, 256, device),
            m_down3: Down::new(256, 512, device),
            m_body: Body::new(512, device),
            m_up3: Up::new(512, 256, device),
            m_up2: Up::new(256, 128, device),
            m_up1: Up::new(128, 64, device),
            m_tail: conv3(64, 3, device),
        }
    }

    /// Denoise a 4-channel input (`[RGB, noise-level map]`, [0,1]) → 3-channel
    /// estimate ([0,1]). Input spatial dims must be multiples of 8.
    pub fn forward(&self, x0: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.m_head.forward(x0);
        let x2 = self.m_down1.forward(x1.clone());
        let x3 = self.m_down2.forward(x2.clone());
        let x4 = self.m_down3.forward(x3.clone());
        let x = self.m_body.forward(x4.clone());
        let x = self.m_up3.forward(x + x4);
        let x = self.m_up2.forward(x + x3);
        let x = self.m_up1.forward(x + x2);
        self.m_tail.forward(x + x1)
    }
}

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    #[test]
    #[ignore = "requires Vulkan; needs RUST_MIN_STACK=33554432 (burn autotune stack overflow on RADV)"]
    fn drunet_output_shape_matches_input() {
        let device = WgpuDevice::DiscreteGpu(0);
        let m = Drunet::<BurnBackend>::new(&device);
        let [n, c, h, w] = [1, 4, 64, 64];
        let x = Tensor::<BurnBackend, 4>::from_data(
            TensorData::new(vec![0.5f32; n * c * h * w], [n, c, h, w]),
            &device,
        );
        let out = m.forward(x);
        assert_eq!(out.dims(), [n, 3, h, w]);
    }

    /// Numerical check against the torch UNetRes (tools/drunet_verify.py writes
    /// `x.bin`/`ref.bin` as f32 little-endian). Loads the real f16 burnpack.
    #[test]
    #[ignore = "requires Vulkan + models/drunet_color.pth.f16.bpk + torch ref bins (tools/drunet_verify.py); needs RUST_MIN_STACK=33554432"]
    fn drunet_matches_torch_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir = std::env::var("SENMEI_DRUNET_VERIFY_DIR")
            .unwrap_or_else(|_| "/tmp/drunet_verify".into());
        let read = |name: &str, n: usize| -> Vec<f32> {
            let data = std::fs::read(format!("{dir}/{name}")).expect("missing ref bin");
            assert_eq!(data.len(), n * 4, "bad {name} size");
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let [n, c, h, w] = [1usize, 4, 64, 64];
        let x_v = read("x.bin", n * c * h * w);
        let ref_v = read("ref.bin", n * 3 * h * w);

        let mut m = Drunet::<BurnBackend<f16>>::new(&device);
        let mut store = BurnpackStore::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/drunet_color.pth.f16.bpk"
        ));
        let res = m.load_from(&mut store).unwrap();
        println!(
            "load: applied={} missing={} unused={}",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
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
        assert!(mae < 0.01, "mae too large: {mae}");
    }
}
