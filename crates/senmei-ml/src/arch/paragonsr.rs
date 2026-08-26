//! ParagonSR (Nano) re-implemented in burn. Clean port from the MIT
//! `Phhofm/ParagonSR2` reference (`paragonsr_arch.py`). Used by the
//! `paragonsr-nano-x2` checkpoint (num_feat 24 / 3 residual groups × 2
//! blocks / ffn_expansion 1.5 / scale 2).
//!
//! Fused inference graph (confirmed against the exported ONNX): `conv_in`
//! (3→24) → 3× `ResidualGroup` (2× `ParagonBlock`, group residual) →
//! `conv_fuse` (24→24) + shallow skip → `upsampler` (24→96) →
//! PixelShuffle(2) → `conv_out` (24→3). The torch arch's MagicKernel
//! upsampler fuses into conv(24→96)+PixelShuffle at inference. Each
//! `ParagonBlock` is two residuals: `+ls1(InceptionDWConv(norm1(x)))` then
//! `+ls2(GatedFFN(norm2(x)))`. `norm` = GroupNorm(1,24): per-location channel
//! norm with the same scaled-domain fp16 trick as SAFMN, eps 1e-5.

use burn::module::{Module, Param, ParamId};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::mish;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// PixelShuffle: `[N, C·r², H, W] → [N, C, H·r, W·r]` (torch `pixel_shuffle`).
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, r: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let oc = c / (r * r);
    x.reshape([n, oc, r, r, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([n, oc, h * r, w * r])
}

fn conv2d<B: Backend>(
    in_c: usize,
    out_c: usize,
    k: [usize; 2],
    pad: [usize; 4],
    groups: usize,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], k)
        .with_padding(PaddingConfig2d::Explicit(pad[0], pad[1], pad[2], pad[3]))
        .with_groups(groups)
        .init(device)
}

/// Per-channel scale (torch LayerScale): multiply by the learned `gamma`.
#[derive(Module, Debug)]
pub struct LayerScale<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerScale<B> {
    fn new(c: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32; c], [c]), device),
            ),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, c, _, _] = x.dims();
        x * self.gamma.val().reshape([1, c, 1, 1])
    }
}

/// GroupNorm(1, C): per-sample mean/var over all of (C, H, W) — the ONNX
/// export reshapes `(N,C,H,W) → (N,1,C·H·W)` and runs InstanceNorm (eps
/// 1e-5), i.e. NOT a per-location channel norm. Variance computed in an
/// `x/S`-scaled domain (fp16-safe).
#[derive(Module, Debug)]
pub struct GroupNorm1<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> GroupNorm1<B> {
    fn new(c: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32; c], [c]), device),
            ),
            bias: Param::initialized(ParamId::new(), Tensor::<B, 1>::zeros([c], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, c, _, _] = x.dims();
        let s: f32 = 128.0;
        // Mean over dims 1..3 → [n,1,1,1].
        let mu = (x.clone() / s).mean_dim(1).mean_dim(2).mean_dim(3) * s;
        let d = x - mu;
        let ds = d.clone() / s;
        let m = (ds.clone() * ds).mean_dim(1).mean_dim(2).mean_dim(3); // var / s^2
        let inv = (m + 1e-5 / (s * s)).sqrt().recip() / s;
        (d * inv) * self.weight.val().reshape([1, c, 1, 1]) + self.bias.val().reshape([1, c, 1, 1])
    }
}

/// InceptionDWConv2d (branch_ratio 0.125): split channels 15/3/3/3, run the
/// three 3-channel branches through 3×3 / 1×11 / 11×1 depthwise convs, concat.
#[derive(Module, Debug)]
pub struct InceptionDwConv<B: Backend> {
    dwconv_hw: Conv2d<B>,
    dwconv_w: Conv2d<B>,
    dwconv_h: Conv2d<B>,
}

impl<B: Backend> InceptionDwConv<B> {
    fn new(gc: usize, device: &B::Device) -> Self {
        Self {
            dwconv_hw: conv2d(gc, gc, [3, 3], [1, 1, 1, 1], gc, device),
            dwconv_w: conv2d(gc, gc, [1, 11], [0, 5, 0, 5], gc, device),
            dwconv_h: conv2d(gc, gc, [11, 1], [5, 0, 5, 0], gc, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let parts = x.split_with_sizes(vec![15, 3, 3, 3], 1);
        Tensor::cat(
            vec![
                parts[0].clone(),
                self.dwconv_hw.forward(parts[1].clone()),
                self.dwconv_w.forward(parts[2].clone()),
                self.dwconv_h.forward(parts[3].clone()),
            ],
            1,
        )
    }
}

/// GatedFFN (ffn_expansion 1.5): project_in_g/i (1×1 → 36), depthwise 3×3
/// spatial mixer (groups 36), Mish gate, project_out (1×1 → 24).
#[derive(Module, Debug)]
pub struct GatedFfn<B: Backend> {
    project_in_g: Conv2d<B>,
    project_in_i: Conv2d<B>,
    spatial_mixer: Conv2d<B>,
    project_out: Conv2d<B>,
}

impl<B: Backend> GatedFfn<B> {
    fn new(in_c: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            project_in_g: conv2d(in_c, hidden, [1, 1], [0, 0, 0, 0], 1, device),
            project_in_i: conv2d(in_c, hidden, [1, 1], [0, 0, 0, 0], 1, device),
            spatial_mixer: conv2d(hidden, hidden, [3, 3], [1, 1, 1, 1], hidden, device),
            project_out: conv2d(hidden, in_c, [1, 1], [0, 0, 0, 0], 1, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let g = self
            .spatial_mixer
            .forward(self.project_in_g.forward(x.clone()));
        let i = self.project_in_i.forward(x);
        self.project_out.forward(mish(g) * i)
    }
}

/// ParagonBlock: two residuals — `+ls1(context(norm1(x)))` then
/// `+ls2(transformer(norm2(x)))`.
#[derive(Module, Debug)]
pub struct ParagonBlock<B: Backend> {
    norm1: GroupNorm1<B>,
    norm2: GroupNorm1<B>,
    context: InceptionDwConv<B>,
    transformer: GatedFfn<B>,
    ls1: LayerScale<B>,
    ls2: LayerScale<B>,
}

impl<B: Backend> ParagonBlock<B> {
    fn new(c: usize, ffn_expansion: f32, device: &B::Device) -> Self {
        let hidden = (c as f32 * ffn_expansion) as usize;
        Self {
            norm1: GroupNorm1::new(c, device),
            norm2: GroupNorm1::new(c, device),
            context: InceptionDwConv::new(3, device),
            transformer: GatedFfn::new(c, hidden, device),
            ls1: LayerScale::new(c, device),
            ls2: LayerScale::new(c, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self
            .ls1
            .forward(self.context.forward(self.norm1.forward(x.clone())))
            + x.clone();
        self.ls2
            .forward(self.transformer.forward(self.norm2.forward(y.clone())))
            + y
    }
}

/// ResidualGroup: `blocks(x) + x`.
#[derive(Module, Debug)]
pub struct ResidualGroup<B: Backend> {
    blocks: Vec<ParagonBlock<B>>,
}

impl<B: Backend> ResidualGroup<B> {
    fn new(c: usize, n_blocks: usize, ffn_expansion: f32, device: &B::Device) -> Self {
        Self {
            blocks: (0..n_blocks)
                .map(|_| ParagonBlock::new(c, ffn_expansion, device))
                .collect(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut y = x.clone();
        for b in &self.blocks {
            y = b.forward(y);
        }
        y + x
    }
}

/// ParagonSR (Nano): conv_in → 3×2 blocks (group residuals) → conv_fuse +
/// shallow skip → upsampler(24→96) → PixelShuffle(2) → conv_out.
#[derive(Module, Debug)]
pub struct ParagonSrNet<B: Backend> {
    conv_in: Conv2d<B>,
    body: Vec<ResidualGroup<B>>,
    conv_fuse: Conv2d<B>,
    upsampler: Conv2d<B>,
    conv_out: Conv2d<B>,
    scale: usize,
}

impl<B: Backend> ParagonSrNet<B> {
    pub fn new(
        scale: usize,
        num_feat: usize,
        num_groups: usize,
        num_blocks: usize,
        ffn_expansion: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            conv_in: conv2d(3, num_feat, [3, 3], [1, 1, 1, 1], 1, device),
            body: (0..num_groups)
                .map(|_| ResidualGroup::new(num_feat, num_blocks, ffn_expansion, device))
                .collect(),
            conv_fuse: conv2d(num_feat, num_feat, [3, 3], [1, 1, 1, 1], 1, device),
            upsampler: conv2d(
                num_feat,
                num_feat * scale * scale,
                [3, 3],
                [1, 1, 1, 1],
                1,
                device,
            ),
            conv_out: conv2d(num_feat, 3, [3, 3], [1, 1, 1, 1], 1, device),
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shallow = self.conv_in.forward(x);
        let mut deep = shallow.clone();
        for g in &self.body {
            deep = g.forward(deep);
        }
        let fused = self.conv_fuse.forward(deep) + shallow;
        let up = pixel_shuffle(self.upsampler.forward(fused), self.scale);
        self.conv_out.forward(up)
    }
}

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::*;
    use crate::BurnBackend;
    use burn::tensor::{f16, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};
    use burn_wgpu::WgpuDevice;

    /// Numerical check against ONNX Runtime on the fused inference graph
    /// (spandrel has no ParagonSR arch): `tools/paragonsr_verify.py` writes
    /// `x.bin`/`ref.bin` (f32, 32×32, fp16-exact on the ONNX side).
    #[test]
    #[ignore = "needs tools/paragonsr_verify.py reference + converted bpk"]
    fn paragonsr_matches_onnx_reference() {
        let device = WgpuDevice::DiscreteGpu(0);
        let dir = std::env::var("SENMEI_PARAGONSR_VERIFY_DIR")
            .unwrap_or_else(|_| "/tmp/paragonsr_verify".into());
        let read = |name: &str, n: usize| -> Vec<f32> {
            let data = std::fs::read(format!("{dir}/{name}")).expect("missing ref bin");
            assert_eq!(data.len(), n * 4, "bad {name} size");
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let [n, c, h, w] = [1usize, 3, 32, 32];
        let x_v = read("x.bin", n * c * h * w);
        let ref_v = read("ref.bin", n * c * h * 2 * w * 2);

        let mut m = ParagonSrNet::<BurnBackend<f16>>::new(2, 24, 3, 2, 1.5, &device);
        let mut store = BurnpackStore::from_file(format!("{dir}/paragonsr-nano.f16.bpk"));
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
        println!("mae vs onnx = {mae}");
        assert!(mae < 0.02, "mae too high: {mae}");
    }
}
