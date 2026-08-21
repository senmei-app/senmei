//! SPAN (Swift Parameter-free Attention Network) — clean burn port from the
//! Apache-2.0 BasicSR reference (hongyuanyu/SPAN). Loads Phhofm (flat keys,
//! norm on) and TNTwise (`params` wrapper) checkpoints; stale fused
//! `eval_conv` ignored. f16-safe on real frames (overflow only on synthetic
//! noise); bf16 broken on RADV. Output is [0,1] for norm-on checkpoints.

use burn::module::{Module, Param};
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
///
/// The final 1×1 `conv2` has `2*c_out` input channels — 96 for the 48ch
/// models, where cubek#519 returns wrong f16 results at H·W ≥ 32768. `pad_k96`
/// rebinds it into a K=128 conv (zero-padded weight) so the kernel takes the
/// correct path; forward then pads the input to 128 channels.
#[derive(Module, Debug)]
pub struct Conv3Xc<B: Backend> {
    conv0: Conv2d<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    sk: Conv2d<B>,
    pad_k96: bool,
}

impl<B: Backend> Conv3Xc<B> {
    pub fn new(c_in: usize, c_out: usize, device: &B::Device) -> Self {
        Self {
            conv0: conv2d(c_in, c_in * 2, 1, 0, device),
            conv1: conv2d(c_in * 2, c_out * 2, 3, 1, device),
            conv2: conv2d(c_out * 2, c_out, 1, 0, device),
            sk: conv2d(c_in, c_out, 1, 0, device),
            pad_k96: false,
        }
    }

    /// Workaround for cubek#519 (upstream-issues.md §6): a f16 1×1 conv with
    /// K=96 in-channels is wrong at H·W ≥ 32768. Zero-pad the weight into a
    /// K=128 conv (a verified-correct path) and pad the input at forward.
    /// Only the weight Param is swapped — burn derives the conv's in/out
    /// channels from the weight shape, and the bias is unchanged.
    pub fn pad_k96(&mut self, device: &B::Device) {
        let [o, c, kh, kw] = self.conv2.weight.val().dims();
        if c != 96 {
            return;
        }
        let w = self.conv2.weight.val().detach();
        let padded = Tensor::cat(
            vec![w, Tensor::<B, 4>::zeros([o, 128 - c, kh, kw], device)],
            1,
        );
        self.conv2.weight = Param::initialized(burn::module::ParamId::new(), padded);
        self.pad_k96 = true;
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(self.conv0.forward(x.clone()));
        let h = if self.pad_k96 { pad_channels_to(h, 128) } else { h };
        let out = self.conv2.forward(h);
        out + self.sk.forward(x)
    }
}

/// Zero-pad `x` to `target` channels (cubek#519 pad path).
fn pad_channels_to<B: Backend>(x: Tensor<B, 4>, target: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    if c >= target {
        return x;
    }
    let zeros = Tensor::<B, 4>::zeros([b, target - c, h, w], &x.device());
    Tensor::cat(vec![x, zeros], 1)
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

    fn pad_k96(&mut self, device: &B::Device) {
        self.c1_r.pad_k96(device);
        self.c2_r.pad_k96(device);
        self.c3_r.pad_k96(device);
    }

    /// `(out, out1_act, att)`; `out1_act` (post-SiLU) feeds the head concat.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let out1 = self.c1_r.forward(x.clone());
        let out1_act = silu(out1.clone());
        let out2 = self.c2_r.forward(out1_act.clone());
        let out2_act = silu(out2);
        let out3 = self.c3_r.forward(out2_act);
        let att = sigmoid(out3.clone()).sub_scalar(0.5);
        let out = (out3 + x) * att.clone();
        (out, out1_act, att)
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
    no_norm: bool,
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
            no_norm: false,
        }
    }

    /// `no_norm` checkpoints feed [0,1] input directly (norm=False).
    pub fn set_no_norm(&mut self, no_norm: bool) {
        self.no_norm = no_norm;
    }

    /// cubek#519 workaround for 48ch models (conv2 K=96): pads every conv2 to
    /// K=128. No-op for 64ch models (their conv2 is already K=128).
    pub fn pad_k96(&mut self, device: &B::Device) {
        self.conv_1.pad_k96(device);
        self.block_1.pad_k96(device);
        self.block_2.pad_k96(device);
        self.block_3.pad_k96(device);
        self.block_4.pad_k96(device);
        self.block_5.pad_k96(device);
        self.block_6.pad_k96(device);
        self.conv_2.pad_k96(device);
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // (x - mean) * 255 — norm-on checkpoints; mean (0.4488, 0.4371, 0.4040).
        let x = if self.no_norm {
            x
        } else {
            let mean = Tensor::<B, 1>::from_floats([0.4488, 0.4371, 0.4040], &x.device())
                .cast(x.dtype())
                .reshape([1, 3, 1, 1]);
            (x - mean).mul_scalar(255.0)
        };

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
    #[ignore = "needs Vulkan; standalone repro of cubek-convolution f16 1x1 conv bug (upstream-issues.md §6)"]
    fn conv1x1_repro() {
        use burn::module::Param;
        let device = WgpuDevice::DiscreteGpu(0);

        // Deterministic LCG so the repro never depends on external files.
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16_777_216.0
        };

        // K=96 with H*W >= 32768 is broken; other K are fine at any N.
        // 128/192/256 probe the pad targets (96→128 for conv2; conv_cat is
        // K=192 at full frame).
        let cases = [
            (96usize, 128usize, 128usize),
            (96, 128, 256),
            (96, 240, 320),
            (64, 240, 320),
            (128, 240, 320),
            (192, 240, 320),
            (256, 240, 320),
        ];
        println!("cubek-convolution f16 1x1 conv repro (K=96 x N>=32768 broken):");
        for (k, h, w) in cases {
            let n = h * w;
            let mut wv = vec![0.0f32; 48 * k];
            let mut bv = vec![0.0f32; 48];
            let mut xv = vec![0.0f32; k * n];
            for v in &mut wv {
                *v = (rnd() - 0.5) * 0.16;
            }
            for v in &mut bv {
                *v = (rnd() - 0.5) * 0.1;
            }
            for v in &mut xv {
                *v = (rnd() - 0.5) * 6.0;
            }

            // f32 CPU reference (1x1 conv = per-pixel matmul).
            let mut refv = vec![0.0f32; 48 * n];
            for j in 0..48 {
                for p in 0..n {
                    let mut acc = 0.0f32;
                    for c in 0..k {
                        acc += wv[j * k + c] * xv[c * n + p];
                    }
                    refv[j * n + p] = acc + bv[j];
                }
            }

            let wt = Tensor::<BurnBackend<f16>, 4>::from_data(
                TensorData::new(wv, [48, k, 1, 1]).convert::<f16>(),
                &device,
            );
            let b = Tensor::<BurnBackend<f16>, 1>::from_data(
                TensorData::new(bv, [48]).convert::<f16>(),
                &device,
            );
            let x = Tensor::<BurnBackend<f16>, 4>::from_data(
                TensorData::new(xv, [1, k, h, w]).convert::<f16>(),
                &device,
            );

            let mut conv = Conv2dConfig::new([k, 48], [1, 1]).init(&device);
            conv.weight = Param::from_tensor(wt);
            conv.bias = Some(Param::from_tensor(b));

            let out: Vec<f32> = conv.forward(x).into_data().convert::<f32>().to_vec().unwrap();
            let mut maxe = 0.0f32;
            let mut mae = 0.0f32;
            for (o, r) in out.iter().zip(&refv) {
                let e = (o - f16::from_f32(*r).to_f32()).abs();
                maxe = maxe.max(e);
                mae += e;
            }
            mae /= out.len() as f32;
            println!("  K={k} N={n} ({h}x{w}): max_abs={maxe:.5} mean_abs={mae:.6}");
        }

        // Verify the pad-96→128 workaround on the broken K=96/N=76800 case:
        // padding the weight into a K=128 conv + padding the input must match
        // the f32 reference (the raw K=96 path is wrong, the padded path not).
        {
            let (k, h, w) = (96usize, 240usize, 320usize);
            let n = h * w;
            let mut wv = vec![0.0f32; 48 * k];
            let mut bv = vec![0.0f32; 48];
            let mut xv = vec![0.0f32; k * n];
            for v in &mut wv {
                *v = (rnd() - 0.5) * 0.16;
            }
            for v in &mut bv {
                *v = (rnd() - 0.5) * 0.1;
            }
            for v in &mut xv {
                *v = (rnd() - 0.5) * 6.0;
            }
            let mut refv = vec![0.0f32; 48 * n];
            for j in 0..48 {
                for p in 0..n {
                    let mut acc = 0.0f32;
                    for c in 0..k {
                        acc += wv[j * k + c] * xv[c * n + p];
                    }
                    refv[j * n + p] = acc + bv[j];
                }
            }
            // Pad weight [48,96]→[48,128] (zeros per row) and input
            // [1,96]→[1,128] (zeros per channel), matching the module's cat.
            let mut wp = Vec::with_capacity(48 * 128);
            for j in 0..48 {
                wp.extend_from_slice(&wv[j * 96..(j + 1) * 96]);
                wp.extend(std::iter::repeat(0.0).take(32));
            }
            let mut xp = Vec::with_capacity(128 * n);
            for c in 0..96 {
                xp.extend_from_slice(&xv[c * n..(c + 1) * n]);
            }
            xp.resize(128 * n, 0.0);

            let wt = Tensor::<BurnBackend<f16>, 4>::from_data(
                TensorData::new(wp, [48, 128, 1, 1]).convert::<f16>(),
                &device,
            );
            let b = Tensor::<BurnBackend<f16>, 1>::from_data(
                TensorData::new(bv, [48]).convert::<f16>(),
                &device,
            );
            let x = Tensor::<BurnBackend<f16>, 4>::from_data(
                TensorData::new(xp, [1, 128, h, w]).convert::<f16>(),
                &device,
            );
            let mut conv = Conv2dConfig::new([128, 48], [1, 1]).init(&device);
            conv.weight = Param::from_tensor(wt);
            conv.bias = Some(Param::from_tensor(b));

            let out: Vec<f32> = conv.forward(x).into_data().convert::<f32>().to_vec().unwrap();
            let mut maxe = 0.0f32;
            let mut mae = 0.0f32;
            for (o, r) in out.iter().zip(&refv) {
                let e = (o - f16::from_f32(*r).to_f32()).abs();
                maxe = maxe.max(e);
                mae += e;
            }
            mae /= out.len() as f32;
            println!("  PAD K=96→128 N={n} (240x320): max_abs={maxe:.5} mean_abs={mae:.6}");
            assert!(maxe < 0.02, "padded conv deviates from f32 reference");
        }

        // Perf impact of the pad: time the K=96 vs the padded K=128 1×1 conv
        // at N=76800 (same weights/input, padded variant). Sync each iter.
        {
            let (k, h, w) = (96usize, 240usize, 320usize);
            let n = h * w;
            let mut wv = vec![0.0f32; 48 * k];
            let mut xv = vec![0.0f32; k * n];
            for v in &mut wv {
                *v = (rnd() - 0.5) * 0.16;
            }
            for v in &mut xv {
                *v = (rnd() - 0.5) * 6.0;
            }
            let b = Tensor::<BurnBackend<f16>, 1>::from_data(
                TensorData::new(vec![0.0f32; 48], [48]).convert::<f16>(),
                &device,
            );
            let x = Tensor::<BurnBackend<f16>, 4>::from_data(
                TensorData::new(xv.clone(), [1, k, h, w]).convert::<f16>(),
                &device,
            );
            let mut conv96 = Conv2dConfig::new([k, 48], [1, 1]).init(&device);
            conv96.weight = Param::from_tensor(
                Tensor::<BurnBackend<f16>, 4>::from_data(
                    TensorData::new(wv.clone(), [48, k, 1, 1]).convert::<f16>(),
                    &device,
                ),
            );
            conv96.bias = Some(Param::from_tensor(b.clone()));

            let mut wp = Vec::with_capacity(48 * 128);
            for j in 0..48 {
                wp.extend_from_slice(&wv[j * 96..(j + 1) * 96]);
                wp.extend(std::iter::repeat(0.0).take(32));
            }
            let xp: Vec<f32> = {
                let mut v = Vec::with_capacity(128 * n);
                for c in 0..96 {
                    v.extend_from_slice(&xv[c * n..(c + 1) * n]);
                }
                v.resize(128 * n, 0.0);
                v
            };
            let mut conv128 = Conv2dConfig::new([128, 48], [1, 1]).init(&device);
            conv128.weight = Param::from_tensor(
                Tensor::<BurnBackend<f16>, 4>::from_data(
                    TensorData::new(wp, [48, 128, 1, 1]).convert::<f16>(),
                    &device,
                ),
            );
            conv128.bias = Some(Param::from_tensor(b));
            let x128 = Tensor::<BurnBackend<f16>, 4>::from_data(
                TensorData::new(xp, [1, 128, h, w]).convert::<f16>(),
                &device,
            );

            let iters = 100usize;
            let time = |conv: &burn::nn::conv::Conv2d<BurnBackend<f16>>, inp: &Tensor<BurnBackend<f16>, 4>| {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    conv.forward(inp.clone()).into_data();
                }
                t0.elapsed().as_secs_f64() * 1e3 / iters as f64
            };
            let t96 = time(&conv96, &x);
            let t128 = time(&conv128, &x128);
            println!(
                "  PERF N={n}: K=96 {t96:.3} ms, K=128-padded {t128:.3} ms, delta {:.1}%",
                (t128 / t96 - 1.0) * 100.0
            );
        }
    }

    #[test]
    #[ignore = "needs Vulkan; verifies the cubek#519 pad reaches every conv2 in a 48ch Span"]
    fn pad_k96_pads_all_conv2() {
        let device = WgpuDevice::DiscreteGpu(0);
        let mut m = Span::<BurnBackend<f16>>::new(48, 2, &device);
        m.pad_k96(&device);

        // 1 (conv_1) + 18 (6 Spab × 3) + 1 (conv_2) = 20 Conv3Xc, all ch=48
        // → all 20 conv2 must become K=128. conv_cat (K=192) / upsampler
        // (K=48) are untouched by design.
        let mut padded = 0usize;
        let mut check = |c: &Conv3Xc<BurnBackend<f16>>| {
            if c.pad_k96 {
                assert_eq!(c.conv2.weight.val().dims()[1], 128);
                padded += 1;
            }
        };
        check(&m.conv_1);
        check(&m.conv_2);
        for b in [&m.block_1, &m.block_2, &m.block_3, &m.block_4, &m.block_5, &m.block_6] {
            check(&b.c1_r);
            check(&b.c2_r);
            check(&b.c3_r);
        }
        assert_eq!(padded, 20, "expected all 20 conv2 padded to K=128");
        assert_eq!(m.conv_cat.weight.val().dims()[1], 192);
        assert_eq!(m.upsampler.weight.val().dims()[1], 48);

        // pad_k96 queues async device tensors; force a sync so the wgpu
        // teardown doesn't crash on the pending queue at exit (test-only).
        let _ = m.conv_1.conv2.weight.val().into_data();
        println!("pad_k96: all 20 conv2 → K=128; conv_cat/upsampler untouched");
        drop(m);
    }

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

    #[test]
    #[ignore = "needs Vulkan + /tmp/senmei_models/2xNomosUni_span_multijpg_ldl.f16.bpk + real_512.rgb; needs RUST_MIN_STACK=33554432"]
    fn span_phhofm_loads_and_outputs_unit_range() {
        let device = WgpuDevice::DiscreteGpu(0);
        let mut m = Span::<BurnBackend<f16>>::new(48, 2, &device);
        let mut store =
            BurnpackStore::from_file("/tmp/senmei_models/2xNomosUni_span_multijpg_ldl.f16.bpk");
        let res = m.load_from(&mut store).unwrap();
        assert!(res.missing.is_empty(), "missing: {:?}", res.missing);

        let rgb = std::fs::read("/tmp/senmei_models/real_512.rgb").expect("missing frame");
        assert_eq!(rgb.len(), 512 * 512 * 3);
        let v: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
        let x = Tensor::<BurnBackend<f16>, 4>::from_data(
            TensorData::new(v, [1, 3, 512, 512]).convert::<f16>(),
            &device,
        );
        let out = m.forward(x);
        let o: Vec<f32> = out.into_data().convert::<f32>().to_vec().unwrap();
        let (nans, infs) = o.iter().fold((0usize, 0usize), |(a, b), f| {
            (a + f.is_nan() as usize, b + f.is_infinite() as usize)
        });
        let mn = o.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = o.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("min={mn:.3} max={mx:.3} nan={nans} inf={infs}");
        assert_eq!((nans, infs), (0, 0));
        assert!(mn > -1.0 && mx < 2.0, "out of range {mn}..{mx}");
    }
}
