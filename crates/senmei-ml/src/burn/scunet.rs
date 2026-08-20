//! SCUNet (cszn/SCUNet, Apache-2.0) — clean burn port.
//!
//! Swin-Conv-UNet denoiser: each ConvTransBlock runs a windowed multi-head
//! self-attention (WMSA, 8×8 windows, 32-dim heads) branch + a residual 3×3
//! conv branch, concatenated through 1×1 convs. U-Net backbone (config
//! [2,2,2,2,2,2,2], dim 64) with stride-2 conv down / convT up and skip
//! additions. Input padded to a multiple of 64 (replication), cropped back.
//!
//! Op mapping vs torch (network_scunet.py): `torch.roll` = `roll`, window
//! partition/reverse = reshape+permute, relative-position bias = flat `gather`
//! on the (heads, 225) table, SW mask = `mask_where` with `-inf`.

use burn::module::Module;
use burn::module::{Param, ParamId};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::PaddingConfig2d;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData};

const WS: usize = 8; // window size
const HEAD_DIM: usize = 32;
const NP: usize = WS * WS; // tokens per window
const BIAS_BASE: usize = 2 * WS - 1; // 15

fn conv3<B: Backend>(in_c: usize, out_c: usize, bias: bool, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(bias)
        .init(device)
}

fn conv1<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [1, 1])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
        .init(device)
}

fn conv_down<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [2, 2])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
        .with_bias(false)
        .init(device)
}

fn conv_up<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> ConvTranspose2d<B> {
    ConvTranspose2dConfig::new([in_c, out_c], [2, 2])
        .with_stride([2, 2])
        .with_padding([0, 0])
        .with_bias(false)
        .init(device)
}

/// Flat gather index (len NP·NP) for the (heads, 15, 15) relative table:
/// `idx[p][q] = (pi - qi + ws-1)*15 + (pj - qj + ws-1)`.
fn relative_index() -> Vec<i64> {
    let mut idx = vec![0i64; NP * NP];
    for p in 0..NP {
        let (pi, pj) = (p / WS, p % WS);
        for q in 0..NP {
            let (qi, qj) = (q / WS, q % WS);
            let di = pi as i64 - qi as i64 + (WS as i64 - 1);
            let dj = pj as i64 - qj as i64 + (WS as i64 - 1);
            idx[p * NP + q] = di * BIAS_BASE as i64 + dj;
        }
    }
    idx
}

/// SW-MSA cross-boundary patterns (flat len NP·NP). Row cross: query/key rows
/// straddle the shift; column cross: columns straddle. Only windows on the
/// last row / last column use them (torch `generate_mask`).
fn sw_cross_patterns() -> (Vec<bool>, Vec<bool>) {
    let s = WS / 2; // 4
    let mut row = vec![false; NP * NP];
    let mut col = vec![false; NP * NP];
    for p in 0..NP {
        let (pi, pj) = (p / WS, p % WS);
        for q in 0..NP {
            let (qi, qj) = (q / WS, q % WS);
            row[p * NP + q] = (pi < s && qi >= s) || (pi >= s && qi < s);
            col[p * NP + q] = (pj < s && qj >= s) || (pj >= s && qj < s);
        }
    }
    (row, col)
}

/// Windowed multi-head self-attention (Swin W-MSA / SW-MSA).
#[derive(Module, Debug)]
struct Wmsa<B: Backend> {
    input_dim: usize,
    embedding_layer: Linear<B>,
    linear: Linear<B>,
    relative_position_params: Param<Tensor<B, 3>>, // [n_heads, 2*ws-1, 2*ws-1]
}

impl<B: Backend> Wmsa<B> {
    fn new(input_dim: usize, device: &B::Device) -> Self {
        let n_heads = input_dim / HEAD_DIM;
        Self {
            input_dim,
            embedding_layer: LinearConfig::new(input_dim, 3 * input_dim).init(device),
            linear: LinearConfig::new(input_dim, input_dim).init(device),
            relative_position_params: Param::initialized(
                ParamId::new(),
                Tensor::<B, 3>::zeros([n_heads, BIAS_BASE, BIAS_BASE], device),
            ),
        }
    }

    /// x is NHWC `[b, h, w, c]` (torch Block layout). burn's `roll(s)` shifts
    /// opposite to torch's (`cat([x[s..], x[..s]])` vs torch `out[i]=x[i-s]`),
    /// so torch's SW shift `-4`/`+4` become burn `+4`/`-4`. Windows 8×8, hd 32.
    fn forward(&self, x: Tensor<B, 4>, sw: bool) -> Tensor<B, 4> {
        let [b, h, w, c] = x.dims();
        let device = self.relative_position_params.device();
        let n_heads = self.input_dim / HEAD_DIM;        let hh = h / WS;
        let ww = w / WS;
        let nw = hh * ww;
        let shift = (WS / 2) as i64;

        let mut x = x;
        if sw {
            x = x.roll(&[shift, shift], &[1, 2]);
        }
        // Window partition: [b, hh, ww, WS, WS, c] → [b, nw, NP, c].
        let win = x
            .reshape([b, hh, WS, ww, WS, c])
            .permute([0, 1, 3, 2, 4, 5])
            .reshape([b, nw, NP, c]);

        // qkv split on the channel dim, then [b,nw,np,heads,hd] → [heads,b,nw,np,hd].
        let hc = n_heads * HEAD_DIM;
        let qkv = self.embedding_layer.forward(win); // [b, nw, np, 3c]
        let q = qkv
            .clone()
            .slice([0..b, 0..nw, 0..NP, 0..hc])
            .reshape([b, nw, NP, n_heads, HEAD_DIM])
            .permute([3, 0, 1, 2, 4]);
        let k = qkv
            .clone()
            .slice([0..b, 0..nw, 0..NP, hc..2 * hc])
            .reshape([b, nw, NP, n_heads, HEAD_DIM])
            .permute([3, 0, 1, 2, 4]);
        let v = qkv
            .slice([0..b, 0..nw, 0..NP, 2 * hc..3 * hc])
            .reshape([b, nw, NP, n_heads, HEAD_DIM])
            .permute([3, 0, 1, 2, 4]);

        let scale = (HEAD_DIM as f32).powf(-0.5);
        let sim = q.matmul(k.transpose()) * scale; // [heads, b, nw, np, np]

        // Relative position bias [heads, np, np] → broadcast. torch advanced-
        // indexes the [n_heads, 15, 15] param; burn gather needs same-rank
        // indices, so gather each head row (2D [1,225] with [1,4096] idx) and
        // reshape to [np, np], then stack.
        let rel_rows: Vec<Tensor<B, 2>> = (0..n_heads)
            .map(|h| {
                let row = self
                    .relative_position_params
                    .val()
                    .clone()
                    .slice([h..h + 1, 0..BIAS_BASE, 0..BIAS_BASE])
                    .flatten::<2>(1, 2); // [1, 225]
                let idx = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(relative_index(), [1, NP * NP]),
                    &device,
                ); // [1, 4096]
                row.gather(1, idx).reshape([NP, NP])
            })
            .collect();
        let rel: Tensor<B, 3> = Tensor::stack(rel_rows, 0); // [heads, np, np]
        let sim = sim + rel.unsqueeze_dim::<4>(1).unsqueeze_dim::<5>(1); // broadcast [heads,1,1,np,np]

        let sim = if sw { self.sw_mask(sim, hh, ww) } else { sim };
        let probs = softmax(sim, 4);

        let out = probs.matmul(v); // [heads, b, nw, np, hd]
        let out = out.permute([1, 2, 3, 0, 4]).reshape([b, nw, NP, self.input_dim]);
        let out = self.linear.forward(out);

        // Window reverse.
        let out = out
            .reshape([b, hh, ww, WS, WS, c])
            .permute([0, 1, 3, 2, 4, 5])
            .reshape([b, h, w, c]);
        if sw {
            out.roll(&[-shift, -shift], &[1, 2])
        } else {
            out
        }
    }

    /// -inf for cross-window (shifted) token pairs; layout [heads, b, nw, np, np].
    fn sw_mask(&self, sim: Tensor<B, 5>, hh: usize, ww: usize) -> Tensor<B, 5> {
        let device = self.relative_position_params.device();        let (row, col) = sw_cross_patterns();
        let row_t = Tensor::<B, 2, Bool>::from_data(TensorData::new(row, [NP, NP]), &device);
        let col_t = Tensor::<B, 2, Bool>::from_data(TensorData::new(col, [NP, NP]), &device);
        let mut mask = Tensor::<B, 4, Bool>::zeros([hh, ww, NP, NP], &device);
        if hh > 0 {
            mask = mask.slice_assign(
                [hh - 1..hh, 0..ww, 0..NP, 0..NP],
                row_t.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0).expand([1, ww, NP, NP]),
            );
        }
        if ww > 0 {
            mask = mask.slice_assign(
                [0..hh, ww - 1..ww, 0..NP, 0..NP],
                col_t.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0).expand([hh, 1, NP, NP]),
            );
        }
        let nw = hh * ww;
        let mask = mask.reshape([nw, NP, NP]).unsqueeze_dim::<4>(0).unsqueeze_dim::<5>(0);
        let neg_inf = Tensor::<B, 5>::full([1, 1, nw, NP, NP], f32::NEG_INFINITY, &device);
        let [heads, b, _, np1, np2] = sim.dims();
        sim.mask_where(
            mask.expand([heads, b, nw, np1, np2]),
            neg_inf.expand([heads, b, nw, np1, np2]),
        )
    }
}

/// Swin transformer block: ln1 → WMSA → +x; ln2 → MLP(GELU) → +x.
#[derive(Module, Debug)]
struct SwinBlock<B: Backend> {
    ln1: LayerNorm<B>,
    msa: Wmsa<B>,
    ln2: LayerNorm<B>,
    mlp0: Linear<B>,
    mlp2: Linear<B>,
    sw: bool,
}

impl<B: Backend> SwinBlock<B> {
    fn new(dim: usize, sw: bool, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(dim).init(device),
            msa: Wmsa::new(dim, device),
            ln2: LayerNormConfig::new(dim).init(device),
            mlp0: LinearConfig::new(dim, 4 * dim).init(device),
            mlp2: LinearConfig::new(4 * dim, dim).init(device),
            sw,
        }
    }

    /// NCHW in/out; burn LayerNorm normalizes the last dim, so permute to NHWC
    /// (torch Block runs `b h w c`) around the swin branch.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = x.permute([0, 2, 3, 1]);
        let n1 = self.ln1.forward(x.clone());
        let a = self.msa.forward(n1, self.sw);
        let x = x + a;
        let n2 = self.ln2.forward(x.clone());
        let m = self.mlp2.forward(gelu(self.mlp0.forward(n2)));
        (x + m).permute([0, 3, 1, 2])
    }
}

/// Conv branch: 3×3 conv (bias=false) + ReLU + 3×3 conv, residual.
#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    c0: Conv2d<B>,
    c2: Conv2d<B>,
}

impl<B: Backend> ConvBlock<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            c0: conv3(dim, dim, false, device),
            c2: conv3(dim, dim, false, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.c2.forward(burn::tensor::activation::relu(self.c0.forward(x.clone())));
        x + h
    }
}

/// ConvTransBlock: conv1_1 splits channels into conv/trans halves; the swin
/// branch runs on the trans half, the conv branch on the conv half; conv1_2
/// merges; residual.
#[derive(Module, Debug)]
struct ConvTransBlock<B: Backend> {
    conv_dim: usize,
    trans_dim: usize,
    conv1_1: Conv2d<B>,
    conv1_2: Conv2d<B>,
    conv_block: ConvBlock<B>,
    trans_block: SwinBlock<B>,
}

impl<B: Backend> ConvTransBlock<B> {
    fn new(conv_dim: usize, trans_dim: usize, sw: bool, device: &B::Device) -> Self {
        let both = conv_dim + trans_dim;
        Self {
            conv_dim,
            trans_dim,
            conv1_1: conv1(both, both, device),
            conv1_2: conv1(both, both, device),
            conv_block: ConvBlock::new(conv_dim, device),
            trans_block: SwinBlock::new(trans_dim, sw, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let split = self.conv1_1.forward(x.clone());
        let conv_x = split.clone().slice([0..1, 0..self.conv_dim]);
        let trans_x = split.slice([0..1, self.conv_dim..self.conv_dim + self.trans_dim]);
        let conv_out = self.conv_block.forward(conv_x);
        let trans_out = self.trans_block.forward(trans_x);
        let merged = Tensor::cat(vec![conv_out, trans_out], 1);
        let res = self.conv1_2.forward(merged);
        x + res
    }
}

/// SCUNet U-Net (config [4,4,4,4,4,4,4], dim 64, window 8, head_dim 32).
/// Block arrays index 0-3; levels alternate W/SW (even W, odd SW).
#[derive(Module, Debug)]
pub struct Scunet<B: Backend> {
    m_head: Conv2d<B>,
    m_down1: [ConvTransBlock<B>; 4],
    m_down1_down: Conv2d<B>,
    m_down2: [ConvTransBlock<B>; 4],
    m_down2_down: Conv2d<B>,
    m_down3: [ConvTransBlock<B>; 4],
    m_down3_down: Conv2d<B>,
    m_body: [ConvTransBlock<B>; 4],
    m_up3_up: ConvTranspose2d<B>,
    m_up3: [ConvTransBlock<B>; 4],
    m_up2_up: ConvTranspose2d<B>,
    m_up2: [ConvTransBlock<B>; 4],
    m_up1_up: ConvTranspose2d<B>,
    m_up1: [ConvTransBlock<B>; 4],
    m_tail: Conv2d<B>,
}

impl<B: Backend> Scunet<B> {
    pub fn new(device: &B::Device) -> Self {
        // conv_dim / trans_dim per level (dim=64): head 64, down1 32/32,
        // down2 64/64, down3 128/128, body 256/256; 4 blocks alternate W/SW.
        let lvl = |conv_dim, trans_dim, device: &B::Device| {
            [
                ConvTransBlock::new(conv_dim, trans_dim, false, device),
                ConvTransBlock::new(conv_dim, trans_dim, true, device),
                ConvTransBlock::new(conv_dim, trans_dim, false, device),
                ConvTransBlock::new(conv_dim, trans_dim, true, device),
            ]
        };
        Self {
            m_head: conv3(3, 64, false, device),
            m_down1: lvl(32, 32, device),
            m_down1_down: conv_down(64, 128, device),
            m_down2: lvl(64, 64, device),
            m_down2_down: conv_down(128, 256, device),
            m_down3: lvl(128, 128, device),
            m_down3_down: conv_down(256, 512, device),
            m_body: lvl(256, 256, device),
            m_up3_up: conv_up(512, 256, device),
            m_up3: lvl(128, 128, device),
            m_up2_up: conv_up(256, 128, device),
            m_up2: lvl(64, 64, device),
            m_up1_up: conv_up(128, 64, device),
            m_up1: lvl(32, 32, device),
            m_tail: conv3(64, 3, false, device),
        }
    }

    pub fn forward(&self, x0: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, _, h, w] = x0.dims();
        let pad_h = (h + 63) / 64 * 64;
        let pad_w = (w + 63) / 64 * 64;
        // replication pad to a multiple of 64 (`right` extra rows/cols: `rep`
        // starts as one copy, so the loop adds `right - 1` more).
        let mut x0 = x0;
        if pad_h > h {
            let right = pad_h - h;
            let pad = x0.clone().slice([0..b, 0..3, (h - 1)..h, 0..w]);
            let mut rep = pad.clone();
            for _ in 1..right {
                rep = Tensor::cat(vec![rep, pad.clone()], 2);
            }
            x0 = Tensor::cat(vec![x0, rep], 2);
        }
        if pad_w > w {
            let right = pad_w - w;
            let pad = x0.clone().slice([0..b, 0..3, 0..pad_h, (w - 1)..w]);
            let mut rep = pad.clone();
            for _ in 1..right {
                rep = Tensor::cat(vec![rep, pad.clone()], 3);
            }
            x0 = Tensor::cat(vec![x0, rep], 3);
        }

        let x1 = self.m_head.forward(x0);
        let x = self.m_down1.iter().fold(x1.clone(), |acc, b| b.forward(acc));
        let x2 = self.m_down1_down.forward(x);
        let x = self.m_down2.iter().fold(x2.clone(), |acc, b| b.forward(acc));
        let x3 = self.m_down2_down.forward(x);
        let x = self.m_down3.iter().fold(x3.clone(), |acc, b| b.forward(acc));
        let x4 = self.m_down3_down.forward(x);
        let x = self.m_body.iter().fold(x4.clone(), |acc, b| b.forward(acc));
        let x = self.m_up3.iter().fold(self.m_up3_up.forward(x + x4), |acc, b| b.forward(acc));
        let x = self.m_up2.iter().fold(self.m_up2_up.forward(x + x3), |acc, b| b.forward(acc));
        let x = self.m_up1.iter().fold(self.m_up1_up.forward(x + x2), |acc, b| b.forward(acc));
        let x = self.m_tail.forward(x + x1);

        x.slice([0..b, 0..3, 0..h, 0..w])
    }
}
