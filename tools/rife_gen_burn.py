#!/usr/bin/env python3
"""Generate the burn `rife.rs` module from the rife-v4.6 ncnn `flownet.param`.

Maps the flattened ncnn graph (215 layers) to a straight-line burn forward:
Conv2d / ConvTranspose2d fields + op helpers (interp, pixel-shuffle, warp,
binary ops, channel crop, concat). Blob variables are `b_<name>`; non-final
uses of a blob are cloned to satisfy burn's move semantics.

Usage: python3 tools/rife_gen_burn.py ref/rife-v4.6/flownet.param > crates/senmei-ml/src/burn/rife.rs
"""

import re
import sys

L = sys.argv[1] if len(sys.argv) > 1 else "ref/rife-v4.6/flownet.param"
lines = open(L).read().splitlines()


def parse_layer(line):
    p = line.split()
    lt, name = p[0], p[1]
    bottom, top = int(p[2]), int(p[3])
    inputs = p[4 : 4 + bottom]
    outputs = p[4 + bottom : 4 + bottom + top]
    params = {}
    for tok in p[4 + bottom + top :]:
        m = re.match(r"(-?\d+)=(.*)", tok)
        if m:
            params[int(m.group(1))] = m.group(2)
    return lt, name, inputs, outputs, params


def mat(v):
    if not v:
        return []
    return [int(x) for x in v.split(",")][1:]


layers = [parse_layer(l) for l in lines[2:]]

# --- shape inference (NCHW) to know conv in-channels ---
H, W = 256, 256
blobs = {"in0": (3, H, W), "in1": (3, H, W), "in2": (1, H, W)}
for lt, name, inputs, outputs, params in layers:
    inp = [blobs.get(b) for b in inputs]
    outs = []
    if lt == "Convolution":
        _, h, w = inp[0]
        k = int(params.get(1, 0)); s = int(params.get(3, 1)); p = int(params.get(4, 0))
        ho = (h + 2 * p - (k - 1) - 1) // s + 1
        outs = [(int(params.get(0)), ho, ho)]
    elif lt == "Deconvolution":
        _, h, w = inp[0]
        k = int(params.get(1, 0)); s = int(params.get(3, 1)); p = int(params.get(4, 0))
        ho = (h - 1) * s + k - 2 * p
        outs = [(int(params.get(0)), ho, ho)]
    elif lt == "Interp":
        c, h, w = inp[0]
        scale = float(params.get(1, 1.0))
        outs = [(c, int(round(h * scale)), int(round(w * scale)))]
    elif lt == "PixelShuffle":
        c, h, w = inp[0]
        outs = [(c // 4, h * 2, w * 2)]
    elif lt in ("BinaryOp", "Eltwise", "rife.Warp", "ReLU", "Sigmoid", "Split", "Reorg"):
        outs = [inp[0]] * len(outputs) if inp and inp[0] else []
    elif lt == "Concat":
        outs = [(sum(x[0] for x in inp), inp[0][1], inp[0][2])]
    elif lt == "Crop":
        c, h, w = inp[0]
        co, ho, wo = c, h, w
        for axis, s, e in zip(mat(params.get(-23311)), mat(params.get(-23309)), mat(params.get(-23310))):
            if axis == 0:
                co = (e if e else c) - (s if s else 0)
            elif axis == 1:
                ho = (e if e else h) - (s if s else 0)
            else:
                wo = (e if e else w) - (s if s else 0)
        outs = [(co, ho, wo)]
    else:
        outs = [inp[0]] * len(outputs) if inp and inp[0] else []
    for o, sh in zip(outputs, outs):
        blobs[o] = sh

# --- blob use counts for move semantics ---
# A Split consumes its input once but needs a copy per output edge.
uses = {}
for lt, _, inputs, outputs, _ in layers:
    for b in inputs:
        uses[b] = uses.get(b, 0) + (len(outputs) if lt == "Split" else 1)
consumed = {b: 0 for b in uses}

def take(blob):
    consumed[blob] += 1
    last = consumed[blob] == uses[blob]
    return f"b_{blob}" if last else f"b_{blob}.clone()"

# --- forward body ---
body = []
for lt, name, inputs, outputs, params in layers:
    lines_out = []
    def assign(o, expr):
        lines_out.append(f"        let b_{o} = {expr};")
    if lt == "Input":
        for o in outputs:
            assign(o, o)
    elif lt == "Split":
        for o in outputs:
            assign(o, take(inputs[0]))
    elif lt == "Convolution":
        expr = f"self.{name}.forward({take(inputs[0])})"
        if params.get(9) == "2":
            expr = f"leaky_relu({expr}, 0.2)"
        assign(outputs[0], expr)
    elif lt == "Deconvolution":
        assign(outputs[0], f"self.{name}.forward({take(inputs[0])})")
    elif lt == "ReLU":
        assign(outputs[0], f"leaky_relu({take(inputs[0])}, {params.get(0, '0.2')})")
    elif lt == "Sigmoid":
        assign(outputs[0], f"sigmoid({take(inputs[0])})")
    elif lt == "Concat":
        assign(outputs[0], f"Tensor::cat(vec![{', '.join(take(b) for b in inputs)}], 1)")
    elif lt == "Interp":
        assign(outputs[0], f"interp({take(inputs[0])}, {params.get(1)})")
    elif lt == "PixelShuffle":
        assign(outputs[0], f"pixel_shuffle({take(inputs[0])})")
    elif lt == "rife.Warp":
        assign(outputs[0], f"warp({take(inputs[0])}, {take(inputs[1])})")
    elif lt == "Crop":
        s = mat(params.get(-23309))[0] if mat(params.get(-23309)) else 0
        e = mat(params.get(-23310))[0] if mat(params.get(-23310)) else -1
        assign(outputs[0], f"slice_c({take(inputs[0])}, {s}, {e})")
    elif lt == "BinaryOp":
        op = int(params.get(0, 0))
        scalar = params.get(2)
        a = take(inputs[0])
        b = take(inputs[1]) if len(inputs) > 1 else None
        if op == 2 and scalar:
            assign(outputs[0], f"{a} * {scalar}")
        elif op == 2:
            assign(outputs[0], f"{a} * {b}")
        elif op == 3 and scalar:
            assign(outputs[0], f"{a} / {scalar}")
        elif op == 7 and scalar:
            assign(outputs[0], f"sub_r({a}, {scalar})")
        else:
            assign(outputs[0], f"{a} + {b}")
    elif lt == "Eltwise":
        coeffs = [float(x) for x in params.get(-23301, "").split(",")[1:]]
        terms = " + ".join(f"{take(b)} * {c}" for c, b in zip(coeffs, inputs))
        assign(outputs[0], terms)
    else:
        lines_out.append(f"        // unhandled {lt} {name}")
    body.append("\n".join(lines_out))

forward = "\n".join(body)

# --- struct fields + new() + load_from_ncnn from shape inference ---
fields = []
new_body = []
load_body = []
weighted = []
for lt, name, inputs, outputs, params in layers:
    if lt == "Convolution":
        in_c = blobs[inputs[0]][0]
        out_c = int(params.get(0))
        k = int(params.get(1, 3))
        s = int(params.get(3, 1))
        bias = int(params.get(5, 0)) == 1
        fields.append(f"    {name}: Conv2d<B>,")
        new_body.append(f"        {name}: conv2d({in_c}, {out_c}, {s}, device),")
        load_body.append(f"        let (w, b) = rd({out_c}, {in_c * out_c * k * k}, {in_c}, {k}, {str(bias).lower()}, false);")
        load_body.append(f"        self.{name}.weight = Param::from_tensor(w);")
        load_body.append(f"        self.{name}.bias = b.map(Param::from_tensor);")
        weighted.append((name, out_c, in_c * out_c * k * k, in_c, k, bias))
    elif lt == "Deconvolution":
        in_c = blobs[inputs[0]][0]
        out_c = int(params.get(0))
        k = int(params.get(1, 4))
        bias = int(params.get(5, 0)) == 1
        fields.append(f"    {name}: ConvTranspose2d<B>,")
        new_body.append(f"        {name}: deconv2d({in_c}, {out_c}, device),")
        load_body.append(f"        let (w, b) = rd({out_c}, {in_c * out_c * k * k}, {in_c}, {k}, {str(bias).lower()}, true);")
        load_body.append(f"        self.{name}.weight = Param::from_tensor(w);")
        load_body.append(f"        self.{name}.bias = b.map(Param::from_tensor);")
        weighted.append((name, out_c, in_c * out_c * k * k, in_c, k, bias))

fields_txt = "\n".join(fields)
new_txt = "\n".join(new_body)
load_txt = "\n".join(load_body)
n_layer = len(weighted)

print(f"""//! RIFE v4.6 (`flownet`) — clean burn port, generated from the ncnn graph.
//!
//! Generated by `tools/rife_gen_burn.py` from `ref/rife-v4.6/flownet.param`
//! (nihui/rife-ncnn-vulkan, MIT). Do not hand-edit — regenerate instead.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}};
use burn::tensor::backend::Backend;
use burn::tensor::activation::{{leaky_relu, sigmoid}};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{{InterpolateMode, InterpolateOptions}};
use burn::tensor::{{Int, Tensor}};

use super::warp::grid_sample;

fn conv2d<B: Backend>(in_c: usize, out_c: usize, stride: usize, device: &B::Device) -> Conv2d<B> {{
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}}

fn deconv2d<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> ConvTranspose2d<B> {{
    ConvTranspose2dConfig::new([in_c, out_c], [4, 4])
        .with_stride([2, 2])
        .with_padding([1, 1])
        .init(device)
}}

/// Channel-axis slice [s..e) (ncnn Crop on axis 0).
fn slice_c<B: Backend>(x: Tensor<B, 4>, s: usize, e: usize) -> Tensor<B, 4> {{
    let [n, _c, h, w] = x.dims();
    x.slice([0..n, s..e, 0..h, 0..w])
}}

/// Bilinear resize by a scale factor (ncnn Interp, type 2).
fn interp<B: Backend>(x: Tensor<B, 4>, scale: f32) -> Tensor<B, 4> {{
    let [_, _, h, w] = x.dims();
    let oh = ((h as f32) * scale).round() as usize;
    let ow = ((w as f32) * scale).round() as usize;
    interpolate(
        x,
        [oh, ow],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    )
}}

/// PixelShuffle upscale by 2.
fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {{
    let [n, c, h, w] = x.dims();
    x.reshape([n, c / 4, 2, 2, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([n, c / 4, h * 2, w * 2])
}}

/// rife.Warp: backward bilinear warp by a 2-channel flow (align_corners=true,
/// border padding) — matches `warp.comp`.
fn warp<B: Backend>(img: Tensor<B, 4>, flow: Tensor<B, 4>) -> Tensor<B, 4> {{
    let [n, _c, h, w] = img.dims();
    let fx = flow.clone().slice([0..n, 0..1, 0..h, 0..w]);
    let fy = flow.slice([0..n, 1..2, 0..h, 0..w]);

    // pixel coordinates 0..W-1 / 0..H-1 broadcast over batch/spatial
    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, &img.device()).float().reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, &img.device()).float().reshape([1, 1, h, 1]);

    // sample = coord + flow, normalized to [-1,1] (align_corners=True)
    let sx = (xs + fx) / ((w - 1) as f64 / 2.0) - 1.0;
    let sy = (ys + fy) / ((h - 1) as f64 / 2.0) - 1.0;
    let grid = Tensor::cat(vec![sx.permute([0, 2, 3, 1]), sy.permute([0, 2, 3, 1])], 3);
    grid_sample(img, grid)
}}

/// scalar - tensor (ncnn BinaryOp RSUB).
fn sub_r<B: Backend>(x: Tensor<B, 4>, s: f32) -> Tensor<B, 4> {{
    Tensor::ones_like(&x) * s - x
}}

#[derive(Module, Debug)]
pub struct RifeNet<B: Backend> {{
{fields_txt}}}

impl<B: Backend> RifeNet<B> {{
    pub fn new(device: &B::Device) -> Self {{
        Self {{
{new_txt}        }}
    }}

    /// Interpolate frame `in0` -> `in1` at `in2` (timestep in [0,1]).
    pub fn forward(&self, in0: Tensor<B, 4>, in1: Tensor<B, 4>, in2: Tensor<B, 4>) -> Tensor<B, 4> {{
{forward}        b_out0
    }}

    /// Load weights from the rife-v4.6 ncnn `flownet.bin`.
    ///
    /// Format (per weighted layer, in .param order):
    /// `[tag u32 = 0x01306B47][weights wsize x f16][bias out x f32 if bias_term]`
    pub fn load_from_ncnn(&mut self, bin: &[u8], device: &B::Device) -> Result<(), String> {{
        use burn::module::Param;
        use burn::tensor::{{f16, TensorData}};
        let mut pos = 0usize;
        let mut rd = |out: usize, wsize: usize, in_c: usize, k: usize, bias: bool, transpose: bool|
            -> (Tensor<B, 4>, Option<Tensor<B, 1>>) {{
            pos += 4; // fp16 tag
            let w: Vec<f32> = (0..wsize)
                .map(|i| f16::from_bits(u16::from_le_bytes([bin[pos + 2 * i], bin[pos + 2 * i + 1]])).to_f32())
                .collect();
            pos += 2 * wsize;
            let b = if bias {{
                let bv: Vec<f32> = (0..out)
                    .map(|i| f32::from_le_bytes([bin[pos + 4 * i], bin[pos + 4 * i + 1], bin[pos + 4 * i + 2], bin[pos + 4 * i + 3]]))
                    .collect();
                pos += 4 * out;
                Some(Tensor::from_data(TensorData::new(bv, [out]), device))
            }} else {{
                None
            }};
            // ncnn stores deconv weights out-major [out, in, k, k]; burn's
            // ConvTranspose2d expects [in, out, k, k].
            let wt = Tensor::from_data(TensorData::new(w, [out, in_c, k, k]), device);
            let wt = if transpose {{ wt.permute([1, 0, 2, 3]) }} else {{ wt }};
            (wt, b)
        }};
{load_txt}        if pos != bin.len() {{
            return Err(format!("ncnn bin: consumed {{pos}} of {{}} bytes", bin.len()));
        }}
        Ok(())
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    use burn::tensor::TensorData;

    /// Structural check: the graph runs end-to-end and preserves resolution.
    /// Weights are random (unloaded), so this validates shapes/wiring only.
    #[test]
    #[ignore = "requires Vulkan"]
    fn rife_net_forward_preserves_shape() {{
        use burn_wgpu::{{Vulkan, WgpuDevice}};
        let device = WgpuDevice::DiscreteGpu(0);
        let net = RifeNet::<Vulkan<f32>>::new(&device);
        let in0 = Tensor::<Vulkan<f32>, 4>::from_data(TensorData::new(vec![0.5f32; 3 * 64 * 64], [1, 3, 64, 64]), &device);
        let in1 = Tensor::<Vulkan<f32>, 4>::from_data(TensorData::new(vec![0.6f32; 3 * 64 * 64], [1, 3, 64, 64]), &device);
        let t = Tensor::<Vulkan<f32>, 4>::from_data(TensorData::new(vec![0.5f32; 64 * 64], [1, 1, 64, 64]), &device);
        let out = net.forward(in0, in1, t);
        assert_eq!(out.dims(), [1, 3, 64, 64]);
        let v: Vec<f32> = out.into_data().convert::<f32>().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }}}}""")
