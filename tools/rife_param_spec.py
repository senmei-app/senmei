#!/usr/bin/env python3
"""Dump a readable spec (with shape inference) for an ncnn RIFE flownet .param.

The rife-v4.6 model is a single network (flownet) — no separate contextnet/
fusionnet. This script translates the ncnn graph into the layer-by-layer
blueprint used to port the network to burn, simulating every blob's
(channels, height, width) through the graph.

Usage: python3 tools/rife_param_spec.py ref/rife-v4.6/flownet.param
"""

import re
import sys


def parse_param(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    magic = lines[0]
    assert magic == "7767517", f"not an ncnn param: {magic}"
    layer_count, blob_count = map(int, lines[1].split())
    layers = []
    for line in lines[2:]:
        parts = line.split()
        layer_type = parts[0]
        name = parts[1]
        bottom = int(parts[2])
        top = int(parts[3])
        inputs = parts[4 : 4 + bottom]
        outputs = parts[4 + bottom : 4 + bottom + top]
        params = {}
        for tok in parts[4 + bottom + top :]:
            m = re.match(r"(-?\d+)=(.*)", tok)
            if m:
                params[int(m.group(1))] = m.group(2)
        layers.append(
            {"type": layer_type, "name": name, "inputs": inputs, "outputs": outputs, "params": params}
        )
    return layers, layer_count, blob_count


def parse_shape(v):
    # "1,2" -> [1,2] ; "0" or missing -> None (keep/unspecified)
    return [int(x) for x in v.split(",")] if v else None


def conv_out(h, w, k, s, p, d=1):
    ho = (h + 2 * p - d * (k - 1) - 1) // s + 1
    return ho


def infer_shapes(layers, H, W):
    blobs = {"in0": (3, H, W), "in1": (3, H, W), "in2": (1, H, W)}
    rows = []
    for L in layers:
        t, n = L["type"], L["name"]
        p = L["params"]
        inp = [blobs.get(b) for b in L["inputs"]]
        outs = []
        if t == "Input":
            pass  # pre-seeded
        elif t == "Convolution":
            c = int(p.get(0, 0)); k = int(p.get(1, 0)); s = int(p.get(3, 1)); pad = int(p.get(4, 0))
            ci, h, w = inp[0]
            ho = conv_out(h, w, k, s, pad)
            outs = [(c, ho, ho)]
        elif t == "Deconvolution":
            c = int(p.get(0, 0)); k = int(p.get(1, 0)); s = int(p.get(3, 1)); pad = int(p.get(4, 0))
            ci, h, w = inp[0]
            ho = (h - 1) * s + k - 2 * pad
            outs = [(c, ho, ho)]
        elif t == "Interp":
            ci, h, w = inp[0]
            scale = float(p.get(1, 1.0))
            outs = [(ci, int(round(h * scale)), int(round(w * scale)))]
        elif t == "PixelShuffle":
            ci, h, w = inp[0]
            r = int(p.get(0, 2))
            outs = [(ci // (r * r), h * r, w * r)]
        elif t == "rife.Warp":
            outs = [inp[0]]
        elif t == "BinaryOp" or t == "Eltwise":
            a = inp[0] if inp[0] else None
            b = inp[1] if len(inp) > 1 and inp[1] else None
            res = a or b
            outs = [res]
        elif t == "Concat":
            c = sum(x[0] for x in inp)
            outs = [(c, inp[0][1], inp[0][2])]
        elif t == "Split" or t == "ReLU" or t == "Sigmoid" or t == "Reorg":
            outs = [inp[0]] * len(L["outputs"]) if inp and inp[0] else []
        elif t == "Crop":
            ci, h, w = inp[0]
            c0 = parse_shape(p.get(-23309)); h0 = parse_shape(p.get(-23310)); w0 = parse_shape(p.get(-23311))
            co = (c0 or [ci])[0]
            ho = (h0 or [h])[0]
            wo = (w0 or [w])[0]
            outs = [(co, ho, wo)]
        else:
            outs = [inp[0]] * len(L["outputs"]) if inp and inp[0] else []
        for b, o in zip(L["outputs"], outs):
            blobs[b] = o
        rows.append((t, n, L["inputs"], L["outputs"], p, inp, outs))
    return rows


def fmt(shape):
    return f"{shape[0]}x{shape[1]}x{shape[2]}" if shape else "?"


def main():
    path = sys.argv[1]
    layers, lc, bc = parse_param(path)
    rows = infer_shapes(layers, 256, 256)  # reference resolution 256x256
    print(f"# rife-v4.6 flownet — {lc} layers, {bc} blobs (inferred @256x256)")
    print()
    for t, n, ins, outs, p, pin, pout in rows:
        act = ""
        if t == "Convolution":
            a = p.get(9, "0")
            act = " relu" if a == "1" else f" leaky{p.get('-23310','?')}" if a == "2" else ""
        print(f"{n:22s} {t:12s} in={[f'{i}({fmt(pin[k])})' if k < len(pin) and pin[k] else i for k, i in enumerate(ins)]}")
        print(f"{'':22s} {'':12s} out={[f'{o}({fmt(pout[k])})' if k < len(pout) and pout[k] else o for k, o in enumerate(outs)]}"
              f"  {dict(sorted(p.items()))}{act}")


if __name__ == "__main__":
    main()
