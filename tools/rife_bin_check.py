#!/usr/bin/env python3
"""Verify and summarize a rife-v4.6 ncnn `flownet.bin` against its `.param`.

The .bin format (fp16 storage): per weighted layer, in .param order —
[tag u32 = 0x01306B47][weights wsize × f16][bias out × f32 if bias_term].

Usage: python3 tools/rife_bin_check.py ref/rife-v4.6/flownet.param flownet.bin
"""

import re
import struct
import sys

F16_TAG = 0x01306B47


def weighted_layers(param_path):
    out = []
    for line in open(param_path).read().splitlines()[2:]:
        p = line.split()
        toks = p[4 + int(p[2]) + int(p[3]):]
        params = {}
        for tok in toks:
            m = re.match(r"(-?\d+)=(.*)", tok)
            if m:
                params[int(m.group(1))] = m.group(2)
        if p[0] in ("Convolution", "Deconvolution"):
            out.append((p[0], p[1], int(params.get(0, 0)), int(params.get(6, 0)), int(params.get(5, 0))))
    return out


def main():
    param_path, bin_path = sys.argv[1], sys.argv[2]
    layers = weighted_layers(param_path)
    data = open(bin_path, "rb").read()
    pos = 0
    print(f"# {bin_path}: {len(layers)} weighted layers, {len(data)} bytes")
    for layer_type, name, out, wsize, bias_term in layers:
        tag = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        assert tag == F16_TAG, f"{name}: expected fp16 tag 0x{F16_TAG:08X}, got 0x{tag:08X}"
        wt = struct.unpack(f"<{wsize}e", data[pos : pos + 2 * wsize])
        pos += 2 * wsize
        bias = None
        if bias_term:
            bias = struct.unpack(f"<{out}f", data[pos : pos + 4 * out])
            pos += 4 * out
        print(
            f"{name:16s} {layer_type[:4]:4s} out={out:4d} w={wsize:7d} "
            f"w[0:2]={[round(x,3) for x in wt[:2]]} "
            + (f"b[0:2]={[round(x,4) for x in bias[:2]]}" if bias else "bias=none")
        )
    assert pos == len(data), f"EOF mismatch: {pos} != {len(data)}"
    print("EOF ok: exactly", pos, "bytes consumed")


if __name__ == "__main__":
    main()
