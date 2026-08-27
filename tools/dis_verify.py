#!/usr/bin/env python3
"""DIS (Direct Image Supersampling) verification for the burn port.

Rebuilds the DIS arch (Kim2091/DIS — head + PReLU, FastResBlock body, fusion,
single PixelShuffleUpsampler, tail, bilinear global residual), loads the real
weights, and writes `x.bin` (input) + `ref.bin` (torch output) as f32
little-endian for the `arch::dis::dis_matches_torch_reference` test.

Usage: dis_verify.py [scale=2] [safetensors=...] [outdir=/tmp/dis_verify] [num_blocks=8]
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Prelu(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.full((n,), 0.25))

    def forward(self, x):
        return F.prelu(x, self.weight)


class FastResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = Prelu(channels)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * scale**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act = Prelu(out_c)

    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))


class Dis(nn.Module):
    def __init__(self, num_features=32, num_blocks=8, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, num_features, 3, padding=1)
        self.head_act = Prelu(num_features)
        self.body = nn.Sequential(*[FastResBlock(num_features) for _ in range(num_blocks)])
        self.fusion = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsampler = PixelShuffleUpsampler(num_features, num_features, scale)
        self.tail = nn.Conv2d(num_features, 3, 3, padding=1)

    def forward(self, x):
        if self.scale == 1:
            base = x
        else:
            base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        feat = self.head_act(self.head(x))
        out = self.fusion(self.body(feat)) + feat
        out = self.upsampler(out)
        return self.tail(out) + base


def main() -> None:
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    st = sys.argv[2] if len(sys.argv) > 2 else ""
    outdir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/dis_verify"
    num_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(0)
    net = Dis(num_features=32, num_blocks=num_blocks, scale=scale).eval()
    if st:
        from safetensors.torch import load_file

        sd = load_file(st)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  missing={missing}")
            print(f"  unexpected={unexpected[:10]} ({len(unexpected)} total)")
            raise SystemExit("state dict mismatch")
        print(f"scale {scale} num_blocks {num_blocks}: loaded {len(sd)} tensors from {st}")
    else:
        sd = net.state_dict()
        print(f"scale {scale} num_blocks {num_blocks}: {len(sd)} random tensors")

    x = torch.rand(1, 3, 32, 32, dtype=torch.float32)  # [0,1] input
    with torch.no_grad():
        ref = net(x)
    print(f"input {tuple(x.shape)} -> output {tuple(ref.shape)}")

    x.numpy().astype("<f4").tofile(f"{outdir}/x.bin")
    ref.numpy().astype("<f4").tofile(f"{outdir}/ref.bin")
    print(f"wrote {outdir}/x.bin, {outdir}/ref.bin")


if __name__ == "__main__":
    main()
