#!/usr/bin/env python3
"""SRVGGNetCompact verification for the burn port.

Rebuilds the FOLDED SRVGGNetCompact (the layout the real checkpoints use:
body convs interleaved with one shared PReLU, the last body conv
`num_feat → 3·scale²`, then PixelShuffle — no `upsampler.*`/`conv_last`),
loads the real weights (animevideo-xs or general-x4v3), and writes
`x.bin` (input) + `ref.bin` (torch output) as f32 little-endian for the
`arch::srvgg::srvgg_matches_torch_reference` test.

Usage: srvgg_verify.py [scale=4] [pth=/tmp/srvgg_general/realesr-general-x4v3.pth] [outdir=/tmp/srvgg_verify] [num_conv=32]
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SrvggFolded(nn.Module):
    """Folded SRVGGNetCompact matching the burn `SrvggNet` (1 + num_conv + 1
    body convs, last conv 3·scale², PixelShuffle). One PReLU per mid conv —
    distinct instances, so per-layer checkpoints (general-x4v3) load 1:1 and
    shared checkpoints (animevideo-xs) fill every layer with the same value."""

    def __init__(self, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale**2, 3, 1, 1))
        self.upscale = upscale

    def forward(self, x):
        x0 = x
        for m in self.body:
            x = m(x)
        out = F.pixel_shuffle(x, self.upscale)
        # SRVGG learns the residual: add the nearest-upsampled input.
        base = F.interpolate(x0, scale_factor=self.upscale, mode="nearest")
        return out + base


def main() -> None:
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pth = sys.argv[2] if len(sys.argv) > 2 else ""
    outdir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/srvgg_verify"
    num_conv = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(0)
    net = SrvggFolded(num_feat=64, num_conv=num_conv, upscale=scale).eval()
    if pth:
        obj = torch.load(pth, map_location="cpu", weights_only=False)
        sd = obj.get("params_ema", obj.get("params", obj))
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  missing={missing}")
            print(f"  unexpected={unexpected[:10]} ({len(unexpected)} total)")
            raise SystemExit("state dict mismatch")
        print(f"scale {scale} num_conv {num_conv}: loaded {len(sd)} tensors from {pth}")
    else:
        sd = net.state_dict()
        print(f"scale {scale} num_conv {num_conv}: {len(sd)} random tensors")

    x = torch.rand(1, 3, 32, 32, dtype=torch.float32)  # [0,1] input
    with torch.no_grad():
        ref = net(x)
    print(f"input {tuple(x.shape)} -> output {tuple(ref.shape)}")

    x.numpy().astype("<f4").tofile(f"{outdir}/x.bin")
    ref.numpy().astype("<f4").tofile(f"{outdir}/ref.bin")
    print(f"wrote {outdir}/x.bin, {outdir}/ref.bin")


if __name__ == "__main__":
    main()
