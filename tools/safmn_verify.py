#!/usr/bin/env python3
"""SAFMN verification for the burn port.

Rebuilds the torch SAFMN (dim 128 / 16 blocks / ffn_scale 2.0, clean
re-implementation from the Apache-2.0 sunny2109/SAFMN reference), loads the
real `SAFMN_L_Real_LSDIR_x{scale}-v2` weights (params_ema), and writes
`x.bin` (input) + `ref.bin` (torch output) as f32 little-endian for the
`arch::safmn::safmn_matches_torch_reference` test.

Usage: safmn_verify.py [scale=2] [pth=/tmp/safmn/SAFMN_L_Real_LSDIR_x2-v2.pth] [outdir=/tmp/safmn_verify]
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        return F.gelu(out) * x


class CCM(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        hidden_dim = int(dim * ffn_scale)
        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1), nn.GELU(), nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.safm = SAFM(dim)
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class SAFMN(nn.Module):
    def __init__(self, dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


def main() -> None:
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pth = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f"/tmp/safmn/SAFMN_L_Real_LSDIR_x{scale}-v2.pth"
    )
    outdir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/safmn_verify"
    os.makedirs(outdir, exist_ok=True)

    net = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=scale).eval()
    obj = torch.load(pth, map_location="cpu", weights_only=False)
    sd = obj.get("params_ema", obj.get("params", obj))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  missing={missing}")
        print(f"  unexpected={unexpected[:10]} ({len(unexpected)} total)")
        raise SystemExit("state dict mismatch")
    print(f"scale {scale}: loaded {len(sd)} tensors")

    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32, dtype=torch.float32)  # [0,1] input
    with torch.no_grad():
        ref = net(x)
    print(f"input {tuple(x.shape)} -> output {tuple(ref.shape)}")

    x.numpy().astype("<f4").tofile(f"{outdir}/x.bin")
    ref.numpy().astype("<f4").tofile(f"{outdir}/ref.bin")
    print(f"wrote {outdir}/x.bin, {outdir}/ref.bin")


if __name__ == "__main__":
    main()
