#!/usr/bin/env python3
"""Numerical reference for the BSRGAN burn port (RRDBNet, 23 blocks, scale 4).

Mirrors the BasicSR RRDBNet exactly, with the *BSRGAN checkpoint key naming*
(`RRDB_trunk.{i}.RDB{j}.conv{k}`, `trunk_conv`, `upconv1/2`, `HRconv`,
`conv_first`, `conv_last`) so `load_state_dict` validates against the real
`BSRGAN.pth` (KAIR v1.0, MIT). Writes f32 bins for the Rust verification test.

usage: python3 tools/bsrgan_verify.py [outdir]   (default /tmp/bsrgan_verify)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), 1)), 0.2)
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)), 0.2)
        x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)), 0.2)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Three dense blocks; attribute names match the BSRGAN keys (RDB1..3)."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.RDB2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.RDB3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.RRDB_trunk = nn.ModuleList(
            [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.HRconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.scale = scale

    def forward(self, x):
        feat = self.conv_first(x)
        body = feat
        for b in self.RRDB_trunk:
            body = b(body)
        feat = feat + self.trunk_conv(body)
        # scale 4 = two nearest-2x + conv stages (upconv1 then upconv2)
        feat = F.leaky_relu(self.upconv1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        feat = F.leaky_relu(self.upconv2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        out = self.conv_last(F.leaky_relu(self.HRconv(feat), 0.2))
        return out


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bsrgan_verify"
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(0)
    n, c, h, w = 1, 3, 32, 32
    # Smooth gradient input (fp16-safe; flat/constant inputs stress the model).
    x = torch.linspace(0.1, 0.9, h * w).reshape(1, 1, h, w).expand(n, c, h, w).contiguous()

    model = RRDBNet(num_block=23, scale=4)
    d = torch.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "BSRGAN.pth"),
        map_location="cpu",
    )
    sd = d.get("params", d)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    model.eval()
    with torch.no_grad():
        ref = model(x)

    for name, t in [("x.bin", x), ("ref.bin", ref)]:
        with open(os.path.join(out, name), "wb") as f:
            f.write(t.numpy().astype("<f4").tobytes())
    print(f"wrote {out}  ref={list(ref.shape)}  range=[{float(ref.min()):.4f},{float(ref.max()):.4f}]")


if __name__ == "__main__":
    main()
