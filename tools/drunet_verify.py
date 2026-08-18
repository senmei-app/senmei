#!/usr/bin/env python3
"""Numerical reference for the DRUNet burn port.

Mirrors KAIR's `UNetRes` (network_unet.py + basicblock.py): strideconv downsample
(Conv2d k2 s2 p0), convtranspose upsample (ConvTranspose2d k2 s2 p0), ResBlock
(Conv-ReLU-Conv + skip via `res` sequential), all convs bias=False, in_nc=4
(RGB + constant noise-level map), out_nc=3. The `nn.Sequential` layout matches
the official state-dict keys (m_down1.0.res.0, m_down1.4, m_up3.0, ...). Loads
drunet_color.pth and writes f32 bins for the Rust verification test.

usage: python3 tools/drunet_verify.py [outdir]   (default /tmp/drunet_verify)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, nc, bias=False):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1, bias=bias),
        )

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4, bias=False):
        super().__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=bias)
        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0], bias) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=bias),
        )
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1], bias) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=bias),
        )
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2], bias) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=bias),
        )
        self.m_body = nn.Sequential(*[ResBlock(nc[3], bias) for _ in range(nb)])
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=bias),
            *[ResBlock(nc[2], bias) for _ in range(nb)],
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=bias),
            *[ResBlock(nc[1], bias) for _ in range(nb)],
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=bias),
            *[ResBlock(nc[0], bias) for _ in range(nb)],
        )
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=bias)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/drunet_verify"
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(0)
    n, c, h, w = 1, 4, 64, 64
    x = torch.rand(n, c, h, w, dtype=torch.float32)

    model = UNetRes(in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4, bias=False)
    sd = torch.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "drunet_color.pth"),
        map_location="cpu",
    )
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        ref = model(x)

    for name, t in [("x.bin", x), ("ref.bin", ref)]:
        with open(os.path.join(out, name), "wb") as f:
            f.write(t.numpy().astype("<f4").tobytes())
    print(f"wrote {out}  ref={list(ref.shape)}  range=[{float(ref.min()):.4f},{float(ref.max()):.4f}]")


if __name__ == "__main__":
    main()
