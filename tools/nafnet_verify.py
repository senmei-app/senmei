#!/usr/bin/env python3
"""Numerical reference for the NAFNet burn port.

Mirrors megvii-research/NAFNet `basicsr/models/archs/NAFNet_arch.py` exactly:
NAFBlock = LayerNorm2d → conv1(1x1, c→2c) → conv2(depthwise 3x3) → SimpleGate
(chunk-2 multiply) → SCA (`x * conv1x1(avgpool(x))`, no sigmoid) → conv3; residual
scaled by beta; then FFN (norm2 → conv4 → SimpleGate → conv5) scaled by gamma.
Top level pads to multiples of 16, encoder/down pyramid, middle block, decoder
with Conv1x1+PixelShuffle(2) ups, `ending + inp`. Loads the GoPro-width32
weights (`params` dict) and writes f32 bins for the Rust verification test.

usage: python3 tools/nafnet_verify.py [outdir]   (default /tmp/nafnet_verify)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(x, mu, var, weight, bias)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y * weight + bias
        return y.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, var, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        d = x - mu
        inv = (var + eps).rsqrt()
        N, C, H, W = d.size()
        g = grad_output.permute(0, 2, 3, 1) * weight
        g = g.permute(0, 3, 1, 2)
        gx = g * inv
        gvar = (g * d * inv * inv * -0.5).sum(1, keepdim=True)
        gmu = (g * -inv).sum(1, keepdim=True) + gvar * (d * -2.0).mean(1, keepdim=True)
        return gx + (gvar * (d * 2.0) + gmu) / (C * H * W), None, None, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        # Simplified Channel Attention: avg pool + single 1x1 conv (no sigmoid).
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nafnet_verify"
    os.makedirs(out, exist_ok=True)

    # Deterministic, structured input (smooth sinusoid rings + gradient) that is
    # fp16-safe for this model: random noise drives the GoPro-width32 encoder
    # activations past fp16 max (65504), but real image content stays far below
    # (see docs/upstream-issues.md §6). 66 is not a multiple of 16 → exercises pad/crop.
    n, c, h, w = 1, 3, 64, 66
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2)
    p = (
        0.5
        + 0.25 * np.sin(2 * np.pi * xx / 21.0) * np.cos(2 * np.pi * yy / 17.0)
        + 0.15 * np.sin(2 * np.pi * r / 13.0)
    )
    p = np.clip(p, 0, 1).astype(np.float32)
    x = torch.from_numpy(p[None, None].repeat(c, axis=1).copy())

    model = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                   enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1))
    d = torch.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "NAFNet-GoPro-width32.pth"),
        map_location="cpu",
    )
    sd = d["params"]
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
