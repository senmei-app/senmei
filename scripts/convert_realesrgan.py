#!/usr/bin/env python3
"""Convert a Real-ESRGAN RRDBNet checkpoint (.pth state dict) to TorchScript
so senmei's TorchEngine (libtorch) can load it.

Handles the classic RRDBNet variants (RealESRGAN_x4plus, x4plus_anime_6B,
no pixel_unshuffle) and the scale-2 variant (RealESRGAN_x2plus, which
pixel-unshuffles the input). num_block and input layout are detected from
the checkpoint, so no flags are needed.

Usage: python3 scripts/convert_realesrgan.py <input.pth> <output.pt>
"""
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

# RRDBNet from basicsr/archs/rrdbnet_arch.py (BSD-3-Clause, same as the model).


def make_layer(basic_block, num_basic_block, **kwarg):
    return nn.Sequential(*(basic_block(**kwarg) for _ in range(num_basic_block)))


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h, w = hh // scale, hw // scale
    return x.view(b, c, h, scale, w, scale).permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch, unshuffle):
        super().__init__()
        self.unshuffle = unshuffle
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = pixel_unshuffle(x, self.unshuffle) if self.unshuffle else x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Two 2x nearest upsamples: 4x, or 2x after a 2x unshuffle.
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def detect(state_dict):
    in_ch = state_dict["conv_first.weight"].shape[1]
    unshuffle = None
    scale = 4
    if in_ch != 3:
        unshuffle = int(math.sqrt(in_ch // 3))
        scale = unshuffle
    body = [k for k in state_dict if k.startswith("body.")]
    num_block = max(int(k.split(".")[1]) for k in body) + 1
    return in_ch, unshuffle, scale, num_block


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "models/realesrgan-x4plus.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "models/realesrgan-x4plus.pt"

    ckpt = torch.load(src, map_location="cpu")
    state_dict = ckpt["params_ema"] if isinstance(ckpt, dict) and "params_ema" in ckpt else ckpt
    in_ch, unshuffle, scale, num_block = detect(state_dict)
    print(f"in_ch={in_ch} unshuffle={unshuffle} scale={scale} num_block={num_block}")

    model = RRDBNet(in_ch, 3, num_feat=64, num_block=num_block, num_grow_ch=32, unshuffle=unshuffle)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        traced = torch.jit.trace(model, x)
        eager = model(x)
        scripted = traced(x)
        diff = (eager - scripted).abs().max().item()
        print(f"trace/eager max abs diff: {diff:.3e}")
        assert diff < 1e-3, "traced model diverges from eager"

    traced.save(dst)
    print(f"saved: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
