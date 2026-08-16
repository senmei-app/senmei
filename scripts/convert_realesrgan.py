#!/usr/bin/env python3
"""Convert the Real-ESRGAN x4plus checkpoint (a .pth state dict) to TorchScript
so senmei's TorchEngine (libtorch) can load it.

Requires: torch (any backend). Usage:
    python3 scripts/convert_realesrgan.py [input.pth] [output.pt]
Defaults to models/realesrgan-x4plus.pt in and out (overwrites the state dict).
"""
import sys

import torch
from torch import nn
from torch.nn import functional as F

# RRDBNet from realesrgan/archs/srdn_arch.py (BSD-3-Clause, same as the model).


def make_layer(basic_block, num_basic_block, **kwarg):
    return nn.Sequential(*(basic_block(**kwarg) for _ in range(num_basic_block)))


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
    def __init__(self, num_in_ch, num_out_ch, scale, num_feat, num_block, num_grow_ch):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = scale

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Two 2x nearest upsamples give the 4x scale of RealESRGAN_x4plus.
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "models/realesrgan-x4plus.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "models/realesrgan-x4plus.pt"

    model = RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["params_ema"], strict=True)
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
