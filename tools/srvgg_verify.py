#!/usr/bin/env python3
"""SRVGGNetCompact (animevideo-xs) verification for the burn port.

Builds a random-weight SRVGGNetCompact (clean re-implementation from the
BSD-3-Clause xinntao/Real-ESRGAN reference), saves the .pth, and writes
`x.bin` (input) + `ref.bin` (torch output) as f32 little-endian for the
`arch::srvgg::srvgg_matches_torch_reference` test.

Usage: srvgg_verify.py [scale=4] [outdir=/tmp/srvgg_verify]
"""
import os
import sys

import torch
import torch.nn as nn


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        activation = nn.PReLU(num_parameters=num_feat)  # shared for every layer
        self.body.append(activation)
        for _ in range(num_conv - 2):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation)
        self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))

        self.upsampler = nn.Sequential()
        self.upsampler.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
        self.upsampler.append(nn.PixelShuffle(2))
        if upscale >= 4:
            self.upsampler.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            self.upsampler.append(nn.PixelShuffle(2))
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        for m in self.body:
            x = m(x)
        x = self.upsampler(x)
        return self.conv_last(x)


def main() -> None:
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/srvgg_verify"
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(0)
    net = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=scale).eval()
    sd = net.state_dict()
    print(f"scale {scale}: {len(sd)} tensors")
    for k, v in sorted(sd.items()):
        print(f"  {k} {tuple(v.shape)}")

    pth = f"{outdir}/srvgg_x{scale}.pth"
    torch.save(sd, pth)

    x = torch.rand(1, 3, 32, 32, dtype=torch.float32)  # [0,1] input
    with torch.no_grad():
        ref = net(x)
    print(f"input {tuple(x.shape)} -> output {tuple(ref.shape)}")

    x.numpy().astype("<f4").tofile(f"{outdir}/x.bin")
    ref.numpy().astype("<f4").tofile(f"{outdir}/ref.bin")
    print(f"wrote {pth}, x.bin, ref.bin")


if __name__ == "__main__":
    main()
