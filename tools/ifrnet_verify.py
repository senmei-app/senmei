#!/usr/bin/env python3
"""Numerical reference for the IFRNet burn port.

Builds the official IFRNet (base) model from `ref/ifrnet/`, loads the real
Vimeo90K weights, runs inference on a synthetic frame pair, and writes the
inputs + reference output as raw f32 little-endian bins for the Rust
verification test `ifrnet_matches_torch_reference` in
`crates/senmei-ml/src/burn/ifrnet.rs`.

usage: python3 tools/ifrnet_verify.py [outdir]   (default /tmp/ifrnet_verify)
"""
import os
import sys
import types

# The reference IFRNet.py imports training-only losses; stub them so the
# module loads (inference never touches the loss heads).
loss = types.ModuleType("loss")
for _n in ["Charbonnier_L1", "Ternary", "Charbonnier_Ada", "Geometry"]:
    setattr(loss, _n, lambda *a, **k: object())
sys.modules["loss"] = loss

# The reference utils.py imports imageio only for its imread/imwrite helpers
# (not used by inference) — stub it so the vendored file stays pristine.
imageio = types.ModuleType("imageio")
imageio.imread = lambda *a, **k: None
imageio.imwrite = lambda *a, **k: None
sys.modules["imageio"] = imageio

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "ref", "ifrnet"))

import torch
import torch.nn.functional as F  # noqa: E402  (after the loss stub + sys.path)
from utils import warp  # noqa: E402
import IFRNet  # noqa: E402  (after the loss stub + sys.path)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ifrnet_verify"
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(0)
    h = w = 64
    a = torch.rand(1, 3, h, w, dtype=torch.float32)
    b = torch.rand(1, 3, h, w, dtype=torch.float32)
    embt = torch.tensor([[0.5]], dtype=torch.float32)  # [1,1]

    model = IFRNet.Model()
    sd = torch.load(
        os.path.join(_here, "..", "models", "IFRNet_Vimeo90K.pth"),
        map_location="cpu",
    )
    model.load_state_dict(sd)
    model.eval()

    # Intermediates for step-wise debugging of the burn port.
    with torch.no_grad():
        mean_ = torch.cat([a, b], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        a_ = a - mean_
        b_ = b - mean_
        f0_1, f0_2, f0_3, f0_4 = model.encoder(a_)
        f1_1, f1_2, f1_3, f1_4 = model.encoder(b_)
        out4 = model.decoder4(f0_4, f1_4, embt)
        d4_in = torch.cat([f0_4, f1_4, embt.repeat(1, 1, f0_4.shape[2], f0_4.shape[3])], 1)
        d4_cb0 = model.decoder4.convblock[0](d4_in)
        rb = model.decoder4.convblock[1]
        rb_s1 = rb.conv1(d4_cb0)
        rb_s2 = rb_s1.clone()
        rb_s2[:, -32:] = rb.conv2(rb_s2[:, -32:].clone())
        rb_s3 = rb.conv3(rb_s2)
        rb_s4 = rb_s3.clone()
        rb_s4[:, -32:] = rb.conv4(rb_s4[:, -32:].clone())
        rb_s5 = rb.prelu(d4_cb0 + rb.conv5(rb_s4))
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]
        out3 = model.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * F.interpolate(up_flow0_4, scale_factor=2.0, mode="bilinear", align_corners=False)
        up_flow1_3 = out3[:, 2:4] + 2.0 * F.interpolate(up_flow1_4, scale_factor=2.0, mode="bilinear", align_corners=False)
        ft_2_ = out3[:, 4:]
        out2 = model.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * F.interpolate(up_flow0_3, scale_factor=2.0, mode="bilinear", align_corners=False)
        up_flow1_2 = out2[:, 2:4] + 2.0 * F.interpolate(up_flow1_3, scale_factor=2.0, mode="bilinear", align_corners=False)
        ft_1_ = out2[:, 4:]
        out1 = model.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * F.interpolate(up_flow0_2, scale_factor=2.0, mode="bilinear", align_corners=False)
        up_flow1_1 = out1[:, 2:4] + 2.0 * F.interpolate(up_flow1_2, scale_factor=2.0, mode="bilinear", align_corners=False)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        img0_warp = warp(a_, up_flow0_1)
        img1_warp = warp(b_, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        ref = torch.clamp(imgt_merge + up_res_1, 0, 1)

    tensors = {
        "mean.bin": mean_,
        "a_sub.bin": a_,
        "b_sub.bin": b_,
        "f0_1.bin": f0_1,
        "f0_2.bin": f0_2,
        "f0_3.bin": f0_3,
        "f0_4.bin": f0_4,
        "f1_1.bin": f1_1,
        "f1_2.bin": f1_2,
        "f1_3.bin": f1_3,
        "f1_4.bin": f1_4,
        "w_c1.bin": model.decoder4.convblock[1].conv1[0].weight,
        "w_c2.bin": model.decoder4.convblock[1].conv2[0].weight,
        "w_c5.bin": model.decoder4.convblock[1].conv5.weight,
        "w_pl.bin": model.decoder4.convblock[1].prelu.weight,
        "d4_in.bin": d4_in,
        "d4_cb0.bin": d4_cb0,
        "rb_s1.bin": rb_s1,
        "rb_s2.bin": rb_s2,
        "rb_s3.bin": rb_s3,
        "rb_s4.bin": rb_s4,
        "rb_s5.bin": rb_s5,
        "out4.bin": out4,
        "out3.bin": out3,
        "out2.bin": out2,
        "out1.bin": out1,
        "up_flow0_1.bin": up_flow0_1,
        "up_mask_1.bin": up_mask_1,
        "up_res_1.bin": up_res_1,
        "img0_warp.bin": img0_warp,
        "ref.bin": ref,
    }
    for name, t in tensors.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(t.detach().numpy().astype("<f4").tobytes())
    print(f"wrote {out}  ref={list(ref.shape)}  range=[{float(ref.min()):.4f},{float(ref.max()):.4f}]")


if __name__ == "__main__":
    main()
