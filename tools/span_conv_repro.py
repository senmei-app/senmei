#!/usr/bin/env python3
"""Reference for the cubek-convolution f16 1x1 conv bug (docs/upstream-issues.md §6).

Mirrors the self-contained Rust repro `conv1x1_repro` in
`crates/senmei-ml/src/burn/span.rs`: same LCG, same shapes, same f32 reference.
Run with torch to confirm the torch f16 conv is correct at every shape (the
burn Vulkan<f16> conv is wrong for K=96 x N>=32768).

    python3 tools/span_conv_repro.py [--torch]
"""
import sys

import numpy as np

# Same LCG as the Rust test.
seed = 0x9E3779B9


def rnd():
    global seed
    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return (seed >> 8) / 16_777_216.0


CASES = [(96, 128, 128), (96, 128, 256), (96, 240, 320), (64, 240, 320)]


def gen(k, h, w):
    wv = np.array([(rnd() - 0.5) * 0.16 for _ in range(48 * k)], dtype=np.float32)
    bv = np.array([(rnd() - 0.5) * 0.1 for _ in range(48)], dtype=np.float32)
    xv = np.array([(rnd() - 0.5) * 6.0 for _ in range(k * h * w)], dtype=np.float32)
    ref = (wv.reshape(48, k) @ xv.reshape(k, h * w) + bv[:, None]).astype(np.float32)
    return wv, bv, xv, ref.reshape(48, h, w)


def main():
    use_torch = "--torch" in sys.argv
    print("cubek-convolution f16 1x1 conv reference (K=96 x N>=32768 broken in burn):")
    for k, h, w in CASES:
        wv, bv, xv, ref = gen(k, h, w)
        if use_torch:
            import torch

            out = (
                torch.nn.functional.conv2d(
                    torch.from_numpy(xv.reshape(1, k, h, w)).half(),
                    torch.from_numpy(wv.reshape(48, k, 1, 1)).half(),
                    torch.from_numpy(bv).half(),
                )
                .float()
                .numpy()
            )
            err = np.abs(out - ref.astype(np.float16).astype(np.float32))
            print(f"  K={k} N={h*w} ({h}x{w}): torch max_abs={err.max():.5f}")
        else:
            print(f"  K={k} N={h*w} ({h}x{w}): ref generated (run with --torch to compare)")


if __name__ == "__main__":
    main()
