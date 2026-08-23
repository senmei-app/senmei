#!/usr/bin/env python3
"""ParagonSR-Nano verification for the burn port.

Runs the FUSED ONNX export (the inference graph the burn `ParagonSrNet`
replicates: conv_in → 3×2 blocks → conv_fuse+skip → upsampler(24→96) →
PixelShuffle(2) → conv_out) with ONNX Runtime, and writes `x.bin` (input) +
`ref.bin` (onnx output) as f32 little-endian for the
`arch::paragonsr::paragonsr_matches_onnx_reference` test.

spandrel has no ParagonSR arch, so the ONNX Runtime output is the reference
(the weights are the same fused f16 weights the converter reads).

Usage: paragonsr_verify.py [onnx=/tmp/paragon_nano.onnx] [outdir=/tmp/paragonsr_verify]
"""
import os
import sys

import numpy as np
import onnxruntime as ort


def main() -> None:
    onnx = sys.argv[1] if len(sys.argv) > 1 else "/tmp/paragon_nano.onnx"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/paragonsr_verify"
    os.makedirs(outdir, exist_ok=True)

    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 32, 32), dtype=np.float32)  # [0,1] input
    # The export is fp16 end-to-end (burn runs f16 too), so feed fp16.
    ref = sess.run([oname], {iname: x.astype(np.float16)})[0]
    ref = ref.astype(np.float32)
    print(f"input {x.shape} -> output {ref.shape}")

    x.astype("<f4").tofile(f"{outdir}/x.bin")
    ref.astype("<f4").tofile(f"{outdir}/ref.bin")
    print(f"wrote {outdir}/x.bin, {outdir}/ref.bin")


if __name__ == "__main__":
    main()
