# rife-v4.6 reference (ncnn)

Clean-port reference for the burn implementation of the RIFE v4 architecture.

## Source & license

- Topology: `flownet.param` from [`nihui/rife-ncnn-vulkan`](https://github.com/nihui/rife-ncnn-vulkan) (`models/rife-v4.6/`), **MIT**.
- `flownet.bin` (the weights) is **not** committed — AGENTS.md forbids committing model weights.
- Upstream model is Practical-RIFE v4.6 (MIT weights).

## Structure

- `flownet.param` — the ncnn layer graph (215 layers, 276 blobs).
- `flownet.spec.md` — generated blueprint: every layer with inputs/outputs and
  inferred `C×H×W` per blob (at 256×256). Regenerate with
  `python3 tools/rife_param_spec.py ref/rife-v4.6/flownet.param`.
- RIFE v4 is a **single network** (`flownet`) — no separate contextnet/fusionnet.

## Architecture notes (from the spec)

- Inputs: `in0`/`in1` (3ch frames), `in2` (timestep, 1ch, broadcast).
- `cat_0` → 7ch, `Interp` 0.125 → 1/8 res, two stride-2 convs → 1/32.
- Flow/mask estimated at low res through ~residual blocks, then upsampled back
  to full res (`Interp`/`Deconvolution`/`PixelShuffle`).
- Synthesis: warp both frames by the flow (`rife.Warp` = grid_sample), mask the
  two warps and sum — `out = (1-m)·warp(in0) + m·warp(in1)`.
