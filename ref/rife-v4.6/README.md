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
- `flownet.bin` (not committed) — the weights, ncnn binary format.
- RIFE v4 is a **single network** (`flownet`) — no separate contextnet/fusionnet.

## ncnn `.bin` format (reverse-engineered)

The model is fp16-storage. Per weighted layer (`Convolution`/`Deconvolution`, in
`.param` declaration order):

```
[tag: u32 = 0x01306B47]        fp16 data marker (little-endian 47 6b 30 01)
[weights: wsize × f16]         wsize = num_output × in_channels × k × k  (= param 6)
[bias: num_output × f32]       only when bias_term = 1 (param 5)
```

- Verified: walking all 44 weighted layers lands exactly on EOF; every weight
  block starts with the `0x01306B47` marker; fp16 weights / fp32 biases read as
  plausible values.
- `wsize` (= param `6=`) counts **fp32 elements**; stored as fp16 → `2×wsize` bytes.

## Architecture notes (from the spec)

- Inputs: `in0`/`in1` (3ch frames), `in2` (timestep, 1ch, broadcast).
- `cat_0` → 7ch, `Interp` 0.125 → 1/8 res, two stride-2 convs → 1/32.
- Flow/mask estimated at low res through ~residual blocks, then upsampled back
  to full res (`Interp`/`Deconvolution`/`PixelShuffle`).
- Synthesis: warp both frames by the flow (`rife.Warp` = grid_sample), mask the
  two warps and sum — `out = (1-m)·warp(in0) + m·warp(in1)`.
