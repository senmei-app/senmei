# Benchmarks — Inference Engines on RX 9070

Comparative inference benchmarks behind the engine decision (final: 2026-08-17).
Target device = the actual dev machine, not a synthetic proxy.

## Final decision (2026-08-17)

- **Shipped: burn (`burn-wgpu`) Vulkan fp16.** Real-CUGAN up2x **302 ms**
  @1080p, ShuffleCugan **103 ms** @1080p — beats ncnn (398 ms) and is portable.
  Weighed costs: heavy build (~800 crates / 1.6 GB `target/`), a `burn-fusion`
  f32 crash at 1080p (fp16 path is fine), the f32→f16 weight workflow.
- **Fastest measured, not shipped:** torch-ROCm (7.14-built) fp16 ShuffleCugan
  **41.8 ms** @1080p — 3.8× faster than burn. Not portable (ROCm 7.x + RDNA4 +
  matching torch), heavy libtorch → kept as an optional `InferenceEngine`
  backend at most.
- **Dropped:** ncnn/Vulkan (superseded), candle/ROCm, burn-ROCm (cubecl fp16
  kernel gap on RDNA4).

## Key numbers

| Path | 1080p x2 | Notes |
|---|---|---|
| **burn-Vulkan fp16 (shipped)** | up2x **302 ms** · ShuffleCugan **103 ms** | ShuffleCugan ~5× faster: UNet runs at half-res |
| torch-ROCm 7.14 fp16 | ShuffleCugan **41.8 ms** (23.9 FPS) | not portable |
| Full app (end-to-end) | **~6.5 FPS** | 1080p→2160p, tiled-fused step + x264 veryfast |
| Fallin soft / strong (fused step) | 176 / 177 ms | full-frame, pre-tiling; ~2× faster than real-cugan-x2 (380 ms) |

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 9070 (Navi 48, RDNA4, `gfx1201`), 16 GB VRAM |
| CPU / RAM | Granite Ridge iGPU; 30 GiB RAM |
| OS | Fedora/Nobara (fc44, dnf) |
| ROCm | installed 7.1 driving the **7.14 RDNA4 port** (`/opt/therock-tarball/install`); torch 2.13.0+rocm7.1 + nightly 2.15.0.dev+rocm7.14 |
| Vulkan | Mesa/RADV |
| Rust | 1.96.1 |

## tch (libtorch) — f16 fused path (2026-08-25)

After the libtorch backend moved to `LibTorch<f16>` + the shared fused RGB8
path (device-side tiling, native-scale accumulation, GPU re-sample). App path:
`real-cugan-pro-conservative-x2`, 576×432 DVD frames → 4× (2304×1728).

| Path | ms/frame | FPS |
|---|---|---|
| tch f32 (previous, CPU roundtrip) | 106.7 | 9.4 |
| **tch f16 fused (current)** | **59.6** | **16.8** |
| real app render (tch f16, VA-API encode) | ~66 | ~15 |

Frame batching (MIOpen) does **not** help — measured with
`bench_upscale_batch_dvd` (576×432 → 4×, pipelined depth 2 reference):

| Batch | vs. per-frame |
|---|---|
| 1 | 100 % (baseline) |
| 2 | 102 % |
| 4 | 104 % |
| 8 | 109 % |

Larger batched matmuls regress on this backend too (same as Vulkan/burn,
docs/benchmarks.md above) — the fused path already carries all frames in the
batch dim per tile position, and the model floor (~44 ms @ native 2×) is the
bottleneck, not launch/readback overhead.


Caveats: heavy build (~800 crates / 1.6 GB `target/`); `PytorchStore` can't cast
f32→f16 at load (pre-convert weights, or `BurnpackStore` + `HalfPrecisionAdapter`).

## Autotune failures — root cause & fix (2026-08-18)

Full-frame fused `infer_rgb8` OOM'd autotune on a large matmul (m=1024, n=4M,
f16) then panicked (upstream-issues.md §1+2). **Fix:** tile `infer_rgb8`
(640px) so no full-frame matmul reaches autotune; disabling autotune is ~5×
slower.

## Tile size & GPU stitch (2026-08-18)

The 512px tiled-fused path (329 ms / 3.0 FPS fallin-soft) was ~2× slower than
the pre-tiling full-frame fused path (176 ms) — that drop was the price of
avoiding the autotune OOM (Bug 3). Tried **1024px tiles** (6 tiles @1080p vs 15
@512): **regression to 762 ms / 1.3 FPS** — the larger per-tile matmul is
pathologically slower on this backend.

**GPU stitch (2026-08-18):** instead of reading each tile's u8 bytes back and
stitching on the CPU, tiles are accumulated into one f16 canvas on the GPU
(`slice_assign` overlap averaging) and read back as a single packed frame — one
readback instead of 15 plus a CPU stitch. `bench_upscale_step` (fallin-soft):
329 → **234.7 ms / 4.3 FPS** (512px).

**640px default (2026-08-18):** re-tuned tile size after the GPU stitch (the
old cost model — 15 u8 readbacks + CPU stitch — is gone). `bench_upscale_step`
(fallin-soft):

| tile | tiles @1080p | step |
|---|---|---|
| 512 | 15 | 247.8 ms / 4.0 FPS |
| **640** | **8** | **186.1 ms / 5.4 FPS** |
| 768 | 6 | 210.2 ms / 4.8 FPS |

640 is the sweet spot: halved tile count (15→8) before the per-tile matmul gets
pathological (768 already regresses). Default is 640, override via `SENMEI_TILE`.
Full-frame fused path (176 ms) is the floor once the autotune OOM is fixed
upstream.

## Multi-frame batch path — verdict (2026-08-22)

`bench_upscale_batch`, fallin-soft 1080p→2160p, burn-Vulkan fp16 (RDNA4),
autotune + fusion on, `EngineBackend::Vulkan` forced:

| Path | ms/frame | FPS | vs per-frame |
|---|---|---|---|
| per-frame (fused single RGB8) | 285.2 | 3.5 | — |
| batch 2 | 275.4 | 3.6 | 97 % |
| batch 4 | 310.8 | 3.2 | 109 % |
| batch 8 | 378.4 | 2.6 | 133 % |

**Batching regresses on RDNA4/Vulkan** — the larger per-tile batched matmuls are
pathologically slower (same effect as the 1024px tile regression above). The
pipeline's `BATCH_SIZE` defaults to **1** (off); the fused multi-frame path
stays for backends where it could win, but is not exercised on the shipped one.
Per-frame here (285 ms) reads higher than the 2026-08-18 `bench_upscale_step`
(186 ms): same tile path, run-to-run / thermal variance + this run interleaves
batch sizes on one hot GPU.

**Fusion coverage audit (2026-08-22):** the upscale hot path is fully fused
(`infer_rgb8`/`infer_rgb8_batch`: GPU NCHW→RGB8 + GPU tile stitch, one readback).
**Gap:** the fused path only fires when requested scale == model scale — fallin
soft/strong are x2, so rendering at **x4 takes the slow path** (`infer_tiled` +
CPU `tensor_to_frame` + CPU bilinear). Denoise/Deblur/interp also still convert
on the CPU. Closing the x4 gap (GPU-side re-scale in `infer_rgb8`) is the real
win for the shipped use case; not done yet.

## Readback pipelining (2026-08-22)

`bench_upscale_pipelined` (fallin-soft 1080p, Vulkan fp16, autotune + fusion):
the forward is split from the readback (`infer_rgb8_submit` → `Rgb8Batch`); the
next batch's GPU work is queued **before** the oldest readback resolves, so the
GPU stays busy during the transfer.

| Path | ms/frame | FPS |
|---|---|---|
| per-frame (sync readback) | 285.2 | 3.5 |
| pipelined depth 1 | **221.6** | **4.5** |
| pipelined depth 2 | 220.3 | 4.5 |
| pipelined depth 3 | 221.4 | 4.5 |

~22 % faster than the sync path. Depth 1 (double-buffered readback) captures
nearly all of it — depth 2/3 add nothing here because the bench has no encoder
write to hide behind the GPU (the full app may benefit).

### Depth 2 default (2026-08-24)

Same bench, heavier model (`real-cugan-pro-conservative-x2`, 1080p@2 — the
shipped grain model): depth now matters.

| Depth | ms/frame | FPS | vs depth 1 |
|---|---|---|---|
| 1 | 777.5 | 1.3 | — |
| **2** | **607.1** | **1.6** | **−22 %** |
| 3 | 596.3 | 1.7 | −23 % |

Depth 2 is the sweet spot (depth 3 adds ~1 %). The heavier model's bigger
canvas/readback makes the overlap worthwhile where the light model's didn't.
Default depth = **2** (was 1), `0` in settings = owning default.

## Real-frame upscaler sweep (2026-08-23)

`bench_upscalers_real_frames` (`cargo test -p senmei-pipeline --release --test
bench -- --ignored --nocapture bench_upscalers_real_frames`): every loadable
`upscale` model at its native scale on two real DVD frames (720×576 rgb24,
`models.bat/frame_*.png`, override via `BENCH_FRAMES`). Burn-Vulkan fp16 on
RX 9070, fused RGB8 `Upscale` step (the app path). Each model's upscaled frame
is saved next to the inputs as `<id>.png`. At 720×576 every model fits the
fused VRAM guard except where noted. Sorted by ms/frame.

| model | scale | ms/frame | FPS |
|---|---|---|---|
| **paragonsr-nano-x2** | **2** | **60.9** | **16.4** |
| fallin-soft | 2 | 60.9 | 16.4 |
| fallin-strong | 2 | 61.6 | 16.2 |
| realesrgan-animevideo-x2 | 2 | 90.0 | 11.1 |
| realesrgan-animevideo-x4 | 4 | 136.3 | 7.3 |
| real-cugan-x2 | 2 | 140.8 | 7.1 |
| real-cugan-pro-no-denoise-x2 | 2 | 141.4 | 7.1 |
| real-cugan-pro-denoise3x-x2 | 2 | 142.1 | 7.0 |
| real-cugan-pro-conservative-x2 | 2 | 142.4 | 7.0 |
| real-cugan-hfa2k-x2 | 2 | 143.2 | 7.0 |
| realesrgan-general-x4v3 | 4 | 197.3 | 5.1 |
| span-2x-modern-spanimation-v1 | 2 | 286.0 | 3.5 |
| span-2x-modern-spanimation-v1.5 | 2 | 286.3 | 3.5 |
| span-2x-nomosuni-ldl | 2 | 304.8 | 3.3 |
| span-2x-nomosuni-multijpg | 2 | 305.5 | 3.3 |
| span-2x-hfa2k-ludvae | 2 | 305.7 | 3.3 |
| span-2x-hfa2k | 2 | 306.3 | 3.3 |
| span-2x-modern-spanimation-v2 | 2 | 306.9 | 3.3 |
| span-2x-bhi-small | 2 | 307.1 | 3.3 |
| **realesrgan-x2plus** | **2** | **671.9** | **1.5** |
| safmn-real-x2 | 2 | 927.7 | 1.1 |
| safmn-real-x4 | 4 | 973.1 | 1.0 |
| realesrgan-x4plus-anime | 4 | 1096.4 | 0.9 |
| real-plksr-2x-public | 2 | 1155.3 | 0.9 |
| 4x-nomoswebphoto-realplksr | 4 | 1167.1 | 0.9 |
| 4x-alchemy | 4 | 1266.6 | 0.8 |
| 4x-bhi-realplksr-otf | 4 | 1282.5 | 0.8 |
| 4x-mssim-realplksr | 4 | 1283.5 | 0.8 |
| 4x-nature-realplksr | 4 | 1284.1 | 0.8 |
| 4x-bhi-realplksr-real | 4 | 1284.2 | 0.8 |
| 4x-hfa2k-realplksr | 4 | 1285.1 | 0.8 |
| 4x-nomos2-realplksr | 4 | 1286.5 | 0.8 |
| bsrgan | 4 | 2864.3 | 0.3 |

(`*tiled` = the fused RGB8 path's free-VRAM guard tripped, fell back to raw
tiled infer; the fused-vs-tiled choice varies run-to-run, so those numbers are
approximate — same model can show lower ms/frame on the tiled path.)

Takeaways:
- **paragonsr-nano-x2 is the fastest 2× overall** (~61 ms / 16.4 FPS, tied
  with fallin) — 2.3× faster than real-cugan-x2 (141 ms). Verified vs ONNX
  Runtime fp16 (mae 0.0009 on random input, 0.0014 / 57 dB PSNR on the real
  DVD frame). Phhofm ParagonSR-Nano GAN, MIT, 24-feat / 3×2-block
  ParagonSrNet.
- **Real-CUGAN-Pro 2× family** (`real-cugan-pro-{no-denoise,conservative,
  denoise3x}-x2`, official bilibili 2022-05, Apache-2.0): same `UpCunet2x`
  arch and speed as real-cugan-x2 (~142 ms / 7 FPS). Verified vs spandrel
  (mae 0.79/255, 50 dB PSNR on the real DVD frame). conservative = balanced
  preset for real film.
- **realesrgan-general-x4v3 is the fast real-film 4× pick**: ~110-196 ms /
  5-9 FPS — ~5-14× faster than every other 4× model and competitive with 2×
  (real-cugan-x2: 142 ms). Real-photo training + compact SRVGGNetCompact.
- **realesrgan-animevideo-x2/x4 are the fast anime picks** (94 / 141 ms), anime-
  trained.
- fallin soft/strong stay the fastest 2× overall (61-63 ms, ~16 FPS).
- **realesrgan-x2plus** (real-photo RRDBNet 23-block x2, pixel_unshuffle
  variant): correct but slow — 651 ms / 1.5 FPS, verified vs spandrel
  (mae 0.43). A quality 2× option, not a fast one.
- The RealPLKSR 4× family clusters at ~656 ms (1.5 FPS, tiled) — no fast option.
- **safmn-real-x2/x4 are surprisingly slow** (912 / 479 ms) — ~15× slower than
  fallin despite the "lightweight" claim; the SAFM block (depthwise convs +
  per-level pool/interp + GELU) maps poorly to this backend.
- `span-2x-nomosuni-multijpg` used to panic in burn-ir (`DTypeMismatch`): its
  bpk was stored F32 — `HalfPrecisionAdapter` gates on the burn module type,
  which `PytorchStore` snapshots lack, so the span convs were never cast.
  Converter now saves through an unconditional `ToF16` adapter; the re-converted
  bpk is F16 (3.67 MB, was 7.32 MB) and loads (297 ms / 3.4 FPS).

### SRVGG residual fix (2026-08-23)

`SRVGGNetCompact` learns the **residual**: the conv net's PixelShuffle output is
added to the nearest-upsampled input (`out += F.interpolate(x, scale_factor,
nearest)` — spandrel's `forward`). The burn port (and `tools/srvgg_verify.py`,
which was the same bug — the "mae 0.0004" reference omitted the base too)
dropped that, so animevideo-x2/x4 and general-x4v3 rendered the near-black
residual alone (means ~2/255). Added the base in `SrvggNet::forward`; burn now
matches torch incl. residual (mae 0.0004) and output brightness matches the
input (animevideo-x2 R=82.5/G=57.2/B=45.7 vs input 81/56/44).

## Backend history (archive)

- **torch-ROCm 7.14 (2026-08-17):** the RDNA4 fp16 gap was a ROCm-7.1 software
  issue; the 7.14 port unlocks WMMA — ShuffleCugan fp16 1080p→2160p **41.8 ms
  (23.9 FPS)** vs burn-Vulkan 157 ms (3.8×), still the fastest measured but not
  portable (ROCm 7.x + RDNA4 + matching torch). **FP8 unusable on RDNA4** (torch
  never wires fp8; the hipBLASLt fp8 kernel crashes). **fp16 is input-range
  sensitive** (NaN outside 0..1) — the app clamps 0..1.
- **ncnn/Vulkan — dropped:** 249/398 ms @720p/1080p, superseded by burn-Vulkan;
  survives only as RIFE's weight format (`flownet.bin`).
- **candle/ROCm — dropped:** f32 im2col memory cliff, f16 ~6× slower.
- **burn-ROCm — dropped:** slow f32; fp16/bf16 matmul `LLVM Cannot select` on
  gfx1201 (ROCm-7.1 software gap).
- **torch-ROCm original (2026-08-16, superseded):** pathological 7153 ms @1080p
  (MIOpen/RDNA4 collapse) — a software gap fixed by the 7.14 port above.
- **burn-Vulkan shipped path (2026-08-17):** up2x fp16 136/302 ms @720p/1080p
  (f32 crashed @1080p — burn-fusion bug), ShuffleCugan fp16 46/103 ms (half-res
  UNet). Superseded by the model sweep below.
- **Full-app pipeline (2026-08-17):** 1080p→2160p ShuffleCugan f16 Vulkan —
  sequential 5.1 → threaded 5.9 → fused step 6.4 → +x264 veryfast **6.5 FPS**;
  superseded by the tch/ROCm + VA-API path (~15 FPS). CPU wins still in code:
  LUT `frame_to_tensor` (3.7→1.9 ms), saturating `tensor_to_frame` (35.7→9.7 ms).
- **Fallin vs real-cugan (2026-08-18):** fallin ~2× faster at ~half the VRAM
  (8.1 vs 14.6 GB peak). Superseded by the real-frame sweep (2026-08-23).

See `docs/PLAN.md` for the current engine/roadmap status.

## Model sweep — fused step @1080p→2160p ×2 (burn-Vulkan, 2026-08-27)

`bench_upscale_step`, current registry models, RX 9070:

| Model | Arch | ms/frame | FPS |
|---|---|---|---|
| real-cugan-x2 | upcunet2x | 510 | 2.0 |
| realesrgan-animevideo-x2 | srvgg | 306 | 3.3 |
| **fallin-soft** | fallin-cugan | **184** | **5.4** |
| span-2x-hfa2k | span | 1137 | 0.9 |
| real-plksr-2x-public | real-plksr | 4556 | 0.2 |
| realesrgan-x2plus | rrdb | 2260 | 0.4 |

fallin-soft is **2.7× faster than real-cugan-x2** within the same catalog —
model selection is the biggest free FPS lever (with tch/ROCm +1.5× → ~9 FPS).

## tch/ROCm fused path — re-evaluated (2026-08-27)

The shared fused RGB8 path (fused f16 pad+cast+upload, dropped coverage
canvas, feather-mask cache) applies to both engines, so the tch/ROCm backend
was re-measured against burn-Vulkan on the fused app step
(`bench_upscale_step`, 1080p→2160p x2, RX 9070):

| Model | burn-Vulkan | tch/ROCm | speedup |
|---|---|---|---|
| fallin-soft | 178 ms (5.6 FPS) | 112.7 ms (8.9 FPS) | **1.58×** |
| real-cugan-x2 | 503 ms (2.0 FPS) | 327 ms (3.1 FPS) | **1.54×** |

Stable in these runs (no hang); the 2026-08-22 RDNA4 mode1-reset risk in the
tch/ROCm path still stands (see engine-decision notes). Runtime: local
`SENMEI_LIBTORCH_ENV` + ROCm-7.14 nightly `LIBTORCH` venv + rock dir
`LD_LIBRARY_PATH`. `engine_for_model(Auto)` still prefers tch when loadable.

## Aux-stack sweep — interp / denoise / deblur (2026-08-27)

`bench_aux_stacks` (new in `bench.rs`): throughput + quality (PSNR dB / SSIM)
against a known reference per stack, plus the no-op baseline. Burn-Vulkan fp16,
RX 9070. Inputs: testsrc2 720×576 (interp, dropped middle of a triplet vs the
real middle) and a real DVD frame (denoise/deblur, synthetic degradation).

**Interpolation** (factor 2):

| path | ms | FPS | PSNR | SSIM |
|---|---|---|---|---|
| linear blend (no model) | — | — | 21.3 | 0.929 |
| rife-v4.6 | 13.6 | 73.5 | 26.2 | 0.925 |
| ifrnet-vimeo90k | 23.8 | 42.0 | 28.9 | 0.943 |
| ifrnet-gopro | 23.4 | 42.7 | 17.3 | 0.754 |

Caveat: testsrc2 is a synthetic high-frequency worst case for flow models —
RIFE under-scores here (SSIM ≈ blend despite +5 dB PSNR) but is 1.7× faster
than IFRNet; IFRNet-Vimeo90K is the quality pick. Re-check on real motion
before trusting RIFE's low score (and IFRNet-GoPro is blur-trained — worst on
this content).

**Denoise** (real DVD frame + Gaussian σ=0.1):

| path | ms | FPS | PSNR | SSIM |
|---|---|---|---|---|
| noisy (no model) | — | — | 20.2 | 0.204 |
| drunet-color | 58.7 | 17.1 | 37.9 | 0.955 |
| dncnn-color | 23.6 | 42.3 | 36.6 | 0.940 |
| ffdnet-color | 10.9 | 91.6 | 37.1 | 0.948 |

**FFDNet is the denoise pick**: ~5× faster than DRUNet at ~equal quality
(37.1 vs 37.9 dB, 0.948 vs 0.955 SSIM); all beat the noisy baseline by
~17 dB. SCUNet skipped (weights not in `models/`).

**Deblur**: NAFNet-GoPro-width32 weights not downloaded → skipped; blurred
baseline 40.3 dB / 0.975 — the σ≈1.5 blur is mild, a stronger blur would
differentiate the model better.
