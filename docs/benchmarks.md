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

## Shipped path — burn-Vulkan (2026-08-17)

Real models (`upcunet_v3`, `sudo_shuffle_cugan`), weights via
`burn-store::PytorchStore` (key remap `conv.0`/`conv.2` → `conv`/`conv2`),
numerically verified vs torch (f32 max diff ~6e-6; fp16 ~1.7e-2).

| Model | Dtype | 720p x2 | 1080p x2 | Notes |
|---|---|---|---|---|
| up2x | f32 | 966 ms | crash | `burn-fusion` bug `Ordering is bigger than operations` @1080p |
| up2x | fp16 | **136 ms** | **302 ms** | beats ncnn (249 / 398 ms) |
| ShuffleCugan | fp16 | **46 ms** | **103 ms** | pixel-unshuffle ⇒ UNet at half-res |

Caveats: heavy build (~800 crates / 1.6 GB `target/`); `PytorchStore` can't cast
f32→f16 at load (pre-convert weights, or `BurnpackStore` + `HalfPrecisionAdapter`).

## Full-app render pipeline (2026-08-17)

`senmei-pipeline/tests/bench.rs` (`cargo test -p senmei-pipeline --release --test
bench -- --ignored --nocapture`; env `BENCH_MODEL`, `SENMEI_X264_PRESET`).
Workload: 1080p testsrc → 2160p x2, ShuffleCugan f16, Vulkan, 48 frames.

| Path | total | FPS |
|---|---|---|
| before (sequential) | 197 ms | 5.1 |
| after (optimized) | 168 ms | 5.9 |
| **fused GPU RGB8 step** | **157 ms** | **6.4** |
| full threaded + x264 veryfast | — | **6.5** |

- Biggest CPU win: 4K `tensor_to_frame` 35.7→9.7 ms via saturating
  `(x*255+0.5) as u8`; `frame_to_tensor` uses a `x/255` LUT (3.7→1.9 ms).
- Decode/encode run on threads so CPU I/O hides behind GPU inference; encode uses
  x264 `-preset veryfast` (was medium — the 2160p bottleneck at 4.7 FPS).
- **`infer_rgb8`:** NCHW→NHWC, clamp 0..1, scale 0..255, cast to u8 **on GPU**;
  tiles accumulated on the GPU (overlap-averaged) and read back as one packed
  frame. The full-frame variant OOM'd autotune + panicked — tiling avoids it
  (below).
- 157 ms is burn-Vulkan's floor on RDNA4 (GPU 100 % / 3.2 GHz, no throttling);
  torch-ROCm fp16 runs the same model at 111.5 ms / 9.0 FPS.

### Autotune failures — root cause & fix (2026-08-18)

Full-frame fused `infer_rgb8` OOM'd autotune on a large matmul (m=1024, n=4M,
f16), then panicked `Ordering is bigger than operations` (docs/upstream-issues.md
§1+2). **Fix:** tile `infer_rgb8` (640px) so no full-frame matmul reaches
autotune. Guarded by `infer_rgb8_tiled_is_reliable_and_correct`. Disabling
autotune also works but is ~5× slower.

### Tile size & GPU stitch (2026-08-18)

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

## Fallin vs real-cugan (2026-08-18)

`bench.rs`, 1080p→2160p x2, Vulkan fp16, autotune + fusion on. Fused step =
`Upscale` + `infer_rgb8`. **Table = earlier full-frame fused step**; the current
tiled-fused step measures ~186 ms / 5.4 FPS (fallin-soft), full threaded
pipeline 2.8 FPS.

| Model | infer | total | FPS | fused step | step FPS | VRAM peak |
|---|---|---|---|---|---|---|
| real-cugan-x2 | 359 ms | 374 ms | 2.7 | 380 ms | 2.6 | 14.6 GB |
| fallin-soft | 193 ms | 205 ms | 4.9 | 176 ms | 5.7 | 8.1 GB |
| fallin-strong | 196 ms | 208 ms | 4.8 | 177 ms | 5.7 | 8.1 GB |

Fallin Soft/Strong ~2× faster than real-cugan-x2 at ~half the VRAM. VRAM via
`/sys/class/drm/card1/device/mem_info_vram_used` (baseline ~1.3 GB).

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
write to hide behind the GPU (the full app may benefit). Default depth = 1,
configurable via `pipeline_depth`.

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
| fallin-strong | 2 | 61.3 | 16.3 |
| **paragonsr-nano-x2** | **2** | **62.0** | **16.1** |
| fallin-soft | 2 | 62.7 | 15.9 |
| realesrgan-animevideo-x2 | 2 | 93.9 | 10.6 |
| realesrgan-animevideo-x4 | 4 | 141.4 | 7.1 |
| real-cugan-x2 | 2 | 142.5 | 7.0 |
| real-cugan-hfa2k-x2 | 2 | 146.7 | 6.8 |
| realesrgan-general-x4v3 | 4 | 196.5 | 5.1 |
| span-2x-modern-spanimation-v1.5 | 2 | 285.1 | 3.5 |
| span-2x-modern-spanimation-v1 | 2 | 285.9 | 3.5 |
| span-2x-hfa2k | 2 | 304.4 | 3.3 |
| span-2x-nomosuni-ldl | 2 | 305.1 | 3.3 |
| span-2x-nomosuni-multijpg | 2 | 306.5 | 3.3 |
| span-2x-modern-spanimation-v2 | 2 | 307.3 | 3.3 |
| span-2x-bhi-small | 2 | 307.5 | 3.3 |
| span-2x-hfa2k-ludvae | 2 | 307.5 | 3.3 |
| **realesrgan-x2plus** | **2** | **670.7** | **1.5** |
| safmn-real-x2 | 2 | 926.3 | 1.1 |
| safmn-real-x4 | 4 | 977.6 | 1.0 |
| realesrgan-x4plus-anime | 4 | 1094.5 | 0.9 |
| real-plksr-2x-public | 2 | 1156.4 | 0.9 |
| 4x-nomoswebphoto-realplksr | 4 | 1165.4 | 0.9 |
| 4x-nomos2-realplksr | 4 | 1281.1 | 0.8 |
| 4x-nature-realplksr | 4 | 1283.5 | 0.8 |
| 4x-mssim-realplksr | 4 | 1284.3 | 0.8 |
| 4x-hfa2k-realplksr | 4 | 1284.6 | 0.8 |
| 4x-bhi-realplksr-real | 4 | 1287.3 | 0.8 |
| 4x-bhi-realplksr-otf | 4 | 1289.2 | 0.8 |
| 4x-alchemy | 4 | 1290.7 | 0.8 |
| bsrgan | 4 | 2861.1 | 0.3 |

(`*tiled` = the fused RGB8 path's free-VRAM guard tripped, fell back to raw
tiled infer; the fused-vs-tiled choice varies run-to-run, so those numbers are
approximate — same model can show lower ms/frame on the tiled path.)

Takeaways:
- **paragonsr-nano-x2 is now a top-3 fastest 2×** (~62 ms / 16 FPS, tied with
  fallin) — 2.3× faster than real-cugan-x2 (142 ms). Verified vs ONNX Runtime
  fp16 (mae 0.0009 on random input, 0.0014 / 57 dB PSNR on the real DVD
  frame). Phhofm ParagonSR-Nano GAN, MIT, 24-feat / 3×2-block ParagonSrNet.
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

## Alternatives

### torch-ROCm (2026-08-17, ROCm 7.14)

The 2026-08-16 "fp16 impossible on RDNA4" verdict was a **ROCm-7.1 software
gap** — the machine runs the **7.14 RDNA4 port**. The 7.14-built nightly
(`2.15.0.dev+rocm7.14`, dedicated 7.14 venv) unlocks RDNA4 fp16
kernels: WMMA is present; the earlier `LLVM Cannot select` was cubecl-hip/7.1.

| Test | torch 2.13 (7.1-built) | **torch 2.15 (7.14-built)** |
|---|---|---|
| fp16 matmul 2048² | 0.29 ms | **0.18 ms** |
| bf16 matmul 2048² | 0.28 ms | **0.18 ms** |
| fp32 matmul 2048² | 1.24 ms | 1.43 ms |
| **ShuffleCugan fp16 1080p→2160p** | 111.5 ms → 9.0 FPS | **41.8 ms → 23.9 FPS** |
| ShuffleCugan fp32 1080p→2160p | 569.0 ms → 1.8 FPS | 94.2 ms → 10.6 FPS |

- **FP8 not usable on RDNA4 yet** (two gaps): torch never wires fp8 on ROCm
  (`addmm_cuda not implemented for Float8_*`, `_scaled_mm` CUDA-only), and a
  direct hipBLASLt 1.4 probe crashes every `hipblasLtMatmul` with a GPU memory
  fault despite fp8 kernels being compiled for gfx1201 → needs newer ROCm.
- **fp16 is input-range sensitive:** outside 0..1 (e.g. randn) fp16 collapses
  to all-NaN; fp32 clean. With normalized 0..1 video (what the app feeds) fp16
  is clean (max diff vs fp32 0.075, mean 0.0006). **Clamp input to 0..1.**
- Not portable (ROCm 7.x + RDNA4 + matching torch), heavy libtorch, first-kernel
  JIT; end-to-end gain smaller than model-only (decode/encode dominate).

### ncnn/Vulkan — dropped

Prebuilt `realcugan-ncnn-vulkan` (auto-tile, fp16, **includes PNG codec
overhead**): 249 ms @720p, 398 ms @1080p. Was the 2026-08-16 winner but is
superseded by burn-Vulkan fp16; the C++ shim (`senmei-ncnn`) was removed. ncnn
survives only as a **weight format** for the RIFE port (`flownet.bin`).

### candle/ROCm — dropped

The `xmiksay/feat/rocm-backend` fork is numerically correct but f32 convs
materialize im2col (memory cliff from ~640p), f16 ~6× slower than burn-Vulkan,
and the ShuffleCugan port OOMs even at 64×64.

### burn-ROCm — dropped

f32 works but is slow (up2x 1119 / 2197 ms @720p / 1080p, linear scaling, no
1080p collapse). fp16/bf16 matmul hits cubecl/ROCm-7.1 `LLVM Cannot select` on
gfx1201 — a **software gap** (HW WMMA works, see torch-ROCm); bf16 is slower
than fp16 anyway. Early smoke rows (57 / 2.19 ms) were a 3-conv toy, not
SR-representative.

### torch-ROCm original rows (2026-08-16, superseded)

Full-image, direct tensor I/O: 139.26 ms @720p, **7153.6 ms @1080p**
(pathological MIOpen/RDNA4 collapse: 51× slower than 720p despite 2.25× pixels);
tiling OOM'd (tile 512) or hard-faulted the GPU (tile 256, core dump). RealESRGAN
x4plus 640×360→1440p: 924.6 ms.

See `docs/PLAN.md` for the current engine/roadmap status.
