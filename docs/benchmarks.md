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
- **`infer_rgb8`:** NCHW→NHWC, clamp 0..1, scale 0..255, cast to u8 **on GPU**
  per 512px tile; only packed u8 crosses back; tiles overlap-averaged. The
  full-frame variant OOM'd autotune + panicked — tiling avoids it (below).
- 157 ms is burn-Vulkan's floor on RDNA4 (GPU 100 % / 3.2 GHz, no throttling);
  torch-ROCm fp16 runs the same model at 111.5 ms / 9.0 FPS.

### Autotune failures — root cause & fix (2026-08-18)

Full-frame fused `infer_rgb8` OOM'd autotune on a large matmul (m=1024, n=4M,
f16), then panicked `Ordering is bigger than operations` (docs/burn-bugs.md
Bug 1+3). **Fix:** tile `infer_rgb8` (512px) so no full-frame matmul reaches
autotune. Guarded by `infer_rgb8_tiled_is_reliable_and_correct`. Disabling
autotune also works but is ~5× slower.

### Tile size & GPU stitch (2026-08-18)

The 512px tiled-fused path (329 ms / 3.0 FPS fallin-soft) was ~2× slower than
the pre-tiling full-frame fused path (176 ms) — that drop was the price of
avoiding the autotune OOM (Bug 3). Tried **1024px tiles** (6 tiles @1080p vs 15
@512, fewer u8 readbacks + less stitch work): **regression to 762 ms / 1.3 FPS**
— the larger per-tile matmul is pathologically slower on this backend.

**GPU stitch (2026-08-18):** instead of reading each 512px tile's u8 bytes back
and stitching on the CPU, tiles are accumulated into one f16 canvas on the GPU
(`slice_assign` overlap averaging) and read back as a single packed frame — one
readback instead of 15 plus a CPU stitch. `bench_upscale_step` (fallin-soft):
329 → **234.7 ms / 4.3 FPS**. 512px stays; no dynamic/`per-settings` tile size
needed.

## Fallin vs real-cugan (2026-08-18)

`bench.rs`, 1080p→2160p x2, Vulkan fp16, autotune + fusion on. Fused step =
`Upscale` + `infer_rgb8`. **Table = earlier full-frame fused step**; the current
tiled-fused step measures ~235 ms / 4.3 FPS (fallin-soft), full threaded
pipeline 2.8 FPS.

| Model | infer | total | FPS | fused step | step FPS | VRAM peak |
|---|---|---|---|---|---|---|
| real-cugan-x2 | 359 ms | 374 ms | 2.7 | 380 ms | 2.6 | 14.6 GB |
| fallin-soft | 193 ms | 205 ms | 4.9 | 176 ms | 5.7 | 8.1 GB |
| fallin-strong | 196 ms | 208 ms | 4.8 | 177 ms | 5.7 | 8.1 GB |

Fallin Soft/Strong ~2× faster than real-cugan-x2 at ~half the VRAM. VRAM via
`/sys/class/drm/card1/device/mem_info_vram_used` (baseline ~1.3 GB).

## Alternatives

### torch-ROCm (2026-08-17, ROCm 7.14)

The 2026-08-16 "fp16 impossible on RDNA4" verdict was a **ROCm-7.1 software
gap** — the machine runs the **7.14 RDNA4 port**. The 7.14-built nightly
(`2.15.0.dev+rocm7.14`, `$HOME/torch714-venv`) unlocks RDNA4 fp16
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
