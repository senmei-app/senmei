# Benchmarks — Inference Engines on RX 9070

Comparative inference benchmarks behind the engine decision (2026-08-16).
Target device = the actual dev machine, not a synthetic proxy.

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 9070 (Navi 48, RDNA4, `gfx1201`), 16 GB VRAM |
| CPU / RAM | Granite Ridge iGPU; 30 GiB RAM |
| OS | Fedora/Nobara (fc44, dnf) |
| ROCm | 7.1 (hipcc, rocminfo, `/dev/kfd`, `libamdhip64`), torch 2.13.0+rocm7.1 |
| Vulkan | Mesa/RADV |
| Rust | 1.96.1 |

## Workload

- Real-CUGAN up2x (upcunet_v3, no-denoise), fp16, no tiling unless noted.
- Inputs: 720p and 1080p still frames → x2.

## Results

| Engine | Workload | Time/frame | Notes |
|---|---|---|---|
| torch-ROCm | 720p x2 | 139.26 ms | full-image, direct tensor I/O |
| torch-ROCm | 1080p x2 | 7153.6 ms | pathological: 51× slower than 720p despite 2.25× pixels (MIOpen/RDNA4) |
| torch-ROCm (tiling) | tile 512 | OOM | 16 GB VRAM exceeded |
| torch-ROCm (tiling) | tile 256 | GPU hard fault | "Memory access fault by GPU node-1"; core dump (`gpucore.*`) |
| torch-ROCm | RealESRGAN x4plus 640×360→1440p | 924.6 ms | |
| ncnn-Vulkan | 720p x2 | 249.24 ms | prebuilt `realcugan-ncnn-vulkan`; auto-tile, fp16; **includes PNG codec overhead** |
| ncnn-Vulkan | 1080p x2 | 397.66 ms | 18× faster than torch at 1080p |
| burn-ROCm (smoke) | 3× Conv2d 32ch 256×256 | 57.11 ms | JIT artifact |
| burn-ROCm (smoke) | same, 512×512 | 2.19 ms | proves ROCm launches on RDNA4; not SR-representative |

### Burn re-benchmark (2026-08-17)

The original burn-ROCm rows were a 3-conv toy. Re-tested with the **real
Real-CUGAN `upcunet_v3`** (up2x-no-denoise `.pth`) and the **ShuffleCugan**
alternative (`sudo_shuffle_cugan`), weights loaded via `burn-store::PytorchStore`
(key remap `conv.0`/`conv.2` → `conv`/`conv2`), outputs numerically verified
against the torch reference (f32 max diff ~6e-6; fp16 max diff ~1.7e-2).
Setup: `burn` 0.21.0, `burn-rocm`, `burn-wgpu` (Vulkan via RADV).

| Engine | Model | Dtype | 720p x2 | 1080p x2 | Notes |
|---|---|---|---|---|---|
| burn-ROCm | up2x | f32 | 1119 ms | 2197 ms | linear scaling; no torch-style 1080p collapse |
| burn-ROCm | any | fp16/bf16 | — | — | cubecl-hip kernels hit LLVM `Cannot select: %llvm.amdgcn.wmma.f32.16x16x16.{f16,bf16}` on gfx1201 — a ROCm-7.1/cubecl **software** gap; the HW WMMA works (see torch-ROCm re-test below) |
| burn-Vulkan | up2x | f32 | 966 ms | crash | 1080p f32: burn-fusion bug `Ordering is bigger than operations` |
| burn-Vulkan | up2x | fp16 | **136 ms** | **302 ms** | **beats ncnn** (249 / 398 ms) |
| burn-Vulkan | ShuffleCugan | f32 | 313 ms | — | |
| burn-Vulkan | ShuffleCugan | fp16 | **46 ms** | **103 ms** | **~5× faster than ncnn**; pixel-unshuffle input ⇒ UNet runs at half resolution |

## Read honestly

- ncnn rows include PNG encode/decode + auto-tiling overhead; torch rows use
  direct tensor I/O — the real ncnn gap vs torch is smaller than the table
  suggests, but at 1080p ncnn is still ~an order of magnitude faster.
- bf16 on ROCm is slower than fp16 (verified); fp16 is the default.
- burn-ROCm (2026-08-16 rows) is a 3-conv toy — not SR-representative.
- **2026-08-17:** burn's real SR numbers come from the **Vulkan backend, not
  ROCm**. `burn-rocm`'s fp16/bf16 matmul hits a cubecl/ROCm-7.1 `LLVM ERROR` on
  RDNA4 (`gfx1201`); even a bare 256×256 matmul fails. ROCm f32 works but is
  slow. **This is a software gap, not a hardware one** — see the torch-ROCm
  fp16 re-test below: with the ROCm 7.14 RDNA4 runtime, fp16 WMMA works and
  beats burn-Vulkan.
- burn-Vulkan fp16 runs the real upcunet at 136/302 ms — **faster than ncnn's
  249/398 ms** on the same GPU (ncnn rows include PNG overhead; burn is pure
  inference). The ShuffleCugan variant is ~5× faster still (46/103 ms) because
  the heavy UNet processes half-resolution tensors.
- Practical burn caveats: ~800 crates / 1.6 GB `target/`; `PytorchStore` cannot
  cast f32→f16 at load (pre-convert weights, or use `SafetensorsStore`/
  `BurnpackStore` + `HalfPrecisionAdapter`); a `burn-fusion` bug crashes Vulkan
  f32 at 1080p.

## Full-app render pipeline (2026-08-17)

Real end-to-end numbers from `senmei-pipeline/tests/bench.rs`
(`cargo test -p senmei-pipeline --release --test bench -- --ignored --nocapture`,
optional `BENCH_MODEL` and `SENMEI_X264_PRESET` env). Workload: 1080p testsrc →
2160p x2, ShuffleCugan f16 burnpack, Vulkan, 48 frames.

| Path | convert-in | infer | convert-out | total | FPS |
|---|---|---|---|---|---|
| before (sequential) | 3.7 ms | 158 ms | 35.7 ms | 197 ms | 5.1 |
| after (optimized) | 1.9 ms | 157 ms | 9.7 ms | 168 ms | 5.9 |
| **fused GPU RGB8 step** | 1.9 ms | 157 ms | — (on GPU) | **157 ms** | **6.4** |
| full threaded pipeline + x264 veryfast + GPU RGB8 | — | — | — | — | **6.5** |

- The 4K `tensor_to_frame` (35.7 → 9.7 ms) was the biggest CPU win: replaced
  `round().clamp()` per element with a saturating `(x*255+0.5) as u8`
  (autovectorizes); `frame_to_tensor` uses a `x/255` LUT (3.7 → 1.9 ms).
- `Pipeline::run` now runs decode/encode on threads so CPU I/O hides behind the
  GPU inference; the encode thread uses x264 `-preset veryfast` (was default
  medium — that had become the bottleneck at 2160p: 4.7 FPS).
- **GPU-side RGB conversion (`infer_rgb8`, the PyTorch way):** after the model
  forward, the output is transposed NCHW→NHWC, scaled to 0..255 and cast to U8
  **on the GPU** (`permute` + `cast(IntDType::U8)`), then only the packed RGB
  bytes (24.8 MB) cross the PCIe bus. This removes the ~100 MB f32 download +
  the CPU interleave pass entirely → 168 → 157 ms, end-to-end **5.1 → 6.5 FPS**.
  (A naive CPU f16→u8 loop was much slower — `f16::to_f32()` doesn't
  autovectorize; the GPU cast is the right call.)
- **Model inference (157 ms) is burn-Vulkan's floor** on RDNA4: ~6.4 FPS pure,
  ~6.5 FPS end-to-end. GPU hits 100 % / 3.2 GHz during inference (verified via
  rocm-smi); no throttling. **It is not an absolute floor** — torch-ROCm fp16
  (ROCm 7.14 runtime) runs the same model at 111.5 ms / 9.0 FPS (see below).
  Closing to TensorRT-class speed (36 FPS) still needs hand-tuned kernels.
- Real-CUGAN up2x and ShuffleCugan both measure identically — the per-frame
  transfer cost dominates, not the weights.

## torch-ROCm re-test (2026-08-17, ROCm 7.14)

The 2026-08-16 verdict ("fp16 impossible on RDNA4") was wrong. It was a
**ROCm-7.1 software gap** — the machine runs the **ROCm 7.14** RDNA4 port
(`/opt/therock-tarball/install`). Two torch builds tested: the installed
`2.13.0+rocm7.1` binary driving the 7.14 runtime, and a fresh nightly built
**for** 7.14 (`2.15.0.dev+rocm7.14`, in `$HOME/torch714-venv`).

| Test | torch 2.13 (7.1-built) | **torch 2.15 (7.14-built)** |
|---|---|---|
| fp16 matmul 2048² | 0.29 ms | **0.18 ms** |
| bf16 matmul 2048² | 0.28 ms | **0.18 ms** |
| fp32 matmul 2048² | 1.24 ms | 1.43 ms |
| **ShuffleCugan fp16 1080p→2160p** | 111.5 ms → 9.0 FPS | **41.8 ms → 23.9 FPS** |
| ShuffleCugan fp32 1080p→2160p | 569.0 ms → 1.8 FPS | 94.2 ms → 10.6 FPS |

- WMMA (FP16/BF16/FP8/BF8) is **present on RDNA4**; the earlier `LLVM Cannot
  select` was a **cubecl-hip / ROCm-7.1 kernel gap**. The **7.14-built torch
  unlocks RDNA4 fp16 conv kernels**: ShuffleCugan fp16 drops 111.5 → 41.8 ms —
  **3.8× faster than burn-Vulkan fp16 (157.4 ms / 6.4 FPS)**. fp32 also drops
  569 → 94.2 ms.
- **FP8 (e4m3/e5m2/fnuz) does NOT work via torch on ROCm** — not even in the
  7.14-built nightly: `addmm_cuda not implemented for Float8_*`, and
  `_scaled_mm` is CUDA-only (cuBLASLt error) on ROCm builds. The hipBLASLt 1.4
  library in ROCm 7.14 *does* support fp8 on gfx1201 (fp8 datatypes + "default
  mode for fp8"); torch just never calls it. fp8 needs a direct hipBLASLt path
  or a runtime that wires it (vLLM/ONNXRuntime-style) — not stock torch.
- **Caveat — fp16 is input-range sensitive:** on inputs outside 0..1 (e.g.
  randn) the fp16 path collapses to all-NaN; fp32 stays clean. With realistic
  video (normalized 0..1, what the app feeds) fp16 is clean: max diff vs fp32
  0.075, mean 0.0006, no NaN. **Clamp input to 0..1 before the fp16 forward.**
- Other caveats: needs ROCm 7.x + RDNA4 + a matching torch install (not
  portable like Vulkan) + libtorch size + first-kernel JIT. End-to-end gain is
  smaller than the model-only number (decode/encode + transfers dominate).

## Decision (2026-08-16, revised 2026-08-17)

- **torch/ROCm (7.14-built, fp16) is now the fastest engine measured** on RDNA4:
  ShuffleCugan fp16 1080p→2160p at **41.8 ms / 23.9 FPS** — **3.8× faster than
  burn-Vulkan fp16 (157.4 ms / 6.4 FPS)**. The 2026-08-16 "not viable" verdict
  was a ROCm-7.1 + fp32 artifact (incl. the tile OOM/hard-fault). FP8 is not
  reachable from stock torch on ROCm (CUDA-only path).
  Portability cost: needs ROCm 7.x + RDNA4 + matching torch (not portable like
  Vulkan) + libtorch size, so **burn-Vulkan stays the shipped default**;
  torch-ROCm-7.14 fp16 is a strong optional AMD backend (worth wiring behind
  the `InferenceEngine` trait) — clamp inputs to 0..1.
- **ncnn/Vulkan** won on 2026-08-16 (398 ms @1080p) and remains the **shipped
  engine** via the C++ shim.
- **2026-08-17 re-benchmark:** the earlier "burn set aside" verdict rested on a
  toy model on the wrong backend (ROCm). With the **real upcunet + Vulkan +
  fp16**, burn beats ncnn (302 vs 398 ms @1080p; 136 vs 249 ms @720p), and
  ShuffleCugan is ~5× faster still. **burn is re-opened as a candidate** —
  adoption must weigh the heavy build (~800 crates/1.6 GB), the fusion bug, and
  the f32→f16 load workflow against ncnn's tiny shim + ready-made ports.
- **candle/ROCm dropped (2026-08-17)** — evaluated the `xmiksay/feat/rocm-backend`
  candle fork (rocBLAS GEMM + im2col conv). Numerically correct (HIP vs CPU
  ~1e-5), but f32 convs always materialize the im2col matrix → memory cliff from
  ~640p (700→1107 ms @640p→720p; multi-GB buffers crash the desktop on
  shared-display GPUs; SD/FLUX VAE decode OOMs at 1024²), f16 scales linearly
  yet stays ~6× slower than burn-Vulkan fp16 (290 vs 46 ms @720p ShuffleCugan),
  and the ShuffleCugan port OOMs even at 64×64 (fork conv bug). Not pursued.
- Engine = **NCNN/Vulkan via C++ shim** with CPU fallback (unchanged until a
  maintainer decision on burn). See `docs/PLAN.md` §15.
