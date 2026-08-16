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

## Read honestly

- ncnn rows include PNG encode/decode + auto-tiling overhead; torch rows use
  direct tensor I/O — the real ncnn gap vs torch is smaller than the table
  suggests, but at 1080p ncnn is still ~an order of magnitude faster.
- bf16 on ROCm is slower than fp16 (verified); fp16 is the default.
- burn-ROCm is a 3-conv toy, not an SR model — it only proves the backend runs
  on RDNA4.

## Decision (2026-08-16)

- **torch/ROCm is not viable on RDNA4 for SR** (1080p perf + tile OOM/hard fault).
- **ncnn/Vulkan wins** — proven 398 ms @1080p x2, ready-made community ports,
  BSD-3, CPU fallback included.
- **candle dropped** (no ROCm backend; per-model Rust ports).
- **burn** set aside (fusion/JIT immature for SR; no reason to finish the port
  against ncnn's ready-made models).
- Engine = **NCNN/Vulkan via C++ shim** with CPU fallback. See `docs/PLAN.md` §15.
