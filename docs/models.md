# Model Adoption Notes

Research & adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing
policy) and `models/metadata.json` (registry). Last verified: 2026-08-16.

Adoption rule (from `AGENTS.md`/`PLAN.md`): only permissive weights
(BSD/MIT/Apache), never AGPL-derived (RVE/TAS off-limits). Each adopted model
gets a `metadata.json` entry with license + source + pinned HF repo/commit.

## Model flow (2026-08-16, v2)

- Inference stack: **NCNN/Vulkan** via C++ shim (`cxx`/bindgen), CPU fallback.
  No libtorch, no ONNX, no TorchScript, **no candle**.
- Models are **downloaded** as ncnn `.param`/`.bin` (community ports) from pinned
  repos (repo + commit SHA) — **no conversion, no Python, no Rust arch ports**.
- The per-model cost is finding a **permissively-licensed NCNN port**; the Rust
  side only shells out through the shim.
- candle/burn were evaluated and **dropped** (candle: no ROCm backend; burn:
  fusion/JIT immature for SR). `.safetensors`/`.pth`/`.pt` and the
  `scripts/convert_*.py` pipeline are obsolete. Evidence: `docs/benchmarks.md`.

## Engine / integration status

- **Interpolation** is a placeholder: `senmei-ml::interpolate` only does linear
  `blend` + scene-cut detection (duplicate frames). RIFE planned
  (`rife-4.26`, MIT, `hzwer/Practical-RIFE`) but requires a **2-input** infer
  (frame pair + `t`) — the current `InferenceEngine::infer(input, opts)` is
  single-tensor and must be extended.
- **NcnnEngine** (Vulkan + CPU) is the **single engine** (decision 2026-08-16);
  it is a stub — wiring the C++ shim is the M6 work item.
- **Deduplication** is a pipeline/UI step (frame-similarity filter), not an ML
  `ModelKind` — no neural model needed.

## NCNN port availability

Adoption depends on a permissively-licensed **NCNN port** (`.param`/`.bin`), not
on spandrel coverage. `nihui/ncnn-assets` and the `*-ncnn-vulkan` releases
(Real-ESRGAN, Real-CUGAN, RIFE, SPAN, …) cover the core tasks; any other model
needs a per-checkpoint license check before adoption.

## Upscaler candidates

| Model | License | NCNN port | Character | Verdict |
|---|---|---|---|---|
| **SPAN** (`hongyuanyu/SPAN`) | Apache-2.0 | ✓ (community) | Lightweight real-time SR; NTIRE 2023 ESR 1st, CVPR-W 2024 oral | **Adopt** (default upscaler) |
| **SAFMN** (`sunny2109/SAFMN`) | Apache-2.0 | ✓ (+`SAFMNBCIE`) | Lightweight real-time SR (ICCV 2023); Real x2/x4 + BCIE variants | **Adopt** (upscale + decompress) |
| **RGT** (`zhengchen1999/RGT`) | Apache-2.0 | △ (port needed) | PSNR-oriented transformer (ICLR 2024); ~13M params, ~251 GFLOPs @x4 | Skip — heavy, smooth output |
| **DRCT** (`ming053l/DRCT`) | MIT | △ (port needed) | PSNR-oriented transformer (CVPR 2024 NTIRE); DRCT-L 27.6M; also Real-DRCT-GAN | Skip — heavy; only Real-DRCT-GAN interesting |
| **SeemoRe** (`eduardzamfir/seemoredetails`) | **conflict** | — | Efficient MoE real-world SR (ICML 2024) | **Exclude** — GitHub shows Apache-2.0 but README states CC BY-NC-SA 4.0 research-only |

## Adopted (NCNN `.param`/`.bin`)

| Model | License | Scale | Source |
|---|---|---|---|
| Real-ESRGAN x4plus / x4plus-anime / x2plus | BSD-3-Clause | 4 / 4 / 2 | `xinntao/Real-ESRGAN` · NCNN port |
| Real-CUGAN up2x (no-denoise) | MIT | 2 | bilibili/ailab · `nihui/realcugan-ncnn-vulkan` (verified: 1080p x2 in 398 ms) |
| SwinIR x2 / x4 | Apache-2.0 | 2 / 4 | `JingyunLiang/SwinIR` · NCNN port |
| RIFE 4.26 | MIT | 1 | `hzwer/Practical-RIFE` · NCNN port (pending) |

Weights are never committed (`models/*.param`/`*.bin` gitignored); the C++ shim
loads them from `models/`.

## Notes

- **SPAN variants**: "SPAN" is an overloaded name — unrelated nets exist
  (e.g. Spatial Pyramid Attention Network for manipulation localization). Pin to
  `hongyuanyu/SPAN` + its NCNN port. Multiple checkpoints share the arch (scales
  x2/x3/x4); each checkpoint gets its own `metadata.json` entry. Distinct
  sub-variants: **SPAN-F** (2025, lighter config) and **SwiftSRGAN** (separate
  GAN-based real-world SR,
  `Koushik0901/Swift-SRGAN`, **CC0-1.0**).
- **License is per artifact, not per arch**: each checkpoint carries its own
  license (code license ≠ weight license; community fine-tunes are sometimes
  non-commercial). Record the weight license per model in `metadata.json` and
  only adopt permissive (BSD/MIT/Apache, CC0 ok).
- **SPAN**: `PLAN.md` §14 previously excluded it ("no license"). Outdated — the
  repo now ships `LICENSE.txt` (Apache-2.0) and README confirms it.
- **SeemoRe**: license conflict (Apache-2.0 badge vs. CC BY-NC-SA 4.0 README
  clause, "academic research use only"). Re-evaluate only after the authors
  clarify.
- **SAFMN** also covers the decompress task via the `SAFMN_BCIE` checkpoint
  (blind compressed-image enhancement, v0.1.1 release).
- PSNR-oriented models (RGT/DRCT) are benchmark-optimized: they produce smooth
  reconstruction, not the "enhancement" look, and are too slow per frame for a
  video enhancer.
