# Model Adoption Notes

Research & adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing
policy) and `models/metadata.json` (registry). Last verified: 2026-08-17.

Adoption rule (from `AGENTS.md`/`PLAN.md`): only permissive weights
(BSD/MIT/Apache), never AGPL-derived (RVE/TAS off-limits). Weights and arch are
separate licenses: a clean arch re-implementation does not relicense the
weights. Each adopted model gets a `metadata.json` entry with license + source +
download URL (+ sha256 where known).

## Model flow (2026-08-17, v3)

- Inference stack: **burn (`burn-wgpu`) on the Vulkan backend, fp16**, CPU
  fallback. No libtorch, no ONNX Runtime, no TorchScript, **no candle, no ncnn**.
- Every architecture is a **clean Rust re-implementation** in burn (from the spec
  or a permissively-licensed reference), never translated from AGPL/unclear
  code. Weights are loaded as `.pth` (via `burn-store`) and converted once to
  f16 `.bpk` burnpacks; the engine (`BurnEngine`) consumes the `.bpk`.
- Weights workflow: `senmei-ml-convert <arch> <model.pth> <out.bpk>` converts
  f32 `.pth` → f16 `.bpk` (maintainer step, proven on the real up2x weights);
  in-app, the `download_model` command downloads the `.pth` (sha256-verified
  when pinned) and converts it automatically.
- The per-model cost is: (a) the burn arch port, (b) a permissive-weight license
  check, (c) the one-time `.pth → f16 .bpk` conversion.
- Engine decision (2026-08-17): **burn** beats ncnn (302 vs 398 ms @1080p up2x;
  ShuffleCugan 46/103 ms) — see `docs/benchmarks.md`.

## Engine / integration status

- **BurnEngine** (Vulkan fp16) implements `InferenceEngine` and dispatches on
  `ModelRef::arch`: **`upcunet2x`** (verified in `rust-sr-bench`), **`upcunet2x-fast`**
  (ShuffleCugan, verified), **`realesrgan`** (`RRDBNet`, BSD-3 port) and
  **`rife46`** (`RifeNet`, generated from the ncnn `flownet` graph, MIT weights).
  `real-cugan-x2`, the 3× Real-ESRGAN models and `rife-v4.6` are `loadable: true`;
  SCUNet, Real-PLKSr and Anime1080Fixer (license verify) are pending.
- **Interpolation** is real (RIFE v4.6): `infer_interp(a, b, t)` runs the
  flow-based network on Vulkan, verified (symmetric, directionally correct).
- **Deduplication** is a pipeline/UI step (frame-similarity filter), not an ML
  `ModelKind` — no neural model needed.

## Model sources

- **`styler00dollar/VSGAN-tensorrt-docker`** `models` release tag — hosts many
  `.pth`/`.onnx` checkpoints (Real-CUGAN, Real-ESRGAN, SCUNet, ShuffleCugan,
  GMFSS, waifu2x, …) with direct release download URLs.
- **`NevermindNilas/TAS-Models-Host`** `main` release — hosts the models
  TheAnimeScripter uses (`.pth` + ONNX). TAS itself is AGPL — never copy its
  arch code; only its model *host* URLs may be referenced.
- TheAnimeScripter's model list (upscale/interpolate/restore) is used as an
  *inspiration* for what to adopt, not as a source of arch code or licensing.

## Adopted (burn)

| Model | License | Scale | Arch status | Source |
|---|---|---|---|---|
| Real-CUGAN up2x no-denoise | Apache-2.0 | 2 | **loadable** (UpCunet2x port, verified) | `bilibili/ailab` · `cugan_up2x-latest-no-denoise.pth` (VSGAN) |
| ShuffleCugan | unclear (SUDO) | 2 | **loadable** (upcunet2x-fast; prototype opt-in) | `sudo_shuffle_cugan_9.584.969.pth` (VSGAN) |
| Real-ESRGAN animevideo x2 / x4 | BSD-3-Clause | 2 / 4 | **loadable** (RRDBNet, 4 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| Real-ESRGAN x4plus-anime (6B) | BSD-3-Clause | 4 | **loadable** (RRDBNet, 6 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| RIFE v4.6 | MIT | 1 | **loadable** (RifeNet port, ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| SCUNet denoise | verify (cszn) | 1 | port pending | `cszn/SCUNet` · `scunet_color_15.pth` |
| Real-PLKSr DeJPG / DeH264 | verify (Phhofm) | 1 | port pending | `Phhofm/models` · TAS-Models-Host |
| Anime1080Fixer | verify (Zarxrax) | 1 | port pending | `Zarxrax/Anime1080Fixer` · VSGAN |

Weights are never committed (`models/*` gitignored); the app downloads them
(download-on-demand, sha256-verified) and converts to f16 `.bpk`.

## Notes

- **License is per artifact, not per arch**: each checkpoint carries its own
  license (code license ≠ weight license; community fine-tunes are sometimes
  non-commercial). Record the weight license per model in `metadata.json` and
  only adopt permissive (BSD/MIT/Apache, CC0 ok). Models flagged "verify" need a
  license check before `loadable: true`.
- **ShuffleCugan** is the fastest SR variant (46/103 ms @720p/1080p in
  `rust-sr-bench`) but its weights carry no clear license → flagged
  "prototype opt-in" in the catalog until the author clarifies.
- **f16 is the default** (Vulkan): half the memory and 3–6× faster than f32 on
  the reference GPU. `PytorchStore` cannot cast f32→f16 at load, so weights are
  pre-converted to f16 `.bpk` (`BurnpackStore` + `HalfPrecisionAdapter`).
