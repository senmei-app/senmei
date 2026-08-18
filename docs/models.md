# Model Adoption Notes

Research & adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing
policy) and `models/metadata.json` (registry). Last verified: 2026-08-17.

Adoption rule (from `AGENTS.md`/`PLAN.md`): only permissive weights
(BSD/MIT/Apache), never AGPL-derived (RVE/TAS off-limits). Weights and arch are
separate licenses: a clean arch re-implementation does not relicense the
weights. Each adopted model gets a `metadata.json` entry with license + source +
download URL (+ sha256 where known).

## Status at a glance

| Kind | Adopted & loadable | Port pending | License verify |
|---|---|---|---|
| Upscale | Real-CUGAN up2x (2×) · Fallin Soft/Strong (2×) · 4x_Alchemy (4×) · Real-ESRGAN animevideo x2/x4 · x4plus-anime 6B | — | — |
| Interpolation | RIFE v4.6 | — | — |
| Denoise / restore | — | SCUNet | Real-PLKSr DeJPG/DeH264 · Anime1080Fixer |

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
- Engine decision (2026-08-17): **burn** beats ncnn (302 vs 398 ms @1080p up2x) — see `docs/benchmarks.md`.

## Engine / integration status

- **BurnEngine** (Vulkan fp16) implements `InferenceEngine` and dispatches on
  `ModelRef::arch`: **`upcunet2x`** (verified in `rust-sr-bench`), **`realesrgan`**
  (`RRDBNet`, BSD-3 port) and **`rife46`** (`RifeNet`, generated from the ncnn
  `flownet` graph, MIT weights).
  `real-cugan-x2`, the 3× Real-ESRGAN models and `rife-v4.6` are `loadable: true`;
  SCUNet, Real-PLKSr and Anime1080Fixer (license verify) and Fallin/4x_Alchemy
  (arch port pending) are not loadable yet.
- **Interpolation** is real (RIFE v4.6): `infer_interp(a, b, t)` runs the
  flow-based network on Vulkan, verified (symmetric, directionally correct).
- **Deduplication** is a pipeline/UI step (frame-similarity filter), not an ML
  `ModelKind` — no neural model needed.

## Model sources

- **`renarchi/Re-SISR`** releases — Fallin Soft/Strong (2× CUGAN) and 4x_Alchemy
  (4× RealPLKSR), all **CC-BY-4.0**; Fallin is ONNX-only, 4x_Alchemy ships a `.pth`.
- **`styler00dollar/VSGAN-tensorrt-docker`** `models` release tag — hosts many
  `.pth`/`.onnx` checkpoints (Real-CUGAN, Real-ESRGAN, SCUNet, GMFSS, waifu2x, …)
  with direct release download URLs.
- **`NevermindNilas/TAS-Models-Host`** `main` release — hosts the models
  TheAnimeScripter uses (`.pth` + ONNX). TAS itself is AGPL — never copy its
  arch code; only its model *host* URLs may be referenced.
- TheAnimeScripter's model list (upscale/interpolate/restore) is used as an
  *inspiration* for what to adopt, not as a source of arch code or licensing.

## Adopted (burn)

| Model | License | Scale | Arch status | Source |
|---|---|---|---|---|
| Real-CUGAN up2x no-denoise | Apache-2.0 | 2 | **loadable** (UpCunet2x port, verified) | `bilibili/ailab` · `cugan_up2x-latest-no-denoise.pth` (VSGAN) |
| Fallin Soft | CC-BY-4.0 | 2 | **loadable** (UpCunet2x_fast, pad 38; ONNX initializer import) | `renarchi/Re-SISR` · `2x_Fallin_soft_renarchi_fp16.onnx` |
| Fallin Strong | CC-BY-4.0 | 2 | **loadable** (UpCunet2x_fast, pad 38; oversharpened sibling) | `renarchi/Re-SISR` · `2x_Fallin_strong_renarchi_fp16.onnx` |
| 4x_Alchemy | CC-BY-4.0 | 4 | port pending (RealPLKSR_Dysample, `.pth`) | `renarchi/Re-SISR` · `4x_Alchemy.pth` |
| Real-ESRGAN animevideo x2 / x4 | BSD-3-Clause | 2 / 4 | **loadable** (RRDBNet, 4 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| Real-ESRGAN x4plus-anime (6B) | BSD-3-Clause | 4 | **loadable** (RRDBNet, 6 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| RIFE v4.6 | MIT | 1 | **loadable** (RifeNet port, ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| SCUNet denoise | verify (cszn) | 1 | port pending | `cszn/SCUNet` · `scunet_color_15.pth` |
| Real-PLKSr DeJPG / DeH264 | verify (Phhofm) | 1 | port pending | `Phhofm/models` · TAS-Models-Host |
| Anime1080Fixer | verify (Zarxrax) | 1 | port pending | `Zarxrax/Anime1080Fixer` · VSGAN |

Weights are never committed (`models/*` gitignored); the app downloads them
(download-on-demand, sha256-verified) and converts to f16 `.bpk`.

## Backlog (candidates to evaluate)

Goal: ~4–5 models per stack, each needing a clean burn port + a permissive
weight license before `loadable: true`. Candidates come from
[`chaiNNer-org/spandrel`](https://github.com/chaiNNer-org/spandrel) (permissive
arch reference) and the sources above — this is a research list, not a
commitment.

| Stack | Candidates | License check |
|---|---|---|
| Interpolation | RIFE family (more variants) | MIT ok |
| Denoise | SCUNet, Real-PLKSr DeJPG/DeH264 | cszn verify · Phhofm verify |
| Restoration | Real-ESRGAN family, Anime1080Fixer | BSD-3 ok · Zarxrax verify |
| Depth map (use-case open) | MiDaS / Depth-Anything class | TBD — is it useful for video enhancement? |

## Notes

- **License is per artifact, not per arch**: each checkpoint carries its own
  license (code license ≠ weight license; community fine-tunes are sometimes
  non-commercial). Record the weight license per model in `metadata.json` and
  only adopt permissive (BSD/MIT/Apache, CC0 ok). Models flagged "verify" need a
  license check before `loadable: true`.
- **`shuffle-cugan` (SUDO) was removed (2026-08-18)** — its weights had no clear
  license. It is replaced by the CC-BY-4.0 Fallin Soft/Strong (2×) and 4x_Alchemy
  (4×); the default upscaler is now `real-cugan-x2` (Apache-2.0).
- **f16 is the default** (Vulkan): half the memory and 3–6× faster than f32 on
  the reference GPU. `PytorchStore` cannot cast f32→f16 at load, so weights are
  pre-converted to f16 `.bpk` (`BurnpackStore` + `HalfPrecisionAdapter`).
- **ONNX-only sources (Fallin) load without ONNX Runtime (2026-08-18)** — a
  built-in protobuf reader (`senmei_ml::onnx`) extracts the `initializer`
  tensors (the ONNX is only a weight container; the arch is the existing
  `UpCunet2x_fast`), then the `.pth`/`.onnx` → `.bpk` converter produces the
  burnpack. `download_model` detects `.onnx` sources automatically.
