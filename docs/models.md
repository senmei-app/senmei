# Model Adoption Notes

Research & adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing
policy) and `models/metadata.json` (registry). Last verified: 2026-08-18.

Adoption rule (from `AGENTS.md`/`PLAN.md`): only permissive weights
(BSD/MIT/Apache), never AGPL-derived (RVE/TAS off-limits). Weights and arch are
separate licenses: a clean arch re-implementation does not relicense the
weights. Each adopted model gets a `metadata.json` entry with license + source +
download URL (+ sha256 where known).

## Model flow (v3)

- Inference stack: **burn (`burn-wgpu`) Vulkan fp16**, CPU fallback. No
  libtorch, ONNX Runtime, TorchScript, candle or ncnn.
- Every arch is a **clean Rust re-implementation** in burn (spec or permissive
  reference), never translated from AGPL/unclear code. Weights load as
  `.pth`/`.onnx` and convert once to f16 `.bpk`; `BurnEngine` consumes the `.bpk`.
- Convert: `senmei-ml-convert <arch> <model.pth> <out.bpk>` (f32 → f16,
  maintainer step). In-app, `download_model` downloads (sha256-verified when
  pinned) and converts automatically.
- Per-model cost: (a) burn arch port, (b) permissive-license check, (c) one-time
  `.pth → f16 .bpk` conversion.
- Engine decision: burn beats ncnn (302 vs 398 ms @1080p up2x) — see
  `docs/benchmarks.md`.

## Adopted (burn)

| Model | License | Scale | Status | Source |
|---|---|---|---|---|
| Real-CUGAN up2x no-denoise | Apache-2.0 | 2 | **loadable** (UpCunet2x) | `bilibili/ailab` · `cugan_up2x-latest-no-denoise.pth` (VSGAN) |
| Fallin Soft | CC-BY-4.0 | 2 | **loadable** (UpCunet2x_fast, pad 38) | `renarchi/Re-SISR` · `.onnx` |
| Fallin Strong | CC-BY-4.0 | 2 | **loadable** (UpCunet2x_fast, pad 38) | `renarchi/Re-SISR` · `.onnx` |
| 4x_Alchemy | CC-BY-4.0 | 4 | **loadable** (RealPLKSR_Dysample) | `renarchi/Re-SISR` · `4x_Alchemy.pth` |
| Real-ESRGAN animevideo x2 / x4 | BSD-3-Clause | 2 / 4 | **loadable** (RRDBNet, 4 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| Real-ESRGAN x4plus-anime (6B) | BSD-3-Clause | 4 | **loadable** (RRDBNet, 6 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| RIFE v4.6 | MIT | 1 | **loadable** (RifeNet, ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| Real-PLKSr DeJPG / DeH264 | verify (Phhofm) | 1 | **loadable** (RealPLKSR); download gated until license reviewed | `Phhofm/models` · TAS-Models-Host |
| SCUNet denoise | verify (cszn) | 1 | arch port pending | `cszn/SCUNet` · `scunet_color_15.pth` |
| Anime1080Fixer | verify (Zarxrax) | 1 | arch port pending | `Zarxrax/Anime1080Fixer` · VSGAN |

Weights are never committed (`models/*` gitignored); the app downloads them
(download-on-demand, sha256-verified) and converts to f16 `.bpk`.

## Engine / integration status

- **BurnEngine** (Vulkan fp16) implements `InferenceEngine` and dispatches on
  `ModelRef::arch`: `upcunet2x`, `realesrgan` (RRDBNet, BSD-3), `rife46`
  (RifeNet from ncnn `flownet`, MIT), `fallin-cugan` (UpCunet2x_fast) and
  `real-plksr`.
- **Interpolation** is real (RIFE v4.6): `infer_interp(a, b, t)` on Vulkan,
  verified (symmetric, directionally correct).
- **Deduplication** is a pipeline/UI step (frame-similarity filter), not an ML
  `ModelKind` — no neural model needed.

## Model sources

- **`renarchi/Re-SISR`** releases — Fallin Soft/Strong (2× CUGAN) and 4x_Alchemy
  (4× RealPLKSR), all **CC-BY-4.0**; Fallin is ONNX-only, 4x_Alchemy ships a `.pth`.
- **`styler00dollar/VSGAN-tensorrt-docker`** `models` release tag — hosts many
  `.pth`/`.onnx` checkpoints (Real-CUGAN, Real-ESRGAN, SCUNet, …) with direct
  release download URLs.
- **`NevermindNilas/TAS-Models-Host`** `main` release — hosts the models
  TheAnimeScripter uses (`.pth` + ONNX). TAS itself is AGPL — never copy its
  arch code; only its model *host* URLs may be referenced.
- TheAnimeScripter's model list (upscale/interpolate/restore) is used as an
  *inspiration* for what to adopt, not as a source of arch code or licensing.

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
| Restoration | Real-ESRGAN family, Anime1080Fixer, SPAN | BSD-3 ok · Zarxrax verify · SPAN Apache-2.0 |
| Depth map (use-case open) | MiDaS / Depth-Anything class | TBD — is it useful for video enhancement? |

## Notes

- **License is per artifact, not per arch**: each checkpoint carries its own
  license (code license ≠ weight license; community fine-tunes are sometimes
  non-commercial). Only adopt permissive (BSD/MIT/Apache, CC0 ok); "verify"
  models need a license check before `loadable: true`.
- **ONNX sources load without ONNX Runtime**: a built-in protobuf reader
  (`senmei_ml::onnx`) extracts the `initializer` tensors (the ONNX is only a
  weight container; the arch is the existing `UpCunet2x_fast`);
  `download_model` auto-detects `.onnx`.
- **f32→f16:** `PytorchStore` can't cast f32→f16 at load — pre-convert to f16
  `.bpk` (`BurnpackStore` + `HalfPrecisionAdapter`).
