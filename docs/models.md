# Model Adoption Notes

Adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing) and
`models/metadata.json` (registry). Last verified: 2026-08-19.

Rule: permissive weights only (BSD/MIT/Apache/CC0), never AGPL-derived
(RVE/TAS off-limits). Weights and arch are separate licenses; each adopted
model gets a `metadata.json` entry (license + source + download URL + sha256).

## Adopted (burn)

| Model | Kind | License | Scale | Status | Source |
|---|---|---|---|---|---|
| Real-CUGAN up2x no-denoise | upscale | Apache-2.0 | 2 | loadable (UpCunet2x) | `bilibili/ailab` · VSGAN |
| Fallin Soft | upscale | CC-BY-4.0 | 2 | loadable (UpCunet2x_fast, pad 38) | `renarchi/Re-SISR` · `.onnx` |
| Fallin Strong | upscale | CC-BY-4.0 | 2 | loadable (UpCunet2x_fast, pad 38) | `renarchi/Re-SISR` · `.onnx` |
| 4x_Alchemy | upscale | CC-BY-4.0 | 4 | loadable (RealPLKSR_Dysample) | `renarchi/Re-SISR` · `.pth` |
| Real-ESRGAN animevideo x2/x4 | upscale | BSD-3-Clause | 2/4 | loadable (RRDBNet, 4 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| Real-ESRGAN x4plus-anime (6B) | upscale | BSD-3-Clause | 4 | loadable (RRDBNet, 6 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| RIFE v4.6 | interpolate | MIT | 1 | loadable (RifeNet, ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| IFRNet (Vimeo90K / GoPro) | interpolate | MIT | 1 | arch port done, numeric verify pending | HF `pavlichenko/ifrnet_*` |
| Real-PLKSr DeJPG / DeH264 | decompress | verify (Phhofm) | 1 | loadable, download gated | `Phhofm/models` · TAS host |

Weights never committed (`models/*` gitignored); download-on-demand + sha256,
converted once to f16 `.bpk`.

## Sources

- `renarchi/Re-SISR` — Fallin Soft/Strong + 4x_Alchemy, CC-BY-4.0 (Fallin ONNX-only).
- `styler00dollar/VSGAN-tensorrt-docker` `models` tag — many `.pth`/`.onnx` checkpoints.
- `pavlichenko/ifrnet_vimeo` / `ifrnet_gopro` (HF) — IFRNet weights, MIT (re-uploads of `ltkong218/IFRNet`).
- `NevermindNilas/TAS-Models-Host` — host only; TAS arch code is AGPL (off-limits).

## Backlog

Candidates per stack; each needs a clean burn port + permissive license before
`loadable: true`. Reference: `chaiNNer-org/spandrel` (permissive archs only).

> Re-SISR "Adore"+ = CC-BY-NC-SA → blocked (only pre-Adore CC-BY-4.0 adoptable).
> spandrel `(+)` archs (Restormer, CodeFormer, …) restrictive → skip. VFI never in spandrel.

| Stack | Candidate | License | Recommendation |
|---|---|---|---|
| Interpolation | RIFE variants (Lite/Fast/Max/S, anime) | MIT | adopt (arch exists) |
| Interpolation | GMFSS_Fortuna (union, anime) | MIT (repo) · weights verify | adopt |
| Interpolation | EMA-VFI | MIT · weights verify | maybe (cross-frame attention) |
| Interpolation | AnimeInterp | MIT · weights verify | maybe |
| Interpolation | FILM | Apache-2.0 repo · TF weights | maybe (TF conversion) |
| Interpolation | AMT | MIT · weights verify | no (transformer) |
| Denoise | SCUNet (GAN/PSNR) | Apache-2.0 | adopt (Swin port) |
| Denoise | DRUNet (DPIR) | MIT (KAIR) | adopt (conv, simple) |
| Denoise | DnCNN / FFDNet | MIT (KAIR) | adopt (trivial) |
| Denoise | NAFNet | MIT (HF nyanko7) | maybe (PSNR-oriented) |
| Denoise | IRCNN | MIT (KAIR) | maybe (denoise/deblur/deblock) |
| Denoise | FBCNN | verify | maybe (DeJPEG overlap) |
| Denoise | VRT / RVRT | verify · not in spandrel | no (temporal/transformer) |
| Restoration | Real-ESRGAN animevideov3 + general-x4v3 | BSD-3 | adopt (SRVGGNetCompact) |
| Restoration | BSRGAN / BSRNet | MIT (KAIR) | adopt (RRDBNet exists) |
| Restoration | Anime1080Fixer | verify | adopt (RRDBNet exists) |
| Restoration | IMDN x4 | MIT (KAIR) | maybe (lightweight) |
| Restoration | SAFMN Real x2/x4 | Apache-2.0 | adopt |
| Restoration | SPAN | Apache-2.0 | adopt |
| Restoration | USRNet / USRGAN | MIT | maybe (non-blind, kernel+noise) |
| Restoration | PLKSR more | verify (new Re-SISR NC-blocked) | maybe |
| Restoration | HAT | MIT · weights verify | maybe (transformer) |
| Restoration | OmniSR | MIT · weights verify | no (deformable conv) |
| Deblur | NAFNet-GoPro (width32/64) | MIT (HF mirror) | adopt (first ML deblur; NAFBlock) |

## Notes

- License is per artifact (code ≠ weight); adopt permissive only.
- ONNX loads without ONNX Runtime (built-in protobuf reader).
- NAFNet fp16 (NAFBlock): SimpleGate = channel split × multiply (no activation);
  LayerNorm2d in fp16 overflows — compute channel reduction in scaled `x/S`
  (S≈128) and back; channel attention = `mean(3).mean(2)`, upsample =
  Conv1×1 + PixelShuffle(2). MIT confirmed (GoPro width32 weights).
- f32→f16: pre-convert to `.bpk` (BurnpackStore + HalfPrecisionAdapter).
