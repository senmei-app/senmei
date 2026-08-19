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
| BSRGAN | upscale (restoration) | MIT | 4 | loadable (RRDBNet, 23 blocks; torch-verified mae 0.001) | `cszn/KAIR` · `BSRGAN.pth` |
| RIFE v4.6 | interpolate | MIT | 1 | loadable (RifeNet, ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| IFRNet (Vimeo90K / GoPro) | interpolate | MIT | 1 | loadable (IfrNet) | HF `pavlichenko/ifrnet_*` |
| DRUNet color (DPIR) | denoise | MIT | 1 | loadable (UNetRes, in_nc=4 sigma-map; torch-verified mae 0.001); wired into the Denoise step | `cszn/KAIR` · `drunet_color.pth` |
| NAFNet-GoPro width32 | deblur | MIT | 1 | loadable (NafNet; torch-verified mae 0.0007; fp16-safe on real images) | HF `nyanko7/nafnet-models` · `NAFNet-GoPro-width32.pth` |
| Real-PLKSr DeJPG / DeH264 | decompress | verify (Phhofm) | 1 | loadable, download gated | `Phhofm/models` · TAS host |

Weights never committed (`models/*` gitignored); download-on-demand + sha256,
converted once to f16 `.bpk`.

## Sources

- `renarchi/Re-SISR` — Fallin Soft/Strong + 4x_Alchemy, CC-BY-4.0 (Fallin ONNX-only).
- `styler00dollar/VSGAN-tensorrt-docker` `models` tag — many `.pth`/`.onnx` checkpoints.
- `pavlichenko/ifrnet_vimeo` / `ifrnet_gopro` (HF) — IFRNet weights, MIT (re-uploads of `ltkong218/IFRNet`).
- `cszn/KAIR` v1.0 — DRUNet/DnCNN/FFDNet/BSRGAN weights, MIT (direct release URLs).
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
| Denoise | DRUNet (DPIR) | MIT (KAIR) | **adopted** — loadable (UNetRes, in_nc=4 sigma-map), Denoise-step wiring pending |
| Denoise | DnCNN / FFDNet | MIT (KAIR) | adopt (trivial) |
| Denoise | NAFNet | MIT (HF nyanko7) | maybe (PSNR-oriented) |
| Denoise | IRCNN | MIT (KAIR) | maybe (denoise/deblur/deblock) |
| Denoise | FBCNN | verify | maybe (DeJPEG overlap) |
| Denoise | VRT / RVRT | verify · not in spandrel | no (temporal/transformer) |
| Restoration | Real-ESRGAN animevideov3 + general-x4v3 | BSD-3 | adopt (SRVGGNetCompact) |
| Restoration | BSRGAN / BSRNet | MIT (KAIR) | **adopted** — BSRGAN loadable (RRDBNet 23, BasicSR key remap); BSRNet port open |
| Restoration | Anime1080Fixer | verify | adopt (RRDBNet exists) |
| Restoration | IMDN x4 | MIT (KAIR) | maybe (lightweight) |
| Restoration | SAFMN Real x2/x4 | Apache-2.0 | adopt |
| Restoration | SPAN | Apache-2.0 | adopt |
| Restoration | USRNet / USRGAN | MIT | maybe (non-blind, kernel+noise) |
| Restoration | PLKSR more | verify (new Re-SISR NC-blocked) | maybe |
| Restoration | HAT | MIT · weights verify | maybe (transformer) |
| Restoration | OmniSR | MIT · weights verify | no (deformable conv) |
| Deblur | NAFNet-GoPro (width32/64) | MIT (HF mirror) | **adopted** — loadable (NafNet, width32); width64 untested in fp16 |

## Notes

- License is per artifact (code ≠ weight); adopt permissive only.
- ONNX loads without ONNX Runtime (built-in protobuf reader).
- NAFNet fp16 (NAFBlock): SimpleGate = channel split × multiply (no activation);
  LayerNorm2d computes the channel reduction in a scaled `x/S` (S≈128) domain
  and back (fp16-safe); channel attention = `mean(3).mean(2)` (no sigmoid),
  decoder up = Conv1×1 + PixelShuffle(2). Verified against torch on a
  realistic input (mae 0.0007) and on the encoder block-by-block. The internal
  activations are fp16-safe on real images (enc3 max ~2000, bottleneck ~175),
  but **overflow fp16 (~70000) on pathological high-noise input** (σ ≥ ~0.1) —
  torch-fp16's LayerNorm silently overflows to 0 there, so it is not a faithful
  fp16 reference (only f32 is). MIT confirmed (GoPro width32 weights).
- f32→f16: pre-convert to `.bpk` (BurnpackStore + HalfPrecisionAdapter).
