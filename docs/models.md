# Model Adoption Notes

Adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing) and
`models/metadata.json` (registry). Last verified: 2026-08-19.

Rule: permissive weights only (BSD/MIT/Apache/CC0), never AGPL-derived
(RVE/TAS off-limits). Weights and arch are separate licenses; each adopted
model gets a `metadata.json` entry (license + source + download URL + sha256).

## Adopted (burn)

| Stack | Model | Scale | License | Status | Source |
|---|---|---|---|---|---|
| Interpolation | RIFE v4.6 | 1 | MIT | loadable (RifeNet, ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| Interpolation | IFRNet (Vimeo90K / GoPro) | 1 | MIT | loadable (IfrNet) | HF `pavlichenko/ifrnet_*` |
| Denoise | DRUNet color (DPIR) | 1 | MIT | loadable (UNetRes, in_nc=4 sigma-map; torch-verified mae 0.001); wired into the Denoise step | `cszn/KAIR` · `drunet_color.pth` |
| Denoise | Real-PLKSr DeJPG / DeH264 | 1 | verify (Phhofm) | loadable, download gated | `Phhofm/models` · TAS host |
| Restoration | Real-CUGAN up2x no-denoise | 2 | Apache-2.0 | loadable (UpCunet2x) | `bilibili/ailab` · VSGAN |
| Restoration | Fallin Soft | 2 | CC-BY-4.0 | loadable (UpCunet2x_fast, pad 38) | `renarchi/Re-SISR` · `.onnx` |
| Restoration | Fallin Strong | 2 | CC-BY-4.0 | loadable (UpCunet2x_fast, pad 38) | `renarchi/Re-SISR` · `.onnx` |
| Restoration | 4x_Alchemy | 4 | CC-BY-4.0 | loadable (RealPLKSR_Dysample) | `renarchi/Re-SISR` · `.pth` |
| Restoration | Real-ESRGAN animevideo x2/x4 | 2/4 | BSD-3-Clause | loadable (RRDBNet, 4 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| Restoration | Real-ESRGAN x4plus-anime (6B) | 4 | BSD-3-Clause | loadable (RRDBNet, 6 blocks) | `xinntao/Real-ESRGAN` · VSGAN |
| Restoration | BSRGAN | 4 | MIT | loadable (RRDBNet, 23 blocks; torch-verified mae 0.001) | `cszn/KAIR` · `BSRGAN.pth` |
| Deblur | NAFNet-GoPro width32 | 1 | MIT | loadable (NafNet; torch-verified mae 0.0007; fp16-safe on real images) | HF `nyanko7/nafnet-models` · `NAFNet-GoPro-width32.pth` |

Weights never committed (`models/*` gitignored); download-on-demand + sha256,
converted once to f16 `.bpk`.

## Sources

- `renarchi/Re-SISR` — Fallin Soft/Strong + 4x_Alchemy, CC-BY-4.0 (Fallin ONNX-only).
- `styler00dollar/VSGAN-tensorrt-docker` `models` tag — many `.pth`/`.onnx` checkpoints.
- `pavlichenko/ifrnet_vimeo` / `ifrnet_gopro` (HF) — IFRNet weights, MIT (re-uploads of `ltkong218/IFRNet`).
- `cszn/KAIR` v1.0 — DRUNet/DnCNN/FFDNet/BSRGAN weights, MIT (direct release URLs).
- `NevermindNilas/TAS-Models-Host` — host only; TAS arch code is AGPL (off-limits).
- `Phhofm/models` GitHub releases — <https://github.com/Phhofm/models/releases> —
  the canonical Phhofm (Philip Hofmann) model source: SPAN / DAT2 / RealPLKSR /
  Real-ESRGAN weights, all **CC-BY-4.0**, `.pth` + `.safetensors`; also mirrored
  on HF as `Phips/…`.

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
| Denoise | DnCNN / FFDNet | MIT (KAIR) | adopt (trivial) |
| Denoise | NAFNet | MIT (HF nyanko7) | maybe (PSNR-oriented) |
| Denoise | IRCNN | MIT (KAIR) | maybe (denoise/deblur/deblock) |
| Denoise | FBCNN | verify | maybe (DeJPEG overlap) |
| Denoise | VRT / RVRT | verify · not in spandrel | no (temporal/transformer) |
| Restoration | Real-ESRGAN animevideov3 + general-x4v3 | BSD-3 | adopt (SRVGGNetCompact) |
| Restoration | BSRNet | MIT (KAIR) | maybe (BSRGAN adopted; port open) |
| Restoration | Anime1080Fixer | verify | adopt (RRDBNet exists) |
| Restoration | IMDN x4 | MIT (KAIR) | maybe (lightweight) |
| Restoration | SAFMN Real x2/x4 | Apache-2.0 | adopt |
| Restoration | SPAN | Apache-2.0 arch · CC-BY-4.0 weights | adopt — burn port done + torch-verified, **f16/bf16-blocked** (see Notes); Phhofm `2xNomosUni_span_multijpg_ldl` + `2xBHI_small_span_pretrain`, TNTwise `ModernSpanimation` V1/V1.5/V2 + `DeH264_SPAN` |
| Restoration | USRNet / USRGAN | MIT | maybe (non-blind, kernel+noise) |
| Restoration | PLKSR more | verify (new Re-SISR NC-blocked) | maybe |
| Restoration | 4x BHI RealPLKSR-dysample (`_real`/`_multi`/`_otf`…) | CC-BY-4.0 | adopt (arch exists, quick; 5 variants in one release) |
| Restoration | 4x BHI DAT2 (`_real`, `_multiblurjpg`) | CC-BY-4.0 | maybe (best 4× quality, but **transformer port** = high cost; separate milestone) |
| Restoration | HAT | MIT · weights verify | maybe (transformer) |
| Restoration | OmniSR | MIT · weights verify | no (deformable conv) |
| Deblur | NAFNet-GoPro width64 | MIT (HF mirror) | maybe (width32 adopted; width64 untested in fp16) |

## Notes

- License is per artifact (code ≠ weight); adopt permissive only.
- RVE-hosted SPAN weights are **license-blocked**: `TNTwise/real-video-enhancer-models`
  ships many community SPAN variants (e.g. `2x_BHI_SpanPlusDynamic_Light.pth`)
  with **no license metadata**. "BHI" is Phhofm's (Philip Hofmann's) model series
  (CC-BY-4.0, e.g. `2xBHI_small_span_pretrain`), but the exact
  `SpanPlusDynamic_Light` checkpoint is not published under that name on
  HF/Phhofm, so the RVE-hosted copy stays **unverified → blocked** (2026-08-19).
  The SPAN arch (Apache-2.0, `hongyuanyu/SPAN`) stays adoptable via a clean port.
- SPAN is **not f16-safe**: torch-verified intermediates reach ~1e5 (block_3
  out2 ≈ 7.4e4, c2_r ≈ 1.1e5) > f16 max 65504 → NaN; bf16 is all-NaN on RADV.
  The burn port matches torch (f32) exactly but stays **gated** (no registry
  entry) until a precision-safe backend exists — re-evaluate with the deferred
  tch/libtorch f32 engine. V1/V1.5 use 64 feature channels (V2: 48).
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
