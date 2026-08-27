# Model Adoption Notes

Adoption matrix for ML models. Companion to `PLAN.md` §14 (licensing) and
`models/metadata.json` (registry). Last verified: 2026-08-21.

Rule: permissive weights only (BSD/MIT/Apache/CC0), never AGPL-derived
(RVE/TAS off-limits). Weights and arch are separate licenses; each adopted
model gets a `metadata.json` entry (id/kind/family/arch + license + source +
download URL + sha256).

## Adopted (burn)

| Stack | Family | Model | Scale | Arch | License | Status | Source |
|---|---|---|---|---|---|---|---|
| Interpolation | RIFE | RIFE v4.6 | 1 | RifeNet | MIT | loadable (ncnn `flownet.bin`) | `nihui/rife-ncnn-vulkan` |
| Interpolation | IFRNet | IFRNet (Vimeo90K / GoPro) | 1 | IfrNet | MIT | loadable | HF `pavlichenko/ifrnet_*` |
| Denoise | DRUNet | DRUNet color (DPIR) | 1 | UNetRes (in_nc=4 σ-map) | MIT | loadable (torch-verified mae 0.001); wired into the Denoise step | `cszn/KAIR` · `drunet_color.pth` |
| Denoise | DnCNN | DnCNN color blind | 1 | Dncnn (20 conv, no BN) | MIT | loadable (spandrel-verified mae ~0.001); wired into the Denoise step | `cszn/KAIR` · `dncnn_color_blind.pth` |
| Denoise | FFDNet | FFDNet color | 1 | Ffdnet (12 conv, nc=96, pixel-unshuffle+σ) | MIT | loadable (torch-verified mae 0.0004); wired into the Denoise step | `cszn/KAIR` · `ffdnet_color.pth` |
| Denoise | SCUNet | SCUNet color σ=15 | 1 | Scunet (Swin-Conv-UNet [4,4,4,4,4,4,4], dim 64, win 8) | Apache-2.0 | loadable (torch-verified mae 0.0018; pth preprocessed contiguous); wired into the Denoise step | `cszn/SCUNet` · `scunet_color_15.pth` |
| Decompress | RealPLKSR | 1× DeNoise / DeJPG / DeH264 (otf, +DeJPG _60) | 1 | RealPLKSR (otf, 1×) | CC-BY-4.0 | loadable (kind `decompress` = 1× de-artifact; wired into the Decompress step) — license confirmed 2026-08-20, _60 contiguous-preprocessed, torch mae ~0.0003 | `Phhofm/models` releases |
| Restoration | Real-CUGAN | Real-CUGAN up2x no-denoise | 2 | UpCunet2x | Apache-2.0 | loadable | `bilibili/ailab` · VSGAN |
| Restoration | Real-CUGAN | Fallin Soft | 2 | UpCunet2x_fast (pad 38) | CC-BY-4.0 | loadable | `renarchi/Re-SISR` · `.onnx` |
| Restoration | Real-CUGAN | Fallin Strong | 2 | UpCunet2x_fast (pad 38) | CC-BY-4.0 | loadable | `renarchi/Re-SISR` · `.onnx` |
| Restoration | Real-CUGAN | Real-CUGAN-Pro 2× (no-denoise / conservative / denoise3x) | 2 | UpCunet2x | Apache-2.0 | loadable (same arch as real-cugan-x2, flat keys; spandrel-verified mae 0.79/255 f16) | `bilibili/ailab` Real-CUGAN-Pro 2022-05 · VSGAN mirror `.pth` |
| Restoration | RealPLKSR | 4x_Alchemy | 4 | RealPLKSR_Dysample | CC-BY-4.0 | loadable | `renarchi/Re-SISR` · `.pth` |
| Restoration | RealPLKSR | 2× Public (LayerNorm) | 2 | RealPLKSR_Dysample LayerNorm | CC-BY-4.0 | loadable (ONNX-verified mae ~0.018 f16 — DySample grid-sample f16-limited; spandrel f32 0.00015; flat contiguous `.pth` converts directly) | `Phhofm/models` · `2xPublic_realplksr_dysample_layernorm_real_nn` |
| Restoration | RealPLKSR | 4× NomosWebPhoto | 4 | RealPLKSR pixel-shuffle | CC-BY-4.0 | loadable (ONNX-verified mae 0.0007 f16; GroupNorm + pixel-shuffle tail, `dysample=false` variant) | `Phhofm/models` · `4xNomosWebPhoto_RealPLKSR` |
| Restoration | Real-ESRGAN | animevideo x2/x4 | 2/4 | SRVGGNetCompact (num_conv 16, folded) | BSD-3-Clause | loadable | `xinntao/Real-ESRGAN` · VSGAN |
| Restoration | Real-ESRGAN | general-x4v3 | 4 | SRVGGNetCompact (num_conv 32, folded) | BSD-3-Clause | loadable (tiny + fast, real scenes; torch mae 0.0004 f16) | `xinntao/Real-ESRGAN` v0.2.5.0 |
| Restoration | Real-ESRGAN | animevideov3 | 4 | SRVGGNetCompact (num_conv 16, per-layer PReLU) | BSD-3-Clause | loadable (official XS anime-video model; torch mae 0.0004 f16) | `xinntao/Real-ESRGAN` v0.2.5.0 |
| Restoration | Real-ESRGAN | x4plus-anime (6B) | 4 | RRDBNet (6 blocks) | BSD-3-Clause | loadable | `xinntao/Real-ESRGAN` · VSGAN |
| Restoration | ESRGAN | BSRGAN | 4 | RRDBNet (23 blocks) | MIT | loadable (torch-verified mae 0.001) | `cszn/KAIR` · `BSRGAN.pth` |
| Deblur | NAFNet | NAFNet-GoPro width32 | 1 | NafNet | MIT | loadable (torch-verified mae 0.0007; fp16-safe on real images) | HF `nyanko7/nafnet-models` · `NAFNet-GoPro-width32.pth` |
| Restoration | SPAN | SPAN 2× (NomosUni multijpg, _ldl, HFA2k, HFA2k LUDVAE, ModernSpanimation V1) | 2 | Span (feature_channels 48/64) | CC-BY-4.0 · MIT | loadable (f16-safe) | `Phhofm/models` · `TNTwise/Models` |
| Restoration | SAFMN | SAFMN-L Real (LSDIR, x2/x4 v2) | 2/4 | SafmnNet (dim 128 / 16 blocks / ffn_scale 2.0) | Apache-2.0 | loadable (clean burn port; torch mae 0.008 x2 / 0.027 x4 f16 on worst-case random input) | `sunny2109/SAFMN` v0.1.0 · HF mirror `Meloo/SAFMN` |
| Restoration | ParagonSR | ParagonSR-Nano GAN 2× | 2 | ParagonSrNet (24 feat / 3×2 blocks / ffn 1.5) | MIT | loadable, but **numerically unstable on high-freq content (see Notes)** | `Phhofm/ParagonSR2` · `Phhofm/models` release `2xParagonSR_Nano_gan` (fused `.safetensors`) |

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
| Denoise | SCUNet (GAN/PSNR) | Apache-2.0 | done (Swin port adopted 2026-08-20) |
| Denoise | FFDNet | MIT (KAIR) | adopt (pixel-shuffle denoise; DnCNN done) |
| Denoise | NAFNet | MIT (HF nyanko7) | maybe (PSNR-oriented) |
| Denoise | IRCNN | MIT (KAIR) | maybe (denoise/deblur/deblock) |
| Denoise | FBCNN | verify | maybe (DeJPEG overlap) |
| Denoise | VRT / RVRT | verify · not in spandrel | no (temporal/transformer) |
| Restoration | BSRNet | MIT (KAIR) | maybe (BSRGAN adopted; port open) |
| Restoration | IMDN x4 | MIT (KAIR) | maybe (lightweight) |
| Restoration | SPAN remaining (ModernSpanimation V1.5/V2, `DeH264_SPAN`; Phhofm `2xBHI_small_span_pretrain`) | Apache-2.0 arch · CC-BY-4.0/MIT weights | adopt (weights-only, arch exists) — NomosUni multijpg/_ldl, ModernSpanimation V1 done |
| Restoration | USRNet / USRGAN | MIT | maybe (non-blind, kernel+noise) |
| Restoration | PLKSR more | verify (new Re-SISR NC-blocked) | maybe |
| Restoration | RealPLKSR_Dysample family — 2× BHI small (6), 4× BHI (5 variants/release), Nomos2, Nature, ArtFaces, HFA2k, mssim | CC-BY-4.0 | adopt dim-64/GroupNorm set (weights-only); small/large need an arch variant first — NomosWebPhoto done (pixel-shuffle tail variant) |
| Restoration | 4x BHI DAT2 (`_real`, `_multiblurjpg`) | CC-BY-4.0 | maybe (best 4× quality, but **transformer port** = high cost; separate milestone) |
| Restoration | HAT | MIT · weights verify | maybe (transformer) |
| Restoration | OmniSR | MIT · weights verify | no (deformable conv) |
| Deblur | NAFNet-GoPro width64 | MIT (HF mirror) | maybe (width32 adopted; width64 untested in fp16) |

## Notes

- SRVGGNetCompact (Real-ESRGAN) has two folded layouts that share the
  `SrvggNet` arch and differ in depth + PReLU sharing: animevideo-xs
  (num_conv 16, ONE shared PReLU — every `body.{odd}.weight` is identical) and
  general-x4v3 (num_conv 32, a distinct PReLU per layer); animevideov3 is the
  16-conv variant with a distinct PReLU per layer (general layout). The arch
  holds one `Prelu` per mid conv; the converter remaps `body.{2k+1}.weight` →
  `prelu.{k}.weight` (shared checkpoints fill every entry with the same value)
  BEFORE the conv remap (that order matters — the conv remap would otherwise
  produce keys the PReLU patterns would steal). `download_model` passes
  `num_conv` for `srvgg` archs. Torch mae on random 32×32: 0.0004
  (general-x4v3, animevideov3).

- SAFMN-L Real (Apache-2.0, `sunny2109/SAFMN` v0.1.0): the `SAFMN_L_Real_LSDIR_*`
  checkpoints are the official "Real" (LSDIR-trained) SAFMN-L weights
  (dim 128 / 16 blocks / ffn_scale 2.0 / SAFM n_levels 4). The state dict is
  wrapped under `params`/`params_ema` (EMA preferred). The burn port uses the
  torch-exact pixel-shuffle permutation `(0,1,4,2,5,3)` — the earlier shared
  helper used the wrong permute (latent bug, fixed 2026-08-23; also affected
  the SRVGG `realesrgan-animevideo` outputs). SAFM max-pools the channel groups
  to `h/2^i` with kernel=stride=2^i, so input H/W are edge-padded to a
  multiple of 8 inside `forward`. fp16 mae vs torch on random 32×32 input:
  0.008 (x2) / 0.027 (x4, larger output accumulates more f16 error); real
  video frames are smoother and land lower. HF mirror `Meloo/SAFMN`
  (apache-2.0) hosts the non-v2 weights.

- License is per artifact (code ≠ weight); adopt permissive only.
- Synthetic input = worst case (max high-frequency): re-test numeric issues on
  real frames before deeming a model unsafe.
- RVE-hosted SPAN weights are **license-blocked**: `TNTwise/real-video-enhancer-models`
  ships many community SPAN variants (e.g. `2x_BHI_SpanPlusDynamic_Light.pth`)
  with **no license metadata**. "BHI" is Phhofm's (Philip Hofmann's) model series
  (CC-BY-4.0, e.g. `2xBHI_small_span_pretrain`), but the exact
  `SpanPlusDynamic_Light` checkpoint is not published under that name on
  HF/Phhofm, so the RVE-hosted copy stays **unverified → blocked** (2026-08-19).
  The SPAN arch (Apache-2.0, `hongyuanyu/SPAN`) stays adoptable via a clean port.
- ShuffleCUGAN is **license-blocked**: the original `blesslus/ShuffleCUGAN`
  repo is gone (404, unverifiable → blocked) and its commonly attributed
  GPL-3.0 is copyleft (`gpl` in the license gate). The `sudo_shuffle_cugan`
  weights were dropped 2026-08-18 as unclear/SUDO (CHANGELOG). The fast arch
  is covered by `UpCunet2x_fast` (= ShuffleCugan's half-res pixel-unshuffle
  UNet) via **fallin** (renarchi/Re-SISR, CC-BY-4.0), so no FPS lever is lost.
- SPAN f16: intermediates fit f16 (no overflow), but a cubek-convolution f16 1×1
  conv bug (K=96 × N≥32768, `docs/upstream-issues.md` §6) degraded the 48ch
  checkpoints — HFA2k_LUDVAE worst (corr 0.57 vs torch), 2xHFA2kSPAN 0.82,
  multijpg 0.68. **Workaround 2026-08-21**: `Span::pad_k96` zero-pads every 1×1
  conv2 96→128 (K=128/K=192 verified correct at N≥32768), which unblocks all
  48ch models (nomosuni-ldl/multijpg, HFA2k, HFA2k_LUDVAE, ModernSpanimation
  V2 re-enabled; LUDVAE = flat channels-last pth, contiguous-preprocessed).
  Measured: the padded K=128 path is not slower than the broken K=96 (−9% on the
  conv; K=128 tiles better). V1.5 (64ch) matches torch exactly (corr 1.00).
  bf16 all-NaN on RADV. V1/V1.5 = 64 channels, V2 = 48.
- RealPLKSR_Dysample family (CC-BY-4.0, `Phhofm/models` releases; arch ported via
  4x_Alchemy = dim 64 / 28 blocks / GroupNorm4 → **weights-only only for that config**):
  1× DeNoise/DeJPG/DeH264 (+ DeJPG `_60` q60 variant, 2026-08-20) + 4×
  BHI/Nomos2/NomosWebPhoto/Nature/HFA2k/mssim fit.
  Anime-first: BHI (4×) + Nomos. Realistic: 4xNature/HFA2k/mssim + 1× DeNoise/DeJPG/DeH264.
  ArtFaces = faces only (skip). 2× Public layernorm adopted (2026-08-20: RealPlk
  layer_norm variant, per-pixel channel LayerNorm; converter `.norm.`→`.layer_norm.`
  remap added 2026-08-21 so it converts at all — before it errored on the missing
  norm; f16 DySample mae ~0.018 vs ONNX, spandrel f32 0.00015 = f16-limited).
  2× BHI small (dim 32) and large (dim 96) still need an arch variant — not
  weights-only (todos). Registered 4× weights-only (2026-08-20): Nomos2, Nature,
  HFA2k_ludvae, mssim, BHI-real. BHI-otf = channels-last (fixed 2026-08-20:
  contiguous-preprocessed, loadable). NomosWebPhoto adopted 2026-08-21: the flat
  `4xNomosWebPhoto_RealPLKSR.pth` is GroupNorm(4) + a pixel-shuffle tail (no
  DySample, no LayerNorm — the ONNX `InstanceNormalization` + reshape is just
  GroupNorm semantics), so it uses the new `dysample=false` RealPlk variant
  (converter `dysample=0`); ONNX-verified mae 0.0007 f16.
  Skip redundant BHI sub-variants (otf_nn/multiblur).
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
- ParagonSR-Nano GAN (`paragonsr-nano-x2`) is **numerically unstable on real
  content** (2026-08-26): the fused release weights produce out-of-range output
  of ±26k–84k (torch fp32 reference, faithful arch, original fp32 safetensors)
  at high-frequency regions (e.g. burned-in subtitles), so the render shows a
  black band / white specks there and the values wander frame-to-frame. The
  arch also uses **GroupNorm(1,C) global over H·W** → NOT translation-equivariant,
  so it must not be tiled. Neither tiling nor full-frame avoids the blowup
  (full-frame even NaNs on some frames in the tch engine). Do not recommend
  this model for video with subs; fallin-soft / real-cugan-pro / animevideo-xs
  are stable.
