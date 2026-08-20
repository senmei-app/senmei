# Senmei — Changelog

> Implementation log (was §15 of PLAN.md). Newest on top.

> Kept in sync with actual implementation. Update on every significant change.
> Each release gets a `## x.y.z (YYYY-MM-DD)` heading; release notes are
> generated from the section above the latest heading.

## Unreleased

## 0.1.3 (2026-08-20)

- **feat: wire tch backend end-to-end (2026-08-20)** — `tch` cargo feature
  threaded senmei → senmei-app → senmei-server (senmei-ml patch deps on the
  senmei-app torch-sys/tch/burn-tch forks); Settings page enables the LibTorch
  backend button only when compiled and a CUDA/ROCm device is present.

- **feat: ml: TchEngine on runtime-dlopen libtorch (CUDA/ROCm) (2026-08-20)**
  — runs the shared archs on `burn-tch` over the on-demand libtorch runtime
  (dlopen via `torch-sys` loader, no build-time link). Resolver pins 2.11.0
  (`rocm7.1`/`cu128`; 2.9.0 has no ROCm-7 build) and detects the system ROCm
  runtime dir so a bundled older HIP is shadowed at load. `backend_info()`
  probes CUDA/ROCm via dlopen (no loader init); `engine_for_model` takes the
  data dir, `Auto` prefers libtorch on a GPU. GPU roundtrip (save/load/infer
  128→256) verified against the downloaded ROCm build.

- **ml: add 2× Public RealPLKSR LayerNorm (2026-08-20)** — `RealPlk` now takes
  a `layer_norm: bool`; the layernorm variant swaps the tail GroupNorm for a
  per-pixel channel `LayerNorm` (eps 1e-6, weight/bias [64]) at the block
  start (`PlkBlock.layer_norm: Option<LayerNorm>` / `norm: Option<GroupNorm>`,
  record keys unchanged for existing GroupNorm models). `ModelRef` gained a
  `layer_norm` flag (parsed from `metadata.layer_norm`); the real-plksr
  converter drops the `params` top-level-key requirement (a `^params\.` → ""
  remap handles both wrapped and flat state dicts, so the flat 2xPublic pth
  converts directly) and the CLI gained a 6th `[layer_norm]` arg. Registered
  `real-plksr-2x-public` (Phhofm, CC-BY-4.0, sha256-pinned); ONNX-verified
  mae ~0.0016. Note: the f16 error explodes on synthetic pure-noise input
  (real-noise model amplifies it) — the verification uses a smooth
  representative input.

- **feat: FFmpeg between-filter step (2026-08-20)** — new `Filter` step in
  `Vec<Step>` that runs each frame through a free-form FFmpeg `-vf` graph over
  a rawvideo pipe (1:1 frame-preserving; filters changing the size are
  rejected). Wired as `filter.ffmpegFilter` in the server `RenderConfig` and
  the Tauri `FilterParams`, plus a new "filter" step in the processing-stack
  UI (single text field, `hue=h=10,unsharp`-style). Gives the whole FFmpeg
  video-filter catalog without per-filter ports. End-to-end verified via
  `/api/render` (`negate` inverts pixels).

- **compliance: adopt cargo-deny as license/bans/advisories gate (2026-08-20)**
  — new `deny.toml` (targets = shipped product triples, so UEFI-only
  `r-efi`/getrandom is excluded) + `bun run deny` script. Allowlist covers the
  full permissive set (MIT/Apache/BSD/ISC/MPL/CC0/CDLA-Permissive-2.0/bzip2-1.0.6/…);
  advisories: 0 CVEs, 13 `unmaintained` (gtk-rs GTK3 webview, paste,
  proc-macro-error, unic-*, bincode) ignored with justification. `zip`
  duplicate-unify deferred (touches libtorch WIP).

- **ml: add SCUNet color denoiser (2026-08-20)** — new `scunet` burn arch:
  Swin-Conv-UNet (config [4,4,4,4,4,4,4], dim 64, window 8, head_dim 32) with
  W/SW windowed multi-head self-attention (relative-position bias via per-head
  gather + stack, cyclic-shift roll reversed vs torch, SW cross-window -inf
  mask) + residual 3×3 conv branch, 1×1 conv merge, stride-2 U-Net with skip
  adds; internal 64-replication pad/crop. Registered `scunet-denoise` (cszn,
  Apache-2.0, sha256-pinned); `relative_position_params` is a transposed view
  in the pth so it must be contiguous-preprocessed before conversion, and the
  `Wmsa` module was added to the half-precision adapter set (else it loads f32
  into the f16 model → DTypeMismatch); wired into the Denoise step;
  torch-verified mae 0.0018. Fixed a replication-pad off-by-one (built
  `right+1` pad rows, so non-64-multiple heights like 360 → 385 broke the
  window partition) — covered by a new 66×50 non-aligned regression test
  (mae 0.0017); headless server needs `RUST_MIN_STACK` (see AGENTS.md).

- **ml: add FFDNet color denoiser (2026-08-20)** — new `ffdnet` burn arch
  (nb=12/nc=96: even-only replication pad, pixel-unshuffle(2) → 12ch + σ map
  = 13ch, 12 convs ReLU, no BN, pixel-shuffle(2)); registered `ffdnet-color`
  (KAIR, MIT, sha256-pinned); wired into the Denoise step; torch-verified mae
  0.0004 (reference regenerated with fresh conv modules — the earlier mae 0.21
  was a buggy Python `*[Conv,ReLU]*10` shared-module reconstruction).

- **ml: add DnCNN color denoiser (2026-08-20)** — new `dncnn` burn arch (20
  conv layers, ReLU between, no BN, residual `x - model(x)`); registered
  `dncnn-color` (KAIR, MIT, sha256-pinned); wired into the Denoise step;
  spandrel-verified mae ~0.001.

- **ml: make 4× RealPLKSR BHI-otf loadable (2026-08-20)** — channels-last
  `.pth` contiguous-preprocessed, converted to f16 `.bpk`, sha256 pinned; arch
  confirmed identical to BHI-real (346 keys / same shapes); torch (spandrel)
  mae ~0.005, no NaNs.

- **docs: headless HTTP/web path for other agents (2026-08-20)** — AGENTS.md
  gets a "Headless web" section (`senmei-server --http` start + REST surface +
  frontend `backend/` transports); PLAN.md §16.7 documents the HTTP adapter +
  web UI as the second transport (browsers *and* agents, same license/confirm
  gates); the OpenCode senmei agent now lists the REST endpoints as an MCP
  alternative.

- **feat: decompress step in the processing stack (M7) (2026-08-20)** — new
  `Decompress` step type picks a 1× RealPLKSR de-artifact model (DeH264 / DeJPG /
  DeJPG `_60` / DeNoise); mapped to `RenderConfig.decompress_model_id` and run
  as a scale-1 ML pass ahead of upscaling in `render`.

- **fix: web E2E — audio no-op, add-video button, sample render path (2026-08-20)** —
  `extractAudio` in the HTTP backend now no-ops (browser `<video>` handles
  sound) instead of throwing; the MediaLibrary `+` button was missing an
  `onClick` and now opens the file picker; sample renders in web mode write to
  `<input dir>/sample/` (the project dir is browser-only) instead of failing
  with permission denied. Verified end-to-end in the browser against
  `senmei-server --http` (render feature): import → probe → sample render →
  done → output file.

- **feat: path-input dialog for web file access (Dateizugriff B) (2026-08-20)** —
  new promise-based `PathDialog` (mounted at the app root) replaces the
  `window.prompt` fallbacks in the HTTP backend: `pickVideoFiles`,
  `pickFolder`, `pickSaveFile`, `pickFile` now open a proper modal for entering
  server-side paths (comma-separated for multiple files). Verified end-to-end
  in the browser against `senmei-server --http` (import + preview frame +
  output-folder picker).

- **fix: app sluggish after backend migration (2026-08-20)** —
  `useFfmpeg` created a fresh `getStatus` closure every render; since the
  StatusBar re-renders every second (hardware poll) and `useDownloadable`
  re-runs its refresh effect whenever `getStatus` changes identity, that meant
  a `getFfmpegStatus` IPC call per render → the whole UI lagged. Wrapped the
  callbacks in `useCallback` so the effect runs once.

- **fix: native video preview never engaged after backend migration (2026-08-20)** —
  `nativeSrc` raced the asset-protocol grant: the `<video>` mounted with
  `convertFileSrc(file)` before `probe_video` had run `allow_file`, so it
  `onError`'d and the preview fell back to slow FFmpeg frames permanently.
  The native URL is now set only after `probeVideo` succeeds (file granted).

- **ml: register RealPLKSR 1× DeJPG `_60` (q60) (2026-08-20)** — weights-only
  on the existing real-plksr arch: contiguous-preprocessed state dict, f16
  `.bpk` conversion, sha256 pinned; verified vs torch (spandrel) mae ~0.0003.

- **fix: preview frames broken after backend migration (2026-08-20)** —
  `tauri.ts readFrame` parsed the bridge result as an object, but
  `read_frame` resolves to a plain path string → empty `convertFileSrc("")` →
  "asset protocol not configured" + broken frame thumbs. Also restored the
  `?v=` cache-busting query for `asset://` frame URLs (stable filename + webview
  cache) that the migration had dropped.

- **feat: transport-agnostic frontend backend — one UI, two transports (2026-08-20)** —
  new `packages/app/src/backend/` abstraction (`types.ts` contract + `tauri.ts`
  IPC / `http.ts` REST impls + `mock.ts` dev backend via `VITE_SENMEI_MOCK=1`);
  all components (`App`, `Monitor`, `Inspector`, `LogsPanel`, `useBatch`,
  `useDownloadable`, `useFfmpeg`) now call `backend.*` only — no scattered
  `isTauri()`/`loadDemo()` in components. Covers settings/projects, file
  pickers (native dialog in Tauri, path prompt in web), audio, hardware status,
  render, model downloads, logs, drag&drop. `demo.ts` removed.

- **fix: menu bar shifted right when opening a menu (2026-08-20)** —
  `space-x-4` added a margin to every menu button once the click-away overlay
  became the nav's first child; switched to `gap-4`.

- **feat: live hardware usage in status bar (2026-08-20)** — new
  `hardware_status` Tauri command samples system CPU/RAM (sysinfo) and the
  primary GPU via `/sys/class/drm` (`gpu_busy_percent`, `mem_info_vram_*`,
  `vendor`; adapter with most VRAM wins). Status bar polls once per second and
  shows GPU utilization/VRAM, CPU %, and RAM.

- **feat: senmei-server http feature — headless web UI + REST (2026-08-20)** —
  `--http` (or `SENMEI_HTTP`) starts an axum server on `127.0.0.1:8765` (env
  `SENMEI_HTTP_PORT`) that serves the built web UI (`packages/app/dist`, env
  `SENMEI_WEB_DIR`) plus a REST API over the shared `core`: `/api/health`,
  `/api/models`, `/api/settings-schema`, `/api/ffmpeg`, `/api/backend-info`,
  `/api/probe`, `/api/frame` (base64 PNG), `/api/compare`, `/api/download-model`,
  `/api/render` (+ `/status`, `/cancel`). Same license/confirm gates as MCP;
  headless (no Wayland/X needed). Verified over curl: probe, frame, async render
  (30/30 frames → done), static UI.

- **feat: backend switch + libtorch status (2026-08-20)** — new `EngineBackend`
  setting (`auto` | `vulkan` | `libTorch`) wired through `engine_for_model`;
  `backend_info` command reports compiled backends, libtorch version, and
  CUDA/ROCm availability. Settings UI gets an inference-backend picker with the
  libtorch version/device line.

- **ml: integrate optional burn-tch (libtorch) engine behind `tch` feature (2026-08-20)** —
  `TchEngine` (upcunet2x / upcunet2x-fast / fallin-cugan / realesrgan / rife) on
  `LibTorch<f32>` with portable `TchDevice` (Auto/Cpu/Cuda/Mps). Shared archs
  extracted to `crates/senmei-ml/src/arch/` (engine-agnostic, used by burn + tch).
  ROCm/C++20/removed-ops patches moved to org forks `senmei-app/torch-sys`
  (v0.22.0-senmei) and `senmei-app/burn-tch` (v0.21.0-senmei), referenced via
  `[patch.crates-io]` — no in-repo vendoring. CPU roundtrip test green; ROCm
  build recipe unchanged (LIBTORCH + ROCm 7.14 runtime).

- **ml: register 4× RealPLKSR weights-only batch (2026-08-20)** — Nomos2,
  Nature, HFA2k_ludvae, mssim, BHI-real added (dim 64/28 blocks, contiguous,
  verified convert + sha256). BHI-otf listed but `loadable:false` (channels-last
  .pth); NomosWebPhoto skipped (non-dysample tail).

- **backend: autotune stays ON (2026-08-20)** — decision: keep the default
  autotune enabled; the full-frame OOM (upstream-issues §2) is avoided by the
  640px tiled infer path, so no opt-out or vendor patch needed.

- **ml: disable 48ch SPAN models hit by cubek conv bug (2026-08-20)** —
  `span-2x-nomosuni-ldl`, `span-2x-hfa2k`, `span-2x-modern-spanimation-v2`
  (multijpg already off) set `loadable: false`; they render degraded in f16
  (corr 0.82–0.94) due to `cubek-convolution` f16 1×1 conv bug (K=96×N≥32768,
  cubek#519). 64ch V1/V1.5 + BHI (corr ≥ 0.99) stay loadable.

- **docs: SPAN f16 root cause isolated — cubek-convolution kernel bug (2026-08-20)** —
  op-by-op bit-exact diff (burn f16 vs torch ROCm f16) proved weights + norm are
  bit-identical and silu/sigmoid/accumulation are innocent; the first SPAB's
  `conv2` (96→48 1×1) returns wrong values. Minimal repro: `Conv2d([96,48],[1,1])`
  f16 breaks when K=96 **and** N=H·W≥32768 (K∈{48,64,80,97,112,128} all correct).
  Documented in `docs/upstream-issues.md` §6 with paste-ready text.

- **ml: swap HFA2k_LUDVAE_SPAN → 2xHFA2kSPAN (2026-08-20)** — HFA2k_LUDVAE's
  f16 render is degraded by the conv kernel bug (corr 0.57 vs torch, artifacts);
  registry entry removed. Replaced with the official Phhofm `2xHFA2kSPAN`
  (48ch, params-wrapped, corr 0.82 in f16).

- **ml: RealPLKSR 1× family unlocked — CC-BY-4.0 confirmed (2026-08-20)** —
  Phhofm release pages + assets verified: `1xDeJPG_realplksr_otf`,
  `1xDeH264_realplksr`, `1xDeNoise_realplksr_otf` are CC-BY-4.0. Registry
  `real-plksr-deh264`/`real-plksr-dejpg` switched from `verify (Phhofm)`
  (license-blocked) to `CC-BY-4.0` with canonical Phhofm download URLs (DeH264
  sha already pinned; DeJPG sha pinned); new `real-plksr-denois` entry added.
  All three download+convert+load now.

- **fix: SPAN SPAB head concat must use the SiLU'd out1 (2026-08-20)** — the
  head concat `[feat, b6, b1, b5_2]` feeds the post-SiLU `out1` in
  span_arch/ONNX (`SiLU(inplace=True)`); the port returned the raw conv output,
  so every SPAN model rendered washed-out/inverted (burn out mean ~2.3 vs
  torch/ONNX ~0.56). Now matches (HFA2k mean 143 vs ONNX 142). Also adds
  no_norm support (`Span::set_no_norm` + `ModelRef.no_norm`) for norm-off
  checkpoints (ModernSpanimation V2, 2xBHI_small, DeH264_SPAN).

- **feat: senmei-server visual sample frames (2026-08-20)** — `render_sample`
  now returns the before/after PNGs as MCP image content blocks (base64) next
  to the text summary, so multimodal clients can visually compare the sample
  against the source.

- **test: senmei-server agent-loop e2e (2026-08-20)** — ignored integration
  test `tests/agent_loop.rs` drives the full MCP loop over stdio (probe →
  render_sample → compare_sample → propose_render → confirm_render → poll
  status → assert output). Needs Vulkan + converted fallin-soft .bpk.

- **feat: senmei-server validate ranges + tool allowlist (2026-08-20)** —
  `validate()` now enforces the full schema ranges (scale 1..=4, fps 1..=16,
  tonemap enum, dedup 0..=1, resize/output_resize > 0, end > start). The tool
  set is whitelist-only: fixed tools, render tools behind the `render` feature.

- **feat: senmei-server sample-compare (2026-08-20)** — `render_sample`
  (synchronous range render, no confirm gate; returns output + before/after
  PNG frames at the range midpoint) and `compare_sample` (PSNR dB + SSIM on
  the original resolution via FFmpeg psnr/ssim filters; VMAF deferred to a
  libvmaf build). E2E-verified over stdio: testsrc2 320×240 → 2× fallin-soft
  sample + PSNR/SSIM against the source.

- **feat: SPAN family with per-model feature_channels (2026-08-20)** —
  `ModelRef`/registry carry `feature_channels` (default 48); the engine builds
  `Span::new(ch, …)` from it (64 for TNTwise ModernSpanimation V1) and the
  convert CLI reuses the 5th arg as the channel count for `span`. Registered:
  `span-2x-nomosuni-multijpg`, `span-2x-hfa2k-ludvae` (Phhofm 48ch,
  CC-BY-4.0) and `span-2x-modern-spanimation-v1` (TNTwise 64ch, MIT) next to
  `span-2x-nomosuni-ldl`; all converted to f16 `.bpk`. E2E render over MCP
  stdio verified with the 64ch model (→ 640×480).

- **feat: senmei-server settings schema tool (2026-08-20)** — new
  `get_settings_schema` MCP tool (works without the `render` feature): the
  render-config JSON Schema (schemars, fields now documented + ranged), the
  model slots (`model_id`/`interp_model`/denoise/deblur → registry models) and
  the hard constraints (license + confirm gate). Also gates `RenderStatus`'s
  `Default` impl behind the feature, fixing the default build.

- **docs: MCP status — in progress + next steps (2026-08-20)** — PLAN §16
  "planned, not started" → "in progress": headless entry + confirm gate done
  (e2e render verified), sample-compare / settings-schema / tool-ranges open;
  §16.6 orders the remaining work; todos.md points to it.

- **feat: register SPAN — 2xNomosUni_span_multijpg_ldl (2026-08-20)** — the
  f16/bf16 block was a synthetic-input artifact: real frames stay ~2e4–3e4
  (f16-safe; bf16 still NaN on RADV). Phhofm weights are flat + norm-on
  (output [0,1]); the `span` convert branch now strips an optional `params`
  wrapper. Registered as `span-2x-nomosuni-ldl` (CC-BY-4.0, 48ch 2×,
  sha256-pinned), converted to f16 `.bpk` and smoke-tested on Vulkan (real
  512² frame: min −0.12, max 1.34, no NaN/inf).

- **feat: senmei-server async render + status polling (2026-08-20)** —
  `confirm_render` now starts the render on a worker thread and returns
  immediately (the stdio loop stays responsive, so `cancel_render` works
  mid-render); new `get_render_status` tool polls
  `{state: idle|running|done|failed, framesProcessed, totalFrames, error}`.
  E2E-verified over stdio: 320×240 testsrc2 → 2× upscale (fallin-soft,
  burn/Vulkan) → 640×480 h264+aac in ~27 s.

- **feat: senmei-server render — confirm gate (2026-08-20)** — new `render`
  cargo feature pulls `senmei-pipeline` + `senmei-ml/burn`. `core` gains
  `RenderConfig`/`validate` (path/ranges + permissive-license model
  allowlist), `engine_for_model` (license gate, hard), and `render` mirroring
  the GUI's pipeline assembly. MCP tools: `propose_render` (validates + parks,
  does NOT start), `confirm_render` (runs the pending render), `cancel_render`.
  Default build stays light; the tools report "render not compiled in" without
  the feature. Verified over stdio: handshake, tools/list, validation errors,
  and a real pipeline run (burn/Vulkan up, ffmpeg stage).

- **docs: update milestones + record senmei-server decision (2026-08-20)** —
  PLAN.md: M5 audio is rodio-based (every codec), M7 model downloader/IFRNet
  done, M8 bundles + release CI done (auto-updater + static FFmpeg bundle
  pending); §16 records the `senmei-server` decision. todos.md: close the burn
  macOS scaffold todo (Metal backend landed, 0.1.1), add the RealPLKSR
  adoption batch, align the SPAN entry with the f16-safe finding.

- **feat: senmei-server scaffold — MCP stdio (2026-08-20)** — headless
  `senmei-server` crate (PLAN §16): transport-agnostic `core` (data/models
  dirs, registry, ffmpeg, probe, list_models) + MCP stdio adapter on the
  official `rmcp` SDK (v3, Apache-2.0). Read-only tools: `health`,
  `probe_video`, `list_models`, `get_ffmpeg_status`. HTTP stays an optional
  feature (YAGNI); license gate lives in core so every transport gets it.

- **ml: SPAN burn port (2026-08-19)** — clean burn
  port of SPAN (Apache-2.0 BasicSR) with Conv3XC/SPAB, `(x−mean)·255`
  normalization and `no_norm` handling; torch-verified (matches f32 up to f16
  limits). Not registered: intermediates reach ~1e5 and overflow f16; bf16 is
  all-NaN on RADV. Blocked on synthetic inputs only — un-gated 2026-08-20 (entry above).

## 0.1.2 (2026-08-19)

- **docs: notes stay short — kurz, bündig, knackig, no novels (2026-08-19)** —
  AGENTS.md now caps todos at ~135 chars and mandates concise notes; long
  entries in `docs/todos.md` trimmed, detail stays in PLAN/CHANGELOG.
- **docs: defer the CEF webview switch, keep WebKitGTK + rodio (2026-08-19)** —
  re-evaluate Tauri's CEF backend (`feat/cef`) in a few months; the
  audio-streaming/kira milestone is deferred with it (a CEF switch would
  obsolete it), so the rodio full-track MP3 preview stays (PLAN §12).
- **fix: plain-click on a media-library video now previews it (2026-08-19)** — the
  monitor used `files[0]` as its current file, so selecting another video only
  highlighted it without changing the preview. A dedicated `currentFile` state is
  set on plain click (multi-select/toggle unchanged); a sync effect keeps it
  valid (falls back to the first file when removed/cleared), and the queue order
  (`files`) is untouched.

- **fix: stale preview audio after switching videos (2026-08-19)** — on a file
  change the rodio sink is now cleared (`audio_clear`) instead of only paused, so
  the previous track can't play while the new one is extracted; a ref guard drops
  a stale `extractAudio` promise so it can't load the old track after a switch.

- **docs: record Phhofm/models as a model source + backlog update (2026-08-19)** —
  `models.md`: Phhofm GitHub releases (CC-BY-4.0) as canonical source; SPAN gets
  concrete weights (`2xNomosUni_span_multijpg_ldl`, `2xBHI_small_span_pretrain`);
  4x BHI RealPLKSR-dysample marked quick-adopt (arch exists); 4x BHI DAT2 marked
  high-cost transformer milestone; BHI note corrected (BHI = Phhofm series, the
  RVE-hosted `SpanPlusDynamic_Light` copy stays unverified → blocked).
- **docs: note RVE-hosted SPAN weights are license-blocked (2026-08-19)** —
  `models.md` Notes: `TNTwise/real-video-enhancer-models` SPAN variants (e.g.
  `2x_BHI_SpanPlusDynamic_Light.pth`) have no license metadata and no verifiable
  source → blocked; the SPAN arch (Apache-2.0, `hongyuanyu/SPAN`) stays
  adoptable via a clean port.

- **ui: slim fullscreen control bar + 320k preview audio (2026-08-19)** — in
  fullscreen the media fills the screen and a minimal overlay (play/pause,
  position, scrubber, volume) sits at the bottom instead of the full timeline.
  Preview audio is now transcoded at 320 kbps (was 128k); MP3 stays the target
  because AAC/M4A crashes rodio 0.20.1 (symphonia isomp4 init SeekError).

- **feat: preview audio via native playback + stable frames (2026-08-19)** —
  WebKitGTK can't play media over Tauri's `asset://` scheme at all (its
  GStreamer backend doesn't know the scheme), so the webview `<audio>` was
  always silent regardless of codec. Audio is now decoded to MP3 and played
  natively through rodio, driven by IPC (`audio_load/play/pause/seek/
  set_volume`), with a volume slider in the timeline. Frames are written to a
  stable per-source file with atomic overwrite and re-fetched via a cache-
  busting query (no more mid-fetch prune races), and the preview dir is capped
  at 400 files. Adds `libasound2-dev`/`alsa-lib-devel` as a Linux build dep.

- **ui: fullscreen fills the screen (2026-08-19)** — the single-view media used
  `max-h-full max-w-full` and stayed small in fullscreen; it now uses
  `h-full w-full` like Compare. Fullscreen also drops the monitor padding/
  border and hides the timeline bar, so the image/video fills the whole screen.

- **ui: F11 fullscreen hotkey + Esc closes settings (2026-08-19)** — new
  configurable `toggleFullscreen` hotkey (default `F11`, in Settings →
  Hotkeys) drives the Monitor's native fullscreen; the Settings page now
  closes with `Esc` (while capturing a hotkey, `Esc` only cancels the capture).

- **fix: no console window on Windows release (2026-08-19)** — the `senmei` bin
  was built as a console-subsystem exe, so a terminal popped up on launch.
  Added `#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`;
  debug builds keep the console for logs.

## 0.1.1 (2026-08-19)

- **v0.1.1 — first public release (2026-08-19)** — 3-OS CI green (Linux/macOS
  full tests; Windows skips the `senmei-app` unit-test harness, see ci.yml),
  bundles for deb/rpm/AppImage/dmg/msi/nsis, model catalog bundled without
  tauri `_up_` mangling, GitHub Release publishing on version tags. Docs:
  `docs/RELEASING.md`.

- **feat: Metal backend on macOS (no Vulkan SDK) (2026-08-19)** — the burn
  macOS scaffold landed as a real backend: `senmei-ml` picks `burn_wgpu::Metal`
  on macOS (`Vulkan` everywhere else); macOS runs in CI (no GPU on hosted
  runners — experimental, see RELEASING.md).

- **fix: bundle model catalog cleanly in packaged builds (2026-08-19)** —
  `bundle.resources` referenced `../../models/metadata.json`, which tauri
  mangles to `_up_/_up_/` dirs; the model list was empty in deb/rpm. The
  `senmei` build script now copies the catalog to `resources/metadata.json`
  (gitignored, `..`-free) and `ensure_catalog` finds it by recursive search.

- **docs: add release process doc (2026-08-19)** — `docs/RELEASING.md` documents
  how to cut a release (version bump → pre-flight → tag → CI build + publish →
  post-release) and is linked from the root README's docs index.

- **ci: fix bun cache, bump actions to v5, publish bundles on version tags (2026-08-19)** —
  `setup-bun@v2` now caches `node_modules` keyed on `./bun.lock`; checkout and
  upload-artifact bumped to v5; a new `release` job publishes the built bundles
  to GitHub Releases when a version tag is pushed.

- **docs: clarify macOS FFmpeg support (2026-08-19)** — macOS stays on system
  FFmpeg (Homebrew); `download_ffmpeg` now reports there is no LGPL-compatible
  portable macOS build and points to `brew install ffmpeg` (the only macOS
  prebuilts, evermeet.cx/osxexperts, are GPL and conflict with the LGPL-only
  policy).

- **cleanup: split mock.ts out of the production bundle + drop tiling dead-code (2026-08-19)** —
  the demo backend loads lazily via `loadDemo()` (dynamic import → its own chunk,
  never fetched in Tauri); `crop_rgb24` is `cfg(feature = "burn")`-gated, dropping
  the `#[allow(dead_code)]`.

- **fix: scope IPC file ops to the app data dir + block tar-slip on import (2026-08-19)** —
  `prune_samples` and `export_project` now reject paths outside the app data dir
  (the `delete_project` allowlist pattern, `store::ensure_within_data_dir`);
  `open_project` refuses archives whose entries escape the project dir
  (`unpack_in` silently skips `..` entries — treated as a refusal). New tests:
  allowlist guard + tar-slip refusal; a shared test-env lock fixes the
  `XDG_DATA_HOME` race between the store/models test modules.

- **cleanup: remove the `SENMEI_FORCE_FFMPEG_MISSING` debug hook (2026-08-19)** —
  `get_ffmpeg_status` no longer simulates a missing FFmpeg via an env var.

- **fix: replace production panics with `Err` in the burn engine + decoder (2026-08-19)** —
  `Model::forward`/`interp` now return `Result` (single-input forward on an
  interpolation model, or interpolation on an SR model, are `Err` instead of
  `panic!`); `infer_interp`'s pad selection drops its `unreachable!()`, and the
  FFmpeg decoder errors on an unsupported rotation instead of `unreachable!()`.

- **fix: harden the ONNX reader against malformed input (2026-08-19)** —
  `onnx.rs` no longer panics on out-of-bounds varints, oversized
  length-delimited fields, or length overflow; the low-level helpers are now
  bounds-checked and return `Result`/`Option`, so corrupt bytes surface as an
  `Err` instead of a crash. New tests: truncated length-delimited field +
  truncated varint.

- **feat: make the model catalog work in packaged apps (2026-08-19)** —
  `models_dir()` now prefers the writable data dir in a packaged app; the catalog
  (`models/metadata.json`) is bundled via `bundle.resources` and materialized to
  `$DATA/models/` at setup (`models::ensure_catalog`). Dev keeps using the repo
  `models/` checkout. The on-demand download + f16 `.bpk` conversion flow is
  unchanged.

- **docs: consolidate burn-bugs.md into upstream-issues.md (2026-08-19)** — the
  findings doc is merged into `docs/upstream-issues.md` (one section per issue:
  Finding + paste-ready text); `burn-bugs.md` deleted and all references updated;
  the open autotune-default decision moved to `docs/todos.md`, which now holds
  open items only (done items live in CHANGELOG).

- **docs: trim further duplication (2026-08-19)** — burn-bugs.md Bug 1's embedded
  suggested-issue text now points to the paste-ready copy in `upstream-issues.md`
  §1; PLAN.md §15 status snapshot no longer re-lists adopted models or the LGPL
  encoder chain (cross-references `models.md` / §14.3).

- **docs: de-dup model tables (2026-08-19)** — PLAN.md §14.2 no longer keeps its
  own model matrix (it was a stale subset of `docs/models.md`); models.md's
  backlog no longer repeats adopted items (DRUNet, BSRGAN, NAFNet-GoPro width32)
  — only the not-yet-adopted parts (BSRNet, width64) remain as candidates.

- **docs: fold docs index into root README, link upstream-issues.md (2026-08-19)** —
  no separate `docs/README.md`; the root README's Docs section now carries the
  one-truth-per-fact index + "where does X go" quick reference and lists
  `upstream-issues.md`.

- **fix: make realesrgan BSRGAN test backend-agnostic (2026-08-19)** — the
  `bsrgan_matches_torch_reference` test imported `burn_wgpu::Vulkan` directly,
  which broke the macOS (Metal) `cargo test` build; switch it to the crate's
  `BurnBackend` alias like the other burn tests.

- **build: move the gpu-allocator windows-0.62 pin into a fork repo (2026-08-19)** —
  the Windows DX12 fix (wgpu-hal 29 needs windows 0.62; upstream 0.28.0's range
  `>=0.53, <=0.62` unified to 0.61 with tauri) no longer vendors the whole crate
  into `third_party/`. The one-line pin now lives in
  `senmei-app/gpu-allocator` (tag `v0.28.0-windows-0.62`), referenced via
  `[patch.crates-io]` as a git dependency; `third_party/gpu-allocator/` removed.

- **feat: BSRGAN loadable (RRDBNet 23, restoration) (2026-08-19)** — BSRGAN
  (KAIR v1.0, MIT) reuses the existing `RrdbNet` arch (RRDBNet, 23 blocks,
  scale 4); the converter now also maps its older BasicSR key naming
  (`RRDB_trunk.{i}.RDB{j}.conv{k}`, `trunk_conv`, `upconv1/2`, `HRconv`), which
  leaves standard Real-ESRGAN pths (`body.*`, `conv_body`, `conv_up*`, `conv_hr`)
  untouched. Torch-verified (applied=702, mae 0.001). Registry `bsrgan`
  (kind upscale/restoration, num_block 23), `tools/bsrgan_verify.py`. Works as a
  restoration upscaler through the existing Upscale step.

- **feat: wire NAFNet into the Deblur step (ML deblur) (2026-08-19)** — the
  Deblur step now runs an ML model (NAFNet) when a deblur model is selected,
  otherwise falls back to the unsharp mask (engine errors also fall back).
  NAFNet is single-input (scale 1, 3ch→3ch) and pads internally to multiples of
  16 — the step reuses the generic `infer_tiled` path, no new trait method.
  `FilterParams.deblur_model_id` → `render()` builds the engine; frontend: the
  Deblur step now has a model dropdown (kind `deblur`), `steps.ts` default,
  `useBatch` sends `deblurModelId`; bindings regenerated. GPU test
  `infer_nafnet_deblurs_via_generic_infer`.

- **feat: NAFNet-GoPro burn arch port (deblur, 2026-08-19)** —
  `burn/nafnet.rs`: NAFNet-GoPro-width32 (megvii, MIT) as a clean
  re-implementation — NAFBlock (LayerNorm2d, SimpleGate, SCA `x·conv(avgpool(x))`
  without sigmoid, depthwise conv, FFN with beta/gamma scales), encoder/down
  pyramid, middle, decoder with Conv1×1+PixelShuffle(2) ups, `ending+inp`.
  LayerNorm2d computes the channel reduction fp16-safely in `x/S` (S=128).
  Torch-verified (mae 0.0007 on a realistic input; encoder block-exact) —
  **first ML deblur, loadable**. `senmei-ml-convert` `nafnet` arch
  (capture-group key remap), Registry `nafnet-gopro-width32` (MIT,
  nyanko7 mirror), `tools/nafnet_verify.py`. New burn-bugs.md finding (Bug 7):
  the internal activations only overflow fp16 on pathological noise input
  (~70000); torch-fp16's LayerNorm silently overflows to 0 there (not a
  faithful fp16 reference).

- **feat: wire DRUNet into the Denoise step (ML denoise) (2026-08-19)** — the
  Denoise step now runs an ML denoiser (DRUNet) when a denoise model is
  selected, otherwise falls back to box blur. New: `InferenceEngine::infer_denoise`
  (+ `infer_denoise_tiled` over the shared `run_tiled`); `BurnEngine` appends
  DRUNet's sigma map (4th channel), pads to multiples of 8, and crops back.
  `FilterParams.denoise_model_id` → `render()` builds the engine; frontend: the
  Denoise step now has a model dropdown (kind `denoise`), `steps.ts` default,
  `useBatch` sends `denoiseModelId`; bindings regenerated. Sigma = radius/20
  (the existing radius setting drives the noise level). GPU test
  `infer_denoise_drunet_pads_and_crops` (66×64).

- **fix: ONNX reader reads `Constant`-node weights (2026-08-19)** — the
  dependency-free protobuf reader (`senmei_ml::onnx`, no ONNX Runtime) only read
  `graph.initializer`; models whose weights live only in `Constant` nodes (value
  attribute) silently yielded an empty bpk. Now also reads `Constant` nodes
  (keyed by the node's output name, since the inner `TensorProto.name` is
  `"value"`/empty), reports an empty result as an error, and rejects external
  data (`data_location == EXTERNAL`) — the three points from the `onnx-ir`
  issue comment (tracel-ai/burn-onnx#456). 4 new unit tests.

- **feat: tile size configurable in Settings (2026-08-19)** — the fused RGB8
  upscale tile size (default 640, previously `SENMEI_TILE` env only) is now a
  Settings value (`tileSize`), applied per render via `senmei_ml::set_tile_size`
  and editable in the Settings UI (Appearance, 128–2048 px). `SENMEI_TILE`
  still works as a bench-only fallback. Also removed the top-right settings
  gear (Settings remain reachable via the status bar and menu).

- **fix: IFRNet ResBlock c5 conv + Bug-6 diagnosis withdrawn (2026-08-19)** —
  the ResBlock `forward` omitted the `conv5` conv (`pl(out4 + x)` instead of
  `pl(c5(out4) + x)`; reference: `x + self.conv5(out)`). That was the real
  cause of the alleged burn-fusion Bug-6 deviation — not a backend bug. With
  the fix IFRNet loads cleanly (applied=104, missing=0) and the torch reference
  test passes on fused `Vulkan<f16>` (mae 0.005).
  `ifrnet-vimeo90k`/`ifrnet-gopro` are now `loadable: true`;
  docs/burn-bugs.md Bug 6 removed. (Review of PR #1, copilot-swe-agent.)

- **feat: DRUNet burn arch port (2026-08-19)** — `burn/drunet.rs`: `UNetRes`
  (DPIR, MIT) as a clean re-implementation — 3× stride-2 downsample, 4 ResBlocks
  per level (Conv→ReLU→Conv + skip), 3× ConvTranspose2d upsamples, all convs
  `bias=false`, `in_nc=4` (RGB + constant noise-level map). Torch-verified
  (mae 0.001, all 64 weights loaded) — first ML denoise **loadable**, no fusion
  bug (no channel slices). `senmei-ml-convert` `drunet` arch (capture-group key
  remap), Registry `drunet-color` (MIT, KAIR v1.0), `tools/drunet_verify.py`.
  Wired into the Denoise step (4ch sigma map).

- **fix: surface the real encode error instead of "encode channel closed" (2026-08-19)** —
  the encoder discarded ffmpeg's stderr (`Stdio::null()`), so a failed encode
  only surfaced as "encode channel closed" (the main loop's channel error masked
  the encode thread's real cause). The encoder now captures stderr and includes
  it in the write/finish error, and the pipeline reports the encode thread's
  error (cancellation and step errors still win). Render failures now show the
  actual ffmpeg reason in the Logs panel.

- **docs: IFRNet torch-verified (2026-08-19)** —
  `tools/ifrnet_verify.py` + vendored torch reference (`ref/ifrnet/`, MIT)
  generate reference bins; encoder + weights are exact (mae ~0.0001). The
  ResBlock (side-channel split/cat) diverged between method and inline (mae
  0.0525 vs ~0.0001) — misdiagnosed as burn-fusion Bug 6; the real cause was a
  `conv5` conv missing in `forward` (see fix entry above; Bug 6 withdrawn,
  IFRNet `loadable: true`).

- **feat: HDR→SDR tonemapping (2026-08-18)** — `probe` reads `color_transfer`/
  `color_primaries` and `VideoInfo::is_hdr()` detects PQ/HLG/DCI. The decoder
  applies a zscale+tonemap filter chain for HDR (or `always`) and converts
  correctly to SDR before outputting `rgb24` — previously HDR was clipped
  uncontrolled on decode. New output-step setting `tonemap` (auto/always/off),
  threaded through `RenderConfig` → `Pipeline::set_tonemap` → `Decoder`.
  Tests: `hdr_detection` (unit) + `hdr_source_is_detected_and_tonemapped`
  (integration, libx265-gated).

- **feat: IFRNet burn arch port (2026-08-18)** — `burn/ifrnet.rs`: base variant
  (ltkong218, MIT) as a clean re-implementation — 2× shared 4-level encoder, four
  coarse-to-fine decoders (bilinear, no GRU), side-channel ResBlock, own PReLU
  implementation (missing in burn 0.21), shared `warp`/`grid_sample`.
  Engine dispatch (`Model::IfrNet`, interp path pad 16), `senmei-ml-convert`
  `ifrnet` arch (capture-group key remap), registry entries
  `ifrnet-vimeo90k`/`ifrnet-gopro` (MIT, HF URLs). `loadable: true` after the
  ResBlock c5 fix (see fix entry above).

- **docs: IFRNet weights verified (2026-08-18)** — official checkpoints
  (Vimeo90K + GoPro, 19.9 MB each, MIT) via `pavlichenko/ifrnet_*` on Hugging
  Face with direct resolve URLs; "Weights verify / Repo-URL verify" resolved
  from the backlog.

- **ci: GitHub Actions matrix build (2026-08-18)** — `.github/workflows/ci.yml`:
  Windows/Linux/macOS — system deps, frontend build, `cargo check` +
  `cargo test --workspace` (GPU tests are `#[ignore]`), app bundle via
  `tauri build`, artifact upload.

- **docs: NAFNet fp16 porting notes (2026-08-18)** — litert-community
  conversion (NAFNet-GoPro-width32) confirms MIT + provides port details for
  the burn re-implementation: SimpleGate (no activation), channel attention =
  mean×2, upsample = Conv1×1 + PixelShuffle, and the fp16 LayerNorm overflow
  trap (compute the channel reduction in a scaled domain). In `models.md` Notes.

- **docs: NAFNet-GoPro promoted to deblur candidate (2026-08-18)** — official
  NAFNet weights are available via `nyanko7/nafnet-models` (Hugging Face) with
  direct, sha256-pinnable URLs (no GDrive needed); GoPro-width32 (68.7 MB) as
  the light option. This makes NAFNet-GoPro the first ML-deblur candidate (the
  deblur stack is CPU-only so far); the NAFBlock arch port remains open.

- **docs: KAIR v1.0 + NAFNet models surveyed (2026-08-18)** — more permissive
  weights in the backlog: DRUNet/DnCNN/FFDNet/IRCNN/BSRGAN/IMDN (all MIT via
  KAIR v1.0, direct download URLs), NAFNet SIDD/GoPro/REDS (MIT). First neural
  deblur candidate (NAFNet-GoPro) noted.

- **docs: licenses for denoise/restoration clarified (2026-08-18)** — SCUNet
  **Apache-2.0** (entered in `metadata.json` + `models.md` → no longer
  license-blocked; arch port stays open), DRUNet (DPIR) **MIT** via KAIR
  v1.0, NAFNet **MIT**, USRNet/USRGAN **MIT** (backlog added).

- **refactor: `Inspector.tsx` split (2026-08-18)** — the whole step editor
  (all types incl. the large output editor) extracted to `StepEditor.tsx`;
  `Inspector.tsx` reduced from ~800 to ~370 lines (stack list, drag&drop, add
  menu). All three large files are now split (App.tsx, commands.rs,
  Inspector.tsx).

- **refactor: `commands.rs` split (2026-08-18)** — model helpers moved to
  `models.rs` (`models_dir`/`load_registry`/`engine_for_model`), preview helpers
  to `preview.rs` (decode streams, `read_frame_inner`, PNG prune).
  `commands.rs` reduced from ~800 to ~636 lines; only Tauri commands remain.

- **security: asset scope narrowed + CSP set (2026-08-18)** — the static
  asset-protocol scope was `["$DATA/**", "$HOME/**"]` (whole home readable).
  All media loads go through `probe_video`/`read_frame` anyway, which release
  the file at runtime via `allow_file` (the same scope the asset protocol
  checks), so `["$DATA/**"]` suffices (app data dir for previews/samples/
  projects). Plus a CSP for production (dev untouched).

- **refactor: App.tsx split — batch logic into `useBatch` hook (2026-08-18)** —
  render state + `startBatch`/`cancel`/`togglePause` + `desiredPath` extracted
  from `App.tsx` into `useBatch.ts` (~150 lines fewer). Behavior unchanged
  (demo render + cancel verified).

- **ui: Logs tab next to the Processing Stack (2026-08-18)** — the right panel
  now has a tab toggle "Processing Stack" / "Logs" (`RightPanel`). New
  `LogHub` logger forwards `log` records to the UI as a Tauri event (ring buffer
  500, `get_logs` on open); the panel has a level filter (ALL/ERROR/WARN/INFO),
  clear and auto-scroll. The `env_logger` console behavior stays unchanged
  (error + `wgpu_hal=off`), the panel catches Info+.

- **refactor: platform-safe frontend paths (2026-08-18)** — all manual
  `split("/")` spots replaced with `paths.ts` helpers (`basename`/`dirname`/
  `joinPath`) that handle both `/` and `\` (Windows); joins use `/`, which
  Windows APIs also accept. Affects output path building, the sample folder and
  filename display.

- **refactor: unified FFmpeg-arg parsing (2026-08-18)** — the frontend now
  sends the encoder args as a pre-split array (`RenderConfig.ffmpegArgs:
  string[]`); the duplicate Rust parser `split_ffmpeg_args` was removed. Only
  one parser remains (`splitArgs` in `steps.ts`), shared for preview and render.

- **ui: hotkey settings on the Settings page (2026-08-18)** — new "Shortcuts"
  section (Koharu-style): show actions, reassign on click (next key press),
  reset to default. Overrides persist in the app settings (`Settings.hotkeys`),
  defaults stay in code; app hotkeys + monitor space use the configured combos.

- **ui: About dialog follows the dark theme (2026-08-18)** — the dialog
  rendered outside the `dark` wrapper, so its `dark:` styles never applied
  (always light). Moved into the wrapper.

- **ui: "View" menu with full-video mode (2026-08-18)** — new "View" menu with
  "Full Video Mode"; toggles the same fullscreen as double-clicking the monitor
  (signal to `Monitor.toggleFullscreenSignal`). Translated DE/EN.

- **fix: LGPL-safe codec mapping (2026-08-18)** — the encoder dropdown mapped
  H.264→`libx264`/H.265→`libx265` (both GPL, missing in the pinned
  BtbN-LGPL builds), so H.264/H.265 outputs failed with the LGPL FFmpeg. Now
  H.264→`libopenh264`, H.265→`libkvazaar` (both BSD) and the args are
  codec-aware: CRF only for svtav1/vpx, preset for kvazaar, openh264 is ABR and
  gets its `-b:v` from the backend. `Encoder::open` drops the base codec's
  default args on a `-c:v` override. Test
  `override_codec_sets_bitrate_for_openh264_only`.

- **perf: tile size 512→640 after GPU stitch (2026-08-18)** — the old cost model
  (15 u8 readbacks + CPU stitch) no longer held, so re-measured
  (`bench_upscale_step`, fallin-soft, 1080p→2160p): 512px 247.8 ms,
  **640px 186.1 ms / 5.4 FPS**, 768px 210.2 ms. 640 halves the tile count
  (15→8) before the per-tile matmul becomes pathological. Default 640, override
  via `SENMEI_TILE`; correctness test switched to a single 640 tile. Full-frame
  (176 ms) stays the floor until the upstream autotune-OOM fix.

- **fix: dedup no longer collapses static material (2026-08-18)** — dedup
  dropped unlimited consecutive duplicates; with static/near-static material
  only one frame remained ("Render Sample" with only dedup gave ~0.05 s). Now
  max 5 consecutive drops, then a frame is forced (static 3 s → ~0.5 s instead
  of 0.05 s). Test `dedup_never_collapses_static_run`.

- **perf: GPU stitching in the tiled-fused RGB8 path (2026-08-18)** — instead
  of reading each 512px tile back as u8 and stitching on the CPU, `infer_rgb8`
  now accumulates the tiles on the GPU in an f16 canvas (`slice_assign` overlap
  averaging) and reads back one packed frame — one readback instead of 15 plus
  CPU stitch. `bench_upscale_step` (1080p→2160p, fallin-soft): 329 →
  **234.7 ms / 4.3 FPS**. The now-dead CPU stitch `stitch_rgb24` was removed.
  Correctness + 48-frame reliability via
  `infer_rgb8_tiled_is_reliable_and_correct`.

- **fix: CPU steps process packed `rgb24` instead of planar (2026-08-18)** —
  `Denoise`/`Deblur`/`Resize` sliced `Frame.data` as planar RGB planes, but
  decoder/encoder work with packed `rgb24`. This mixed the channels in the
  denoiser: "Render Sample" drifted apart with an active upscaler, denoiser-only
  produced garbage. The steps now blur/sharpen/resample channel-separately on
  packed data; regression tests `denoise_keeps_channels_separate`,
  `deblur_keeps_channels_separate`, `resize_keeps_channels_separate` (closes the
  maintainability TODO).

- **fix: `prune_samples` deletes by mtime instead of filename (2026-08-18)** —
  sample renders were deleted sorted lexically by path; due to the range tags in
  the name the just-rendered sample could disappear. Now deletes the oldest
  (mtime), keeps the newest `keep`. Test
  `prune_samples_keeps_newest_by_mtime`.

- **ui: "Render Sample" renders only the current video (2026-08-18)** — the
  sample button called `startBatch(false, …)` and created samples for the
  **whole queue** instead of the video in the monitor. `startBatch` now accepts
  an explicit file list; `onRenderSample` passes `[currentFile]`.

- **media: video rotation is handled (2026-08-18)** — `probe` reads the
  rotation (DisplayMatrix side-data or case-insensitive `rotate` tag), reports
  display dims + `VideoInfo.rotation`; `Decoder` sets `-noautorotate` and applies
  the rotation explicitly (90→`transpose=2`, 180→`hflip,vflip`, 270→`transpose=1`),
  verified byte-identical to ffmpeg's autorotation (test
  `probe_and_decode_apply_rotation`). Previously 90°/270° videos were
  mislabeled/distorted (autorotated output ≠ probed dims).

- **docs: PLAN §14/§15 restructure + maintainability backlog (2026-08-18)** —
  `PLAN.md` §14 split into subsections (own code & libs, models, codecs, AGPL
  boundary) with an expanded dependency/license table, §15 rewritten as a status
  snapshot; `models.md` SPAN added to the backlog; `todos.md` gained a
  Maintainability section from a code review (8 open items; AGENTS generated-path
  check confirmed fine).

- **docs: tidy-up + re-sync all docs (2026-08-18)** — `todos.md` entries capped at
  ~135 chars; `benchmarks.md` reorganized decision-first with a key-numbers table;
  `burn-bugs.md` prose tightened (all facts kept); `models.md` deduped
  (status-at-a-glance removed) and loadable status updated to match
  `metadata.json`; `PLAN.md` brought back in sync with the code (engine trait,
  PNG/native-video preview, adopted models/licenses, LGPL-safe encoder, vertical
  layout diagram).

- **ml: RealPLKSR port — 4x-alchemy + decompress models loadable (2026-08-18)** —
  clean burn re-implementation of RealPLKSR (Partial Large Kernel CNNs for
  Efficient Super-Resolution, arXiv 2404.11848; spandrel MIT reference):
  head → 28 PLK blocks (DCCM + partial 17×17 conv + EA + GroupNorm) → tail,
  with the DySample upsampler tail for the 4x model and a pixel-shuffle
  identity for the 1x decompress models. Numerically verified against torch
  on deterministic inputs (deh264/dejpg 1x mae ~0.002 / ~0.0002, alchemy 4x
  mae ~0.002). Two burn-wgpu findings worked around along the way: f16
  `div_scalar(65536)` underflows to 0 (GroupNorm rebuilt on `mean_dim`,
  docs/burn-bugs.md Bug 4) and `repeat`/`reshape` interleaves copies wrongly
  (`repeat_interleave` built explicitly). `4x_Alchemy.pth` stores weights
  channels-last — burn-store ignores strides (docs/burn-bugs.md Bug 5), so
  that conversion needs a `.contiguous()`-fixed pth. `warp.rs` grid sampling
  generalized (align_corners selectable, arbitrary output size).

- **ui: keyboard shortcuts (2026-08-18)** — Ctrl/Cmd+O imports a file, +A
  selects all, +E exports the project, +R renders, Delete removes the
  selection, Space toggles monitor play/pause. Shortcut hints are shown in the
  menu bar; hotkeys are active only in the workspace (not the start screen).
  Also fixes a latent `menu.children` reference in the MenuBar import submenu.

- **ui: meaningful dedup controls (2026-08-18)** — the deduplication step now
  has mode presets (Off / Standard / Aggressive), a threshold slider with a
  live percent readout, and a one-line explanation instead of a bare slider.

- **ui: full-video monitor mode via native WebKit fullscreen (2026-08-18)** —
  double-click on the monitor view opens it fullscreen via the HTML Fullscreen
  API (`requestFullscreen` on the monitor element, supported by WebKitGTK) —
  the same video/frame instance stays mounted, so playback continues and no
  second decoder runs underneath. Works in original / compare / result modes.
  Exit via a second double-click, the ✕ button, or native Esc.

- **perf: tiled-fused RGB8 overlap — tile/8 rejected (2026-08-18)** — tested
  `overlap = tile/4 → tile/8` on the fused RGB8 path (1080p, fallin-soft):
  regression to 394 ms / 2.5 FPS vs 329 ms / 3.0 FPS. With 512px tiles the
  tile count is unchanged (5×3) and the smaller overlap only enlarges the
  padded region, so the CPU stitch/crop does more work. Kept `tile/4`
  (reliability confirmed by `infer_rgb8_tiled_is_reliable_and_correct`). The
  real remaining cost is CPU stitching + per-tile u8 readback — GPU stitching
  tracked as follow-up in docs/todos.md. Bench test-input generation switched
  from GPL `libx264` to the universally available native `mpeg4` (LGPL-safe).

- **fix: LGPL-only FFmpeg + LGPL-safe HEVC encoder (2026-08-18)** — the
  portable download now pins BtbN `-lgpl` builds on a dated tag
  (autobuild-2026-08-17-13-05, N-126188) with per-platform SHA-256
  (linux/win64); the old single `latest`-tag GPL pin was license-noncompliant
  and shared one SHA across platforms. The encoder no longer hardcodes
  `libx264` (GPL-only): `pick_video_encoder` prefers libkvazaar (HEVC, BSD,
  ships in the LGPL builds) → libopenh264 → h264_nvenc → libx264 → native
  h264. kvazaar/x264 use quality-based rate control; libopenh264 gets a
  resolution-based `-b:v` (~14 Mbps @1080p; `extra_args` override). Resolves
  the AGENTS.md GPL-vs-LGPL contradiction. Guarded by
  `encodes_through_selected_codec` (runs against a real ffmpeg via
  SENMEI_FFMPEG).

- **fix: license gate for model download/use (2026-08-18)** — `download_model`
  and the app `engine_for_model` only checked `loadable`, so a model flagged
  `verify`/`unclear` (license review pending) or under a copyleft /
  non-commercial license could be unlocked by flipping `loadable`. Added
  `ModelMetadata::license_blocked()` (blocks `verify`, `unclear`,
  GPL/LGPL/AGPL, CC-BY-NC/ND/SA; missing → blocked) and enforced it in both
  commands, independent of `loadable` — the review gate never auto-unlocks an
  unclear license. Guarded by `license_gate_blocks_unclear_and_copyleft`.

- **fix: tiled-fused RGB8 render path (reliable GPU conversion) (2026-08-18)** —
  the full-frame fused `infer_rgb8` OOM'd burn/cubecl autotune on the large
  full-frame matmul (m=1024, n=4M, f16) and then cascaded into "Ordering is
  bigger than operations" panics (docs/burn-bugs.md Bug 1+3). `infer_rgb8` now
  tiles internally (512px, overlap): per tile the GPU runs forward + NHWC
  permute + clamp + scale + u8 cast, so only packed u8 bytes cross back, and
  tiles are stitched with overlap averaging (`stitch_rgb24`/`crop_rgb24`).
  Structurally immune to the OOM. `Upscale` prefers `infer_rgb8`, falls back
  to `infer_tiled`. Guarded by `infer_rgb8_tiled_is_reliable_and_correct`
  (correctness within fp16 tolerance + 48-frame reliability). Benched
  (1080p→2160p, fallin-soft): step 329 ms / 3.0 FPS, full threaded pipeline
  2.8 FPS. Supersedes the f32-readback-only attempt (ed1b27e). Overlap / GPU
  stitch tuning tracked in docs/todos.md.

- **fix: burn-fusion ordering panic in the fused RGB8 render path (2026-08-18)** —
  `infer_rgb8` read back the RGB8 output as u8, which (like any non-f32
  `to_vec()`) deterministically panics after ~48 frames with "Ordering is
  bigger than operations" (burn-fusion 0.21 + cubecl-autotune), on every model.
  The permute + clamp + scale now still run on the GPU, but the readback is f32
  and the trivial u8 cast happens on the CPU — byte-identical to the reference,
  full autotune speed retained. Added two guarded tests:
  `repeated_infer_rgb8_does_not_panic` and `infer_rgb8_matches_infer_reference`.
  Benched at 1080p→2160p: real-cugan-x2 2.6 FPS / 14.6 GB, fallin-soft 5.7 FPS
  / 8.1 GB, fallin-strong 5.7 FPS / 8.1 GB (fused step).

- **Fallin loadable: UpCunet2x_fast hand-port + built-in ONNX reader (2026-08-18)** —
  `fallin-soft` / `fallin-strong` are the existing `UpCunet2x_fast` arch (same
  38px reflect pad, verified numerically against the ONNX) — no codegen needed.
  The ONNX file is only a weight container: a new dependency-free protobuf
  reader (`senmei_ml::onnx`) extracts the initializers, and
  `convert_onnx_to_bpk` feeds them into the module (torch `.conv.0`/`.conv.2`
  key remap) to build the f16 `.bpk`. `download_model` and
  `senmei-ml-convert` accept `.onnx` sources automatically. Both models are
  now `loadable: true`; engine output matches the ONNX reference within fp16
  tolerance.

- **senmei-app: drop dead IPC + unused deps (2026-08-18)** — removed the
  frontend-unused `remember_project` command (the internal
  `store::remember_project` stays for `export_project`) and the unused
  `base64` / `tauri-plugin-dialog` / `tauri-plugin-opener` dependencies; the
  `num_block` default now comes from `Registry::resolve`.

- **Dead code removed (2026-08-18)** — dropped `tiling::tile` (test-only),
  `Error::Unimplemented`, `Registry::from_json` (test-only), `Decoder::open`
  (bench-only), `preview::extract_frame` (test-only; the smoke test now encodes
  a synthetic frame via `encode_png`), and a stale `#[allow(dead_code)]` on
  `grid_sample` (it is used by the RIFE arch).

- **Inference engine trait simplified (2026-08-18)** — removed the never-read
  `Backend` enum, `EngineCaps.backend`/`half`, `InferOptions.half`, and
  `InferenceEngine::name()`; capabilities/options now carry only what the
  tiling path consumes (`tiles`, `tile_size`).

- **Model registry: drop SUDO shuffle-cugan, add Fallin + 4x_Alchemy (2026-08-18)** —
  removed `shuffle-cugan` (unclear/SUDO weights). Added `fallin-soft` /
  `fallin-strong` (2× CUGAN retrain, CC-BY-4.0, ONNX-only, sha256-pinned) and
  `4x-alchemy` (4× RealPLKSR_Dysample, CC-BY-4.0, `.pth`) — all `loadable: false`
  until their archs are ported. The default upscaler is now `real-cugan-x2`
  (Apache-2.0). Bench/test defaults updated.

- **Sample output + compare sync (2026-08-18)** — sample renders go into the
  project's `sample/` folder with a time-range tag in the name (pruned to the 5
  newest); the sample window follows the playhead (scrub outside it repositions
  it) and snaps to frame boundaries so the rendered result starts on the exact
  source frame; compare updates both sides together (never one ahead) and the
  result/compare timeline shows the sample window in source coordinates.

- **read_frame: async + project preview frames (2026-08-18)** — `read_frame` is
  now async (decode off the main thread) and accepts `project_dir`; preview
  PNGs land in `<project>/preview/`, namespaced per input file with zero-padded
  counters so pruning keeps the actual newest frames. New `prune_samples`
  command keeps only the newest sample renders in a folder.

- **Ranged renders: stable timestamps + container duration (2026-08-18)** — the
  encoder passes `-copyts` so the piped video keeps its 0-based PTS (the muxer
  no longer shifts it by the seeked-and-copied audio start, which broke
  compare/result alignment) and `-shortest` so copied audio cannot over-run a
  ranged render (the container duration no longer over-reports).

- **Persistent preview decode + PNG frames (2026-08-18)** — `senmei-media` keeps
  one long-lived ffmpeg decode stream per file (`PreviewCache`), so the monitor
  reads the next frame from the pipe instead of spawning ffmpeg per frame.
  `encode_png` replaces the mjpeg round-trip (range-safe on every FFmpeg build).

- **Fix runtime asset scope (2026-08-18)** — `probe_video` and `read_frame` now
  also extend the asset-protocol scope at runtime via `app.state::<Scopes>()`
  `allow_file`, so arbitrary video paths (e.g. outside `$HOME`) and freshly
  written preview frames are always loadable by the webview, even before the
  config globs apply.

- **Fix asset protocol scope (2026-08-18)** — the `assetProtocol` scope was
  `["**"]`, which matches almost nothing: Tauri enables `require_literal_separator`
  for the scope (so `**` behaves like `*`) and requires a literal leading dot to
  match hidden dirs like `~/.local`. Now `["$DATA/**", "$HOME/**"]`, which covers
  the preview temp frames (app data dir) and the user's videos under home.
  Fixes `asset protocol not configured to allow the path` in the monitor.
- **Monitor frames via asset protocol, not data: URIs (2026-08-18)** —
  `read_frame` now writes the extracted frame to a temp PNG in the app data dir
  and returns its path; the monitor loads it with `convertFileSrc`. Large
  frames as `data:` URIs could fail to render in WebKitGTK (broken image icon),
  while the asset protocol already works (native video). Old preview frames are
  capped at 30.
- **Preview frames as PNG instead of mjpeg (2026-08-18)** — `extract_frame`
  encodes the preview frame to PNG. The mjpeg encoder refuses limited-range
  (tv) YUV from libx265/HEVC renders ("Non full-range YUV is non-standard")
  unless `-strict unofficial` is passed, which still produced a broken preview
  on some FFmpeg builds; PNG has no such range restriction. Frontend now uses
  `data:image/png` for decoded frames.
- **Fix monitor frame read-back of HEVC/x265 renders (2026-08-18)** —
  `extract_frame` now passes `-strict unofficial` to the mjpeg `image2pipe`
  encode. The mjpeg encoder refuses limited-range (tv) YUV from libx265/HEVC
  renders without it ("Non full-range YUV is non-standard"), which made the
  result/compare preview fail right after rendering with an ffmpeg error.
  (Superseded by the PNG switch above.)
- **Preview uses the pipeline's ffmpeg (2026-08-18)** — `extract_frame` no
  longer resolves ffmpeg from the current directory; the `read_frame` command
  resolves the same binary the pipeline uses (app data dir / bundled) and passes
  it in. Fixes frame read-back of rendered output failing with an ffmpeg error
  when system ffmpeg is missing or differs.
- **Keep render position after rendering (2026-08-18)** — the monitor no
  longer jumps to position 0 after a render. The position is only reset when a
  new file loads; view switches preserve it, and the result view clamps to the
  sample in-point so it shows the rendered moment. The sample range is no
  longer reset on view switches either, and the result frame is read at
  `ms − inMs` (its timeline starts at inMs). Verified: render of a 30–90s sample
  ends in the Result view at 00:00:30 with In/Out preserved.
- **Slider sample-range highlight (2026-08-18)** — the timeline slider's track
  is now transparent with drawn underlays: a slate base, an indigo played fill
  up to the current position, and the sample window as a strong indigo bar with
  a ring. Previously the highlight sat behind the opaque native track and was
  invisible.
- **Compare alignment (2026-08-18)** — in Compare both sides now show the same
  source moment: the original is clamped to the sample in-point (the rendered
  sample has no frames before it) and the result is read at `source − inMs`
  (its timeline starts at inMs). Previously the result was offset by the sample
  start, so Original at 0 vs Result at the render point were misaligned.
- **Monitor preview opacity (2026-08-18)** — a loaded frame/video now shows at
  80% opacity; the pre-load placeholder is 70% and greyscaled, consistently in
  Original / Compare / Result.
- **Monitor placeholder 80% everywhere (2026-08-18)** — the no-frame
  placeholder now uses the same 80%-translucent background in all three views
  (Original / Compare / Result) so the monitor looks consistent from start.
- **Sample selector as segmented control (2026-08-18)** — the monitor's sample
  range picker is now a compact segmented control `[10s | 30s | 60s | Full | ▾]`
  instead of a dropdown field: presets are one-click segments, the ▾ opens a
  small popup with the custom duration editor (55s, 10m, 1m30s, 1h), and an
  active custom range shows its duration next to ▾. No double field anymore.
- **Native video preview + FFmpeg fallback (2026-08-18)** — the monitor source
  preview now uses a native `<video>` element (hardware decode, via the Tauri
  asset protocol + `convertFileSrc`), falling back to the FFmpeg-decoded frame
  path only when the webview cannot load/play the file (`onError`). Play, scrub
  and the sample in/out loop are wired to the video element. The asset protocol
  is enabled in `tauri.conf.json` (scope `**`). Binding decision updated in
  `AGENTS.md` + `PLAN.md` §1/§3.2. Browser demo unchanged (frames path).
- **Monitor playback + sample dropdown (2026-08-18)** — playback now runs the
  time indicator 1:1 real-time with at most one frame decode in flight (frames
  are skipped if the decoder lags, so FFmpeg subprocesses never pile up — this
  also fixes a performance regression/crash). The sample selector is now a
  dropdown menu like the Output folder (10s/30s/60s/Full/Custom…), with the
  custom duration editor supporting `55s`, `10m`, `1m30s`, `1h`. Fixed a bug
  where picking a preset produced `NaN` (unit strings were parsed with
  `Number()` → now `parseInt`). Verified: 30s → Out 00:00:30.00, 60s →
  00:01:00.00, custom 55s → 00:00:55.00, 10m → 00:10:00.00. The sample panel
  now carries `relative z-10` so its upward-opening dropdown paints above the
  positioned preview area, and the menu is a compact 2-column grid (~89 px tall)
  so it no longer covers the preview. The menu is left-aligned (`left-0`) so it
  grows into the free space to the right instead of clipping at the panel's left
  edge.
- **Monitor sample bar (2026-08-18)** — removed the redundant "Preview sample
  (15s)" button, promoted "Render Sample" to a filled primary button like
  Start Render, and made the sample range default to 10 s (highlighted preset).
- **Demo Compare/Result (2026-08-18)** — the browser demo now simulates a
  rendered output per video, so the Compare and Result tabs work immediately
  (previously they stayed disabled until a fake render finished). The simulated
  result gets a subtle saturate/brightness filter so the split visibly differs;
  the real Tauri app still only enables them after an actual render.
- **Docs cleanup (2026-08-18)** — `models.md` gains a status-at-a-glance table
  and a backlog/candidates section (per-stack, spandrel as source);
  `benchmarks.md` gains a TL;DR box with the engine decision and key numbers.
- **UI backlog (2026-08-18)** — About dialog (Help → About: version, engine,
  license, GitHub link), media-library multi-select (plain click selects one,
  Ctrl/Cmd+click or the ⧉ toggle adds/removes), and the version badge moved
  from the top headers to the bottom-right (status bar + project screen).
  Verified in the running app.
- **Color metadata (M4, 2026-08-17)** — the Output step gains a Color group
  (primaries / transfer / matrix) that tags the encode with `-color_primaries`,
  `-color_trc` and `-colorspace`. Verified in the app: bt2020 primaries →
  `-color_primaries bt2020` in the command preview.
- **FFmpeg quality profiles + command preview (M4, 2026-08-17)** — the Output
  step gains a Quality dropdown (Lossless / Very High / High / Medium / Low)
  that sets crf + preset as a bundle ("Custom" when they diverge), persisted as
  `StepParams.quality`, and a live command preview showing the merged ffmpeg
  args. Verified in the app: Lossless → crf 0 / preset slow, preview updates.
- **Render only the sample range (M5, 2026-08-17)** — the render command and
  pipeline accept `startMs`/`endMs`: the decoder seeks with fast `-ss` and caps
  the frame count, the encoder seeks the audio input so it stays in sync, and
  progress totals reflect the range. The Monitor's sample window now drives a
  "Render Sample" button. Test `render_only_time_range`: 200..700 ms of a 10
  fps clip yields exactly 5 frames.
- **Sample preview range (M5, 2026-08-17)** — the Monitor timeline gains an
  in/out sample range: 10s/15s/30s/60s/Full presets set the window from the
  current position, playback loops inside it, the selected range is highlighted
  on the slider and In/Out markers shown below. Verified in the running app
  (10s preset sets Out 00:00:10.00).
- **RIFE e2e verified (M3, 2026-08-17)** — `infer_interp` now pads the input
  to multiples of 32 (matching rife-ncnn-vulkan, whose flow estimation runs at
  1/32 scale) and crops the output back. Non-32 inputs previously hit a `Cat`
  shape mismatch (e.g. 120 vs 128). New pipeline test
  `rife_interpolates_real_model_e2e` runs decode → real `flownet.bin`
  interpolate (Vulkan fp16) → encode: 10 frames @10fps in → 19 frames @20fps
  out (needs `RUST_MIN_STACK=33554432`).
- **Docs reorganization (2026-08-17)** — PLAN.md §15 moved to
  `docs/CHANGELOG.md`; PLAN.md's front sections rewritten for the current
  reality (burn-Vulkan fp16 is the engine; ncnn engine removed from the plan,
  it survives only as a weight format for RIFE). `models.md` and
  `benchmarks.md` cleaned up (consistent RIFE v4.6 status, single clear engine
  verdict).
- **Project export/open (2026-08-17)** — File → "Export Project…" writes the
  project as a **`.tar.xz`** archive (tar + liblzma, same path as the FFmpeg
  download — no zip crate, which would clash on the native `lzma` link). The
  project screen's "Open Project…" button (was "Open other folder…") imports a
  `.tar.xz` back into the app storage and opens it. "Save Project As" was
  dropped — export/open round-trips instead.
- **UI polish (2026-08-17)** — the media-library drop box now only shows when
  no video is loaded (and fills the whole panel height); videos can be added
  by dragging them anywhere in the window (Tauri `onDragDropEvent` with full
  paths; HTML5 fallback in the browser demo). The `↓` arrows between stack
  steps are gone. The top bar shows `project / video` centered (no pill box),
  and a settings gear sits at the bottom-left of the status bar.
- **Reference filter stacks (M7, 2026-08-17)** — the previously disabled
  `denoise`, `deblur` and `deduplication` steps are now implemented with CPU
  references: box-blur denoise (radius), unsharp-mask deblur (amount), and a
  stateful dedup that drops frames below a mean-pixel-diff threshold.
  `Step::process` now returns `Result<bool>` (false = drop the frame), so the
  pipeline `run_step` can skip frames. The `render` command takes a bundled
  `RenderConfig` (specta caps command arity at 10 args — all knobs moved into
  one struct) with an optional `filter: FilterParams`. Also fixed the long-
  standing `value assigned to failed is never read` warning in pipeline.rs.
- **RIFE v4.6 engine wired (M3, 2026-08-17)** — `RifeNet::load_from_ncnn` parses
  the ncnn `flownet.bin` (per layer `[tag u32][weights f16][bias f32]`) and
  assigns params directly (conv weights `[out,in,k,k]`, deconv transposed to
  `[in,out,k,k]` — burn's `ConvTranspose2d` wants weight[0] = input channels,
  while ncnn stores deconv weights out-major). `BurnEngine` gains a `RifeNet`
  model + `infer_interp(a, b, t)`: frames are f16 NCHW, the timestep is a
  broadcast `[1,1,H,W]` filled with `t` (matching ncnn's `in2`), output returns
  as f32. Loaded and verified on Vulkan: weights walk the whole `.bin` exactly
  to EOF; the interpolated frame is flow-based (≠ linear blend), symmetric
  under `(a,b,t)↔(b,a,1-t)`, and directionally correct (t=0.05→a, t=0.95→b;
  the reference short-circuits exact t=0/1 so endpoints aren't a network
  property). Catalog entry renamed to the real artifact: `rife-v4.6`
  (`arch rife46`, MIT, `flownet.bin`).
- **RIFE v4.6 burn port (M3, 2026-08-17, generated)** — `tools/rife_gen_burn.py`
  translates the ncnn `flownet.param` (215 layers, MIT) into a straight-line
  burn network (`senmei_ml::burn::rife::RifeNet`): 40 Conv2d + 4
  ConvTranspose2d + op helpers (warp = `grid_sample`, bilinear interp,
  pixel-shuffle, channel crop, binary ops), with per-output use-counting for
  burn's move semantics. It **compiles and runs end-to-end on Vulkan**,
  preserving `[1,3,H,W]` with finite values (ignored structural test; needs a
  larger thread stack).
- **grid_sample foundation (M3, 2026-08-17)** — new `senmei_ml::burn::grid_sample`
  (bilinear warp, `align_corners=True`, border padding) matching torch
  semantics; each corner is sampled with a single gather over a flattened
  spatial axis (`y*W + x`) because two chained H/W dim gathers re-pair the
  per-pixel indices wrongly. Verified against a CPU reference over in-range
  and out-of-range grid coords (ignored Vulkan test). This is the sampling op
  RIFE's IFNet/FusionNet warps need.
- **RIFE plumbing (M3, 2026-08-17, phase 1)** — `InferenceEngine` gains a
  2-input `infer_interp(a, b, t, opts)` (default `None` → CPU fallback). The
  pipeline `Interpolator` gets `with_engine` and routes each intermediate
  through the engine when present, else falls back to linear blend / scene-cut
  duplication. The `render` command accepts `interp_model`; the Interpolate
  step's Model dropdown now lists **rife-4.25** (Apache-2.0) from the catalog
  and auto-selects it. The RIFE burn arch port + weight conversion (Phase 2)
  is still pending — until then a selected model degrades to the blend.
- **Output filename includes model & scale (2026-08-17)** — rendered files are
  named `{stem}_{label|senmei}_{model}_x{scale}.{ext}` (e.g.
  `Folge 7_senmei_shuffle-cugan_x2.mkv`), so the applied processing is visible
  at a glance. Also fixed: the Start Render button passed its click event as
  `onlySelected` (truthy), which filtered by the empty selection and never
  started the batch — wrapped in `() => startBatch()`.
- **Selection + Edit/Process menus + hotkeys (2026-08-17)** — library rows
  are selectable (ring highlight; click toggles). Ctrl/Cmd+A selects all,
  Delete removes the selection, Ctrl/Cmd+R starts the batch render. The
  menubar gains **Edit** (Select All Videos, Delete Selected) and **Process**
  (Add All / Add Selected to Queue, Process Selected / Queue / All); Process
  Selected renders only the chosen files, Add to Queue switches the media
  panel to the Queue tab.
- **Header & project-screen polish (2026-08-17)** — the app header drops the
  "Senmei" wordmark (logo + version badge suffice) and gains a gear Settings
  button (like Koharu). The project screen header now matches the main app
  (鮮 logo + version badge) and deleting a project uses an in-app themed
  confirm modal instead of `ask()`/`window.confirm`. Step titles in the stack
  show their model & scale ("2. Upscale · shuffle-cugan ×2", "1. Interpolate ×2").
- **Fix: neon color artifacts on hard edges (2026-08-17)** — the GPU output
  path (`infer_rgb8`) cast to U8 **without clamping**, so model values >1.0 at
  hard edges (burnt-in subtitles) wrapped (e.g. 275 → 19 → magenta/cyan).
  Now `out.clamp(0.0, 1.0)` before the 0..255 scale + U8 cast. The CPU path
  already saturates (`as u8`). Regression: `app_render_upscales_real_model`.
- **Batch rendering (2026-08-17, M7)** — `Start Render` now renders **all files
  sequentially** (a single file is a batch of one). The Queue tab lists one job
  per file with status (queued/rendering/done/failed/cancelled), per-file
  progress bar + frames, and batch controls (Pause/Resume, Stop). Errors mark
  the file failed and continue; Stop aborts after the current file; Pause
  freezes the running file. Output paths are derived from the Output-step
  (folder mode / label / container); new `unique_path` command appends
  `_2`, `_3`, … on filename collisions instead of overwriting. No per-file
  save dialog (auto path + dedupe).
- **Stack reorder via drag (2026-08-17)** — the ▲▼ move buttons are replaced by
  a ≡ drag handle; the whole step header is draggable (pointer-based
  mousedown/move/up with a 4px click-vs-drag threshold, target hit-test on
  `data-step-index`). WebKitGTK handles HTML5 DnD unreliably and renders a huge
  ghost, so this avoids both; a `setTimeout(0)` clears the post-drag click
  suppression so the next click still expands.
- **Pause/resume render (2026-08-17)** — the pipeline waits between frames on a
  pause flag (`set_pause`); `pause_render(bool)` command toggles it. The Queue
  tab shows Pause/Resume next to Cancel next to the progress. Regression test
  `passthrough_pause_resume` proves frames stall while paused and resume after.
- **Output naming/flow (2026-08-17)** — the rendered filename includes the
  Output-step `label` when set (`{stem}_{label}.{ext}`, else `{stem}_senmei.{ext}`);
  when a folder mode is configured (Global/Custom) the render writes straight
  into that folder — no save dialog. Dialog only remains for "Same as input".
- **TopBar cleanup (2026-08-17)** — the redundant Cancel button is gone from
  the topbar; Cancel already lives next to the render progress in the media
  library.
- **Resize factor decimal (2026-08-17)** — the factor field is a text input
  (`inputMode=decimal`) that normalizes a comma to a dot, since `type=number`
  silently drops the comma before `onChange`.
- **Version badge with build hash (2026-08-17)** — the TopBar shows
  `v0.1.0-<short-hash>`; Vite injects `__APP_VERSION__`/`__BUILD_HASH__`
  (last commit) via `define`, so every build identifies its exact source.
- **Audio passthrough (2026-08-17)** — the encoder now takes the source file as
  a second ffmpeg input and maps its audio (`-map 0:v:0 -map 1:a:0?`), so the
  rendered file keeps the soundtrack. The Output-step Audio dropdown drives it:
  `Passthrough` → `-c:a copy`, `AAC`/`Opus`/`FLAC` → re-encode, `None` → `-an`.
  Pipeline passes `input` to `Encoder::open`; regression test
  `passthrough_copies_audio`.
- **Dev stale-UI fix (2026-08-17)** — WebKitGTK showed a stale/cached page
  under Wayland; `dev:release`/`dev` now run under XWayland
  (`GDK_BACKEND=x11`), Vite binds `127.0.0.1` explicitly, `devUrl` matches it,
  and `predev`/`predev:release` auto-run `dev:clean` (kill port 1420 + senmei,
  clear WebKit cache) like Koharu's `predev: kill-port`.
- **Structured encoder settings + merge (2026-08-17)** — the `output` step
  gains RVE-style structured fields, all persisted in `StepParams`
  (`crf`/`preset`/`pix_fmt`/`tune` + existing `videoCodec`/`audioCodec`/
  `subtitleMode`): Preset select, CRF number, Pixel-format select, Tune select.
  `buildEncoderArgs()` (frontend) **merges** them with the raw FFmpeg field:
  the custom field wins for any flag it defines (e.g. `-tune grain`), the
  dropdown values fill the rest (e.g. `-pix_fmt`); the merged string is passed
  to `render` as before. The output-step `label` param (renamed from `name`)
  is empty by default → the badge shows only when a real multi-output label is
  set. Output-step also gains **Format** (`container`, default `mkv`) used for
  the save-dialog extension, and **Output folder** mode (`output_mode`:
  `input`/`global`/`custom` + `output_folder` picker) that sets the save
  default target.
- **Custom FFmpeg output options (2026-08-17)** — the `output` step gains a
  **FFmpeg options** textarea (`params.ffmpegArgs`, persisted in `project.json`
  via `StepParams.ffmpeg_args`). `render` accepts `ffmpegArgs: Option<String>`
  (shell-like tokenizer with quote support), `Pipeline::set_encoder_args`
  threads them into `Encoder::open`, which appends them **after** the built-in
  x264 defaults so user codec/filter args override them. Verified end-to-end by
  the app smoke test: `-c:v libx265 -crf 18 -preset ultrafast -pix_fmt
  yuv420p10le` → ffprobe confirms HEVC + 10-bit output. Default output stays
  x264 `veryfast` (overridable via `SENMEI_X264_PRESET`).
- **Pipeline-stack Inspector (2026-08-17)** — Inspector's flat accordion list is
  replaced by a **dynamic layer stack** (order top→bottom = execution order):
  add steps via a "+ Add step" menu, remove (✕), enable/disable (checkbox),
  reorder (▲/▼). Step types: `interpolation`, `upscale`, `denoise`, `deblur`,
  `deduplication`, `resize`, `output` — the **not-yet-implemented** ones
  (denoise/deblur/dedup) are **disabled in the add menu** ("Soon"). `output` is
  a regular step addable anywhere (multi-output design: each carries a `name`
  label + video/audio codec + subtitle mode; the backend renders the last
  active one for now). `ProjectSettings` schema changed
  (`stepsEnabled`/`upscaleModel`/`scale` → ordered `steps: Vec<PipelineStep>`
  with a typed `StepParams`); bindings regenerated via the specta export test.
  Frontend holds `steps[]` in App state, persists per project, and `startRender`
  derives scale/model/resize/fps from the **enabled** steps. Model select
  auto-fills the first loadable upscaler (ShuffleCugan).
- **UX feedback batch (2026-08-17)** — projects are deletable (🗑 on the
  project screen, `delete_project` command, confirm dialog); videos can be
  removed from the library (✕ per row); **cancel render** (TopBar ■ + Queue
  tab) via a shared `AtomicBool` checked between frames — partial output is
  deleted on abort; Monitor gains a **Compare (side-by-side)** mode for the
  source/result frames plus an auto-switch to the Result view when a render
  finishes. Preview extraction now uses the **resolved ffmpeg** (portable
  fallback) instead of bare `ffmpeg`. Tile size raised 256 → 512 to cut GPU
  sync overhead (better GPU utilization at 1080p). `dev:release` script added
  (`cargo tauri dev --release`) — debug builds render 10–50× slower.
- **Prototype polish (2026-08-17)** — per-project persistence extended: selected
  model/scale, imported videos and output folder are saved in `project.json`
  (`ProjectSettings`). **ShuffleCugan** is the default upscaler (converted f16
  `.bpk`; license flagged "prototype opt-in" pending author clarification).
  Output folder is pickable (Media Library 📁) and used as the render save
  default. Queue tab shows the active render + finished output. Monitor gains
  Original/Result tabs (previews the rendered file) + an in-view render progress
  overlay. Language switch removed from the top bar (Settings only).
- **Preview prototype (2026-08-17)** — working Monitor: new `probe_video` +
  `read_frame` commands (`senmei_media::extract_frame`: ffmpeg `-ss pos -i …
  -frames:v 1 -c:v mjpeg -`, base64 JPEG over IPC) drive a canvas `<img>`
  preview with a **timeline scrubber** (debounced seek) + play/pause. Inspector
  gains a **Download weights** button (`download_model` now reachable from the
  UI). Render now honors `stepsEnabled` (default-on; toggling a step off
  disables it). End-to-end proof: ignored pipeline test
  `burn_engine_upscales_real_model` runs decode → real `real-cugan-x2` burn
  Vulkan fp16 (tiled) → encode → 320×240.
- **Engine switch v3 (decision, 2026-08-17)** — ncnn removed; inference = **burn
  (`burn-wgpu`) on the Vulkan backend, fp16**, CPU fallback. Deleted
  `crates/senmei-ncnn` (C++ shim) and `NcnnEngine`; dropped the `ncnn` registry
  field. Replaced `xz2` with `liblzma` in `senmei-media` (resolves the
  `links="lzma"` conflict with `cubecl-cpu`/`tracel-llvm-bundler`). Added
  **`BurnEngine`** (feature `senmei-ml/burn`, wired into `senmei-app`): loads f16
  `.bpk` burnpacks via `BurnpackStore` and runs the clean **`UpCunet2x`** arch
  (port from `~/github/rust-sr-bench`, verified) on `Vulkan<f16>`. Registry
  schema: `ncnn` → `weights` + `download_url`/`sha256`; `models/metadata.json`
  re-catalogued from VSGAN/TAS hosts. Archs ported: **`upcunet2x`**,
  **`upcunet2x-fast`** (ShuffleCugan) and **`realesrgan`** (RRDBNet, scale 2/4
  via `Option` conv_up2, `num_block` from metadata) — `real-cugan-x2` + 3×
  Real-ESRGAN are `loadable`; SCUNet / Real-PLKSr / Anime1080Fixer (license
  verify) and RIFE (+ 2-input API) still pending. `BurnEngine` dispatches on
  `ModelRef::arch`. See `docs/models.md` + `docs/benchmarks.md`.
- **Burn re-benchmark (2026-08-17)** — re-tested burn with the **real** Real-CUGAN
  upcunet (`up2x-no-denoise.pth` via `burn-store::PytorchStore`) instead of the
  3-conv toy. All outputs numerically verified against the torch reference.
  Findings: **burn-ROCm f32** = 1119/2197 ms @720p/1080p and fp16/bf16 are
  **impossible on RDNA4** (cubecl-hip uses CDNA-only WMMA kernels → `LLVM ERROR`);
  **burn-Vulkan fp16** runs the real model at **136/302 ms** (720p/1080p) —
  *faster than ncnn* (249/398 ms) — and the **ShuffleCugan** variant at
  **46/103 ms**. Vulkan f32 1080p crashes on a `burn-fusion` bug. This **revises
  the 2026-08-16 "burn set aside" verdict** (it was a toy on the wrong backend);
  burn is re-opened as a candidate, but adoption must weigh the ~800-crate /
  1.6 GB build, the fusion bug, and the f32→f16 load workflow. Engine stays
  **NCNN/Vulkan** until a maintainer decision. Details: `docs/benchmarks.md`;
  repo: `~/github/rust-sr-bench`.
- **Candle-ROCm evaluation (2026-08-17)** — tried the `xmiksay/feat/rocm-backend`
  candle fork (local `~/github/candle`, branch `test/xmiksay-rocm`; rocBLAS GEMM
  + im2col conv) via a feature-gated `candle` bin in `~/github/rust-sr-bench`.
  Numerically correct (HIP vs CPU ~1e-5), but f32 convs always materialize the
  im2col matrix → memory cliff from ~640p (multi-GB buffers crash the desktop on
  shared-display GPUs; SD/FLUX VAE decode OOMs at 1024²); f16 scales linearly
  but stays ~6× slower than burn-Vulkan fp16 (290 vs 46 ms @720p ShuffleCugan);
  the ShuffleCugan port additionally OOMs at any size (fork conv bug).
  **Not pursued — burn stays the candidate** (Vulkan fp16). Abandoned work
  remains feature-gated/uncommitted in `rust-sr-bench`.
- **Weights workflow (2026-08-17)** — `senmei-ml` gains a feature-gated
  `senmei-ml-convert` bin: loads a torch `.pth` (f32, Vulkan, upcunet key
  remap) and saves the arch as an f16 `.bpk` burnpack (`HalfPrecisionAdapter`).
  Proven end-to-end on the real `up2x-no-denoise.pth` (→ 2.5 MB `.bpk`); an
  ignored GPU test loads the `.bpk` through `BurnEngine` and infers 32×32 →
  64×64. New `download_model` Tauri command: downloads the `.pth`
  (`download_to_temp`, sha256-verified when pinned) and converts it to the
  `.bpk` in-app. Removed dead `extract_zip` from `senmei-media`.
- **Archs (2026-08-17)** — ported **`UpCunet2xFast`** (ShuffleCugan, from
  `rust-sr-bench`) and **`RrdbNet`** (Real-ESRGAN, BSD-3 reference) into
  `senmei-ml::burn`; `BurnEngine` now dispatches on `ModelRef::arch`
  (`upcunet2x` / `upcunet2x-fast` / `realesrgan`). `RrdbNet` uses burn's
  `Vec<Rrdb>` (torch `body.0…`) and `Option<Conv2d>` (`conv_up2` only at
  scale 4). Real-ESRGAN models flipped `loadable`; RRDBNet numerical
  verification vs torch is the next step (rust-sr-bench harness).
- **M6 (foundation, 2026-08-17)** — new `crates/senmei-ncnn` C++ shim (bindgen + cc): `build.rs` builds NCNN `20260526` from `third_party/ncnn` (Vulkan + CPU, auto-cloned if missing; dir is gitignored) and exposes a safe Rust `Engine` (load `.param`/`.bin`, planar NCHW infer). `NcnnEngine` in `senmei-ml` is now real (was a stub). Verified with the Real-CUGAN `up2x-no-denoise` model — its upcunet crops a fixed border (`out = 2·h − 72`), which the shim/engine faithfully returns; border-aware tiling is a follow-up. `metadata.json` pins the real asset name `up2x-no-denoise`. Build deps: `cmake`, `g++`, Vulkan.
- ~~**Inference engine switch v2 (decision, 2026-08-16)**~~ — **superseded 2026-08-17 (v3: burn/Vulkan)** — after benchmarking on the target AMD RX 9070 (RDNA4/`gfx1201`), the engine was **NCNN/Vulkan** via C++ shim (`cxx`/bindgen) with **CPU fallback**. **candle dropped** (no ROCm backend; per-model Rust ports). **burn set aside** (fusion/JIT immature for SR). Model format was ncnn `.param`/`.bin` (community ports) — **no safetensors graph loading, no conversion, no Python, no Rust arch ports**. The per-model cost is finding a permissively-licensed NCNN port. Evidence in `docs/benchmarks.md`: ncnn 1080p x2 = 398 ms vs torch-ROCm 7153 ms (pathological) + tile OOM/hard-fault on RDNA4. Obsolete vs v1: `CandleEngine`, `.safetensors` loading, "port each arch to Rust" plan. Registry schema: `torch` field → `ncnn` (see §6.4).
- ~~**NCNN-only code switch (2026-08-16)**~~ — **superseded 2026-08-17 (v3)** — removed the `torch` feature, `tch` dep, `TorchEngine`, and the `torch`/`download_url`/`sha256` `ModelMetadata` fields. `engine_for_model` mapped only `.param`/`.bin` → `NcnnEngine`; `Registry::resolve` pointed at the `.param` (the `.bin` sat alongside). Registry = 7 NCNN models (`rife-4.26`, Real-ESRGAN ×3, Real-CUGAN up2x, SwinIR x2/x4) — `loadable: false` until the C++ shim lands (M6). Dropped the `download_model` command + `senmei_media::download_model`, the `scripts/convert_*.py` pipeline, and local `models/*.pt`. `Backend` = `Cpu | Vulkan`. Bridge bindings regenerated. Exact NCNN asset filenames still need pinning in M7.
- **Download-on-demand (decision, 2026-08-16)** — model weights are **not bundled or redistributed**. The app downloads `.param`/`.bin` from a pinned upstream URL on first use (M7). Keeps the runtime small and sidesteps redistribution-license questions for models whose ports lack a clear license; `metadata.json` records license + source for transparency.
- **Libtorch downloader/UI cleanup (TODO, 2026-08-16)** — the libtorch provisioning path still exists (`senmei-media/src/libtorch.rs`, `get_libtorch_status`/`download_libtorch`, `useLibtorch`, SettingsPage inference section, i18n strings) and contradicts the engine switch. Remove it in a follow-up; the Settings inference section should later show the NCNN backend instead.
- ~~**Inference engine switch (decision, 2026-08-16)**~~ — **superseded by v2** (candle dropped after benchmarks; engine = NCNN/Vulkan) — libtorch/`tch`/TorchScript is **dropped**. `senmei-ml` moves to **candle** (CPU/CUDA/Metal) + **NCNN/Vulkan** (no ONNX, no TorchScript). Models are **downloaded** as `.safetensors` from pinned HF repos (Koharu-style `model_repository!` pattern, repo + commit SHA) — **no conversion, no Python**. Each architecture is **ported to Rust** (`candle-nn`) once; that is the main per-model cost. Consequence: the ROCm/AMD-Linux accelerated path is dropped (AMD → NCNN/Vulkan). Obsolete: the `torch` feature, `tch` dep, `TorchEngine`, `scripts/convert_*.py`, and the existing `.pt` files (models re-fetched as `.safetensors`).
- **M0 done** — workspace, 5 crates, `InferenceEngine` trait + engine stubs + model registry, Tauri shell (frameless), React UI, `models/metadata.json`, LICENSE (MIT/Apache), crates.io names secured.
- **M1 done** — FFmpeg passthrough: `senmei-media` decoder/encoder (`rawvideo` pipe), `senmei-pipeline` (`Step`, `Passthrough`, `Pipeline::run`), `render` command + progress channel.
- **UI deviations from §3.1** — Settings panel uses **2 top-level tabs** (`Settings` / `Advanced`) with **accordions** inside, instead of the 6 tabs from the plan. Steps: Interpolate, Decompress, Denoise, Deblur, Upscale, Deduplication, Resize, Output Resize (Settings) + Video Encoder, Audio Encoder, Backend (Advanced).
- **Project flow** — start screen with new/open project; projects stored as directories under `~/.local/share/senmei/projects/` (+ JSON index for browsed folders).
- **Settings page** — dedicated page (not modal) with section sidebar; **Appearance** section holds Language (EN/DE) + Theme (light/dark/system), persisted in `~/.local/share/senmei/settings.json`. Extensible for more settings.
- **i18n** — English default + German; switch in the top bar and Settings.
- **Window controls** — minimize/maximize/close on both the project start screen and the main window (frameless).
- **Theme** — light/dark applied across all components via Tailwind `dark:` classes; `system` follows `prefers-color-scheme`.
- **UI fix** — top bar has `z-50` so menu dropdowns render above the live monitor (stacking-context issue with `backdrop-blur`).
- **Window controls fix** — Tauri v2 ACL: `minimize`/`toggleMaximize`/`close` need explicit `core:window:allow-*` permissions in `capabilities/default.json` (not part of `core:default`). **Window dragging** needs `core:window:allow-start-dragging` (also not in `core:default`) — added.
- ~~**libtorch provisioning (decision)**~~ — **superseded 2026-08-16** (libtorch dropped; see engine switch) — libtorch is **not** bundled; downloaded at first run via Settings → Inference: backend auto-detected (CUDA via `nvidia-smi`, else ROCm via `/dev/kfd`, else CPU) and the matching pytorch.org archive is fetched into `~/.local/share/senmei/libtorch/`. **Version:** `tch 0.24` / `torch-sys 0.24` expect **libtorch 2.11.0** (URLs pinned to 2.11.0: CPU / cu126 / **rocm7.1**; newer archives use the `libtorch-shared-with-deps` filename — no `cxx11-abi`). **Note:** `tch` links libtorch at **build time**, so after the download build with `LIBTORCH=~/.local/share/senmei/libtorch cargo build --features senmei-ml/torch`. Runtime dynamic-load is not supported by `tch`.
- ~~**ROCm not a system dependency (decision)**~~ — **superseded 2026-08-16** (ROCm path dropped with libtorch) — the libtorch ROCm archive **bundles its own ROCm runtime libs** (`libamdhip64.so`, `libMIOpen.so`, `librocblas.so`, … resolve inside `libtorch/lib/`, verified via `ldd`). End users therefore need **no system ROCm install**; only a Linux kernel with `amdgpu` + KFD (`/dev/kfd`) and an AMD GPU. Backend detection uses `/dev/kfd`, not an installed ROCm version. At M8 (packaging) ship the bundled libtorch and document „AMD GPU + `/dev/kfd`" as the requirement.
- **FFmpeg sourcing (decision)** — prefer **system FFmpeg** (Linux: x264/x265/NVENC/VAAPI present). If missing/too old: download **portable FFmpeg** (BtbN GPL builds) into `~/.local/share/senmei/bin/` with progress UI (RVE-style). `get_ffmpeg_status` + `download_ffmpeg` commands; no bundling in installer (GPL binary is a separate process, does not affect MIT/Apache code). macOS download TBD at M8. Resolution order (used by status AND the decode/encode pipeline): valid `SENMEI_FFMPEG` env → system `ffmpeg` → portable. `SENMEI_FORCE_FFMPEG_MISSING=1` simulates a missing FFmpeg for testing the download flow.
- **Webview decision (revision)** — CEF dropped; use Tauri platform webview. In-app preview is **frame-based** (FFmpeg decode → canvas) so it is codec-agnostic (incl. H.265); audio via `<audio>` (AAC/Opus). Final output via FFmpeg (x264/x265).
- **tauri-specta** — typed bindings replace the hand-written bridge: `collect_commands!` + `#[specta::specta]`, `bindings.ts` generated (camelCase, Throw errors), bridge re-exports. `export_ts_bindings` test regenerates.
- **Logging** — `log`/`env_logger` initialized in `senmei`; logs in commands/media/pipeline (render, download, ffmpeg install).
- **Tests** — registry (`from_json`, `load_dir`), encoder capability parsing, settings roundtrip/defaults, project dir creation, passthrough, ffmpeg probe (9 tests).
- **Download integrity** — SHA-256 of the portable FFmpeg archive is verified against a pinned constant (`FFMPEG_SHA256`); BtbN provides no stable tag/checksum, so update the constant on bump.
- **De-mocked UI** — render button wired (output dialog + progress in status bar); TopBar/StatusBar/Monitor driven by real state (files, health, FFmpeg version); Inspector model selects populated from the real model registry via `list_models`.
- **UI kit** — `packages/ui` provides theme-aware `Button` (primary/secondary/ghost) and `Chip`; used in ProjectScreen/SettingsPage; added to Tailwind content.
- **M2 (partial, upscaling)** — `senmei-ml`: **tiling** (tile/stitch, overlap, tested) + **reference bilinear scaler** (tested). `senmei-pipeline`: **`Upscale` step** (Frame↔Tensor, engine or reference fallback, tested). `render` accepts `scale`; UI has 2x/3x/4x control (Inspector) and progress; **upscaling works end-to-end** via the reference scaler without ML. **`TorchEngine`** (real `tch`/libtorch) is implemented behind the `torch` feature — **requires a full libtorch install (headers) + a TorchScript model**; not compiled/verified here (local libtorch is runtime-only). Enable with `--features senmei-ml/torch` + `LIBTORCH=<full-libtorch>`.
- **M2 (tiled inference)** — **`infer_tiled`** in `senmei-ml` wraps an engine so large inputs are split into overlapping tiles, inferred per tile, and stitched (overlap-averaged, canvas scaled by the engine's per-tile scale). Used by `Upscale` with a default `tile_size` of 256 when the engine advertises `tiles`. **Tests** (4): identity reconstruction, scaled output dims, skip-tiling on small input, whole-image path for engines without tiling.
- **M2 (engine selection)** — **`engine_for_model`** picks an engine by weight-file format (`.pt` → `TorchEngine`, `.param`/`.bin` → `NcnnEngine`, else error). **`Registry::resolve(id, dir)`** maps a model id to a `ModelRef` pointing at its torch weight file. The `render` command now takes an optional `model_id`: the Inspector's Upscale model select passes it through, so a real engine is loaded and handed to the `Upscale` step (reference scaler remains the fallback when no model is selected). **Tests** (2): factory-by-format, registry-resolves-model-ref.
- **M2 (model download)** — **`senmei_media::download_model`** (reusing the shared `downloader`) fetches a model weight file into `models/`, verifies SHA-256 against `metadata.json`'s `sha256` (temp-file-then-rename, so a mismatch never leaves a corrupt weight). `ModelMetadata` gains `download_url` + `sha256` fields. New `download_model` command + Inspector "Download weights" button per downloadable model (`useModel` hook). Real-ESRGAN `realesrgan-x4plus` has a pinned URL + checksum. **Note:** the official Real-ESRGAN release is a `.pth` state dict — a one-time conversion to TorchScript `.pt` (per PLAN §6) is still required before `TorchEngine` can load it. **Tests** (1): checksum match/mismatch.
- **M2 (resize + encoder dims)** — new **`Resize` step** (planar-RGB bilinear by a **scale factor**, tested: grow/shrink/noop/color). `Pipeline::run` now opens the encoder with the **first processed frame's dims** instead of the decoder's, so any size-changing step (upscale/resize) produces a correctly-sized output. `render` takes optional `resize` + `output_resize` (`f32` factor, applied before/after upscale); Inspector Resize/Output Resize accordions get a single factor input (empty = off), replacing the "— M2" placeholders. If a selected model can't be loaded (missing weights/unsupported format), render logs a warning and falls back to the **reference scaler** instead of aborting. **Tests** (2 e2e): 160x120 → upscale x2 → 320x240; 160x120 → resize 0.5 → 80x60.
- **M2 (review fixes)** — **Frame↔Tensor layout fix**: FFmpeg frames are packed `rgb24`, but `frame_to_tensor`/`tensor_to_frame` did a linear copy into planar NCHW, scrambling every upscaled/resized frame's pixels (dims-only tests hid it). Now de-interleaved/interleaved correctly. **Scale enforcement**: an engine's fixed upscale factor (e.g. x4) is now resized back to the requested scale, so the UI scale choice is authoritative. **`loadable` flag** on `ModelMetadata`: `realesrgan-x4plus` is marked `loadable: false` (`.pth` state dict awaiting TorchScript conversion), so render no longer auto-downloads unusable weights. Non-torch `TorchEngine` stub now reports `Cpu` (was `Cuda`), and `Decoder.total_frames` guards against 0 (progress NaN). **Tests** added: frame↔tensor pixel roundtrip, upscale x1 pixel preservation, engine-scale enforcement.
- **Project settings persistence** — the Inspector's per-step enabled toggles are persisted in **`<project>/project.json`** (`steps_enabled` map), loaded when a project opens and saved on every change (`load_project_settings` / `save_project_settings` commands, tested roundtrip). The Accordion `enabled` state is now controlled from App. Expanding an accordion enables the step; **collapsing leaves it unchanged** (fix: previously every row click also enabled the step).
- **Dev workflow (decision)** — `bun run dev` runs **`cargo tauri dev`** (Koharu-style): Tauri CLI starts Vite (`beforeDevCommand`) and `cargo run`s the app with hot-reload. `tauri.conf.json` (+ `capabilities/`, `icons/`, `build.rs`) live in the **bin crate `crates/senmei`**, not `senmei-app`, because `tauri::generate_context!`/`tauri_build` read the config relative to the crate's manifest dir; `senmei-app` stays a pure lib (`specta_builder` + commands). Root `package.json`: `dev` → `cargo tauri dev`, `ui:dev` → frontend only. **Note:** `beforeDevCommand` runs from the **repo root** (verified), so paths are `packages/app`, not `../`. `default-run = "senmei"` makes `cargo tauri dev` pick the right bin.
- **Dependency security** — **JS:** bumped `vite` `5.4.x → 7.3.x` to clear 4 Dependabot alerts (vite `fs.deny` bypass + `.map` path traversal + launch-editor NTLMv2, and esbuild dev-server — esbuild is only patched ≥0.25, hence Vite 7). `bun audit` reports no vulnerabilities. **Rust:** one open Dependabot alert on `glib 0.18.5` (`VariantStrIter` unsoundness, fixed only in ≥0.20) is an **upstream blocker** — `gtk 0.18.2` (last GTK3 binding) pins `glib ^0.18`, and `tauri 2.11.5` is the latest; no patched 0.18.x exists, so it is unresolvable by `cargo update` until the Tauri/gtk-rs stack moves to glib 0.20. **Accepted risk** (dismissed on GitHub as "Risk is tolerable to this project"): the vulnerable `VariantStrIter` API is never exercised by the app.
- **M3 (interpolation, partial)** — `senmei_ml::interpolate` provides `mean_abs_diff`/`is_scene_cut`/`blend`; a stateful pipeline **`Interpolator`** emits `factor-1` blended intermediates between consecutive frames, or **duplicates across scene cuts** (threshold 0.25 mean-abs-diff). `Pipeline::run` accepts an optional interpolator and scales the encoder **fps** and progress total by the factor; frame↔tensor conversion moved to a shared `frame` module. `render` takes `fps_multiplier`; the Inspector fps buttons (2x/3x/4x, toggleable) drive it, and the interpolate model select no longer writes to the upscale model. **RIFE TorchScript inference is still pending** — the reference path is linear blending (rife-4.26 has no downloadable `.pt` yet). **Tests**: ml blend/scene-cut (4), interpolator factor/scene-cut (4), e2e fps doubling (1).
- **Real-ESRGAN TorchScript conversion** — `scripts/convert_realesrgan.py` converts official Real-ESRGAN RRDBNet checkpoints into loadable TorchScript; it auto-detects `num_block` and the input layout (classic 3-channel, or `pixel_unshuffle` for the x2 model) and traces two 2× nearest upsamples. Registry has three loadable upscalers: `realesrgan-x4plus` (4×), `realesrgan-x4plus-anime` (6B, 4×), `realesrgan-x2plus` (2×). Verified via the ignored `torch_loads_realesrgan_models` test (loads in `TorchEngine`, 64→64·scale). The `.pt`s are **not committed** (`models/*.pt` ignored) and `download_url`/`sha256` are dropped (they pointed at unloadable `.pth`s). Requires `torch` + `libtorch` to build/run with the `torch` feature.
- ~~**Model ingest (decision)**~~ — **superseded 2026-08-16** (models are now downloaded as ncnn `.param`/`.bin`, not converted to TorchScript) — raw `.pth` state dicts are converted **once** to TorchScript, maintainer-side (`scripts/convert_realesrgan.py`, needs Python + torch). End users never convert: the finished `.pt`s are **bundled/downloaded at M8** (like libtorch provisioning). `spandrel` (MIT) can later broaden the converter to more architectures; each adopted model must itself carry a permissive license (BSD/MIT/Apache).
- ~~**More upscalers (spandrel)**~~ — **superseded 2026-08-16** (conversion pipeline dropped; models downloaded as ncnn `.param`/`.bin`) — `scripts/convert_spandrel.py` converts permissively-licensed checkpoints (`.pth`/`.safetensors`) to TorchScript via spandrel (MIT): it retargets window-attention models (SwinIR/HAT) to the runtime tile size (256), re-derives their attention masks, and verifies trace==eager. Registered 4 new loadable upscalers: `real-cugan-x2` (MIT, anime 2×), `swinir-x2` (Apache-2.0, classical 2×), `swinir-x4` (Apache-2.0, real-world 4×), `hat-x4` (Apache-2.0, real-world 4×). **Tiling fix:** traced window-attention transformers are resolution-locked, so `infer_tiled` now pads small inputs to a full tile and `tile()` edge-aligns the last tile — every tile is exactly `tile_size`; padded borders are cropped from the output. `.pt`s are not committed; verified via the ignored `torch_loads_upscaler_models` test.
