# Senmei — Changelog

> Implementation log (was §15 of PLAN.md). Newest on top.

> Kept in sync with actual implementation. Update on every significant change.
> Each release gets a `## x.y.z (YYYY-MM-DD)` heading; release notes are
> generated from the section above the latest heading.

## Unreleased

- **test: real-frame upscaler sweep (2026-08-23)** — `bench_upscalers_real_frames`
  benchmarks every loadable `upscale` model at its native scale on two real DVD
  frames (720×576, `models.bat/`); results in `docs/benchmarks.md`. Sweep is
  panic-isolated per model and falls back to tiled infer when the fused path
  trips the VRAM guard. Findings: `realesrgan-general-x4v3` is the fast real-film
  4× (197 ms), fallin fastest 2× (61 ms), SAFMN surprisingly slow (~1 FPS),
  `span-2x-nomosuni-multijpg` panics with `DTypeMismatch` (stale bpk, needs
  re-convert).

- **feat: Real-ESRGAN general-x4v3 (SRVGGNetCompact, 2026-08-23)** — real-scene
  compact upscaler added to the catalog (`realesrgan-general-x4v3`, BSD-3-Clause,
  flat `realesr-general-x4v3.pth`). The `SrvggNet` arch now carries one `Prelu`
  per mid conv (`num_conv` + 1) instead of one shared PReLU — animevideo-xs
  (num_conv 16, shared) and general-x4v3 (num_conv 32, per-layer) both load; the
  converter remaps `body.{2k+1}.weight` → `prelu.{k}.weight` before the conv
  remap (ordering matters — the conv remap would otherwise collide). `ModelRef`
  gained `num_conv`; `download_model` passes it for `srvgg` archs. Verified vs
  torch: mae 0.0004 (f16).

- **ui: sort the model dropdown (2026-08-23)** — model options are now ordered
  loadable-first, then family → scale → id, so usable models are on top and the
  list is predictable (was insertion order).

- **feat: SAFMN arch + SAFMN-L Real x2/x4 (2026-08-23)** — new `SafmnNet`
  arch (clean burn port of the Apache-2.0 `sunny2109/SAFMN` reference),
  converter path, engine dispatch, and two catalog entries
  (`safmn-real-x2` / `safmn-real-x4`, Apache-2.0 weights from the official
  `SAFMN_L_Real_LSDIR_*-v2` release, HF mirror `Meloo/SAFMN`). Config
  dim 128 / 16 blocks / ffn_scale 2.0 / SAFM n_levels 4; input is edge-padded
  to a multiple of 8 (SAFM's `h/2^i` pools). Verified vs torch: mae 0.008 (x2)
  / 0.027 (x4, f16, worst-case random input). Converter key-contract test
  added.

- **fix: pixel-shuffle permutation scrambled upscalers (2026-08-23)** — the
  shared `pixel_shuffle` helper used the wrong `permute`
  (`(0,1,3,5,2,4)` instead of torch's `(0,1,4,2,5,3)`), scrambling the
  channel/spatial layout of every PixelShuffle upsampler. Fixed in `SrvggNet`
  (the `realesrgan-animevideo` x2/x4 outputs were affected) and in the new
  `SafmnNet`.

## 0.1.8 (2026-08-23)

- **fix: Full Video Mode (2026-08-23)** — reworked onto OS window fullscreen:
  `requestFullscreen()` on webkit2gtk only works for `<video>` media fullscreen,
  whose separate layer swallowed dblclick/controls (uncontrolled toggle + a
  playback hiccup when the sink re-inits). Full Video Mode now fullscreens the
  window (`Backend::setWindowFullscreen`, Tauri) and covers the viewport with
  the monitor (fixed overlay) — the `<video>` stays in the DOM and the Monitor
  is never remounted (position kept, no jump to 0, no dblclick stutter). Toggle
  is a document-level capture-phase `click` listener scoped to the monitor's
  bounding box (500 ms window matching GTK's dblclick timeout) — it counts
  clicks, not dblclicks (webkit2gtk hijacks a real dblclick over the `<video>`
  for its own native fullscreen), and capture phase + coordinate scoping make it
  independent of which element webkit2gtk hit-tests: its hit-testing goes stale
  under a stationary cursor once the transition moves/resizes the window, so a
  dblclick at the same spot used to target a stale element (and toggle nothing)
  until the mouse moved. Requires `core:window:allow-set-fullscreen` (was
  missing → the window never went fullscreen, only the monitor filled the
  window). `Esc` exits Full Video Mode (the OS-window fullscreen has no native
  Esc, which `requestFullscreen()` previously provided). Removes the earlier
  webkit2gtk signal wiring (dead code).

- **fix: sample window follows the playhead during playback (2026-08-23)** — the
  sample range only re-anchored to the playhead on scrubs; playing past the
  window left it stale, so "Render Sample" clipped content from before the
  current position (e.g. the default 10 s window). Playback now re-anchors the
  window to the playhead when it crosses the out-point (source mode, native
  video and decoded frames), matching the scrub behavior.

- **fix: long renders hang once ffmpeg's stderr fills its pipe (2026-08-23)** —
  `Encoder` captured stderr but only read it after `child.wait()`; on long
  encodes (with any steady warning stream) the 64-KiB pipe filled up, ffmpeg
  blocked writing, and `finish()` never returned (queue stuck, output stopped
  growing). stderr is now drained by a background thread (tail kept for error
  messages). Regression test `finish_after_stderr_overflows` fails without the
  drain (deadlocks at ~60 s) and passes with it.

- **fix: render ETA shows `--:--:--` instead of `-1:-1:-1` (2026-08-23)** — the
  remaining-seconds estimate could go negative (frame-count estimate lags the
  actual emission, or no frames processed yet) and `fmtEta` formatted negative
  components. Clamped to ≥ 0; not-yet-estimable shows a placeholder.

- **refactor: dedup zip extraction (2026-08-23)** — shared
  `senmei_media::extract_zip(archive, dest, filter)` replaces the three
  near-identical extract loops (`extract_zip_prefix`, torch.rs
  `extract_wheel_prefixes`/`unzip`); `extract_binary` (find-one) unchanged.

- **docs: PLAN §18 preview/media pipeline (2026-08-23)** — media belongs to
  the app, not the engine: raw-frame `FrameSink` transport, preview decode
  budget, audio FFmpeg→PCM native sink, PreviewCache simplification; phased
  todos added.

- **refactor: shared deps live in `[workspace.dependencies]` (2026-08-23)** —
  log/serde/serde_json/thiserror/anyhow/tokio/zip/clap/env_logger/schemars/
  base64 were duplicated across the crate manifests; now one version per dep
  (`X.workspace = true`). Also unified the `zip` duplicate: senmei bumped to
  zip 8 (was 2.4.2), matching burn-store's 8.6.0 (extraction code
  API-compatible). Tree has 3 zip versions (was 4); 0.6.6/7.2.0 transitives
  remain.

- **test: HTTP/REST adapter unit tests (2026-08-23)** — `senmei-server` router
  smoke tests (9): health, backend-info (camelCase), settings-schema, models,
  SPA-fallback 404 for unknown `/api/*`, error paths (probe/scan-folder → 400),
  CORS, method-not-allowed. Run with `cargo test -p senmei-server --features
  http`; no GPU / no network (system ffmpeg on PATH).

- **fix: sampled renders keep the audio at the sample position (2026-08-23)** —
  muxing the source audio with `-ss`/`-t` between the two ffmpeg inputs +
  `-copyts` was unreliable: the seeked audio kept its source PTS (dropped/
  desynced by `-shortest`) and some containers ignored the seek entirely
  (audio from the start of the file). `Encoder` now extracts the exact audio
  range to a temp `.m4a` first (re-encoded, 0-based) and stream-copies it in —
  verified against the source at the sample position (correlation 0.99).

## 0.1.7 (2026-08-23)

- **fix: U-Net denoisers run full-frame, no tiling (2026-08-23)** — tiled
  SCUNet/DRUNet ghost on moving content: window attention (Swin) + the ÷8
  down/upsample pyramid are not translation-equivariant, so each tile's output
  differs globally from a full-frame run (verified: scunet tiled vs full
  mae ≈ 0.13, ghost copies at tile seams). DnCNN/FFDNet (local convs) tile
  fine. `infer_denoise_tiled` now runs full-frame up to 4K (models pad
  internally); regression test `scunet_tiled_ghosts_at_tile_seams`.
- **fix: srvgg conversion matches the animevideo-xs checkpoints (2026-08-23)** —
  the burn `SrvggNet` arch was the generic SRVGGNetCompact, but
  `realesrgan-animevideo-x2/x4` are the xs variant: 18 body convs, the last
  folded `64 → 3·scale²` upscale conv, **no** `upsampler.*`/`conv_last`, state
  dict under `params` (stripped). Both models convert + load + upscale again
  (download_model). New no-GPU key-contract test
  `srvgg_conversion_key_contract` guards the mapping against drift.

- **docs: testing todos (2026-08-23)** — coverage-review gaps logged for the
  next release: frontend (0 tests), HTTP adapter (0), `#[ignore]` model/GPU
  tests local-only, arch tests (real_plksr et al.).

- **fix: libtorch runtime ignores a stale `LIBTORCH` env (2026-08-23)** — a
  foreign `LIBTORCH` in the launch shell (e.g. a Python venv) no longer hijacks
  the shipped/pinned runtime; the local install is only used when explicitly
  opted in via `SENMEI_LIBTORCH_ENV=1`. Fixes the packaged-app ABI mismatch.
- **feat: adaptive fused VRAM guard (2026-08-23)** — the fused RGB8 peak
  ceiling scales with the system: half the GPU's total VRAM on smaller cards,
  crash-safe 2.5 GiB cap on larger ones (just under the ~3.2 GB wgpu/burn
  single-allocation OOM at 1080p×4). SD/720p×4 and 1080p×2 now render;
  1080p×4 stays blocked (deep burn fix still open).

- **docs: PLAN §17 Auto-Enhance decision (2026-08-23)** — `QualityProfile`
  seam (code-first analyzers, NR-IQA optional behind it); model shortlist
  FACTOR/NIMA (Apache-2.0), PaQ-2-PiQ + CLIP-IQA excluded (NC licenses);
  todos added.

## 0.1.6 (2026-08-22)

- **docs: README — first-run guide, FAQ, download path fix (2026-08-22)** —
  new "First run" walkthrough (wizard screenshots now used), FAQ section,
  corrected "Download weights" path, dev docs renamed "For developers".

- **docs: module structure — senmei-core + senmei-server (2026-08-22)** —
  AGENTS.md + PLAN.md + README architecture now list the two headless crates
  (transport-agnostic core + MCP/HTTP service); the matching `todos.md` entry
  is closed.

- **docs: README + screenshots (2026-08-22)** — hero screenshot, status note,
  Installation + System requirements sections, Quickstart renamed "from source";
  all six UI screenshots re-captured at 1280×800 and cropped to content (no
  letterbox).

- **fix: pipeline trailing empty batch (2026-08-22)** — at decoder EOF the
  pipeline drained an empty batch through the step chain; the deferred upscale
  path treated it as an empty submit (`"empty batch"`) and failed the render
  right at the end whenever an engine-backed upscale ran. The upscale step now
  no-ops on an empty batch and the pipeline skips the trailing empty pass.
  Test: `upscale_process_batch_empty_is_noop`.

- **perf: f16 readback + 8K pre-check (2026-08-22)** — the fused RGB8 readback
  goes f16→u8 directly, skipping the full f32 copy per frame (cubecl-wgpu
  already pools the staging buffers). The VRAM guard now runs *before* the tile
  grid is built, so 8K+ inputs (144+ tiles) are rejected up front with a clear
  error instead of wasting tile/pad work (Koharu `max_pixels`).

- **feat: VRAM guard for the fused RGB8 path (2026-08-22)** — oversized fused
  renders are rejected with a clear error *before* the ~3.2 GB single
  allocation OOM (which lost the wgpu device handle) instead of silently
  falling back to the slow CPU path. Hard 2 GiB peak window (1080p×4 is over
  it — tile size and autotune level don't help; 720p×4 / 1080p×2 stay under)
  plus a free-VRAM budget read from DRM sysfs. Root cause of the 1080p×4
  allocation (wgpu/burn internal) still open; the x2 fused path is unaffected.

- **perf: readback pipelining (2026-08-22)** — the fused RGB8 forward is split
  from its readback (`infer_rgb8_submit` → `Rgb8Batch`); the upscale step keeps
  `pipeline_depth` batches in flight, queuing the next forward **before**
  resolving the oldest readback, so the GPU stays busy during the transfer.
  `bench_upscale_pipelined` (fallin-soft 1080p): 285.2 → **221.6 ms/frame**
  (3.5 → 4.5 FPS, ~22 %). Depth 1 (double-buffer) captures the win; configurable
  via new `pipeline_depth` setting (default 1).

- **perf: batch path measured — disabled on RDNA4/Vulkan (2026-08-22)** —
  `bench_upscale_batch` (fallin-soft 1080p, burn-Vulkan fp16) shows multi-frame
  batching regresses: batch 4 = 310.8 ms (109 % of per-frame 285.2), batch 8 =
  378.4 ms (133 %). Larger batched matmuls are pathologically slower on this
  backend. `BATCH_SIZE` now defaults to **1** (per-frame; the fused single
  `infer_rgb8` path still wins). Audit also notes the fused path only fires at
  the model's native scale — x2 models rendered at x4 take the slow
  CPU-convert path (open; the real win for the shipped use case).

- **fix: log the headless/error path (2026-08-22)** —
  - `list_models` no longer swallows a registry load failure (logs it instead
    of silently returning an empty list).
  - model weight loads are logged ("weights loaded"): a log tail ending before
    that line pinpoints a load-time crash vs. one mid-render.
  - HTTP responses log client rejections (`warn`) and server errors (`error`);
    `json_ok` no longer returns a silent empty 200 on serialize failure.
  - `senmei-server` now writes a rotating `senmei.log` (Info+) to the data dir
    (same scheme as the GUI), so headless/HTTP runs leave a trace.

- **fix: cancel cleans up properly (2026-08-22)** — a cancel used to leave the
  encode ffmpeg to mux the whole output file (`Encoder::finish` → `child.wait`),
  holding the pipeline (and its GPU engine) hostage until it returned; a quick
  re-render then ran two GPU engines at once (the RDNA4 reset pattern). Cancel
  now aborts the encoder immediately (`Encoder::abort` = kill + reap), the
  render gate rejects a new render while the previous one is still running or
  cleaning up, and cancel is logged (GUI command, core, pipeline).

- **fix: sample render drops audio (2026-08-22)** — ranged sample renders copied
  the source audio (`-c:a copy`), which needed `-ss`/`-t`/`-copyts` mux surgery
  and hung at 100% on ranged inputs. Samples now force `-an`: a single
  rawvideo-pipe stream has no mux-sync hazard.

- **perf: multi-frame fused batching (2026-08-22)** — `InferenceEngine` gains
  `infer_rgb8_batch` (fused tiled RGB8 over the batch dim: one tile grid, one
  feather mask, fewer launches/readbacks; bit-identical to N separate
  `infer_rgb8` calls). `Step` gains `process_batch`/`flush`; the pipeline
  accumulates up to 4 frames (`BATCH_SIZE`) and runs the whole chain once per
  batch, with a trailing flush cascade on decoder EOF. `Upscale` uses the batch
  path only for equal-sized frames, falling back to per-frame otherwise.
  `warmup()` runs one tiled forward on load so the autotune cache is warm before
  the first real frame.

- **fix: render/engine edge cases from review (2026-08-22)** —
  - cancel is set up + cleared *before* the (slow) model load, so a cancel
    issued while models load is no longer overwritten to false.
  - `gfx_target_version` parse tolerates a `0x` prefix (some kernels print it).
  - `aotriton.images` is only required for archs with a family wheel
    (gfx11/gfx12); demanding it on gfx9/gfx10 made every launch re-download
    the ~2 GB wheel and fail with "libtorch download incomplete".
  - the wrapper/runtime ABI probe now also guards the CUDA path (was ROCm-only).
  - interpolation progress: `total_frames = 1 + (N-1)*factor` (the interpolator
    emits `factor-1` intermediates per following frame), so progress reaches
    100% instead of capping below it.

- **fix: killing the app no longer freezes the terminal (2026-08-22)** — the
  ffmpeg decode subprocess inherited the terminal's stdin and the encode
  subprocess inherited its stdout, so an orphaned ffmpeg kept the pty held
  after the app was killed (terminal appeared dead until `reset`). Both now
  use `Stdio::null()` for the side they don't use.

- **fix: log libtorch fallback reasons (2026-08-22)** — `engine: auto` silently
  swallowed `TchEngine::runtime` errors, so a failed libtorch (e.g. a stale
  `LIBTORCH` env pointing at a different torch version) fell back to burn-Vulkan
  with no trace. `resolve()` now logs which runtime it picked (LIBTORCH env vs
  cached vs download), `ensure_loaded` logs dlopen/probe/SDK-download failures,
  and the probe error hints at a stale `LIBTORCH` env.

- **fix: ranged render with audio never finishes (2026-08-22)** — the encode
  ffmpeg command maps the source audio as a second input (`-map 1:a:0?`),
  seeked with `-ss` but not duration-limited. For a ranged render the (short)
  video pipe hits EOF but the copied audio input runs to the end of the source,
  so ffmpeg never exits and `Encoder::finish` blocks in `child.wait()` — the
  UI shows 100% (all video frames written) but the render never reaches
  `done`. `Encoder::open` now takes `duration_ms` and bounds the audio input
  with `-t` (pipeline passes `end_ms - start_ms`), so ffmpeg exits after the
  range.

- **fix: ROCm libtorch backend runs (multiple crash bugs, Koharu-style
  (2026-08-22)** — the pytorch.org `2.11.0+rocm7.1` libtorch zip ships
  unversioned ROCm libs but not the versioned SONAMEs (`libMIOpen.so.1`,
  `librocprofiler-sdk.so.1`, `libamdhip64.so.7`) that `libtorch_cpu` dlopens;
  on a bare system libtorch loads incompletely and the wrapper reads
  `TensorOptions` wrongly (dtype/device garbage → heap corruption / SIGSEGV).
  Fixed four root causes:
  1. **Runtime/sdk pair** — the ROCm runtime now comes from the **AMD wheel
     index** (`torch-2.12.0%2Brocm7.14.0` + `amd_torch_device_<gfx>` /
     `_<family>`), matching the pinned SDK 7.14; `runtime/torch.rs` extracts
     `torch/lib` + `torch/.kpack` + `torch/lib/aotriton.images`.
  2. **Wrapper headers** — the tch wrapper must be built against the **same
     torch headers as the runtime** (2.12). Built against 2.13 it reads
     `TensorOptions` wrongly (Half comes back as Int16); `probe_tensor_ok`
     now checks the **dtype** (not just `is_ok()`) so a mismatch is caught
     → clean fallback to burn-Vulkan. CI + local builds pin 2.12 CPU headers.
  3. **System-HIP poisoning** — probing the system HIP (e.g. 7.1) initializes
     HSA, whose runtime state is shared with the kernel driver; the SDK HIP
     7.14 that torch preloads then reports "No CUDA GPUs are available" and
     the render crashes. Linux ROCm detection now reads the **kernel KFD
     topology** (`/sys/class/kfd/…/gfx_target_version` → `gfx1201`), never
     touching HIP; non-Linux keeps the dlopen probe.
  4. **SDK extraction** — `extract_zip_prefix` was stripping the
     `_rocm_sdk_core`/`_rocm_sdk_libraries` root (flat extraction → preload
     found nothing). `TchEngine` preloads the SDK (`rocm_sdk_core`/`_libraries`/
     `_device_<gfx>`, RTLD_LAZY|GLOBAL, full ordered list incl. `host-math` +
     `rocm_sysdeps`) and never touches the system ROCm.

- **fix: no silent resize fallback when the upscale model is missing
  (2026-08-22)** — `build_steps` swallowed `engine_for_model` errors via
  `.ok()`, so a render whose model wasn't downloaded silently degraded to a
  bilinear resize (the sample "looked too fast / no model ran"). The main
  upscale model is now mandatory: a missing/unloadable model fails the render
  with a clear "weights are not downloaded" error; optional aux models
  (decompress/denoise/deblur) and the interpolator keep their reference
  fallback but log a warning. `engine_for_model` checks the weight file
  exists and names the expected path.

- **feat: Windows libtorch (tch) backend via dlopen fork (2026-08-22)** — the
  `senmei-app/tch-rs` fork now builds the wrapper with CMake + MSVC on Windows
  (`tch.dll`, SHARED + `WINDOWS_EXPORT_ALL_SYMBOLS`) instead of bailing; the
  loader preloads libtorch's DLLs (`c10`/`torch_cpu`/`torch`/…) and loads the
  wrapper via LoadLibraryW. The Rust-stream tensor save/load bridge (which
  left the DLL with unresolved symbols and broke the MSVC link) was dropped —
  nothing in senmei/burn-tch used it, and the wrapper is now self-contained.
  The ROCm/HIP `RTLD_GLOBAL` preload in `senmei-ml`'s tch engine is cfg-gated
  to Unix (Windows loads libtorch via LoadLibrary). Tag bumped to
  `v0.22.0-senmei-win`, so the Windows bundle ships the tch backend again.

## 0.1.5 (2026-08-22)

- **fix: A/B keeps its pair when re-rendering the same input (2026-08-21)** —
  `startBatch` no longer clears the previous result for a single-input
  re-render (model A → B); a file switch or multi-file batch still clears it.
  Monitor's single-view fallback is gated off while the A/B/compare panes are
  shown.

- **fix: remove the multi-tile seam grid on SPAN renders (2026-08-21)** —
  `infer_rgb8` now sums tiles with a feather ramp (partition of unity) into
  the canvas instead of `slice_assign`-replacing the overlap. The model emits
  a 1-2px dark line at every tile edge (border context is cut off); replacing
  the overlap left the next tile's edge line visible, showing as a "6-band"
  grid on 1280×720 (3×2 tiles). Weights are ~0 at a tile edge bordering a
  neighbour and 1 at the canvas border; the intermediate slice view is scoped
  so the backend writes in place (no copy-on-write). Seam jumps at x=960/1920
  dropped from ~41 to ~0 on a constant-gray probe; single-tile and warm 6-tile
  cost unchanged (~24 ms stitch). The CPU `tiling::stitch` (used by
  `run_tiled` for >1080p denoise/filter) gets the same feather ramp; the
  weighted sum keeps brightness (no partition-of-unity drift), verified by
  new unit tests.

- **fix: SPAN inversion — burn-store now respects `.pth` strides (2026-08-21)** —
  `PytorchReader` read storage linearly, so TNTwise/Phhofm `params`-wrapped
  SPAN checkpoints (non-contiguous 3×3 `conv1`, strides `(54,1,18,6)`) loaded
  `(out,kh,kw,in)`-scrambled and rendered inverted. All burn crates +
  burn-tch/burn-store now come from the `senmei-app/burn` fork tag
  `v0.21.0-senmei-burn-store-strides` (strides + dlopen), tch/torch-sys from
  `senmei-app/tch-rs` `v0.22.0-senmei-dlopen`; the original non-contiguous
  `.pth` converts correctly (global_mae=0.00001) without preprocess.

- **refactor: split Monitor into `monitor/` sub-components (2026-08-21)** — the
  pure presentational blocks move into `monitor/{CompareView,ModeTabs,Timeline,
  Benchmark}.tsx`; the stateful Monitor keeps the video/transport/sample editor
  and shrinks 903→757. (A/B + compare, mode tabs, scrubber + in/out bar,
  per-step benchmark.)

- **refactor: split frontend i18n + Monitor helpers (2026-08-21)** — the
  en/de message dictionaries move out of `i18n.tsx` into `src/i18n/{en,de}.ts`
  (`i18n/index.tsx` keeps the provider/hook); Monitor's pure time/format
  helpers move to `monitor/format.ts`. TS build unchanged.

- **refactor: split `app/store.rs` into `store/` modules (2026-08-21)** —
  settings (`Settings` + load/save) move to `store/settings.rs`, project
  management (entries, tar.xz import/export, delete) to `store/projects.rs`;
  `store/mod.rs` keeps the shared `data_dir`, re-exports and the tests.

- **refactor: extract `.pth`/`.onnx` converter into `src/convert.rs`
  (2026-08-21)** — `convert_pth_to_bpk`, `convert_onnx_to_bpk` and helpers
  move out of `burn/mod.rs` into a dedicated `convert` module (burn-gated);
  the crate-root re-export keeps `senmei-ml-convert` and `download_model`
  callers unchanged. `burn/mod.rs` is now just the engine + its tests
  (~965 lines, was 1358).

- **refactor: split `pipeline/step.rs` into `steps/` modules (2026-08-21)** —
  each step (`Filter`, `Denoise`, `Deblur`, `Dedup`, `Upscale`, `Resize`)
  moves into its own `steps/<step>.rs`; `steps/mod.rs` keeps the `Step`
  trait, `Passthrough`, the shared `TILE_SIZE`, re-exports and the tests.

- **refactor: unify `download_model` in `senmei-core` (2026-08-21)** — the
  GUI's richer download (ncnn `.bin`, release-zip extract, skip-if-present,
  progress) moves into `core::download_model(model_id, on_progress)`; the GUI
  command and HTTP delegate to it. ~135 lines of duplication removed.

- **refactor: GUI delegates render/models to shared `senmei-core` (2026-08-21)**
  — `commands.rs` no longer assembles the pipeline or duplicates model loading:
  `render` maps its IPC config onto `core::RenderConfig` and calls
  `core::render` (user tile size + backend + cancel/pause flags via
  `RenderOpts`); `list_models`, `scan_folder`, `get_ffmpeg_status` and
  `models::engine_for_model` delegate to the core. Core gains
  `RenderOpts.backend` threaded through `build_steps`,
  `engine_for_model(model_id, backend)`, `list_models` annotating `downloaded`,
  and pause reset at render start. ~235 lines of duplication removed from the
  GUI.

- **refactor: extract transport-free `senmei-core` crate (2026-08-21)** — the
  shared probe/render/models/queue + license/confirm gates move from
  `senmei-server/src/core.rs` into a new `senmei-core` crate (no Tauri, no
  transport); `senmei-server` re-exports it, so the MCP/HTTP adapters are
  unchanged. Foundation for the GUI to delegate to the same core instead of
  duplicating it in `commands.rs`.

- **ui: drop model-download step from onboarding wizard (2026-08-21)** —
  `OnboardingWizard` is now welcome→ffmpeg→engine→done (models are added in
  Settings); removes `RECOMMENDED_MODELS`/`downloadModel` and the dead
  `onboard.model.*` i18n keys.

- **refactor: dedup burn/tch into shared `engine::core` (2026-08-21)** — the
  generic `Model<B>` enum, 13-branch `load_arch`, and `infer` /
  `infer_interp` (pad 32/16) / `infer_denoise` (FFDNet-σ, blind, DRUNet-σ-map
  pad 8) / `infer_rgb8` moved out of `burn/mod.rs` + `tch/mod.rs` into
  `src/engine/core.rs`, backend-generic over `B::FloatElem` (burn f16, tch
  f32). Both engines are now thin wrappers (per-engine: dlopen, `load_rife`,
  store adapters, converter stay). `engine.rs` → `engine/`. ~1030 lines of
  duplication removed; the f32-readback workaround (burn-bug-1) preserved.
  Verified: burn tests, tch GPU roundtrip, `senmei-server` links with tch.

- **ml: add 4× NomosWebPhoto RealPLKSR (2026-08-21)** — the flat
  `4xNomosWebPhoto_RealPLKSR.pth` (Phhofm, CC-BY-4.0, sha256-pinned) is
  GroupNorm(4) + a pixel-shuffle tail (no DySample, no LayerNorm; the ONNX
  `InstanceNormalization` + reshape `[N,4,16·H·W]` is just GroupNorm
  semantics). `RealPlk` gains a `dysample` flag (pixel-shuffle tail at
  scale>1) and the converter a `dysample=0` variant; converter also remaps
  `.norm.`→`.layer_norm.` for the LayerNorm variant (2× Public now converts —
  before it errored on the missing norm). Verified: 256²→1024² vs official
  ONNX mae 0.0007 (f16). Registry 35→36.

- **ml: add SPAN 2× HFA2k LUDVAE (2026-08-21)** — registers
  `span-2x-hfa2k-ludvae` (Phhofm `2xHFA2k_LUDVAE_SPAN`, CC-BY-4.0,
  sha256-pinned), the LUDVAE variant dropped 2026-08-20 for the cubek#519 f16
  bug. Flat channels-last pth (contiguous-preprocessed to convert); the
  `pad_k96` workaround applies at load. Verified: loads + infers 256²→512²,
  no NaN, plausible [0,1] output.

- **ml: cubek#519 workaround — pad 1×1 conv K=96→128, re-enable 48ch SPAN
  (2026-08-21)** — a f16 1×1 conv with 96 in-channels returns wrong values at
  H·W ≥ 32768 (upstream-issues.md §6). `Span::pad_k96` zero-pads every conv2
  weight into a K=128 conv + pads the input at forward (K=128 verified correct
  at N=76800). Measured: the padded path is not slower than the broken K=96
  (−9% on the conv; K=128 tiles better), so the workaround is perf-free. The 4
  disabled 48ch SPAN models (nomosuni-ldl/multijpg, HFA2k, ModernSpanimation
  V2) re-enabled. Covered by `pad_k96_pads_all_conv2` + the extended
  `conv1x1_repro` (PAD-vs-f32-reference check).

- **feat: SRVGGNetCompact port (2026-08-21)** — new `srvgg` arch in the burn
  + tch engines makes the two fast anime upscalers
  (`realesrgan-animevideo-x2/x4`, BSD-3 weights) loadable. Converter maps the
  flat torch body (16 convs + one shared PReLU) onto a `Vec<Conv2d>` + a
  single shared PReLU param. Torch-verified: x2/x4 mae ≈0.016 (f16, random
  weights).

- **docs: upstream answers on burn/cubecl issues (2026-08-21)** — burn#4950
  (ordering panic) fixed on burn `main` (#4962/#5282/#5400); burn#5382
  (GroupNorm f16) #5211 fixed accumulation, denominator division separate;
  cubecl#1531 (autotune OOM) verified on cubecl `main` — server no longer
  stays corrupted after a failed allocation, but the failed reserve still
  panics the server worker (recoverable). Statuses in `upstream-issues.md`.
  burn#4950 probe: the ordering panic does not reproduce in isolation on
  0.21.0 (300–400 fused readbacks + cycling autotune keys, both 0.21.0 and
  `main` clean) — it fired only as a secondary symptom of the #1531 OOM
  corruption, which no longer corrupts the server on cubecl `main`.

- **feat: first-run onboarding wizard (2026-08-21)** — a 5-step setup on
  first launch: welcome, FFmpeg check + download, inference-engine status,
  starter model download, get-started (dismissed once via a localStorage
  flag).

- **feat: model A/B compare (2026-08-21)** — render twice (e.g. with a
  different pipeline); the Monitor keeps the previous result and the new A/B
  tab shows both renders side by side.

- **feat: queue persistence + resume (2026-08-21)** — the batch queue is
  persisted to `<data-dir>/batch-queue.json` when a batch starts and pruned
  as jobs finish; after a restart a banner offers to resume the files that
  never completed (or discard the saved queue).

- **feat: content-aware pipeline defaults (2026-08-21)** — „Suggest“ in the
  stack panel probes the current file (anime vs live-action via a
  flatness/edge heuristic on sampled frames, input resolution, fps) and
  populates a suggested step chain (interpolation, upscaler choice, denoise
  for live footage).

- **feat: hardware encoders (2026-08-21)** — the encoder picker prefers
  verified hardware encoders (`hevc_nvenc`/VAAPI/QSV/AMF on Linux,
  VideoToolbox on macOS, HEVC before H.264); each is confirmed with a cached
  one-frame test encode, so GPU-less machines keep the LGPL-safe software
  chain (libkvazaar → libopenh264 → …). VA-API gets device + `hwupload` args.

- **feat: batch folder processing (2026-08-21)** — „Process folder…“ (Process
  menu) recursively scans a folder for videos via the shared
  `senmei_media::find_videos` (GUI + new `/api/scan-folder` REST endpoint) and
  enqueues every file for batch render.

- **feat: per-step FPS benchmark (2026-08-21)** — the pipeline times each
  step (ms/frame + fps) and reports it in the final render progress event
  (GUI Monitor panel) and in `RenderStatus.steps` (MCP/HTTP); a per-step
  summary is also logged at the end of every render.

- **feat: one-binary headless server (2026-08-21)** — the `senmei` binary
  embeds the built web UI (`rust-embed`) and runs the headless service with
  `--server` (HTTP + REST) / `--mcp-server` (stdio), reusing `senmei-server`
  core; `--server` sets `RUST_MIN_STACK` for burn model loads. Replaces the
  Tauri-sidecar idea.

- **ci: fix release notes generation (2026-08-21)** — git-cliff no longer
  fetches GitHub metadata (private-repo 404 panic); notes via
  `orhun/git-cliff-action@v4`, no pinned binary download.

- **ci: release bundles built with the tch (libtorch) engine (2026-08-21)** —
  `tauri build --features tch` (Linux/Windows; macOS skipped — no CUDA/ROCm)
  ships the CUDA/ROCm engine compiled in; the libtorch runtime itself is
  still downloaded on demand.

- **feat: senmei-server CLI via clap (2026-08-21)** — `--server`/`-s`,
  `--http-port`/`-p`, `--mcp-server`/`-m`, `--web-dir`; env fallbacks
  (SENMEI_HTTP/PORT/WEB_DIR) and `--http` alias kept.

- **ui: maximize/restore window-control icon (2026-08-21)** — toggles
  between a hollow square and the overlapping-rectangle restore glyph,
  synced via the window resize event.

- **ui: slim header to h-10 (2026-08-21)** — matches the Koharu/VS Code
  title-bar height (was h-12).

- **ci: harden build against runner disk/net flakes (2026-08-21)** —
  `CARGO_INCREMENTAL=0` + free check artifacts before tests (runner disk);
  retry `bun install` (network).

- **ci: split release bundle into its own job (2026-08-21)** — check+test and
  the bundle build each get a fresh runner (disk); `Free check artifacts`
  runs in bash on Windows.

- **feat: rotating log file (2026-08-21)** — Info+ records append to
  `<data-dir>/logs/senmei.log` (5 MB cap, 3 rotations) with module/file:line,
  so logs survive crashes (base for the diagnose export).

- **feat: one-click diagnose export (2026-08-21)** — Settings → Info packages
  `logs/*.log` + `diagnostics.json` (settings, backend, ffmpeg) as a
  `.tar.xz` for bug reports.

- **feat: pipeline templates (art presets) (2026-08-21)** — save/load named
  step chains in the stack panel (localStorage, all transports).

- **feat: model manager in Settings (2026-08-21)** — list installed weight
  files with size + sha256 check, delete to free disk.

## 0.1.4 (2026-08-21)

- **feat: logs panel + render timing (2026-08-21)** — the Logs panel gets a
  copy button (single click selects all), a backend `clear_logs` so Clear
  sticks (buffer emptied, no reload on re-mount), and stick-to-bottom
  scrolling that leaves the viewport alone once scrolled up. The render
  pipeline logs per-stage timing (process/encode ms) and the selected engine,
  making slow renders diagnosable.

- **feat: model download end-to-end (2026-08-21)** — `download_model` now
  handles every weight format (`.bpk`/`.pth`/`.onnx`/ncnn-`.bin`/release-zip
  with `extract_suffix`), logs start/result/errors, and skips
  already-present weights. The converter tolerates unnamed ONNX initializers
  and strips `params_ema.`/`params.` key prefixes (fixes
  realesrgan-x4plus-anime). Models expose a `downloaded` flag; every model
  step (upscale, decompress, denoise, deblur, interpolate) gets a download
  button + auto-download, and non-loadable models are disabled in the
  dropdown. Catalog: dropped anime1080fixer, marked the RealESRGANv2-animevideo-xs
  entries not loadable, wired RIFE v4.6 + IFRNet downloads.

- **feat: record model family lineage in the catalog (2026-08-21)** — every
  `metadata.json` entry now carries the `family` it descends from
  (real-esrgan, real-cugan, real-plksr, span, rife, …) alongside `arch`;
  `ModelMetadata.family` flows through the TS bindings, the model dropdown
  shows it, and `docs/models.md` gains Family + Arch columns.

- **docs: align PLAN/README with the optional tch backend (2026-08-20)** —
  binding decisions now note the optional libtorch engine (`tch` feature,
  runtime dlopen, CUDA/ROCm only) next to the shipped burn-Vulkan default.

- **ci: release notes via git-cliff (2026-08-20)** — the release job now
  generates notes from conventional commits (grouped Features/Bug Fixes/…,
  like Koharu) via `git-cliff` (`cliff.toml`) instead of an awk slice of the
  CHANGELOG; `--current` emits exactly the tagged release.

## 0.1.3 (2026-08-20)

- **fix: refresh stale model catalog from bundled metadata (2026-08-20)** —
  `ensure_catalog` now overwrites the data-dir `metadata.json` when the
  bundled one differs, so a packaged app picks up newly added models instead
  of keeping an old copy.

- **fix: About version derived from package.json (2026-08-20)** —
  `__APP_VERSION__` was hardcoded to 0.1.0 in `vite.config.ts`; it now reads
  `packages/app/package.json` so the About dialog shows the real version.

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
