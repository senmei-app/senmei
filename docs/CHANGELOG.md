# Senmei — Changelog

> Implementation log (was §15 of PLAN.md). Newest on top.

> Kept in sync with actual implementation. Update on every significant change.

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
