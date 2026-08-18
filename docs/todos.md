# Todos

## AI Stack
- [x] Implement processing stacks (denoise / deblur / dedup — reference CPU)
- [x] Model backlog in `models.md`: RIFE/GMFSS, SCUNet/DRUNet, SRVGGNet/SPAN. Re-SISR from „Adore" CC-BY-NC-SA → blocked
- [x] Source: https://github.com/chaiNNer-org/spandrel (permissive arch reference, documented)
- [x] Depth Map / Detection / Stabilization: **no** — no ML workflow; stabilization only classic via OpenCV (Apache-2.0)
- [x] Upscaler perf (25→12 fps): cause tiling (512px, 329 ms) — price for the autotune-OOM fix; 1024px regression → keep 512px
- [x] Tile re-tuning after GPU-stitch: 640px = sweet spot (186.1 ms / 5.4 FPS) — default 512→640, 768 regresses

## Backend
- [x] burn-tch backend: ROCm-Nightly, RDNA4 fp16; vendored `third_party/` + `[patch.crates-io]`. Open: Fallin bench, app wiring
- [x] Dedup no longer collapses static material (cap on consecutive drops)
- [ ] burn macOS scaffold as an experiment (no guarantees)
- [x] Tiled-fused-RGB8: overlap `tile/8` = regression (394→329 ms) — keep `tile/4`; GPU stitching open
- [x] GPU stitching: accumulate tiles on the GPU (`slice_assign` overlap averaging), one readback instead of 15 — 329→234.7 ms / 4.3 FPS (fallin-soft)
- [x] Remove dead code: `tiling::tile`, `Error::Unimplemented`, `Registry::from_json`, `Decoder::open`, `#[allow(dead_code)]`
- [x] Remove engine-trait plumbing: `name()`, `EngineCaps.backend/half`, `InferOptions.half`, `Backend` enum never read
- [x] Remove unused deps from `senmei-app`: `base64`, `tauri-plugin-dialog`, `tauri-plugin-opener`
- [x] `extract_frame` test-only — smoke test switched to `encode_png`, re-export removed
- [x] Remove dead IPC command `remember_project` (internal `store::remember_project` stays)
- [x] Move `num_block` default out of `commands.rs` into model metadata/converter
- [x] Trim redundant comments: `Monitor.tsx`, batch comment, `.pth`-/asset-protocol comment
- [x] Sample rendered the whole queue instead of only the monitor video — fixed: `startBatch` takes an explicit file list
- [x] Rotation: `probe` reports display dims + `rotation`; `Decoder` -noautorotate + transpose

## UI
- [x] Export project as .tar.xz + „Open Project" loads the archive (Save As removed)
- [x] Drop box only when empty + full height; drag & drop videos everywhere
- [x] Remove arrows between the stacks
- [x] Video name centered (`project / video`), no box
- [x] Settings button bottom left (status bar)
- [x] About page (Help → About dialog with version/engine/license/GitHub)
- [x] Hotkeys: Ctrl/Cmd+O/+A/+E/+R, Delete, Space; menu shows shortcuts; workspace-only
- [x] Version bottom right (status bar + start screen)
- [x] Full-video mode via double-click on the monitor, exit ✕/Esc (all three modes)
- [x] Deduplication: presets (Off/Standard/Aggressive) + slider with % + hint
- [x] Menu: add View for full video mode
- [x] Settings: hotkey settings (view/change, Koharu-style)
- [x] Right side: tab bar next to „Processing Stack" with „Logs" tab (system log)
- [x] About dark theme (dark styles present)

## Docs
- [x] Remove ncnn engine completely from todos/plan (burn is default)
- [x] PLAN.md §15 → moved to docs/CHANGELOG.md
- [x] PLAN.md updated/redesigned
- [x] models.md more readable (status overview + backlog/candidate table)
- [x] benchmarks.md more readable (TL;DR box with decision + key numbers)
- [x] AGENTS.md: generated-code path fixed (`crates/senmei/gen/schemas/`), commit rule → CHANGELOG

## License
- [x] Remove `shuffle-cugan` (unclear/SUDO) → Fallin Soft/Strong + 4x_Alchemy; default: `real-cugan-x2`
- [x] `license_blocked()`: blocks verify/unclear + GPL/LGPL/AGPL + CC-BY-NC/ND/SA, enforced in both commands
- [x] Review gate: „verify → loadable" never unlocks unclear licenses (test `license_gate_blocks_unclear_and_copyleft`)
- [x] FFmpeg download pinned to BtbN `-lgpl` builds + SHA256 (tag `autobuild-2026-08-17-13-05`)
- [x] LGPL-safe encoders: `pick_video_encoder` — libkvazaar → libopenh264 → h264_nvenc → libx264 → h264
- [x] AGENTS.md: media bullet updated to BtbN LGPL builds + libopenh264

## Models
- [x] Fallin arch: hand port instead of burn-onnx codegen (UpCunet2x_fast, pad 38, ONNX-verified)
- [x] Runtime ONNX→fp16-bpk: `senmei_ml::onnx` (dependency-free) + `convert_onnx_to_bpk` + `download_model` branch
- [x] RealPLKSR port → `4x-alchemy` + `real-plksr-deh264/dejpg` loadable (numerically verified)
- [x] Bench: Fallin Soft/Strong vs. real-cugan (1080p→2160p): 176/177 vs 380 ms; fusion panic fixed

## Maintainability (review 2026-08-18)

- [ ] Split large files: `App.tsx` (done: `useBatch` + `RightPanel`), `Inspector.tsx`, `commands.rs`
- [x] CPU steps: `step.rs` sliced planar, FFmpeg provides packed `rgb24` — layout conflict checked/fixed (fixed: packed rgb24)
- [x] Unify duplicate arg parsing: `splitArgs` (TS) and `split_ffmpeg_args` (Rust) — one parser (frontend sends a split array)
- [x] Frontend paths: replace manual `/` splits with platform-safe helpers (Windows)
- [x] Align codec mapping: frontend H.264→libopenh264/H.265→libkvazaar (LGPL-safe) + `-c:v` override in the encoder
- [x] README: „planning phase / M0" → current state (M2–M5)
- [x] `todos.md` fully in English (AGENTS rule: docs in English)

## CI / Packaging
- [ ] Tauri security: evaluate CSP + asset scope `$HOME/**` (media access vs. surface)
- [x] AGENTS.md path checked: `crates/senmei/gen/schemas/` exists (build-generated, gitignored) — AGENTS.md correct

## shortly before release
- [ ] GitHub runner: build + test packages for Windows / Linux / macOS (macOS runner: compile + CPU tests, no Metal on hosted runners)

## after release
- [ ] Project website
- [ ] Make tile size configurable (backlog, requirements open)
- [x] Follow-up: burn feature request „load ONNX initializer" filed (`tracel-ai/burn-onnx#456`), replace own parser later

