# AGENTS.md — Instructions for AI Agents

## Project

Senmei (鮮明) — video enhancer in Rust (frame interpolation + upscaling).
Source of truth for architecture & decisions: [`docs/PLAN.md`](docs/PLAN.md).

## Binding Decisions (do not change)

- **UI host:** Tauri 2 + platform webview (webkit2gtk / WebView2 / WKWebView) — no CEF
- **Frontend:** React + TypeScript + Vite · package manager `bun` · Base UI + Tailwind + lucide-react
- **Inference:** burn (`burn-wgpu`) on the **Vulkan backend, fp16**, CPU fallback — **no libtorch, no ONNX Runtime, no TorchScript, no candle, no ncnn**
- **Preview:** native `<video>` where the webview can load the file (hardware decode; H.264/AAC), FFmpeg-decoded frame fallback for everything else (codec-agnostic, incl. H.265) + `<audio>` (AAC/Opus) — **no WebGPU/WASM**
- **Media:** FFmpeg as subprocess with `rawvideo` pipe · source: **system FFmpeg preferred, portable download fallback** (pinned BtbN **LGPL** builds, dated tag + SHA — separate process, not bundled/linked; encoder picks LGPL-safe HEVC `libkvazaar`, H.264 fallbacks, no GPL `libx264` by default)
- **Pipeline:** composable `Vec<Step>` (`Upscale` + `Resize`) + a stateful interpolation stage (`Interpolator`) that emits blended intermediates (or duplicates across scene cuts) before the step chain
- **License:** MIT OR Apache-2.0 · FFmpeg only as **LGPL build**
- **Model arch = clean re-implementation** (engine-agnostic: applies to any
  burn/candle/ncnn/… port): re-implement every architecture from the spec or a
  permissively-licensed (MIT/Apache/BSD) reference — never translate or copy from
  AGPL or unclear-license code (RVE/TAS, TheAnimeScripter included). Weights and
  arch are **separate licenses**: a clean arch port does not relicense the
  weights; record each artifact's license in `metadata.json` and adopt only
  permissive weights.

## Change Policy

- **No backward compat:** when an API, schema, or config changes, update every in-repo consumer and remove the old form — no aliases, shims, or compatibility parsers.
- **Own responsibilities:** keep responsibilities self-contained; defaults belong to the owning component, not a central list of special cases.
- **Remove dead code:** delete dead abstractions and one-use helpers when direct code is clearer.
- **Generated code:** never hand-edit generated output (`packages/bridge/src/bindings.ts`, `crates/senmei/gen/schemas/`). Change the authoritative input and rerun the generator.
- **Patch upstream via fork, not vendoring:** when a dependency needs a small pin/patch (e.g. gpu-allocator's `windows 0.62` for wgpu-hal 29), fork it under the org and reference it via `[patch.crates-io]` — never vendor the crate into `third_party/` in this repo.

## Module Structure (follows with M0)

- `crates/senmei` — entry, Tauri app shell (`tauri.conf.json`, `build.rs`, `main.rs`), logging
- `crates/senmei-app` — Tauri commands, state, IPC (`Channel<PreviewFrame>`), specta builder (lib)
- `crates/senmei-pipeline` — orchestration, queue, events
- `crates/senmei-ml` — `InferenceEngine` trait, `BurnEngine` (Vulkan fp16), burn archs (`burn/realesrgan`, `burn/rife`, `burn/upcunet`, `burn/warp`), model registry
- `crates/senmei-media` — FFmpeg process, frame pipe, encoder profiles
- `packages/app` — React frontend (3-panel + timeline)
- `packages/ui` — reusable UI kit
- `packages/bridge` — generated types (tauri-specta)

## Conventions

- **Language:** docs & commits English; code identifiers English
- **Comments:** only when necessary, as short as possible; explain ownership, invariants, or deliberate divergence — don't narrate straightforward code.
- **Models:** `.pth` weights (converted to f16 `.bpk` burnpacks) + `metadata.json`
- **Never commit** credentials, model weights, datasets, or machine-specific artifacts.

## Docs

- **Language:** English everywhere (no German fragments).
- **Date everything** that can change: findings, decisions, eval notes get `(YYYY-MM-DD)`.
- **One truth per fact** — cross-reference, don't duplicate. PLAN.md = decisions & architecture; CHANGELOG.md = implementation log; benchmarks.md = numbers; models.md = model matrix; burn-bugs.md = upstream findings; todos.md = open backlog only.
- **Fixed per-file structure:**
  - `models.md`: rule → Adopted table → Backlog table → Sources → Notes.
  - `benchmarks.md`: decision + key numbers + environment at top; dated findings below; tables over prose.
  - `burn-bugs.md`: one `## Bug N` section per finding (Symptom / Reproducer / Root cause / Workaround / Status).
  - `todos.md`: only open items; completed items move to CHANGELOG.

## ML

- Keep a consistent load→infer lifecycle across models; separate weight loading from pre/post-processing.
- Take a device abstraction at the model boundary and convert once; disable gradient tracking during inference.

## Performance

- Benchmark on the real target device with representative inputs; remove redundant transfers/allocations before adding concurrency.
- Load and warm models outside measured regions.

## Commit Rules

- **One logical change per commit** — no mixed commits (e.g. never combine a feature + a fix + docs in one).
- **Conventional prefix:** `feat:` / `fix:` / `refactor:` / `ui:` / `docs:` / `test:` / `cleanup:`.
- **Keep it small:** if a commit would touch many unrelated files or topics, split it.
- **Trivial follow-ups** (typo fixes, small moves, version pins) get squashed into the related commit — don't create standalone micro-commits.
- **`Co-authored-by:` trailer** for the AI assistant on every commit.
- Keep `docs/CHANGELOG.md` in sync with the change in the same commit.

## Check

- `cargo check --workspace`
- Prefer the smallest relevant check in debug; don't rerun unchanged tests/builds.

## Build (follows with M0)

- `bun install`
- `cargo build --workspace`

## Run
- `bun run dev` — full app: `cargo tauri dev` (Vite + Rust, `tauri.conf.json` in `crates/senmei`)
- `bun run ui:dev` — frontend only

Keep this file focused on long-lived decision rules, not current implementation details. Engineering principles adapted from Koharu (MIT OR Apache-2.0).
