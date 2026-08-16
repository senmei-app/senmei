# AGENTS.md — Instructions for AI Agents

## Project

Senmei (鮮明) — video enhancer in Rust (frame interpolation + upscaling).
Source of truth for architecture & decisions: [`docs/PLAN.md`](docs/PLAN.md).

## Binding Decisions (do not change)

- **UI host:** Tauri 2 + platform webview (webkit2gtk / WebView2 / WKWebView) — no CEF
- **Frontend:** React + TypeScript + Vite · package manager `bun` · Base UI + Tailwind + lucide-react
- **Inference:** `tch` (libtorch) + NCNN/Vulkan — **no ONNX Runtime**
- **Preview:** 2D canvas fed by FFmpeg-decoded frames (codec-agnostic, incl. H.265) + `<audio>` (AAC/Opus) — **no WebGPU/WASM**
- **Media:** FFmpeg as subprocess with `rawvideo` pipe
- **Pipeline:** composable `Vec<Step>` (phase 1: only `Interpolate` + `Upscale`)
- **License:** MIT OR Apache-2.0 · FFmpeg only as **LGPL build**
- **No code** from RVE/TAS (AGPL-3.0) — clean re-implementation

## Module Structure (follows with M0)

- `crates/senmei` — entry, logging, version
- `crates/senmei-app` — Tauri commands, state, IPC (`Channel<PreviewFrame>`)
- `crates/senmei-pipeline` — orchestration, queue, events
- `crates/senmei-ml` — `InferenceEngine` trait, `TorchEngine`, `NcnnEngine`, model registry
- `crates/senmei-media` — FFmpeg process, frame pipe, encoder profiles
- `packages/app` — React frontend (3-panel + timeline)
- `packages/ui` — reusable UI kit
- `packages/bridge` — generated types (tauri-specta)

## Conventions

- **Language:** docs & commits English; code identifiers English
- **Comments:** only when necessary, as short as possible
- **Commits:** `Co-authored-by` trailer for the AI assistant
- **Models:** TorchScript `.pt` (libtorch) + `.param/.bin` (NCNN) + `metadata.json`

## Check

- `cargo check --workspace`


## Build (follows with M0)

- `bun install`
- `cargo build --workspace`

## Run
- `bun run dev`
