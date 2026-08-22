# Senmei — Implementation Plan

> Rust re-implementation of a video enhancer modeled after REAL Video Enhancer (RVE).
> **Senmei (鮮明)** — GitHub: [senmei-app](https://github.com/senmei-app)
> GUI concept inspired by [Koharu](https://github.com/mayocream/koharu) and VS Code.

---

## 0. Vision

A fast, modern desktop video enhancer in Rust with:

- **Frame interpolation** (e.g. 24 → 48 fps) and **upscaling** (e.g. 1080p → 4K)
- **GPU inference**: **burn (`burn-wgpu`) on the Vulkan backend, fp16** with CPU fallback — default; optional **libtorch** backend behind the `tch` feature (runtime dlopen, CUDA/ROCm only). No ONNX Runtime, no TorchScript, no candle, no ncnn
- **Consistent HTML/CSS/JS UI** via platform webviews (webkit2gtk / WebView2 / WKWebView)
- **Better FFmpeg settings** than RVE (profile-based, extensible, validated)
- **Sample preview** of 10–60 s directly in the app

---

## 1. Agreed Decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Shell / UI host | **Tauri 2 + platform webview** (webkit2gtk / WebView2 / WKWebView), not CEF |
| 2 | Frontend | **React + TypeScript**, `react-resizable-panels`, Tailwind, lucide-react |
| 3 | Inference | **burn (`burn-wgpu`) on the Vulkan backend, fp16**, CPU fallback — default; optional **libtorch** (`tch` feature, runtime dlopen, CUDA/ROCm). No ONNX Runtime, no candle, no ncnn engine |
| 4 | No ONNX Runtime / no TorchScript | every arch is a **clean burn re-implementation**; weights from a permissive source (torch `.pth` → f16 `.bpk`, ONNX `.onnx` via a built-in reader, or ncnn `.bin` for RIFE) |
| 5 | No WebGPU/WASM | preview via native `<video>` where the webview can play the file; FFmpeg-decoded frame fallback (codec-agnostic, incl. H.265) |
| 6 | Media | **FFmpeg as subprocess** with `rawvideo` pipe; prefer **system FFmpeg**, fallback: portable download (BtbN builds) into data dir |
| 7 | Layout | **3-panel + timeline**: Input \| Monitor \| Settings |
| 8 | Codecs | in-app preview is **codec-agnostic** (FFmpeg decode → canvas, incl. H.265); final file freely selectable (x264/x265) |
| 9 | Phase-1 models | **RIFE** (interpolation) + **SPAN / Real-ESRGAN** (upscale) |
| 10 | Platform order | **Linux first** (AMD/ROCm), then Windows, then macOS |
| 11 | License | **MIT OR Apache-2.0** (Koharu-style), FFmpeg as **LGPL build**, no AGPL code |
| 12 | Name | **Senmei (鮮明)** · GitHub org `senmei-app` · binary `senmei` |

---

## 2. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Shell | Tauri 2 + platform webview | IPC/plugins/windows for free, small footprint |
| Frontend | React 18+ / TypeScript / Vite | simple, Koharu-compatible pattern |
| Package manager | **bun** | like Koharu (fast, bun lockfile) |
| UI kit | Base UI + Tailwind CSS + lucide-react | Koharu pattern, maintainable |
| IPC | Tauri commands + `tauri-specta` + `Channel` | typed bridge, streaming for preview |
| Inference | burn (`burn-wgpu`, Vulkan fp16) | one backend, GPU + CPU fallback; clean Rust ports |
| Media | FFmpeg subprocess (`rawvideo` pipe) | robust, full encoder selection |
| Async | tokio | worker threads for inference |
| Config | serde + JSON (`config.json` + profiles) | persistable, schema-validated |

---

## 3. GUI Concept

### 3.1 Layout (3-Panel + Timeline)

```mermaid
flowchart TB
    subgraph App
        direction TB
        A[Left: Input<br/>file browser · models · queue]
        B[Center: Monitor<br/>live preview · before/after<br/><br/>timeline below: in/out 10–60s]
        C[Right: Settings<br/>tabs: model · interpolate · upscale · FFmpeg · audio · advanced]
    end
```

| Panel | Width (default) | Content |
|---|---|---|
| **Left** | ~18 % | file browser, model library, render queue (batch) |
| **Center** | ~58 % | live monitor, before/after split, below it **timeline** |
| **Right** | ~24 % | **tabbed** settings (no endless scroll like RVE) |

- All panels **collapsible/resizable** (`react-resizable-panels`)
- **Timeline** with in/out markers, preset buttons `10s / 15s / 30s / 60s / custom`
- Settings tabs (Koharu pattern): `Model`, `Interpolate`, `Upscale`, `FFmpeg`, `Audio/Subtitles`, `Advanced`

### 3.2 Preview (no WebGPU/WASM)

| Task | Solution |
|---|---|
| Live monitor (last frame) | Rust → PNG → Tauri `Channel<PreviewFrame>` → 2D `<canvas>` |
| Before/after | two bitmaps, movable divider (CSS `clip-path`) |
| Sample playback | native `<video>` (hardware decode) where supported; else FFmpeg decodes frames → 2D canvas; audio via `<audio>` (AAC/Opus) |

---

## 4. Architecture

```mermaid
flowchart TB
    subgraph Frontend["packages/ (React + TS)"]
        UI[3-panel UI + timeline]
        BRIDGE[tauri-specta types]
    end

    subgraph Rust["crates/"]
        APP[senmei-app<br/>Tauri commands · state · channel]
        PIPE[senmei-pipeline<br/>orchestration · queue · events]
        ML[senmei-ml<br/>InferenceEngine: burn · Vulkan fp16]
        MEDIA[senmei-media<br/>FFmpeg process · frame pipe · probe]
    end

    FF[FFmpeg binary]

    UI -->|commands / channel| APP
    APP --> PIPE
    PIPE --> ML
    PIPE --> MEDIA
    MEDIA <-->|rawvideo pipe| FF
```

A **single process** — no Python-subprocess separation like in RVE.

---

## 5. Workspace Structure

```
senmei/
├─ Cargo.toml                 # workspace
├─ package.json               # frontend root (bun)
├─ .gitignore
├─ README.md
├─ crates/
│  ├─ senmei/                 # entry point, logging, diagnostics, version
│  ├─ senmei-app/             # Tauri commands, app state, IPC (Channel<PreviewFrame>)
│  ├─ senmei-pipeline/        # render orchestration, queue, events, progress
│  ├─ senmei-ml/              # InferenceEngine trait, burn engine (Vulkan fp16), model registry
│  ├─ senmei-media/           # FFmpeg process, frame decode/encode, video probe, encoder profiles
│  ├─ senmei-core/            # transport-agnostic core: probe/render/models/queue + gates
│  └─ senmei-server/          # headless service: MCP (stdio) + HTTP (axum)
├─ packages/
│  ├─ ui/                     # reusable UI kit (Base UI + Tailwind)
│  ├─ bridge/                 # generated types (tauri-specta)
│  └─ app/                    # React frontend (3-panel + timeline)
└─ models/                    # weight files (gitignored) + metadata.json
```

---

## 6. Inference Design (`senmei-ml`)

### 6.1 Engine Trait

```rust
pub trait InferenceEngine: Send + Sync {
    fn capabilities(&self) -> EngineCaps;            // tiles
    fn load(&mut self, model: &ModelRef) -> Result<()>;
    fn infer(&mut self, input: &Tensor, opts: &InferOptions) -> Result<Tensor>;
    // optional: fused RGB8 path (must tile internally), two-input interpolation (RIFE)
    fn infer_rgb8(&mut self, input: &Tensor, scale: u32) -> Option<Result<(Vec<u8>, u32, u32)>>;
    fn infer_interp(&mut self, a: &Tensor, b: &Tensor, t: f32, opts: &InferOptions) -> Option<Result<Tensor>>;
}
```

### 6.2 Engine

| Engine | Backend | Model format |
|---|---|---|
| `BurnEngine` | **burn (`burn-wgpu`), Vulkan fp16**, CPU fallback | f16 `.bpk` burnpack (`BurnpackStore`), or raw ncnn `.bin` (RIFE) |

- **One engine, clean Rust ports.** Every adopted arch is a **clean re-implementation** in `senmei_ml::burn` (UpCunet2x / UpCunet2xFast / RrdbNet / RifeNet) — never translated or copied from AGPL or unclear-license code. Weights and arch are separate licenses (recorded in `metadata.json`).
- Weights come from permissive sources: torch `.pth` converted once to an f16 `.bpk` burnpack (maintainer step, `senmei-ml-convert` / in-app `download_model`), or — for RIFE — the raw ncnn `flownet.bin` (MIT), parsed by the generated loader.
- Benchmark-verified on the target device (AMD RX 9070 / RDNA4): burn-Vulkan fp16 beats the alternatives and runs in half precision; see `docs/benchmarks.md`.

### 6.3 Backend Matrix (honest)

| Platform / GPU | burn-Vulkan fp16 | CPU fallback |
|---|---|---|
| NVIDIA (Win/Linux) | ✅ | ✅ |
| AMD Linux (Mesa/RADV) | ✅ | ✅ |
| AMD Windows | ✅ | ✅ |
| Intel Arc Linux | ✅ | ✅ |
| Apple Silicon | ⚠️ via MoltenVK (experimental) | ✅ |
| CPU-only | ❌ | ✅ |

### 6.4 Model Registry

```json
{
  "id": "rife-v4.6",
  "kind": "interpolate",            // interpolate | upscale | denoise | decompress | deblur
  "scale": 1,
  "arch": "rife46",
  "weights": ["flownet.bin"],
  "license": "MIT",
  "source_url": "https://github.com/nihui/rife-ncnn-vulkan",
  "loadable": true
}
```

---

## 7. Media Design (`senmei-media`)

### 7.1 FFmpeg as Subprocess

- **Decode**: `ffmpeg -ss <in> -i input -f rawvideo -pix_fmt rgb24 -` → frames via stdout
- **Encode**: processed frames via stdin → `ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r FPS -i - <encoder> output`
- Color conversion (YUV→RGB) is done by **`swscale`** directly during decode

### 7.2 "Better FFmpeg settings" (profile-based instead of hard-coded)

Instead of RVE's `if/else` encoder blocks, a **schema-driven profile system**:

```json
{
  "encoders": {
    "libx265": {
      "quality_model": "crf",
      "pixel_formats": ["yuv420p", "yuv420p10le", "yuv444p"],
      "presets": ["placebo", "slow", "medium", "fast", "veryfast"],
      "advanced": ["bf", "refs", "aq-mode", "psy-rd", "tune", "two_pass"]
    }
  },
  "profiles": {
    "output": { "encoder": "libx265", "quality": "high", "pixel_format": "yuv420p10le" }
  }
}
```

Features:
1. Profile presets (Lossless/Ultra/Very High/High/Medium/Low) per encoder
2. Advanced parameters (B-frames, refs, AQ, psy-rd, tune, 10-bit)
3. Two-pass (for target bitrate)
4. Free filter-chain field (e.g. `zscale`, `tonemap`) with live validation
5. HDR→SDR tone mapping + color metadata (colorprim/transfer/matrix)
6. Audio: AAC/Opus/FLAC/passthrough · subtitles: copy/srt/ass/webvtt
7. HW encoders with capability detection (NVENC/VAAPI/AMF/VideoToolbox)
8. **Output profile** (final file); in-app preview is frame-based (no encode needed)
9. Live command preview + validation

---

## 8. Pipeline / Data Flow

```mermaid
sequenceDiagram
    participant UI as UI (React)
    participant APP as senmei-app
    participant PIPE as senmei-pipeline
    participant MEDIA as senmei-media (FFmpeg decode)
    participant ML as senmei-ml (burn)
    participant ENC as senmei-media (FFmpeg encode)

    UI->>APP: startRender(Settings + SampleRange)
    APP->>PIPE: RenderRequest
    PIPE->>MEDIA: decode (seek to in point)
    loop Frame pairs
        MEDIA->>ML: RGB frame (float32/half)
        ML-->>ENC: interpolated/upscaled frame
        ML-->>UI: PNG preview via Channel (throttled)
    end
    ENC-->>UI: progress / ETA / FPS
```

---

## 9. Sample/Preview (10–60 s)

- Timeline with **in/out markers** + preset buttons
- Exact seek via FFmpeg (`-ss` after `-i`)
- Sample playback: native `<video>` (hardware decode) where the webview can play the file; FFmpeg frame fallback for everything else (codec-agnostic, incl. H.265); audio via `<audio>` (AAC/Opus)
- Button **"apply sample settings to full render"**
- Live monitor: last frame as PNG via `Channel`

---

## 10. Milestones

| # | Milestone | Content | Status |
|---|---|---|---|
| **M0** | **Scaffold** | workspace, cargo crates (empty/stub), Tauri shell, React 3-panel, `InferenceEngine` trait | ✅ done |
| **M1** | **FFmpeg passthrough** | decode → frames → encode end-to-end (no ML), first renderable chain | ✅ done |
| **M2** | **Upscaling** | SPAN/Real-ESRGAN via burn-Vulkan, tiling, progress | ✅ real upscale via **burn-Vulkan** (real-cugan-x2, e2e verified 1080p→2160p); NCNN plan superseded |
| **M3** | **Interpolation** | RIFE, scene-change detection, interpolation factor | ✅ RIFE v4.6 wired (burn port, ncnn `.bin` weights) |
| **M4** | **Settings** | FFmpeg profile system, command preview, audio/subtitles/HDR | ✅ profiles + preview + audio/subtitles + color metadata + HDR→SDR tonemapping done |
| **M5** | **Sample/Preview** | timeline in/out, 10–60 s sample, before/after, live monitor | ✅ live monitor + compare + timeline in/out sample presets done; native `<video>` + FFmpeg-frame fallback; audio via rodio (every codec) |
| **M6** | **Engine** | decided 2026-08-17: **burn-Vulkan fp16 is the shipped default**; no C++/ncnn shim | ✅ burn default, ncnn engine dropped |
| **M7** | **Advanced** | GMFSS/GIMM/IFRNet, model downloader, batch queue, reference filter stacks | 🟡 batch queue + filter stacks + model downloader/IFRNet done; more models/backends pending (todos) |
| **M8** | **Packaging** | model bundling/download, static FFmpeg, installer, auto-updater | 🟡 bundles (deb/rpm/AppImage/dmg/msi/nsis) + model catalog + release CI done (v0.1.1); auto-updater + static FFmpeg bundle pending |

> The M-numbers are stable feature labels, not a strict build sequence. The engine decision (2026-08-17) made **burn-Vulkan fp16 the shipped default**; the ncnn C++ shim was dropped. macOS is experimental (MoltenVK).

---

## 11. Risks

1. **Per-model port cost**: every arch is a clean Rust port (UpCunet2x, RrdbNet, RifeNet…) and must be numerically verified against a reference. Weights are **not bundled** — downloaded/converted on demand (`download_url` + `sha256` in `metadata.json`).
2. **burn maturity**: `burn`/`cubecl` is young — expect API churn on upgrade; a `burn-fusion` f32 bug crashes Vulkan at 1080p (fp16 path is fine); the build is heavy (~800 crates). Mitigated by pinning the burn version and running fp16.
3. **Preview codec**: HEVC is **not** supported in webviews — native `<video>` covers H.264/AAC, everything else (incl. H.265) falls back to FFmpeg-decoded frames, so any source codec plays.
4. **Single-backend risk**: burn-Vulkan covers all vendors; the CPU fallback keeps dev/render usable without Vulkan. macOS is experimental only.

---

## 12. Decided Points (after review)

| Point | Decision |
|---|---|
| Frontend build | **Vite** |
| Package manager | **bun** (like Koharu) |
| inference runtime | **burn (`burn-wgpu`), Vulkan fp16** (default); optional **libtorch** behind `tch` feature (runtime dlopen) — no ONNX Runtime / no candle / no ncnn |
| Engine (2026-08-20) | burn-Vulkan fp16 is the shipped default; ncnn C++ shim dropped; **libtorch: optional `tch` feature** — runtime-dlopen, CUDA/ROCm only, wrapper ABI must match the downloaded libtorch |

**Open evaluation (2026-08-19) — Tauri CEF backend (`feat/cef`):**
- **Status:** active branch (2026-08); Koharu already installs the CLI from it.
- **Gain:** Chromium rendering + VAAPI decode in the native `<video>` preview (WebKitGTK lacks it).
- **Cost:** pre-release; Chromium footprint.
- **Action:** revisit §1 „no CEF" when it matures. Not ruled out long-term, but no current need.
- **Decision (2026-08-19):** stay on **WebKitGTK** for now; re-evaluate CEF in a few
  months once `feat/cef` matures. The audio-streaming milestone (rodio→kira, live
  ffmpeg pipe) is **deferred with CEF** — a CEF switch would obsolete it entirely.
- **Interim (media):** WebKitGTK can't play media over Tauri's `asset://` scheme at all
  (GStreamer backend, `error 4` for every codec) — audio is played natively via **rodio**
  (FFmpeg-extracted MP3, full track; joins at the current position when ready), video via
  native `<video>` where possible + FFmpeg frame fallback.
- **Interim (decode):** VAAPI on the FFmpeg decode path (`-hwaccel vaapi`), under our control.

---

## 13. Next Steps

1. **End-to-end RIFE render** through the app (interpolate with `rife-v4.6`, verify the output).
2. **Numeric verification** of RIFE against the ncnn reference binary.
3. More upscalers/interpolators as clean burn ports; mac backend marked experimental.
4. Port + license-verify the backlog models — tracked in `docs/models.md` / `docs/todos.md`.

---

## 14. License

### 14.1 Own code & dependencies

**MIT OR Apache-2.0** (dual license like Koharu); **no AGPL code is adopted** — everything is cleanly re-implemented.

| Component | License | Note |
|---|---|---|
| Own code | **MIT OR Apache-2.0** | user chooses one of the two |
| FFmpeg | **LGPL build** (dynamically linked) | compatible with permissive license; **do not bundle a GPL build** |
| Encoder codec libs | `libkvazaar` **BSD-2-Clause** · `libopenh264` **BSD-2-Clause** | LGPL-safe encoders (see 14.3) |
| burn / cubecl / wgpu | **MIT / Apache-2.0** | inference engine + Vulkan backend |
| Tauri / tauri-specta | **MIT / Apache-2.0** | app shell + typed IPC |
| React / Vite / Base UI / Tailwind / lucide-react | **MIT** | frontend stack |
| tokio / serde | **MIT / Apache-2.0** | async runtime / serialization |
| liblzma (XZ) | **public domain (0BSD)** | `.tar.xz` project export |

### 14.2 Models (weights — separately published, permissive)

RVE itself is AGPL-3.0, but the models it runs are published independently under permissive licenses — we adopt only the **weights + architecture definitions**, never RVE code. **Adopted archs are clean Rust ports** (never translated from AGPL or unclear-license code); TAS's vendored code stays off-limits. `models/metadata.json` is the source of truth; the adopted-models matrix and the candidate backlog live in [`docs/models.md`](models.md).

Weights are **never committed** — only downloaded on demand; `metadata.json` holds id/kind/arch/scale + license/source_url.

### 14.3 Codecs (LGPL-only)

`libx264`/`libx265` need a **GPL** FFmpeg build and conflict with the LGPL-only rule. The encoder prefers LGPL-safe codecs first: `libkvazaar` (HEVC) → `libopenh264` → `h264_nvenc` → `libx264` (system GPL) → native `h264`; the portable download pins **BtbN `-lgpl` builds** (see `docs/CHANGELOG.md`, 2026-08-18).

### 14.4 AGPL boundary

**RVE and TAS are AGPL-3.0** — adopting any of their code parts (FFmpeg logic, conversion scripts, NCNN wrapper) would make the project AGPL-3.0, so they are always cleanly re-implemented.

**Optional:** if a weak copyleft for the own code is wanted later, **LGPL-3.0** is possible as an alternative — the default stays **MIT OR Apache-2.0**.

---

---

## 15. Current Status

> Short status snapshot — the full implementation log lives in
> [`docs/CHANGELOG.md`](CHANGELOG.md) (newest on top), the adopted model matrix in
> [`docs/models.md`](models.md), numbers in [`docs/benchmarks.md`](benchmarks.md).

- **Engine:** burn (`burn-wgpu`) **Vulkan fp16** (shipped default) — see `docs/benchmarks.md`.
- **Interpolation (M3):** RIFE v4.6 wired and verified end-to-end (clean burn port, `flownet.bin` weights).
- **Upscaling (M2):** real upscale on burn-Vulkan fp16 with tiling, verified 1080p→2160p (adopted models per `docs/models.md`).
- **Stacks (M7):** interpolation, upscale, **denoise/deblur/dedup (reference CPU)**, resize, output — all work; batch queue + progress done.
- **UI:** 3-panel + Inspector stack, drag&drop import, queue tab, monitor (native `<video>` + FFmpeg fallback, full-video mode), keyboard shortcuts, export/open project (`.tar.xz`).
- **Sample preview (M5):** timeline in/out presets + "Render Sample" range render; compare/result views.
- **Media/License:** LGPL-only FFmpeg + LGPL-safe encoder chain (§14.3); license gate for model download/use.
- **Next:** end-to-end RIFE render in the app, more model ports — backlog in `docs/todos.md` / `docs/models.md`.

---

## 16. MCP / AI-Agent Control (2026-08-19, status 2026-08-20)

> Status: **core loop done** — scaffold, settings schema, sample-compare, tool
> ranges and the e2e agent loop are all in (§16.3, §16.6); **plus an HTTP
> adapter + full web UI** (headless, no Wayland/X) — agents/browsers can drive
> Senmei over REST or the built UI as well as MCP (§16.7). A real Claude/ChatGPT
> client over MCP remains a manual follow-up.
> Goal: let AI assistants (ChatGPT, Gemini, Claude, …) drive Senmei over
> **MCP**: analyze a video, propose settings, render a sample, compare it
> against the original, then start the full render after user confirmation.

### 16.1 Workflow

```mermaid
sequenceDiagram
    participant AGENT as AI agent (MCP client)
    participant MCP as Senmei MCP server
    participant PIPE as senmei-pipeline
    participant METRIC as sample-compare (metrics)

    AGENT->>MCP: probe_video(input)
    MCP-->>AGENT: VideoInfo (dims/fps/duration/rotation/HDR)
    AGENT->>MCP: list_models() + settings schema
    MCP-->>AGENT: model matrix + StepParams schema
    AGENT->>MCP: render_sample(input, config, range)
    MCP->>PIPE: range render (start_ms..end_ms)
    PIPE-->>METRIC: sample clip + original
    METRIC-->>AGENT: PSNR/SSIM/VMAF + before/after frames
    AGENT-->>AGENT: iterate settings
    AGENT->>MCP: propose config (no auto-start)
    Note over AGENT,MCP: user confirms
    AGENT->>MCP: render(input, output, config)
```

### 16.2 Reuse (already exists)

| Capability | Source |
|---|---|
| Media probe (dims/fps/duration/rotation/HDR) | `probe_video` → `VideoInfo` |
| Range render (sample) | `render` with `start_ms`/`end_ms` |
| Model matrix + license gate | `list_models`, `metadata.json`, `license_blocked()` |
| Settings schema | `StepParams` / `PipelineStep` / `ProjectSettings` (specta → `bindings.ts`) |
| Encoder validation + fallback | `senmei-media` `pick_video_encoder` |

### 16.3 New work

| # | Piece | Status | Notes |
|---|---|---|---|
| 1 | **Headless entry point** | ✅ done | `crates/senmei-server`: transport-agnostic `core` + MCP stdio adapter (rmcp), no Tauri/GUI dep |
| 2 | **Sample-compare tool** | ✅ done | `render_sample` (range render + before/after PNGs as MCP image blocks) + `compare_sample` (PSNR/SSIM on the original res); VMAF deferred (libvmaf build) |
| 3 | **Settings JSON Schema** | ✅ done | `get_settings_schema` tool: schemars render-config schema (documented + ranged) + model slots + constraints |
| 4 | **Confirmation gate** | ✅ done | `propose_render` (validate+park) / `confirm_render` / `cancel_render`; async + `get_render_status` |
| 5 | **Tool allowlist + ranges** | ✅ done | `validate()` enforces the schema ranges on every param; tool set is whitelist-only (fixed tools, render behind the `render` feature) |

**Decision (2026-08-19):** headless crate = **`senmei-server`** — thin, transport-agnostic
`core` service (probe/render/models/queue + license/confirm gates) with adapters:
**MCP (stdio) first**; REST/HTTP as an optional cargo feature, added only when a real
consumer exists (YAGNI). MCP is a transport, not the core — an HTTP API later must not
require a refactor. Both gates live in `core`, so every transport gets them.

### 16.4 Constraints (from AGENTS.md)

- License gate applies to MCP too — the agent can only load permissive, verified weights (`license_blocked()` already enforced).
- No auto-start of long renders without confirmation.
- FFmpeg stays a subprocess; the MCP server uses the same `senmei-media` path (LGPL builds).
- No new inference backends — burn Vulkan fp16, CPU fallback.

### 16.5 Open questions

- **VMAF:** expensive, needs a `libvmaf` FFmpeg build — fall back to PSNR/SSIM if unavailable.
- **Subjective settings** (sharpness, denoise strength): the agent reasons from objective signals; user preference stays a prompt input.
- **Placement:** after release (post-M8), next to the project website (`docs/todos.md`).

### 16.6 Next steps (2026-08-20)

1. ~~**Settings JSON schema** (§16.3 #3)~~ ✅ done — `get_settings_schema`
   exposes the render-config JSON Schema + model slots + constraints.
2. ~~**Sample-compare** (§16.3 #2)~~ ✅ done — `render_sample` (range render +
   before/after PNGs) + `compare_sample` (PSNR/SSIM on the original resolution;
   VMAF deferred to a libvmaf build).
3. ~~**Tool allowlist + ranges** (§16.3 #5)~~ ✅ done — `validate()` now
   enforces the schema ranges (scale/fps/tonemap/dedup/resize/range); the tool
   set is already whitelist-only (fixed tools; render tools behind the
   `render` feature).
4. ~~**E2E agent loop**~~ ✅ done — ignored integration test `tests/agent_loop.rs`
   drives the full loop over stdio (probe → sample → compare → propose →
   confirm → poll); a real Claude/ChatGPT client remains a manual follow-up.

### 16.7 HTTP adapter + full web UI (2026-08-20)

**Decision (2026-08-20):** the HTTP adapter is no longer YAGNI — it is the
display-server-free path for browsers *and* other agents. One `core`, two
transports: **MCP (stdio)** for tool-driven agents, **HTTP (axum)** for the
web UI + REST. Both enforce the same license/confirm gates from `core`.

| Piece | Status | Notes |
|---|---|---|
| HTTP adapter (`--http`) | ✅ | axum 0.8 + tower-http; REST + static UI fallback |
| REST surface | ✅ | `/api/health`, `/api/models`, `/api/ffmpeg`, `/api/backend-info`, `/api/probe`, `/api/frame` (base64 PNG), `/api/download-model`, `/api/render` (+`/status`, `/cancel`), `/api/compare`, `/api/settings-schema` |
| Web UI (headless) | ✅ | frontend `backend/` abstraction — `tauri.ts` IPC / `http.ts` REST / `mock.ts` dev, auto-selected, no `isTauri()` in components |
| Path-input dialogs (Dateizugriff B) | ✅ | `PathDialog` for entering server-side paths in web mode (no native picker) |
| E2E verified | ✅ | browser against `--http`: import → probe → sample render → done; live progress + output file |

Run: `cargo run -p senmei-server --features render,http -- --http`
(port `SENMEI_HTTP_PORT`, web dir `SENMEI_WEB_DIR`; serves the built UI from
`packages/app/dist` by default).
