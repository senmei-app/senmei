# Senmei (鮮明)

**Senmei** (Japanese for "clear, vivid, distinct") is a modern desktop video enhancer in Rust — frame interpolation and AI upscaling, modeled after [REAL Video Enhancer](https://github.com/TNTwise/REAL-Video-Enhancer).

## Status

� **Active development** — milestones **M0–M6 done**, M4/M5/M7 partially. The
full plan lives in [`docs/PLAN.md`](docs/PLAN.md); the implementation log is in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md).

Current state:
- **Upscaling (M2):** real models on **burn-Vulkan fp16** with tiling —
  real-cugan-x2, Fallin Soft/Strong, 4x-Alchemy, Real-ESRGAN (verified
  1080p→2160p).
- **Interpolation (M3):** **RIFE v4.6** (clean burn port) with scene-change
  detection.
- **Processing stack (M7):** interpolate, upscale, denoise/deblur/dedup,
  resize, output; batch queue + progress.
- **Sample/Preview (M5):** timeline in/out + "Render Sample", compare/result
  views.
- **Engine (M6):** burn-Vulkan fp16 is the shipped default; no ncnn/C++ shim.

## Tech Stack (short version)

| Layer | Choice |
|---|---|
| UI | Tauri 2 + platform webview · React + TypeScript + Vite · bun |
| Inference | burn (`burn-wgpu`) · Vulkan fp16 |
| Media | FFmpeg (subprocess, `rawvideo` pipe) |
| License | MIT OR Apache-2.0 · FFmpeg as LGPL build |

## Links

- Plan: [`docs/PLAN.md`](docs/PLAN.md)
- GitHub: [senmei-app/senmei](https://github.com/senmei-app/senmei)

## License

Own code dual-licensed under **MIT OR Apache-2.0** (details in [`docs/PLAN.md`](docs/PLAN.md), section 14).
