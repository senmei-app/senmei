# Senmei (鮮明)

**Senmei** (Japanese for "clear, vivid, distinct") is a modern desktop video enhancer in Rust — frame interpolation and AI upscaling, modeled after [REAL Video Enhancer](https://github.com/TNTwise/REAL-Video-Enhancer).

## Status

🟡 **Planning phase** — the full implementation plan lives in [`docs/PLAN.md`](docs/PLAN.md). Next step is **M0 (Scaffold)**.

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
