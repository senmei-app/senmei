# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Backend
- [ ] SPAN f16 degrades 48ch norm-on (ldl/multijpg/hfa2k 0.69–0.92 vs torch) —
      cause: cubek-convolution f16 1×1 conv bug K=96×N≥32768 (cubek#519,
      `docs/upstream-issues.md` §6). Affected 48ch models disabled in registry;
      re-enable once fixed upstream (still loadable: 64ch V1/V1.5 + BHI)
- [ ] FFmpeg between-filter step (B linear): `Filter` Step in `Vec<Step>`,
      position pre/post/between; frame-preserving only (rawvideo pipe, 1:1)
- [ ] Hardware usage display (like Koharu): live GPU busy % (amdgpu
      `gpu_busy_percent` on RADV/AMD, nvidia-smi fallback) + CPU/RAM + render
      FPS in the UI (status bar / monitor)

## Models

- [ ] RealPLKSR rest: 1× DeJPG _60 (dl+sha256+verify); BHI-otf contiguous fix;
      NomosWebPhoto non-dysample port
- [ ] RealPLKSR 2× Public layernorm (realistic 2×): port LayerNorm PLKBlock
      variant (verify dim), then weights
- [ ] RealPLKSR 2× BHI small (anime 2×): port dim-32 RealPLKSR variant
      (SPAN-Nachfolger)
- [ ] RealPLKSR 2× BHI large + 4xArtFaces: skip unless needed
      (dim-96 port; faces niche)

## Release review (2026-08-19)

> 0.1.0 complete — see `docs/RELEASING.md`. macOS FFmpeg system-only (no LGPL
> prebuilt; `brew install ffmpeg`) — revisit if LGPL rule relaxes.

## gpu-allocator / windows 0.62 pin (revisit ~2026-02)
- [ ] Drop the `senmei-app/gpu-allocator` fork once webview2-com-sys/tao move
      `windows` to 0.62. Tracked: gpu-allocator #310 ("not planned"); no Tauri
      issue yet.

## Webview / media preview (revisit ~2026-11)
- [ ] Re-evaluate Tauri CEF backend for native media playback (VAAPI, no
      `asset://` limit) — would obsolete rodio. Stay WebKitGTK until then
      (PLAN §12).

## after release
- [ ] Project website
- [ ] Flatpak: bundle target + Flathub publishing (FFmpeg via `org.freedesktop.Platform.ffmpeg-full`)

