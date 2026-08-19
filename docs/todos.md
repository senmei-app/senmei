# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Backend
- [ ] burn macOS scaffold as an experiment (no guarantees)
- [ ] Autotune default: keep ON vs OFF vs vendor-patch (see `docs/upstream-issues.md` §2)
- [ ] SPAN: re-evaluate once the tch/libtorch f32 engine lands (f16 overflows,
      bf16 broken on RADV — port done, gated in `senmei-ml/src/burn/span.rs`)

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
- [ ] MCP server: AI agents auto-tune settings → sample → compare vs original →
      confirm → full render (PLAN.md §16)
- [ ] Flatpak: bundle target + Flathub publishing (FFmpeg via `org.freedesktop.Platform.ffmpeg-full`)

