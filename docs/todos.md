# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Backend
- [ ] burn macOS scaffold as an experiment (no guarantees)
- [ ] Autotune default: keep ON vs OFF vs vendor-patch (see `docs/upstream-issues.md` §2)

## Release review (2026-08-19)

> Complete for 0.1.0 — see `docs/RELEASING.md`. macOS portable FFmpeg stays
> **system-only**: no LGPL-compatible macOS prebuilt exists (evermeet.cx/
> osxexperts are GPL, conflicting with the LGPL-only policy); `download_ffmpeg`
> points to `brew install ffmpeg`. Revisit if the LGPL rule is ever relaxed.

## gpu-allocator / windows 0.62 pin (revisit ~2026-02)
- [ ] Drop the `senmei-app/gpu-allocator` fork (`[patch.crates-io]`) once the
      Tauri webview stack (webview2-com-sys/tao) moves `windows` to 0.62 —
      then gpu-allocator + wgpu-hal 29 unify on 0.62 with no fork. Tracked
      upstream: gpu-allocator #310 (closed "not planned"); no Tauri issue yet.

## Webview / media preview (revisit ~2026-11)
- [ ] Re-evaluate the Tauri CEF backend (`feat/cef`) for native media playback
      (VAAPI, no `asset://` limitation). Would obsolete the rodio audio path +
      the audio-streaming idea. Stay on WebKitGTK + rodio until then (PLAN §12).

## after release
- [ ] Project website
- [ ] MCP server: AI agents auto-tune settings → sample → compare vs original →
      confirm → full render (PLAN.md §16)
- [ ] Flatpak: bundle target + Flathub publishing (FFmpeg via `org.freedesktop.Platform.ffmpeg-full`)

