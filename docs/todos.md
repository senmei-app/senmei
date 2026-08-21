# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Docs
- Documentation / user guide

## Findings
- libtorch on ROCm 7.14 / nightly
## Product / roadmap (2026-08-21)
- [ ] Onboarding wizard (first run): FFmpeg, engine check, first model
      downloads, license gate
- [ ] Auto-update via `tauri-plugin-updater` (signed bundles)
- [ ] Model A/B compare: two pipelines side by side (not only source vs result)
- [ ] RealESRGANv2-animevideo-xs (SRVGGNetCompact port, x2/x4) — enable the two
      fast anime upscalers (loadable: false today)

## Models (2026-08-21)
- [ ] anime1080fixer arch + license verify — removed from catalog 2026-08-21; revisit with the RRDBNet port


# Not yet
- [ ] Project website
- [ ] Flatpak: bundle target + Flathub publishing (FFmpeg via `org.freedesktop.Platform.ffmpeg-full`)

## gpu-allocator / windows 0.62 pin (revisit ~2026-02)
- [ ] Drop the `senmei-app/gpu-allocator` fork once webview2-com-sys/tao move
      `windows` to 0.62. Tracked: gpu-allocator #310 ("not planned"); no Tauri
      issue yet.

## Webview / media preview (revisit ~2026-11)
- [ ] Re-evaluate Tauri CEF backend for native media playback (VAAPI, no
      `asset://` limit) — would obsolete rodio. Stay WebKitGTK until then
      (PLAN §12).
## Compliance (2026-08-20)

> cargo-deny 0.20.2 scan (ORT not viable: Cargo analyzer OOM >20 GiB heap +
> config-loading bug on JDK 25). 0 CVEs, no GPL/AGPL/LGPL in tree.

- [ ] Unify `zip` duplicate: 2.4.2 (senmei-media) vs 8.6.0 (burn-store) —
      deferred 2026-08-20: touches `senmei-ml/Cargo.toml` + `Cargo.lock`
      (libtorch WIP), marginal win (0.6.6/7.2.0 transitives remain either way)
## Models

- [ ] RealPLKSR rest: NomosWebPhoto non-dysample port
- [ ] RealPLKSR 2× BHI small (anime 2×): port dim-32 RealPLKSR variant
      (SPAN successor)
- [ ] RealPLKSR 2× BHI large + 4xArtFaces: skip unless needed
      (dim-96 port; faces niche)



## Backend
- [ ] SPAN f16 degrades 48ch norm-on (ldl/multijpg/hfa2k 0.69–0.92 vs torch) —
      cause: cubek-convolution f16 1×1 conv bug K=96×N≥32768 (cubek#519,
      `docs/upstream-issues.md` §6). Affected 48ch models disabled in registry;
      re-enable once fixed upstream (still loadable: 64ch V1/V1.5 + BHI)
- [ ] License policy: opt-in toggle for non-commercial/unverified models (user
      decides, off by default)? — decision deferred 2026-08-20, hard gate stays

## Web / headless
- [ ] Audio in the web UI (senmei-server --http): currently no sound — no
      native `<video>` (no raw-file stream), rodio path is Tauri-only. Prefer
      option A: server streams the raw file with Range requests (new
      `/api/stream` endpoint) so the browser `<video>` plays video+audio;
      wire `nativeVideoUrl` in `http.ts` + `media-src` CSP. (decision
      2026-08-20: deferred, backlog)
- [ ] License policy (web): model download over HTTP uses POST without progress
      events — add progress once streaming lands (or poll download status)
- [ ] Web UI hardware/GPU status: `httpBackend.hardwareStatus` returns `null`
      (Tauri-only for now)
