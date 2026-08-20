# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Backend
- [ ] SPAN f16 degrades 48ch norm-on (ldl/multijpg/hfa2k 0.69–0.92 vs torch) —
      cause: cubek-convolution f16 1×1 conv bug K=96×N≥32768 (cubek#519,
      `docs/upstream-issues.md` §6). Affected 48ch models disabled in registry;
      re-enable once fixed upstream (still loadable: 64ch V1/V1.5 + BHI)
- [ ] FFmpeg between-filter step (B linear): `Filter` Step in `Vec<Step>`,
      position pre/post/between; frame-preserving only (rawvideo pipe, 1:1)
- [ ] License policy: opt-in toggle for non-commercial/unverified models (user
      decides, off by default)? — decision deferred 2026-08-20, hard gate stays
- [ ] Runtime libtorch loading (like Koharu): resolve CUDA/ROCm libtorch at
      runtime via dlopen (`libloading`) + on-demand download to the data dir,
      instead of the build-time `LIBTORCH`/`download-libtorch` link. Pattern:
      `koharu-runtime` `hardware/{cuda,hip}.rs` (probe) + `runtime/loader.rs`
      (lazy `Library`) + `runtime/packages/torch.rs` (pinned download +
      `Store::directory` cache). **Scope: CUDA + ROCm only — NO CPU libtorch**
      (CPU stays on burn-Vulkan; decision 2026-08-20). TchEngine archs
      (upcunet/realesrgan/rife) stay; `TchDevice::Cpu` disabled.

## Web / headless
- [ ] Audio in the web UI (senmei-server --http): currently no sound — no
      native `<video>` (no raw-file stream), rodio path is Tauri-only. Prefer
      option A: server streams the raw file with Range requests (new
      `/api/stream` endpoint) so the browser `<video>` plays video+audio;
      wire `nativeVideoUrl` in `http.ts` + `media-src` CSP. (decision
      2026-08-20: deferred, backlog)
- [ ] License policy (web): model download over HTTP uses POST without progress
      events — add progress once streaming lands (or poll download status)

## Models

- [ ] RealPLKSR rest: NomosWebPhoto non-dysample port
- [ ] RealPLKSR 2× BHI small (anime 2×): port dim-32 RealPLKSR variant
      (SPAN-Nachfolger)
- [ ] RealPLKSR 2× BHI large + 4xArtFaces: skip unless needed
      (dim-96 port; faces niche)

## Release review (2026-08-19)

> 0.1.0 complete — see `docs/RELEASING.md`. macOS FFmpeg system-only (no LGPL
> prebuilt; `brew install ffmpeg`) — revisit if LGPL rule relaxes.

## Compliance (2026-08-20)

> cargo-deny 0.20.2 scan (ORT not viable: Cargo analyzer OOM >20 GiB heap +
> config-loading bug on JDK 25). 0 CVEs, no GPL/AGPL/LGPL in tree.

- [ ] Unify `zip` duplicate: 2.4.2 (senmei-media) vs 8.6.0 (burn-store) —
      deferred 2026-08-20: touches `senmei-ml/Cargo.toml` + `Cargo.lock`
      (libtorch WIP), marginal win (0.6.6/7.2.0 transitives remain either way)

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

