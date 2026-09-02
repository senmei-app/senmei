# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Docs
- Documentation / user guide

## Findings
- libtorch on ROCm 10.0 / stable
## Product / roadmap (2026-08-21)
- [ ] Auto-update via `tauri-plugin-updater` (signed bundles)

## Models (2026-08-21)
- [ ] anime1080fixer arch + license verify — removed from catalog 2026-08-21; revisit with the RRDBNet port

## Models (2026-08-23)
- [ ] Adopt Real-CUGAN up2x conservative (`cugan_up2x-latest-conservative.pth`) — same UpCunet2x arch as real-cugan-x2; balanced preset for real film (Pro-conservative already adopted as real-cugan-pro-conservative-x2)


# Not yet
- [ ] Project website
- [ ] Flatpak: bundle target + Flathub publishing (FFmpeg via `org.freedesktop.Platform.ffmpeg-full`)

## gpu-allocator / windows 0.62 pin (revisit ~2026-02)
- [ ] Drop the `senmei-app/gpu-allocator` fork once webview2-com-sys/tao move
      `windows` to 0.62. Tracked: gpu-allocator #310 ("not planned"); no Tauri
      issue yet.

## Webview / media preview (revisit ~2026-11)
- [ ] Re-evaluate webview backend (CEF VAAPI / Servo canvas) — media is now
      engine-independent via our own FFmpeg pipeline (PLAN §18); switch only
      if hardware decode or canvas throughput demands it

## Preview / Media (2026-08-23, PLAN §18)

> 2026-08-26: warm streams + last-frame-wins landed; web audio (Range-stream
> + transcoded Vorbis/Ogg `<audio>`) done. Phase-3 ring buffer **dropped**
> (warm streams + ±300 ms tolerance already cover scrubbing; a buffer adds
> complexity without real gain) and the per-viewport DPR budget **deferred**
> (the fixed 1280 cap is fine except HiDPI fullscreen).

## Compliance (2026-08-20)

> cargo-deny 0.20.2 scan (ORT not viable: Cargo analyzer OOM >20 GiB heap +
> config-loading bug on JDK 25). 0 CVEs, no GPL/AGPL/LGPL in tree.

## Models

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
- [ ] Fused RGB8 1080p×4 single-allocation OOM (~3.2 GB, tile/autotune
      independent) — VRAM guard now rejects it with a clear error (no CPU
      fallback); root cause is a wgpu/burn internal buffer. Deep burn fix or
      chunked readback needed to actually render 1080p×4 fused
- [ ] Drop in-repo workarounds on the next burn bump: custom `group_norm`
      helper (burn#5410 merged — native mean_dim) + `Span::pad_k96` (cubek#519,
      once fixed); re-check tiling/pipelining after the bump (2026-08-22)

## Auto-Enhance (2026-08-23)
- [ ] Phase 1: `QualityProfile` seam + code analyzers (noise/banding/blocking/sharpness) in senmei-core; replaces suggest_pipeline
- [ ] Phase 2: FACTOR port (NIMA fallback) behind QualityProfile — benchmark vs code first, adopt only if it wins
- [ ] Auto-Enhance transport-agnostic (Tauri + MCP + HTTP; suggest is Tauri-only today)

## Testing (2026-08-23)
- [ ] Frontend: Vitest + React Testing Library, erste Smoke-Tests (step builder, suggest mapping) — UI komplett ungetestet

- [ ] CI: optionaler Workflow für `#[ignore]` GPU/Modell-Tests (torch↔burn mae) — laufen nur lokal
- [ ] senmei-ml: Arch-Unit-Tests für real_plksr (0; auch upcunet/scunet/dncnn)

## Web / headless
- [ ] License policy (web): model download over HTTP uses POST without progress
      events — add progress once streaming lands (or poll download status)
- [ ] Web UI hardware/GPU status: `httpBackend.hardwareStatus` returns `null`
      (Tauri-only for now)

## Refactor (2026-08-23)
- [ ] Dedup feather ramp — shared `feather_ramp()` in tiling.rs for the fused
      (engine/core.rs) + CPU (tiling.rs) stitch — deferred (kein Tiling-
      Experimentieren gerade)

## Refactor / file size (2026-09-02)
> Mechanische Splits auf `suggest-and-tests` committed (Tests aus
> burn/encoder/commands/steps; core.rs-God-File → core/{config,compare,
> download,render}). Noch offen:
- [ ] `engine/core.rs` (1073) → `engine/core/` (model/tensor/infer/rgb8) —
      heißer Inferenz-Pfad: eigener Pass + Benchmark vorher/nachher
- [ ] `http/mod.rs` (~530) — Media-/Stream-Handler raus wäre nett, aber
      blockiert: axum `Handler` bricht für die Whole-Request-Handler
      (`stream`/`audio`) sobald Katalog/Media in Submodule wandern (2026-09-02
      getestet, zurückgerollt). Erledigt: render.rs + tests.rs sind raus.
- [ ] `senmei-ml/src/convert.rs` (~767) — Konverter je Format splitten
- [ ] `core/render.rs` (589) — Subsplit (run/lifecycle) auf <400
- [ ] `encoder/mod.rs` (669) + `commands/mod.rs` (641) — Code-Splits nach
      Test-Auslagerung
- [ ] Frontend `Monitor.tsx` (996) → Hooks (playback/hotkeys/scrub) +
      Overlays; Ziel ~300, tsc-Verify
- [ ] Frontend `App.tsx` (665), `StepEditor` (588), `SettingsPage` (542),
      `Inspector` (486) — Ziel ~300
