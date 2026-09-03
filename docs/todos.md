# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Docs
- Documentation / user guide

## Tooling (2026-09-03)
- [ ] Pre-commit hook: warn if `.rs` files changed but `docs/CHANGELOG.md` not staged (AI agents forget this regularly)


# Not yet
- [ ] Project website
- [ ] Flatpak: bundle target + Flathub publishing (FFmpeg via `org.freedesktop.Platform.ffmpeg-full`)


## Preview / Media (2026-08-23, PLAN §18)

> 2026-08-26: warm streams + last-frame-wins landed; web audio (Range-stream
> + transcoded Vorbis/Ogg `<audio>`) done. Phase-3 ring buffer **dropped**
> (warm streams + ±300 ms tolerance already cover scrubbing; a buffer adds
> complexity without real gain) and the per-viewport DPR budget **deferred**
> (the fixed 1280 cap is fine except HiDPI fullscreen).

## Compliance (2026-08-20)

> cargo-deny 0.20.2 scan (ORT not viable: Cargo analyzer OOM >20 GiB heap +
> config-loading bug on JDK 25). 0 CVEs, no GPL/AGPL/LGPL in tree.

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
      (engine/core.rs) + CPU (tiling.rs) stitch — deferred (no tiling
      experimentation right now)

## Refactor / file size (2026-09-02)
- [ ] `http/mod.rs` (~530) — further media/stream extraction would help but is
      blocked: axum `Handler` breaks for the whole-request handlers
      (`stream`/`audio`) once catalog/media move into submodules (tested
      2026-09-02, reverted). Done: render.rs + tests.rs already extracted.
- [ ] `engine/core.rs` (1073) → `engine/core/` (model/tensor/infer/rgb8) —
      hot inference path: dedicated pass + before/after benchmark
- [ ] `senmei-ml/src/convert.rs` (~767) — split converters per format
- [ ] `core/render.rs` (589) — sub-split (run/lifecycle) below 400
- [ ] `encoder/mod.rs` (669) + `commands/mod.rs` (641) — code splits after the
      test extraction
- [ ] Frontend `Monitor.tsx` (996) → hooks (playback/hotkeys/scrub) +
      overlays; target ~300, tsc-verified
- [ ] Frontend `App.tsx` (665), `StepEditor` (588), `SettingsPage` (542),
      `Inspector` (486) — target ~300
