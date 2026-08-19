# Todos

> Open items only — completed items move to `docs/CHANGELOG.md`.

## Backend
- [ ] burn macOS scaffold as an experiment (no guarantees)
- [ ] Autotune default: keep ON vs OFF vs vendor-patch (see `docs/upstream-issues.md` §2)

## Release review (2026-08-19)
- [ ] Bundle the model catalog (`bundle.resources` + `models_dir()` in a packaged app)
- [ ] Scope broad IPC file ops to the `delete_project` allowlist pattern
- [ ] Harden ONNX reader (`onnx.rs`): bounds-checked `Result` instead of panics
- [ ] Replace `panic!`/`unreachable!()` with `Err` in `burn/mod.rs` + `decoder.rs`
- [ ] Add macOS portable-FFmpeg fallback (system FFmpeg only today)
- [ ] Code-split `mock.ts` out of the bundle; drop `#[allow(dead_code)]` in `tiling.rs`
- [ ] CI: fix `setup-bun@v2` `cache` input; bump checkout/upload-artifact to v5
- [ ] Release step: publish bundles to GitHub Releases on version tags

## after release
- [ ] Project website
- [ ] MCP server: AI agents auto-tune settings → sample → compare vs original →
      confirm → full render (PLAN.md §16)

