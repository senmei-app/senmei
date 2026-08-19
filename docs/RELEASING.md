# Releasing

How to cut a Senmei release. The CI builds bundles and publishes them on
version tags — no manual packaging. Process fits `.github/workflows/ci.yml`.

## Process

1. **Bump the version** in `crates/senmei/tauri.conf.json` and the workspace
   `Cargo.toml` (crate version tracks the app), and add a release entry to
   `docs/CHANGELOG.md` (newest on top).
2. **Pre-flight** (all green before tagging):
   - `cargo check --workspace` + `cargo test --workspace` (CPU/unit tests).
   - Frontend: `bun install --frozen-lockfile` + `bun run build` (tsc + vite).
   - GPU smoke test on a real device (Vulkan fp16): upscale + interpolate +
     denoise/deblur sample render.
   - Packaged app: model catalog present (`models/metadata.json` bundled),
     `download_model` works (weights download on demand).
   - FFmpeg: system FFmpeg works; portable download on Linux/Windows (macOS is
     system-only — LGPL policy, see `docs/CHANGELOG.md`).
3. **Tag + push**:
   ```sh
   git tag v0.1.0 && git push origin v0.1.0
   ```
4. **CI** builds `bundle/*` for Linux/Windows/macOS, uploads them as artifacts,
   and the `release` job publishes a GitHub Release (bundles attached, notes
   auto-generated from the tag).
5. **Post-release**: verify the downloaded bundles run on each platform; log any
   follow-ups in `docs/todos.md`.

## Constraints

- **FFmpeg LGPL-only** (PLAN §14.3): portable builds are BtbN `-lgpl`
  (Linux/Windows); macOS relies on system/Homebrew FFmpeg.
- **Weights never bundled**: downloaded on demand from `models/metadata.json`.
- **macOS is experimental** (Metal backend, no GPU tests on hosted runners).
