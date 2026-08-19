# Releasing

How to cut a Senmei release. The CI builds bundles and publishes them on
version tags — no manual packaging. Process fits `.github/workflows/ci.yml`.

## Process

1. **Bump the version** in `crates/senmei/tauri.conf.json` and the workspace
   `Cargo.toml` (crate version tracks the app), and add a
   `## x.y.z (YYYY-MM-DD)` heading + entry to the top of `docs/CHANGELOG.md`.
   The GitHub release notes are generated from that section (step 4).
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
   generated from the CHANGELOG section above the latest `## x.y.z` heading).
5. **Post-release**: verify the downloaded bundles run on each platform; log any
   follow-ups in `docs/todos.md`.

## Constraints

- **FFmpeg LGPL-only** (PLAN §14.3): portable builds are BtbN `-lgpl`
  (Linux/Windows); macOS relies on system/Homebrew FFmpeg.
- **Weights never bundled**: downloaded on demand from `models/metadata.json`.
- **macOS is experimental** (Metal backend, no GPU tests on hosted runners).

## crates.io

Publishing to crates.io is **optional and deferred** — not every release goes
there. Do it once the API is considered stable (no downstream consumers yet,
pre-1.0, and `senmei-ml`/`senmei-app` carry a `gpu-allocator` fork caveat for
Windows downstream builds). When ready: `cargo publish` in dependency order
`senmei-media → senmei-ml → senmei-pipeline → senmei-app → senmei`.
