# Todos

## AI Stack
- [x] Stacks implementieren (denoise / deblur / dedup — Referenz-CPU)
- weiter Modelle definieren für:
  - Interpolation Models
  - Denoising
  - Restoration Models
    - quelle:
      - https://github.com/chaiNNer-org/spandrel
- macht folgendes Sinn:
  - Depth Map Models
  - Object Detection Models
  - Video Stabilization
- Upscaler-Performance eingebrochen (25→12 fps) — Tile per Settings dynamisch einstellen?

## Backend
- [x] burn-tch Backend: ROCm-Nightly, RDNA4 fp16; vendored `third_party/` + `[patch.crates-io]`. Offen: Fallin-Bench, App-Anbindung
- [ ] burn für macOS Grundgerüst als Experiment (keine Garantie)
- [x] Tiled-Fused-RGB8: Overlap `tile/8` = Regression (394→329 ms) — `tile/4` behalten; GPU-Stitching offen
- [x] Toten Code entfernen: `tiling::tile`, `Error::Unimplemented`, `Registry::from_json`, `Decoder::open`, `#[allow(dead_code)]`
- [x] Engine-Trait-Plumbing entfernt: `name()`, `EngineCaps.backend/half`, `InferOptions.half`, `Backend`-Enum nie gelesen
- [x] Unbenutzte Dependencies aus `senmei-app` entfernen: `base64`, `tauri-plugin-dialog`, `tauri-plugin-opener`
- [x] `extract_frame` Test-only — Smoke-Test auf `encode_png` umgestellt, Re-Export entfernt
- [x] Totes IPC-Command `remember_project` entfernt (internes `store::remember_project` bleibt)
- [x] `num_block`-Default aus `commands.rs` in Modell-Metadaten/Converter verlagern
- [x] Überflüssige Kommentare gekürzt: `Monitor.tsx`, Batch-Kommentar, `.pth`-/Asset-Protocol-Kommentar
- sample mit anderen Stacks geht nicht
- werden die sample video files durch rotiert?

## UI
- [x] Export Project als .tar.xz + „Open Project“ lädt das Archiv (Save As entfernt)
- [x] Drop-Box nur wenn leer + volle Höhe; Drag & Drop von Videos überall
- [x] Pfeile zwischen den Stacks entfernen
- [x] Videoname mittig (`project / video`), keine Box
- [x] Settings-Button unten links (Statusleiste)
- [x] About page (Help → Über Dialog mit Version/Engine/Lizenz/GitHub)
- [x] Hotkeys: Ctrl/Cmd+O/+A/+E/+R, Delete, Space; Menü zeigt Shortcuts; nur im Workspace
- [x] Version unten rechts (Statusleiste + Startseite)
- [x] Full-Video-Modus per Doppelklick auf Monitor, Exit ✕/Esc (alle drei Modi)
- [x] Deduplication: Presets (Aus/Standard/Aggressiv) + Slider mit % + Hinweis
- Menu - View hinzufügne 
- Settings - Hotkeys einstellungen

## Docs
- [x] ncnn-Engine komplett aus Todos/Plan entfernen (burn ist Default)
- [x] PLAN.md §15 → docs/CHANGELOG.md ausgelagert
- [x] PLAN.md aktualisiert/redesignet
- [x] models.md übersichtlicher (Status-Überblick + Backlog/Kandidaten-Tabelle)
- [x] benchmarks.md übersichtlicher (TL;DR-Box mit Entscheidung + Kernzahlen)
- [x] AGENTS.md: Generated-Code-Pfad korrigiert (`crates/senmei/gen/schemas/`), Commit-Regel auf CHANGELOG

## License
- [x] `shuffle-cugan` (unclear/SUDO) entfernt → Fallin Soft/Strong + 4x_Alchemy; Default: `real-cugan-x2`
- [x] `license_blocked()`: blockt verify/unclear + GPL/LGPL/AGPL + CC-BY-NC/ND/SA, durchgesetzt in beiden Commands
- [x] Review-Gate: „verify → loadable“ schaltet unklare Lizenzen nie frei (Test `license_gate_blocks_unclear_and_copyleft`)
- [x] FFmpeg-Download auf BtbN `-lgpl`-Builds + SHA256 gepinnt (Tag `autobuild-2026-08-17-13-05`)
- [x] LGPL-sichere Encoder: `pick_video_encoder` — libkvazaar → libopenh264 → h264_nvenc → libx264 → h264
- [x] AGENTS.md: Media-Bullet auf BtbN LGPL-Builds + libopenh264 aktualisiert

## Models
- [x] Fallin-Arch: Hand-Port statt burn-onnx-Codegen (UpCunet2x_fast, Pad 38, ONNX-verifiziert)
- [x] Laufzeit-ONNX→fp16-bpk: `senmei_ml::onnx` (dependency-frei) + `convert_onnx_to_bpk` + `download_model`-Zweig
- [x] RealPLKSR-Port → `4x-alchemy` + `real-plksr-deh264/dejpg` loadable (numerisch verifiziert)
- [x] Bench: Fallin Soft/Strong vs. real-cugan (1080p→2160p): 176/177 vs 380 ms; Fusion-Panic gefixt

## after release
- Project website
- [ ] Follow-up: burn Feature-Request „ONNX-Initializer laden“ einreichen, eigenen Parser ablösen

