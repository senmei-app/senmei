# Todos

## AI Stack
- [x] Stacks implementieren (denoise / deblur / dedup — Referenz-CPU)
- [x] Model-Backlog in `models.md`: RIFE/GMFSS, SCUNet/DRUNet, SRVGGNet/SPAN. Re-SISR ab „Adore" CC-BY-NC-SA → geblockt
- [x] quelle: https://github.com/chaiNNer-org/spandrel (permissive Arch-Referenz, dokumentiert)
- [x] Depth Map / Detection / Stabilization: **nein** — kein ML-Workflow; Stabilization nur klassisch via OpenCV (Apache-2.0)
- [x] Upscaler-Perf (25→12 fps): Ursache Tiling (512px, 329 ms) — Preis für Autotune-OOM-Fix; 1024px-Regression → 512px bleibt

## Backend
- [x] burn-tch Backend: ROCm-Nightly, RDNA4 fp16; vendored `third_party/` + `[patch.crates-io]`. Offen: Fallin-Bench, App-Anbindung
- [x] Dedup kollabiert statisches Material nicht mehr (Cap aufeinanderfolgender Drops)
- [ ] burn für macOS Grundgerüst als Experiment (keine Garantie)
- [x] Tiled-Fused-RGB8: Overlap `tile/8` = Regression (394→329 ms) — `tile/4` behalten; GPU-Stitching offen
- [x] GPU-Stitching: Tiles auf der GPU akkumulieren (`slice_assign`-Overlap-Averaging), ein Readback statt 15 — 329→234.7 ms / 4.3 FPS (fallin-soft)
- [x] Toten Code entfernen: `tiling::tile`, `Error::Unimplemented`, `Registry::from_json`, `Decoder::open`, `#[allow(dead_code)]`
- [x] Engine-Trait-Plumbing entfernt: `name()`, `EngineCaps.backend/half`, `InferOptions.half`, `Backend`-Enum nie gelesen
- [x] Unbenutzte Dependencies aus `senmei-app` entfernen: `base64`, `tauri-plugin-dialog`, `tauri-plugin-opener`
- [x] `extract_frame` Test-only — Smoke-Test auf `encode_png` umgestellt, Re-Export entfernt
- [x] Totes IPC-Command `remember_project` entfernt (internes `store::remember_project` bleibt)
- [x] `num_block`-Default aus `commands.rs` in Modell-Metadaten/Converter verlagern
- [x] Überflüssige Kommentare gekürzt: `Monitor.tsx`, Batch-Kommentar, `.pth`-/Asset-Protocol-Kommentar
- [x] Sample rendert ganze Queue statt nur Monitor-Video — gefixt: `startBatch` bekommt explizite Dateiliste
- [x] Rotation: `probe` meldet Display-Maße + `rotation`; `Decoder` -noautorotate + Transpose

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
- Menu: View hinzufügen für Full Video Modus
- Settings: Tile ändern
- Settings: Hotkeys einstellungen
- Rechte Seite: Tab-Bar neben „Processing Stack" mit Tab „Logs" (Systemlog)
- About dark Thmea

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

## Maintainability (Review 2026-08-18)

- [ ] Große Dateien splitten: `App.tsx`, `Inspector.tsx`, `commands.rs` (Orchestrierung/State trennen)
- [x] CPU-Steps: `step.rs` slicet planar, FFmpeg liefert packed `rgb24` — Layout-Konflikt prüfen/fixen (gefixt: packed rgb24)
- [ ] Doppelte Arg-Parsing-Logik: `splitArgs` (TS) und `split_ffmpeg_args` (Rust) vereinheitlichen
- [ ] Frontend-Pfade: manuelle `/`-Splits plattform-sicher ersetzen (Windows)
- [ ] Codec-Mapping angleichen: Frontend `H.264→libx264`/`H.265→libx265` an LGPL-safe Policy
- [ ] README: „planning phase / M0“ → aktuellen Stand (M2–M5) aktualisieren
- [ ] `todos.md` komplett auf Englisch (AGENTS-Vorgabe docs in English)
- [ ] Tauri-Security: CSP + Asset-Scope `$HOME/**` bewerten (Media-Zugriff vs. Fläche)
- [x] AGENTS.md-Pfad geprüft: `crates/senmei/gen/schemas/` existiert (Build-generiert, gitignored) — AGENTS.md korrekt

## after release
- [ ] Project website
- [ ] Follow-up: burn Feature-Request „ONNX-Initializer laden“ einreichen, eigenen Parser ablösen

