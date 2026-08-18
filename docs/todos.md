# Todos

## AI Stack
- [x] Stacks implementieren, die noch deaktiviert waren (denoise / deblur / dedup — Referenz-CPU)
- weiter modele definieren für:
  - Interpolation Models
  - Denoising
  - Restoration Models
- quelle:
  - https://github.com/chaiNNer-org/spandrel
- macht folgendes sinn:
  - Depth Map Models
  - Object Detection Models
  - Video Stabilization
- was anderes die performance vom upscaler ist grafieren eingebrochen von 25fps auf so 12fps runter (evtl tile per settings dynamisch einstellen)

## Backend
- [x] burn-tch (libtorch) Backend: Läuft gegen die ROCm-Nightly (torch714-venv, 2.15+rocm7.14) — `LibTorchDevice::Cuda(0)` → RDNA4 fp16. Archs sind Backend-generisch (UpCunet2x/RrdbNet/RifeNet mit `LibTorch<f32>`). Reproduzierbar via Vendored torch-sys/burn-tch unter `third_party/` + `[patch.crates-io]`: torch-sys 0.22 mit C++20 (libtorch ≥2.15 verlangt es), 6 entfernte Ops gemappt (align_as/align_tensors → Stub, cholesky→`at::linalg_cholesky`, qr→`at::linalg_qr`), ROCm-rpath-link-Emmission (Env `SENMEI_ROCM_LIB` oder Autodetect `/opt/rocm|/opt/therock`); burn-tch 0.21 cuda_hack auf ROCm-Signatur `getCurrentCUDABlasHandle(bool)` angepasst. Getestet: `tch_engine_roundtrips_bpk_on_rocm` (skippt ohne CUDA/ROCm). Build: `LIBTORCH=<nightly> LIBTORCH_BYPASS_VERSION_CHECK=1`; Runtime: `LD_LIBRARY_PATH` auf ROCm-Libs. Offen: echtes Fallin-bpk/Perf-Benchmark auf der GPU, App-Anbindung (engine-Umschalter)
- [ ] burn für macOS Grundgerüst als Experiment (keine Garantie)
- [x] Tiled-Fused-RGB8-Pfad optimieren: Overlap `tile/4` → `tile/8` getestet → 1080p-**Regression** (394 vs 329 ms: Tile-Zahl bleibt 15 bei 512er-Tiles, kleineres Overlap vergrößert nur den Pad-Bereich) — `tile/4` behalten; **GPU-Stitching bleibt Follow-up** (CPU-Stitch + u8-Rücklese sind der Rest-Aufwand, nicht der Overlap)
- [x] Toten Code entfernen: `senmei_ml::tiling::tile`, `Error::unimplemented`/`Error::Unimplemented` (0 Aufrufer), `Registry::from_json` (nur Test), `Decoder::open` (nur Bench), veraltetes `#[allow(dead_code)]` an `grid_sample` (wird genutzt)
- [x] Engine-Trait-Plumbing entscheiden: `InferenceEngine::name()`, `EngineCaps.backend`/`half`, `InferOptions.half`, `Backend`-Enum werden nie gelesen — entfernt
- [x] Unbenutzte Dependencies aus `senmei-app` entfernen: `base64` (Rest des alten JPEG-Previews), `tauri-plugin-dialog`/`tauri-plugin-opener` (nur in `senmei` nötig)
- [x] `extract_frame` ist nur noch Test-only — Smoke-Test auf `encode_png` umgestellt, Funktion samt Re-Export entfernt
- [x] Totes IPC-Command entscheiden: `remember_project` hat keinen Frontend-Aufrufer — entfernt (internes `store::remember_project` bleibt, wird von `export_project` genutzt)
- [x] `num_block`-Default (`unwrap_or(4)`) aus `commands.rs` in die Modell-Metadaten/den Converter verlagern (Own-Responsibility)
- [x] Überflüssige Kommentare kürzen: Playback-/Effekt-Kommentare in `Monitor.tsx` (bereits leer), falsch platzierter Batch-Kommentar über `selectFile` zu `startBatch` verschoben, redundanter `.pth`-Kommentar + doppelter Asset-Protocol-Kommentar in `commands.rs` entfernt
- sample mit anderen Stacks geht nicht
- werden die sample video files durch rotiert?

## UI
- [x] Export Project als .tar.xz + „Open Project“ lädt das Archiv (Save As entfernt)
- [x] Drop-Box nur wenn leer + volle Höhe; Drag & Drop von Videos überall
- [x] Pfeile zwischen den Stacks entfernen
- [x] Videoname mittig (`project / video`), keine Box
- [x] Settings-Button unten links (Statusleiste)
- [x] About page (Help → Über Dialog mit Version/Engine/Lizenz/GitHub)
- [x] hotkeys — Ctrl/Cmd+O Import, +A Alle auswählen, +E Projekt exportieren, +R Render, Delete entfernt Auswahl, Space Play/Pause im Monitor; Menü zeigt Shortcuts; nur im Workspace aktiv
- [x] Version unten rechts (Statusleiste + Startseite)
- [x] full video mode für orignal result und compare mittels doppelklick auf monitor view — Doppelklick auf den Monitor → Full-Video-Overlay (Panels verdeckt), Exit per ✕ oder Esc; funktioniert in allen drei Modi
- [x] deduplication nur ein schieber nichts aussage kräftige — jetzt Presets (Aus/Standard/Aggressiv) + Slider mit Prozent-Anzeige + erklärendem Hinweis
- Menu
  - View
- Settings
    den Punkt Hotkeys aufnehemn



## Docs
- [x] ncnn-Engine komplett aus Todos/Plan entfernen (burn ist Default)
- [x] PLAN.md §15 → docs/CHANGELOG.md ausgelagert
- [x] PLAN.md aktualisiert/redesignet
- [x] models.md übersichtlicher (Status-Überblick + Backlog/Kandidaten-Tabelle)
- [x] benchmarks.md übersichtlicher (TL;DR-Box mit Entscheidung + Kernzahlen)
- [x] AGENTS.md aktualisieren: Generated-Code-Pfad `crates/senmei-app/gen/` existiert nicht (korrekt `crates/senmei/gen/schemas/`, `burn/rife.rs` fehlt in der Liste) + Commit-Regel auf CHANGELOG umformulieren

## License
- [x] `shuffle-cugan` (unclear/SUDO) entfernt → ersetzt durch Fallin Soft/Strong + 4x_Alchemy (CC-BY-4.0); Default-Upscaler ist jetzt `real-cugan-x2` (Apache-2.0)
- [x] `download_model`/`engine_for_model` gaten nur auf `loadable`, nicht auf Lizenz — bei `unclear`/`verify` Download/Weights verweigern oder hart warnen — `ModelMetadata::license_blocked()` (blockt verify/unclear + GPL/LGPL/AGPL + CC-BY-NC/ND/SA, fehlend → blockiert), durchgesetzt in beiden Commands
- [x] Review-Gate einführen: „verify → loadable“ darf unklare Lizenzen nie automatisch freischalten — `license_blocked()` ist unabhängig von `loadable`; Test `license_gate_blocks_unclear_and_copyleft` beweist Block auch bei `loadable: true`
- [x] FFmpeg-Download auf BtbN `-lgpl`-Builds umstellen (aktuell GPL trotz LGPL-only-Policy) + SHA256 neu pinnen — datierter Tag `autobuild-2026-08-17-13-05` (N-126188), linux+win64 je eigener SHA; "latest"-Tag-GPL-Pin entfernt
- [x] `libx264` erfordert GPL-FFmpeg — für gebündelte Builds LGPL-sichere Encoder nutzen (`libkvazaar` / `libopenh264` / `h264_nvenc` / `vaapi`) — `pick_video_encoder`: libkvazaar (HEVC, BSD, bundled LGPL) → libopenh264 → h264_nvenc → libx264 (System-GPL) → h264; kvazaar/x264 quality-based, libopenh264 mit Auflösungs-`-b:v`; Test `encodes_through_selected_codec`
- [x] AGENTS.md-Widerspruch auflösen (Media-Bullet sagt GPL-Download, Lizenz-Bullet LGPL-only) — Media-Bullet auf gepinnte BtbN LGPL-Builds + libopenh264 aktualisiert

## Models
- [x] Fallin-Arch: burn-onnx-Codegen vs. Hand-Port (upcunet2x-Struktur) vergleichen → `fallin-soft`/`fallin-strong` loadable machen — Hand-Port gewinnt (Fallin = `UpCunet2x_fast`, Pad 38, numerisch gegen ONNX verifiziert)
- [x] Laufzeit-ONNX→fp16-bpk-Import: `download_model` ONNX-fähig machen (onnx-ir-Parser, kein ONNX Runtime) — `senmei_ml::onnx` (dependency-frei) + `convert_onnx_to_bpk` + `download_model`-ONNX-Zweig
- [x] RealPLKSR-Arch portieren → `4x-alchemy` + `real-plksr-deh264/dejpg` loadable machen — clean burn-Port (dim 64/28 Blöcke/kernel 17/DySample), numerisch gegen torch verifiziert (1x mae ~0.002, 4x mae ~0.002). Zwei burn-bugs dabei gefunden & umgangen: f16-`div_scalar(65536)` → 0 (GroupNorm via `mean_dim` neu gebaut) und `repeat`/`reshape`-Interleaving (repeat_interleave explizit); DySample-`init_pos` exakt nach Referenz-Transformation. `4x_Alchemy.pth` ist channels-last (nicht contiguous) — burn-store ignoriert Strides (Bug 5), Converter braucht `.contiguous()`-Vorverarbeitung
- [x] Bench: Fallin Soft/Strong gegen `real-cugan-x2` messen — fused step 1080p→2160p: real-cugan 380 ms/2.6 FPS/14.6 GB, fallin-soft 176 ms/5.7 FPS/8.1 GB, fallin-strong 177 ms/5.7 FPS/8.1 GB (dabei burn-fusion-Ordering-Panic in `infer_rgb8` gefunden & gefixt: f32-Rücklese statt u8/f16)
- [ ] Follow-up: Feature-Request bei burn einreichen — „burn-store: ONNX-Initializer (weights-only) laden“, damit der eigene ONNX-Parser später durch burn-store ersetzt werden kann

## after release
- Project website

