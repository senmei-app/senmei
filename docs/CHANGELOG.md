# Senmei — Changelog

> Implementation log (was §15 of PLAN.md). Newest on top.

> Kept in sync with actual implementation. Update on every significant change.

- **fix: ONNX-Reader liest auch `Constant`-Node-Weights (2026-08-19)** — der
  eigene protobuf-Reader (`senmei_ml::onnx`, kein ONNX-Runtime) las bisher nur
  `graph.initializer`; Modelle, deren Weights nur in `Constant`-Nodes stecken
  (Value-Attribut), ergaben still ein leeres bpk. Liest jetzt zusätzlich
  `Constant`-Nodes (geteikt per Node-Output-Name, da das innere
  `TensorProto.name` `"value"`/leer ist), meldet ein leeres Ergebnis als
  Fehler und lehnt External Data (`data_location == EXTERNAL`) ab — die
  drei Punkte aus dem `onnx-ir`-Issue-Kommentar (tracel-ai/burn-onnx#456).
  4 neue Unit-Tests.

- **feat: tile size configurable in Settings (2026-08-19)** — the fused RGB8
  upscale tile size (default 640, previously `SENMEI_TILE` env only) is now a
  Settings value (`tileSize`), applied per render via `senmei_ml::set_tile_size`
  and editable in the Settings UI (Appearance, 128–2048 px). `SENMEI_TILE`
  still works as a bench-only fallback. Also removed the top-right settings
  gear (Settings remain reachable via the status bar and menu).

- **fix: IFRNet ResBlock c5-Conv + Bug-6-Diagnose zurückgezogen (2026-08-19)** —
  der ResBlock-`forward` ließ die `conv5`-Conv weg (`pl(out4 + x)` statt
  `pl(c5(out4) + x)`; Referenz: `x + self.conv5(out)`). Das war die echte
  Ursache der angeblichen burn-Fusion-Bug-6-Abweichung — kein Backend-Bug.
  Mit der Korrektur lädt IFRNet sauber (applied=104, missing=0) und der
  Torch-Referenz-Test besteht auf gefusetem `Vulkan<f16>` (mae 0.005).
  `ifrnet-vimeo90k`/`ifrnet-gopro` sind jetzt `loadable: true`;
  docs/burn-bugs.md Bug 6 entfernt. (Review von PR #1, copilot-swe-agent.)

- **feat: DRUNet burn-Arch-Port (2026-08-19)** — `burn/drunet.rs`: `UNetRes`
  (DPIR, MIT) als sauberer Nachbau — 3× Stride-2-Downsample, 4 ResBlocks pro
  Ebene (Conv→ReLU→Conv + Skip), 3× ConvTranspose2d-Upsample, alle Convs
  `bias=false`, `in_nc=4` (RGB + konstante Noise-Level-Map). Torch-verifiziert
  (mae 0.001, alle 64 Weights geladen) — als erster ML-Denoise **loadable**,
  ohne Fusion-Bug (keine Channel-Slices). `senmei-ml-convert` `drunet`-Arch
  (Capture-Group-Key-Remap), Registry `drunet-color` (MIT, KAIR v1.0),
  `tools/drunet_verify.py`. Pipeline-Denoise-Step-Verdrahtung (4ch-Sigma-Map)
  noch offen.

- **fix: surface the real encode error instead of "encode channel closed" (2026-08-19)** —
  the encoder discarded ffmpeg's stderr (`Stdio::null()`), so a failed encode
  only surfaced as "encode channel closed" (the main loop's channel error masked
  the encode thread's real cause). The encoder now captures stderr and includes
  it in the write/finish error, and the pipeline reports the encode thread's
  error (cancellation and step errors still win). Render failures now show the
  actual ffmpeg reason in the Logs panel.

- **docs: IFRNet torch-verifiziert (2026-08-19)** —
  `tools/ifrnet_verify.py` + vendorte torch-Referenz (`ref/ifrnet/`, MIT) erzeugen
  Referenz-Bins; Encoder + Weights sind exakt (mae ~0.0001). Der ResBlock
  (Side-Channel split/cat) wich zwischen Methode und inline ab (mae 0.0525 vs
  ~0.0001) — als burn-Fusion-Bug 6 fehldiagnostiziert; die echte Ursache war
  eine im `forward` fehlende `conv5`-Conv (siehe fix-Entry oben; Bug 6
  zurückgezogen, IFRNet `loadable: true`).

- **feat: HDR→SDR tonemapping (2026-08-18)** — `probe` liest `color_transfer`/
  `color_primaries` und `VideoInfo::is_hdr()` erkennt PQ/HLG/DCI. Der Decoder
  wendet bei HDR (oder `always`) eine zscale+tonemap-Filterkette an und
  konvertiert korrekt nach SDR, bevor `rgb24` ausgegeben wird — vorher wurde
  HDR beim Decode unkontrolliert geclippt. Neues Output-Step-Setting `tonemap`
  (auto/always/off), durchgereicht über `RenderConfig` → `Pipeline::set_tonemap`
  → `Decoder`. Tests: `hdr_detection` (Unit) + `hdr_source_is_detected_and_tonemapped`
  (Integration, libx265-gated).

- **feat: IFRNet burn-Arch-Port (2026-08-18)** — `burn/ifrnet.rs`: Base-Variante
  (ltkong218, MIT) als sauberer Nachbau — 2× geteilter 4-Level-Encoder, vier
  coarse-to-fine Decoder (bilinear, kein GRU), Side-Channel-ResBlock, eigene
  PReLU-Implementierung (fehlt in burn 0.21), geteiltes `warp`/`grid_sample`.
  Engine-Dispatch (`Model::IfrNet`, Interp-Pfad pad 16), `senmei-ml-convert`
  `ifrnet`-Arch (Capture-Group-Key-Remap), Registry-Einträge
  `ifrnet-vimeo90k`/`ifrnet-gopro` (MIT, HF-URLs). `loadable: true` nach der
  ResBlock-c5-Korrektur (siehe fix-Entry oben).

- **docs: IFRNet-Weights verifiziert (2026-08-18)** — offizielle Checkpoints
  (Vimeo90K + GoPro, je 19.9 MB, MIT) via `pavlichenko/ifrnet_*` auf Hugging
  Face mit direkten resolve-URLs; „Weights verify / Repo-URL verify" im
  Backlog aufgelöst.

- **ci: GitHub Actions Matrix-Build (2026-08-18)** — `.github/workflows/ci.yml`:
  Windows/Linux/macOS — System-Deps, Frontend-Build, `cargo check` +
  `cargo test --workspace` (GPU-Tests sind `#[ignore]`), App-Bundle via
  `tauri build`, Artifact-Upload.

- **docs: NAFNet fp16-Port-Hinweise notiert (2026-08-18)** — litert-community-
  Konversion (NAFNet-GoPro-width32) bestätigt MIT + liefert Port-Details für
  den burn-Nachbau: SimpleGate (kein Activation), Channel-Attention = mean×2,
  Upsample = Conv1×1 + PixelShuffle, und die fp16-LayerNorm-Overflow-Falle
  (Kanalreduktion in skaliertem Bereich rechnen). In `models.md` Notes.

- **docs: NAFNet-GoPro als Deblur-Kandidat hochgestuft (2026-08-18)** —
  offizielle NAFNet-Weights sind via `nyanko7/nafnet-models` (Hugging Face)
  mit direkten, sha256-pinnbaren URLs verfügbar (kein GDrive nötig);
  GoPro-width32 (68.7 MB) als leichte Option. Damit ist NAFNet-GoPro der
  erste ML-Deblur-Kandidat (Deblur-Stack ist bisher CPU-only); der
  NAFBlock-Arch-Port bleibt offen.

- **docs: KAIR v1.0 + NAFNet Modelle gesichtet (2026-08-18)** — weitere
  permissive Weights im Backlog: DRUNet/DnCNN/FFDNet/IRCNN/BSRGAN/IMDN
  (alle MIT via KAIR v1.0, direkte Download-URLs), NAFNet SIDD/GoPro/REDS
  (MIT). Erster neuronaler Deblur-Kandidat (NAFNet-GoPro) notiert.

- **docs: Lizenzen für Denoise/Restoration geklärt (2026-08-18)** — SCUNet
  **Apache-2.0** (in `metadata.json` + `models.md` eingetragen → nicht mehr
  lizenz-geblockt; Arch-Port bleibt offen), DRUNet (DPIR) **MIT** via KAIR
  v1.0, NAFNet **MIT**, USRNet/USRGAN **MIT** (Backlog ergänzt).

- **refactor: `Inspector.tsx` aufgeteilt (2026-08-18)** — der komplette
  Step-Editor (alle Typen inkl. großem Output-Editor) nach `StepEditor.tsx`
  extrahiert; `Inspector.tsx` von ~800 auf ~370 Zeilen reduziert (Stack-Liste,
  Drag&Drop, Add-Menü). Damit sind alle drei Groß-Dateien aufgeteilt
  (App.tsx, commands.rs, Inspector.tsx).

- **refactor: `commands.rs` aufgeteilt (2026-08-18)** — Modell-Helfer nach
  `models.rs` (`models_dir`/`load_registry`/`engine_for_model`), Preview-Helfer
  nach `preview.rs` (decode-Streams, `read_frame_inner`, PNG-Prune).
  `commands.rs` von ~800 auf ~636 Zeilen reduziert; nur noch Tauri-Commands.

- **security: Asset-Scope verengt + CSP gesetzt (2026-08-18)** — der statische
  Asset-Protocol-Scope war `["$DATA/**", "$HOME/**"]` (ganzes Home lesbar).
  Alle Media-Loads laufen ohnehin über `probe_video`/`read_frame`, die die
  Datei per `allow_file` zur Laufzeit freigeben (dieselbe Scope, die der
  Asset-Protocol prüft), also reicht `["$DATA/**"]` (App-Datadir für
  Previews/Samples/Projekte). Dazu eine CSP für Produktion (dev bleibt
  unberührt).

- **refactor: App.tsx aufgeteilt — Batch-Logik in `useBatch`-Hook (2026-08-18)** —
  Render-State + `startBatch`/`cancel`/`togglePause` + `desiredPath` aus
  `App.tsx` in `useBatch.ts` extrahiert (~150 Zeilen weniger). Verhalten
  unverändert (Demo-Render + Cancel verifiziert).

- **ui: Logs-Tab neben dem Processing Stack (2026-08-18)** — das rechte Panel
  hat jetzt einen Tab-Umschalter „Processing Stack“ / „Logs“ (`RightPanel`).
  Neuer `LogHub`-Logger leitet `log`-Records als Tauri-Event an die UI
  (Ringbuffer 500, `get_logs` beim Öffnen); das Panel hat Level-Filter
  (ALL/ERROR/WARN/INFO), Clear und Auto-Scroll. Das Konsolenverhalten von
  `env_logger` bleibt unverändert (error + `wgpu_hal=off`), das Panel fängt
  Info+.

- **refactor: Frontend-Pfade plattformsicher (2026-08-18)** — alle manuellen
  `split("/")`-Stellen durch `paths.ts`-Helfer ersetzt (`basename`/`dirname`/
  `joinPath`), die sowohl `/` als auch `\` (Windows) verarbeiten; Joins nutzen
  `/`, das Windows-APIs ebenfalls akzeptieren. Betrifft Output-Pfadbau,
  Sample-Ordner und Dateinamen-Anzeige.

- **refactor: FFmpeg-Args einheitlich geparst (2026-08-18)** — das Frontend
  sendet die Encoder-Args jetzt als vorgesplittetes Array
  (`RenderConfig.ffmpegArgs: string[]`); der doppelte Rust-Parser
  `split_ffmpeg_args` wurde entfernt. Es gibt nur noch einen Parser
  (`splitArgs` in `steps.ts`), geteilt für Vorschau und Render.

- **ui: Hotkey-Einstellungen auf der Settings-Seite (2026-08-18)** — neue
  Sektion „Tastenkürzel“ (Koharu-Stil): Aktionen anzeigen, per Klick neu
  belegen (nächster Tastendruck), auf Standard zurücksetzen. Overrides werden
  in den App-Settings persistiert (`Settings.hotkeys`), Defaults bleiben im
  Code; App-Hotkeys + Monitor-Space nutzen die konfigurierten Combos.

- **ui: About-Dialog folgt dem Dark-Theme (2026-08-18)** — der Dialog wurde
  außerhalb des `dark`-Wrappers gerendert, seine `dark:`-Styles griffen nie
  (immer hell). In den Wrapper verschoben.

- **ui: Menü „View“ mit Full-Video-Modus (2026-08-18)** — neues Menü „View“
  (Ansicht) mit „Full Video Mode“; togglet denselben Fullscreen wie der
  Doppelklick auf den Monitor (Signal an `Monitor.toggleFullscreenSignal`).
  DE/EN übersetzt.

- **fix: Codec-Mapping LGPL-safe (2026-08-18)** — der Encoder-Dropdown mappte
  H.264→`libx264`/H.265→`libx265` (beide GPL, fehlen in den gepinnten
  BtbN-LGPL-Builds), wodurch H.264/H.265-Outputs mit dem LGPL-FFmpeg
  fehlschlugen. Jetzt H.264→`libopenh264`, H.265→`libkvazaar` (beide BSD) und
  die Args sind codec-bewusst: CRF nur für svtav1/vpx, Preset für kvazaar,
  openh264 ist ABR und bekommt sein `-b:v` vom Backend. `Encoder::open`
  verwirft beim `-c:v`-Override die Default-Args des Basis-Codecs. Test
  `override_codec_sets_bitrate_for_openh264_only`.

- **perf: Tile-Größe 512→640 nach GPU-Stitch (2026-08-18)** — das alte
  Kostenmodell (15 u8-Readbacks + CPU-Stitch) galt nicht mehr, daher neu
  gemessen (`bench_upscale_step`, fallin-soft, 1080p→2160p): 512px 247.8 ms,
  **640px 186.1 ms / 5.4 FPS**, 768px 210.2 ms. 640 halbiert die Tile-Zahl
  (15→8), bevor der per-Tile-Matmul pathologisch wird. Default 640,
  Override via `SENMEI_TILE`; Korrektheitstest auf ein einzelnes 640-Tile
  umgestellt. Full-Frame (176 ms) bleibt der Floor bis zum upstream
  Autotune-OOM-Fix.

- **fix: Dedup kollabiert statisches Material nicht mehr (2026-08-18)** —
  Dedup droppte unbegrenzt aufeinanderfolgende Duplikate; bei statischem/
  nahezu statischem Material blieb nur ein Frame übrig („Render Sample“ mit
  nur Dedup ergab ~0,05 s). Jetzt max. 5 aufeinanderfolgende Drops, danach
  wird ein Frame erzwungen (statische 3 s → ~0,5 s statt 0,05 s). Test
  `dedup_never_collapses_static_run`.

- **perf: GPU-Stitching im tiled-fused RGB8-Pfad (2026-08-18)** — statt jedes
  512px-Tile als u8 zurückzulesen und auf der CPU zu stitchen, akkumuliert
  `infer_rgb8` die Tiles jetzt auf der GPU in einem f16-Canvas
  (`slice_assign`-Overlap-Averaging) und liest einmal ein packed Frame zurück —
  ein Readback statt 15 plus CPU-Stitch. `bench_upscale_step` (1080p→2160p,
  fallin-soft): 329 → **234.7 ms / 4.3 FPS**. Der dadurch tote CPU-Stitch
  `stitch_rgb24` wurde entfernt. Korrektheit + 48-Frame-Reliability via
  `infer_rgb8_tiled_is_reliable_and_correct`.

- **fix: CPU-Steps verarbeiten packed `rgb24` statt planar (2026-08-18)** —
  `Denoise`/`Deblur`/`Resize` sliceten `Frame.data` als planare RGB-Ebenen,
  aber Decoder/Encoder arbeiten mit packed `rgb24`. Dadurch mischte der
  Denoiser die Kanäle: „Render Sample" driftete mit aktivem Upscaler
  auseinander, Denoiser-only ergab Müll. Die Steps bluren/schärfen/resamplen
  jetzt kanalgetrennt auf packed Daten; Regressionstests
  `denoise_keeps_channels_separate`, `deblur_keeps_channels_separate`,
  `resize_keeps_channels_separate` (schließt Maintainability-TODO).

- **fix: `prune_samples` löscht nach mtime statt Dateiname (2026-08-18)** —
  Sample-Renderings wurden lexikalisch nach Pfad sortiert gelöscht; durch die
  Range-Tags im Namen konnte so das gerade gerenderte Sample verschwinden.
  Löscht jetzt die ältesten (mtime), behält die `keep` neuesten. Test
  `prune_samples_keeps_newest_by_mtime`.

- **ui: „Render Sample" rendert nur das aktuelle Video (2026-08-18)** — der
  Sample-Button rief `startBatch(false, …)` auf und erzeugte Samples für die
  **ganze Queue** statt für das Video im Monitor. `startBatch` akzeptiert jetzt
  eine explizite Dateiliste; `onRenderSample` übergibt `[currentFile]`.

- **media: Video-Rotation wird verarbeitet (2026-08-18)** — `probe` liest die
  Rotation (DisplayMatrix side-data oder case-insensitives `rotate`-Tag), meldet
  Display-Maße + `VideoInfo.rotation`; `Decoder` setzt `-noautorotate` und wendet
  die Rotation explizit an (90→`transpose=2`, 180→`hflip,vflip`, 270→`transpose=1`),
  byte-identisch zu ffmpegs Autorotation verifiziert (Test
  `probe_and_decode_apply_rotation`). Vorher wurden 90°/270°-Videos
  fehlbeschriftet/verzerrt verarbeitet (autorotierte Ausgabe ≠ probed Maße).

- **docs: PLAN §14/§15 restructure + maintainability backlog (2026-08-18)** —
  `PLAN.md` §14 split into subsections (own code & libs, models, codecs, AGPL
  boundary) with an expanded dependency/license table, §15 rewritten as a status
  snapshot; `models.md` SPAN added to the backlog; `todos.md` gained a
  Maintainability section from a code review (8 open items; AGENTS generated-path
  check confirmed fine).

- **docs: tidy-up + re-sync all docs (2026-08-18)** — `todos.md` entries capped at
  ~135 chars; `benchmarks.md` reorganized decision-first with a key-numbers table;
  `burn-bugs.md` prose tightened (all facts kept); `models.md` deduped
  (status-at-a-glance removed) and loadable status updated to match
  `metadata.json`; `PLAN.md` brought back in sync with the code (engine trait,
  PNG/native-video preview, adopted models/licenses, LGPL-safe encoder, vertical
  layout diagram).

- **ml: RealPLKSR port — 4x-alchemy + decompress models loadable (2026-08-18)** —
  clean burn re-implementation of RealPLKSR (Partial Large Kernel CNNs for
  Efficient Super-Resolution, arXiv 2404.11848; spandrel MIT reference):
  head → 28 PLK blocks (DCCM + partial 17×17 conv + EA + GroupNorm) → tail,
  with the DySample upsampler tail for the 4x model and a pixel-shuffle
  identity for the 1x decompress models. Numerically verified against torch
  on deterministic inputs (deh264/dejpg 1x mae ~0.002 / ~0.0002, alchemy 4x
  mae ~0.002). Two burn-wgpu findings worked around along the way: f16
  `div_scalar(65536)` underflows to 0 (GroupNorm rebuilt on `mean_dim`,
  docs/burn-bugs.md Bug 4) and `repeat`/`reshape` interleaves copies wrongly
  (`repeat_interleave` built explicitly). `4x_Alchemy.pth` stores weights
  channels-last — burn-store ignores strides (docs/burn-bugs.md Bug 5), so
  that conversion needs a `.contiguous()`-fixed pth. `warp.rs` grid sampling
  generalized (align_corners selectable, arbitrary output size).

- **ui: keyboard shortcuts (2026-08-18)** — Ctrl/Cmd+O imports a file, +A
  selects all, +E exports the project, +R renders, Delete removes the
  selection, Space toggles monitor play/pause. Shortcut hints are shown in the
  menu bar; hotkeys are active only in the workspace (not the start screen).
  Also fixes a latent `menu.children` reference in the MenuBar import submenu.

- **ui: meaningful dedup controls (2026-08-18)** — the deduplication step now
  has mode presets (Off / Standard / Aggressive), a threshold slider with a
  live percent readout, and a one-line explanation instead of a bare slider.

- **ui: full-video monitor mode via native WebKit fullscreen (2026-08-18)** —
  double-click on the monitor view opens it fullscreen via the HTML Fullscreen
  API (`requestFullscreen` on the monitor element, supported by WebKitGTK) —
  the same video/frame instance stays mounted, so playback continues and no
  second decoder runs underneath. Works in original / compare / result modes.
  Exit via a second double-click, the ✕ button, or native Esc.

- **perf: tiled-fused RGB8 overlap — tile/8 rejected (2026-08-18)** — tested
  `overlap = tile/4 → tile/8` on the fused RGB8 path (1080p, fallin-soft):
  regression to 394 ms / 2.5 FPS vs 329 ms / 3.0 FPS. With 512px tiles the
  tile count is unchanged (5×3) and the smaller overlap only enlarges the
  padded region, so the CPU stitch/crop does more work. Kept `tile/4`
  (reliability confirmed by `infer_rgb8_tiled_is_reliable_and_correct`). The
  real remaining cost is CPU stitching + per-tile u8 readback — GPU stitching
  tracked as follow-up in docs/todos.md. Bench test-input generation switched
  from GPL `libx264` to the universally available native `mpeg4` (LGPL-safe).

- **fix: LGPL-only FFmpeg + LGPL-safe HEVC encoder (2026-08-18)** — the
  portable download now pins BtbN `-lgpl` builds on a dated tag
  (autobuild-2026-08-17-13-05, N-126188) with per-platform SHA-256
  (linux/win64); the old single `latest`-tag GPL pin was license-noncompliant
  and shared one SHA across platforms. The encoder no longer hardcodes
  `libx264` (GPL-only): `pick_video_encoder` prefers libkvazaar (HEVC, BSD,
  ships in the LGPL builds) → libopenh264 → h264_nvenc → libx264 → native
  h264. kvazaar/x264 use quality-based rate control; libopenh264 gets a
  resolution-based `-b:v` (~14 Mbps @1080p; `extra_args` override). Resolves
  the AGENTS.md GPL-vs-LGPL contradiction. Guarded by
  `encodes_through_selected_codec` (runs against a real ffmpeg via
  SENMEI_FFMPEG).

- **fix: license gate for model download/use (2026-08-18)** — `download_model`
  and the app `engine_for_model` only checked `loadable`, so a model flagged
  `verify`/`unclear` (license review pending) or under a copyleft /
  non-commercial license could be unlocked by flipping `loadable`. Added
  `ModelMetadata::license_blocked()` (blocks `verify`, `unclear`,
  GPL/LGPL/AGPL, CC-BY-NC/ND/SA; missing → blocked) and enforced it in both
  commands, independent of `loadable` — the review gate never auto-unlocks an
  unclear license. Guarded by `license_gate_blocks_unclear_and_copyleft`.

- **fix: tiled-fused RGB8 render path (reliable GPU conversion) (2026-08-18)** —
  the full-frame fused `infer_rgb8` OOM'd burn/cubecl autotune on the large
  full-frame matmul (m=1024, n=4M, f16) and then cascaded into "Ordering is
  bigger than operations" panics (docs/burn-bugs.md Bug 1+3). `infer_rgb8` now
  tiles internally (512px, overlap): per tile the GPU runs forward + NHWC
  permute + clamp + scale + u8 cast, so only packed u8 bytes cross back, and
  tiles are stitched with overlap averaging (`stitch_rgb24`/`crop_rgb24`).
  Structurally immune to the OOM. `Upscale` prefers `infer_rgb8`, falls back
  to `infer_tiled`. Guarded by `infer_rgb8_tiled_is_reliable_and_correct`
  (correctness within fp16 tolerance + 48-frame reliability). Benched
  (1080p→2160p, fallin-soft): step 329 ms / 3.0 FPS, full threaded pipeline
  2.8 FPS. Supersedes the f32-readback-only attempt (ed1b27e). Overlap / GPU
  stitch tuning tracked in docs/todos.md.

- **fix: burn-fusion ordering panic in the fused RGB8 render path (2026-08-18)** —
  `infer_rgb8` read back the RGB8 output as u8, which (like any non-f32
  `to_vec()`) deterministically panics after ~48 frames with "Ordering is
  bigger than operations" (burn-fusion 0.21 + cubecl-autotune), on every model.
  The permute + clamp + scale now still run on the GPU, but the readback is f32
  and the trivial u8 cast happens on the CPU — byte-identical to the reference,
  full autotune speed retained. Added two guarded tests:
  `repeated_infer_rgb8_does_not_panic` and `infer_rgb8_matches_infer_reference`.
  Benched at 1080p→2160p: real-cugan-x2 2.6 FPS / 14.6 GB, fallin-soft 5.7 FPS
  / 8.1 GB, fallin-strong 5.7 FPS / 8.1 GB (fused step).

- **Fallin loadable: UpCunet2x_fast hand-port + built-in ONNX reader (2026-08-18)** —
  `fallin-soft` / `fallin-strong` are the existing `UpCunet2x_fast` arch (same
  38px reflect pad, verified numerically against the ONNX) — no codegen needed.
  The ONNX file is only a weight container: a new dependency-free protobuf
  reader (`senmei_ml::onnx`) extracts the initializers, and
  `convert_onnx_to_bpk` feeds them into the module (torch `.conv.0`/`.conv.2`
  key remap) to build the f16 `.bpk`. `download_model` and
  `senmei-ml-convert` accept `.onnx` sources automatically. Both models are
  now `loadable: true`; engine output matches the ONNX reference within fp16
  tolerance.

- **senmei-app: drop dead IPC + unused deps (2026-08-18)** — removed the
  frontend-unused `remember_project` command (the internal
  `store::remember_project` stays for `export_project`) and the unused
  `base64` / `tauri-plugin-dialog` / `tauri-plugin-opener` dependencies; the
  `num_block` default now comes from `Registry::resolve`.

- **Dead code removed (2026-08-18)** — dropped `tiling::tile` (test-only),
  `Error::Unimplemented`, `Registry::from_json` (test-only), `Decoder::open`
  (bench-only), `preview::extract_frame` (test-only; the smoke test now encodes
  a synthetic frame via `encode_png`), and a stale `#[allow(dead_code)]` on
  `grid_sample` (it is used by the RIFE arch).

- **Inference engine trait simplified (2026-08-18)** — removed the never-read
  `Backend` enum, `EngineCaps.backend`/`half`, `InferOptions.half`, and
  `InferenceEngine::name()`; capabilities/options now carry only what the
  tiling path consumes (`tiles`, `tile_size`).

- **Model registry: drop SUDO shuffle-cugan, add Fallin + 4x_Alchemy (2026-08-18)** —
  removed `shuffle-cugan` (unclear/SUDO weights). Added `fallin-soft` /
  `fallin-strong` (2× CUGAN retrain, CC-BY-4.0, ONNX-only, sha256-pinned) and
  `4x-alchemy` (4× RealPLKSR_Dysample, CC-BY-4.0, `.pth`) — all `loadable: false`
  until their archs are ported. The default upscaler is now `real-cugan-x2`
  (Apache-2.0). Bench/test defaults updated.

- **Sample output + compare sync (2026-08-18)** — sample renders go into the
  project's `sample/` folder with a time-range tag in the name (pruned to the 5
  newest); the sample window follows the playhead (scrub outside it repositions
  it) and snaps to frame boundaries so the rendered result starts on the exact
  source frame; compare updates both sides together (never one ahead) and the
  result/compare timeline shows the sample window in source coordinates.

- **read_frame: async + project preview frames (2026-08-18)** — `read_frame` is
  now async (decode off the main thread) and accepts `project_dir`; preview
  PNGs land in `<project>/preview/`, namespaced per input file with zero-padded
  counters so pruning keeps the actual newest frames. New `prune_samples`
  command keeps only the newest sample renders in a folder.

- **Ranged renders: stable timestamps + container duration (2026-08-18)** — the
  encoder passes `-copyts` so the piped video keeps its 0-based PTS (the muxer
  no longer shifts it by the seeked-and-copied audio start, which broke
  compare/result alignment) and `-shortest` so copied audio cannot over-run a
  ranged render (the container duration no longer over-reports).

- **Persistent preview decode + PNG frames (2026-08-18)** — `senmei-media` keeps
  one long-lived ffmpeg decode stream per file (`PreviewCache`), so the monitor
  reads the next frame from the pipe instead of spawning ffmpeg per frame.
  `encode_png` replaces the mjpeg round-trip (range-safe on every FFmpeg build).

- **Fix runtime asset scope (2026-08-18)** — `probe_video` and `read_frame` now
  also extend the asset-protocol scope at runtime via `app.state::<Scopes>()`
  `allow_file`, so arbitrary video paths (e.g. outside `$HOME`) and freshly
  written preview frames are always loadable by the webview, even before the
  config globs apply.

- **Fix asset protocol scope (2026-08-18)** — the `assetProtocol` scope was
  `["**"]`, which matches almost nothing: Tauri enables `require_literal_separator`
  for the scope (so `**` behaves like `*`) and requires a literal leading dot to
  match hidden dirs like `~/.local`. Now `["$DATA/**", "$HOME/**"]`, which covers
  the preview temp frames (app data dir) and the user's videos under home.
  Fixes `asset protocol not configured to allow the path` in the monitor.
- **Monitor frames via asset protocol, not data: URIs (2026-08-18)** —
  `read_frame` now writes the extracted frame to a temp PNG in the app data dir
  and returns its path; the monitor loads it with `convertFileSrc`. Large
  frames as `data:` URIs could fail to render in WebKitGTK (broken image icon),
  while the asset protocol already works (native video). Old preview frames are
  capped at 30.
- **Preview frames as PNG instead of mjpeg (2026-08-18)** — `extract_frame`
  encodes the preview frame to PNG. The mjpeg encoder refuses limited-range
  (tv) YUV from libx265/HEVC renders ("Non full-range YUV is non-standard")
  unless `-strict unofficial` is passed, which still produced a broken preview
  on some FFmpeg builds; PNG has no such range restriction. Frontend now uses
  `data:image/png` for decoded frames.
- **Fix monitor frame read-back of HEVC/x265 renders (2026-08-18)** —
  `extract_frame` now passes `-strict unofficial` to the mjpeg `image2pipe`
  encode. The mjpeg encoder refuses limited-range (tv) YUV from libx265/HEVC
  renders without it ("Non full-range YUV is non-standard"), which made the
  result/compare preview fail right after rendering with an ffmpeg error.
  (Superseded by the PNG switch above.)
- **Preview uses the pipeline's ffmpeg (2026-08-18)** — `extract_frame` no
  longer resolves ffmpeg from the current directory; the `read_frame` command
  resolves the same binary the pipeline uses (app data dir / bundled) and passes
  it in. Fixes frame read-back of rendered output failing with an ffmpeg error
  when system ffmpeg is missing or differs.
- **Keep render position after rendering (2026-08-18)** — the monitor no
  longer jumps to position 0 after a render. The position is only reset when a
  new file loads; view switches preserve it, and the result view clamps to the
  sample in-point so it shows the rendered moment. The sample range is no
  longer reset on view switches either, and the result frame is read at
  `ms − inMs` (its timeline starts at inMs). Verified: render of a 30–90s sample
  ends in the Result view at 00:00:30 with In/Out preserved.
- **Slider sample-range highlight (2026-08-18)** — the timeline slider's track
  is now transparent with drawn underlays: a slate base, an indigo played fill
  up to the current position, and the sample window as a strong indigo bar with
  a ring. Previously the highlight sat behind the opaque native track and was
  invisible.
- **Compare alignment (2026-08-18)** — in Compare both sides now show the same
  source moment: the original is clamped to the sample in-point (the rendered
  sample has no frames before it) and the result is read at `source − inMs`
  (its timeline starts at inMs). Previously the result was offset by the sample
  start, so Original at 0 vs Result at the render point were misaligned.
- **Monitor preview opacity (2026-08-18)** — a loaded frame/video now shows at
  80% opacity; the pre-load placeholder is 70% and greyscaled, consistently in
  Original / Compare / Result.
- **Monitor placeholder 80% everywhere (2026-08-18)** — the no-frame
  placeholder now uses the same 80%-translucent background in all three views
  (Original / Compare / Result) so the monitor looks consistent from start.
- **Sample selector as segmented control (2026-08-18)** — the monitor's sample
  range picker is now a compact segmented control `[10s | 30s | 60s | Full | ▾]`
  instead of a dropdown field: presets are one-click segments, the ▾ opens a
  small popup with the custom duration editor (55s, 10m, 1m30s, 1h), and an
  active custom range shows its duration next to ▾. No double field anymore.
- **Native video preview + FFmpeg fallback (2026-08-18)** — the monitor source
  preview now uses a native `<video>` element (hardware decode, via the Tauri
  asset protocol + `convertFileSrc`), falling back to the FFmpeg-decoded frame
  path only when the webview cannot load/play the file (`onError`). Play, scrub
  and the sample in/out loop are wired to the video element. The asset protocol
  is enabled in `tauri.conf.json` (scope `**`). Binding decision updated in
  `AGENTS.md` + `PLAN.md` §1/§3.2. Browser demo unchanged (frames path).
- **Monitor playback + sample dropdown (2026-08-18)** — playback now runs the
  time indicator 1:1 real-time with at most one frame decode in flight (frames
  are skipped if the decoder lags, so FFmpeg subprocesses never pile up — this
  also fixes a performance regression/crash). The sample selector is now a
  dropdown menu like the Output folder (10s/30s/60s/Full/Custom…), with the
  custom duration editor supporting `55s`, `10m`, `1m30s`, `1h`. Fixed a bug
  where picking a preset produced `NaN` (unit strings were parsed with
  `Number()` → now `parseInt`). Verified: 30s → Out 00:00:30.00, 60s →
  00:01:00.00, custom 55s → 00:00:55.00, 10m → 00:10:00.00. The sample panel
  now carries `relative z-10` so its upward-opening dropdown paints above the
  positioned preview area, and the menu is a compact 2-column grid (~89 px tall)
  so it no longer covers the preview. The menu is left-aligned (`left-0`) so it
  grows into the free space to the right instead of clipping at the panel's left
  edge.
- **Monitor sample bar (2026-08-18)** — removed the redundant "Preview sample
  (15s)" button, promoted "Render Sample" to a filled primary button like
  Start Render, and made the sample range default to 10 s (highlighted preset).
- **Demo Compare/Result (2026-08-18)** — the browser demo now simulates a
  rendered output per video, so the Compare and Result tabs work immediately
  (previously they stayed disabled until a fake render finished). The simulated
  result gets a subtle saturate/brightness filter so the split visibly differs;
  the real Tauri app still only enables them after an actual render.
- **Docs cleanup (2026-08-18)** — `models.md` gains a status-at-a-glance table
  and a backlog/candidates section (per-stack, spandrel as source);
  `benchmarks.md` gains a TL;DR box with the engine decision and key numbers.
- **UI backlog (2026-08-18)** — About dialog (Help → About: version, engine,
  license, GitHub link), media-library multi-select (plain click selects one,
  Ctrl/Cmd+click or the ⧉ toggle adds/removes), and the version badge moved
  from the top headers to the bottom-right (status bar + project screen).
  Verified in the running app.
- **Color metadata (M4, 2026-08-17)** — the Output step gains a Color group
  (primaries / transfer / matrix) that tags the encode with `-color_primaries`,
  `-color_trc` and `-colorspace`. Verified in the app: bt2020 primaries →
  `-color_primaries bt2020` in the command preview.
- **FFmpeg quality profiles + command preview (M4, 2026-08-17)** — the Output
  step gains a Quality dropdown (Lossless / Very High / High / Medium / Low)
  that sets crf + preset as a bundle ("Custom" when they diverge), persisted as
  `StepParams.quality`, and a live command preview showing the merged ffmpeg
  args. Verified in the app: Lossless → crf 0 / preset slow, preview updates.
- **Render only the sample range (M5, 2026-08-17)** — the render command and
  pipeline accept `startMs`/`endMs`: the decoder seeks with fast `-ss` and caps
  the frame count, the encoder seeks the audio input so it stays in sync, and
  progress totals reflect the range. The Monitor's sample window now drives a
  "Render Sample" button. Test `render_only_time_range`: 200..700 ms of a 10
  fps clip yields exactly 5 frames.
- **Sample preview range (M5, 2026-08-17)** — the Monitor timeline gains an
  in/out sample range: 10s/15s/30s/60s/Full presets set the window from the
  current position, playback loops inside it, the selected range is highlighted
  on the slider and In/Out markers shown below. Verified in the running app
  (10s preset sets Out 00:00:10.00).
- **RIFE e2e verified (M3, 2026-08-17)** — `infer_interp` now pads the input
  to multiples of 32 (matching rife-ncnn-vulkan, whose flow estimation runs at
  1/32 scale) and crops the output back. Non-32 inputs previously hit a `Cat`
  shape mismatch (e.g. 120 vs 128). New pipeline test
  `rife_interpolates_real_model_e2e` runs decode → real `flownet.bin`
  interpolate (Vulkan fp16) → encode: 10 frames @10fps in → 19 frames @20fps
  out (needs `RUST_MIN_STACK=33554432`).
- **Docs reorganization (2026-08-17)** — PLAN.md §15 moved to
  `docs/CHANGELOG.md`; PLAN.md's front sections rewritten for the current
  reality (burn-Vulkan fp16 is the engine; ncnn engine removed from the plan,
  it survives only as a weight format for RIFE). `models.md` and
  `benchmarks.md` cleaned up (consistent RIFE v4.6 status, single clear engine
  verdict).
- **Project export/open (2026-08-17)** — File → "Export Project…" writes the
  project as a **`.tar.xz`** archive (tar + liblzma, same path as the FFmpeg
  download — no zip crate, which would clash on the native `lzma` link). The
  project screen's "Open Project…" button (was "Open other folder…") imports a
  `.tar.xz` back into the app storage and opens it. "Save Project As" was
  dropped — export/open round-trips instead.
- **UI polish (2026-08-17)** — the media-library drop box now only shows when
  no video is loaded (and fills the whole panel height); videos can be added
  by dragging them anywhere in the window (Tauri `onDragDropEvent` with full
  paths; HTML5 fallback in the browser demo). The `↓` arrows between stack
  steps are gone. The top bar shows `project / video` centered (no pill box),
  and a settings gear sits at the bottom-left of the status bar.
- **Reference filter stacks (M7, 2026-08-17)** — the previously disabled
  `denoise`, `deblur` and `deduplication` steps are now implemented with CPU
  references: box-blur denoise (radius), unsharp-mask deblur (amount), and a
  stateful dedup that drops frames below a mean-pixel-diff threshold.
  `Step::process` now returns `Result<bool>` (false = drop the frame), so the
  pipeline `run_step` can skip frames. The `render` command takes a bundled
  `RenderConfig` (specta caps command arity at 10 args — all knobs moved into
  one struct) with an optional `filter: FilterParams`. Also fixed the long-
  standing `value assigned to failed is never read` warning in pipeline.rs.
- **RIFE v4.6 engine wired (M3, 2026-08-17)** — `RifeNet::load_from_ncnn` parses
  the ncnn `flownet.bin` (per layer `[tag u32][weights f16][bias f32]`) and
  assigns params directly (conv weights `[out,in,k,k]`, deconv transposed to
  `[in,out,k,k]` — burn's `ConvTranspose2d` wants weight[0] = input channels,
  while ncnn stores deconv weights out-major). `BurnEngine` gains a `RifeNet`
  model + `infer_interp(a, b, t)`: frames are f16 NCHW, the timestep is a
  broadcast `[1,1,H,W]` filled with `t` (matching ncnn's `in2`), output returns
  as f32. Loaded and verified on Vulkan: weights walk the whole `.bin` exactly
  to EOF; the interpolated frame is flow-based (≠ linear blend), symmetric
  under `(a,b,t)↔(b,a,1-t)`, and directionally correct (t=0.05→a, t=0.95→b;
  the reference short-circuits exact t=0/1 so endpoints aren't a network
  property). Catalog entry renamed to the real artifact: `rife-v4.6`
  (`arch rife46`, MIT, `flownet.bin`).
- **RIFE v4.6 burn port (M3, 2026-08-17, generated)** — `tools/rife_gen_burn.py`
  translates the ncnn `flownet.param` (215 layers, MIT) into a straight-line
  burn network (`senmei_ml::burn::rife::RifeNet`): 40 Conv2d + 4
  ConvTranspose2d + op helpers (warp = `grid_sample`, bilinear interp,
  pixel-shuffle, channel crop, binary ops), with per-output use-counting for
  burn's move semantics. It **compiles and runs end-to-end on Vulkan**,
  preserving `[1,3,H,W]` with finite values (ignored structural test; needs a
  larger thread stack).
- **grid_sample foundation (M3, 2026-08-17)** — new `senmei_ml::burn::grid_sample`
  (bilinear warp, `align_corners=True`, border padding) matching torch
  semantics; each corner is sampled with a single gather over a flattened
  spatial axis (`y*W + x`) because two chained H/W dim gathers re-pair the
  per-pixel indices wrongly. Verified against a CPU reference over in-range
  and out-of-range grid coords (ignored Vulkan test). This is the sampling op
  RIFE's IFNet/FusionNet warps need.
- **RIFE plumbing (M3, 2026-08-17, phase 1)** — `InferenceEngine` gains a
  2-input `infer_interp(a, b, t, opts)` (default `None` → CPU fallback). The
  pipeline `Interpolator` gets `with_engine` and routes each intermediate
  through the engine when present, else falls back to linear blend / scene-cut
  duplication. The `render` command accepts `interp_model`; the Interpolate
  step's Model dropdown now lists **rife-4.25** (Apache-2.0) from the catalog
  and auto-selects it. The RIFE burn arch port + weight conversion (Phase 2)
  is still pending — until then a selected model degrades to the blend.
- **Output filename includes model & scale (2026-08-17)** — rendered files are
  named `{stem}_{label|senmei}_{model}_x{scale}.{ext}` (e.g.
  `Folge 7_senmei_shuffle-cugan_x2.mkv`), so the applied processing is visible
  at a glance. Also fixed: the Start Render button passed its click event as
  `onlySelected` (truthy), which filtered by the empty selection and never
  started the batch — wrapped in `() => startBatch()`.
- **Selection + Edit/Process menus + hotkeys (2026-08-17)** — library rows
  are selectable (ring highlight; click toggles). Ctrl/Cmd+A selects all,
  Delete removes the selection, Ctrl/Cmd+R starts the batch render. The
  menubar gains **Edit** (Select All Videos, Delete Selected) and **Process**
  (Add All / Add Selected to Queue, Process Selected / Queue / All); Process
  Selected renders only the chosen files, Add to Queue switches the media
  panel to the Queue tab.
- **Header & project-screen polish (2026-08-17)** — the app header drops the
  "Senmei" wordmark (logo + version badge suffice) and gains a gear Settings
  button (like Koharu). The project screen header now matches the main app
  (鮮 logo + version badge) and deleting a project uses an in-app themed
  confirm modal instead of `ask()`/`window.confirm`. Step titles in the stack
  show their model & scale ("2. Upscale · shuffle-cugan ×2", "1. Interpolate ×2").
- **Fix: neon color artifacts on hard edges (2026-08-17)** — the GPU output
  path (`infer_rgb8`) cast to U8 **without clamping**, so model values >1.0 at
  hard edges (burnt-in subtitles) wrapped (e.g. 275 → 19 → magenta/cyan).
  Now `out.clamp(0.0, 1.0)` before the 0..255 scale + U8 cast. The CPU path
  already saturates (`as u8`). Regression: `app_render_upscales_real_model`.
- **Batch rendering (2026-08-17, M7)** — `Start Render` now renders **all files
  sequentially** (a single file is a batch of one). The Queue tab lists one job
  per file with status (queued/rendering/done/failed/cancelled), per-file
  progress bar + frames, and batch controls (Pause/Resume, Stop). Errors mark
  the file failed and continue; Stop aborts after the current file; Pause
  freezes the running file. Output paths are derived from the Output-step
  (folder mode / label / container); new `unique_path` command appends
  `_2`, `_3`, … on filename collisions instead of overwriting. No per-file
  save dialog (auto path + dedupe).
- **Stack reorder via drag (2026-08-17)** — the ▲▼ move buttons are replaced by
  a ≡ drag handle; the whole step header is draggable (pointer-based
  mousedown/move/up with a 4px click-vs-drag threshold, target hit-test on
  `data-step-index`). WebKitGTK handles HTML5 DnD unreliably and renders a huge
  ghost, so this avoids both; a `setTimeout(0)` clears the post-drag click
  suppression so the next click still expands.
- **Pause/resume render (2026-08-17)** — the pipeline waits between frames on a
  pause flag (`set_pause`); `pause_render(bool)` command toggles it. The Queue
  tab shows Pause/Resume next to Cancel next to the progress. Regression test
  `passthrough_pause_resume` proves frames stall while paused and resume after.
- **Output naming/flow (2026-08-17)** — the rendered filename includes the
  Output-step `label` when set (`{stem}_{label}.{ext}`, else `{stem}_senmei.{ext}`);
  when a folder mode is configured (Global/Custom) the render writes straight
  into that folder — no save dialog. Dialog only remains for "Same as input".
- **TopBar cleanup (2026-08-17)** — the redundant Cancel button is gone from
  the topbar; Cancel already lives next to the render progress in the media
  library.
- **Resize factor decimal (2026-08-17)** — the factor field is a text input
  (`inputMode=decimal`) that normalizes a comma to a dot, since `type=number`
  silently drops the comma before `onChange`.
- **Version badge with build hash (2026-08-17)** — the TopBar shows
  `v0.1.0-<short-hash>`; Vite injects `__APP_VERSION__`/`__BUILD_HASH__`
  (last commit) via `define`, so every build identifies its exact source.
- **Audio passthrough (2026-08-17)** — the encoder now takes the source file as
  a second ffmpeg input and maps its audio (`-map 0:v:0 -map 1:a:0?`), so the
  rendered file keeps the soundtrack. The Output-step Audio dropdown drives it:
  `Passthrough` → `-c:a copy`, `AAC`/`Opus`/`FLAC` → re-encode, `None` → `-an`.
  Pipeline passes `input` to `Encoder::open`; regression test
  `passthrough_copies_audio`.
- **Dev stale-UI fix (2026-08-17)** — WebKitGTK showed a stale/cached page
  under Wayland; `dev:release`/`dev` now run under XWayland
  (`GDK_BACKEND=x11`), Vite binds `127.0.0.1` explicitly, `devUrl` matches it,
  and `predev`/`predev:release` auto-run `dev:clean` (kill port 1420 + senmei,
  clear WebKit cache) like Koharu's `predev: kill-port`.
- **Structured encoder settings + merge (2026-08-17)** — the `output` step
  gains RVE-style structured fields, all persisted in `StepParams`
  (`crf`/`preset`/`pix_fmt`/`tune` + existing `videoCodec`/`audioCodec`/
  `subtitleMode`): Preset select, CRF number, Pixel-format select, Tune select.
  `buildEncoderArgs()` (frontend) **merges** them with the raw FFmpeg field:
  the custom field wins for any flag it defines (e.g. `-tune grain`), the
  dropdown values fill the rest (e.g. `-pix_fmt`); the merged string is passed
  to `render` as before. The output-step `label` param (renamed from `name`)
  is empty by default → the badge shows only when a real multi-output label is
  set. Output-step also gains **Format** (`container`, default `mkv`) used for
  the save-dialog extension, and **Output folder** mode (`output_mode`:
  `input`/`global`/`custom` + `output_folder` picker) that sets the save
  default target.
- **Custom FFmpeg output options (2026-08-17)** — the `output` step gains a
  **FFmpeg options** textarea (`params.ffmpegArgs`, persisted in `project.json`
  via `StepParams.ffmpeg_args`). `render` accepts `ffmpegArgs: Option<String>`
  (shell-like tokenizer with quote support), `Pipeline::set_encoder_args`
  threads them into `Encoder::open`, which appends them **after** the built-in
  x264 defaults so user codec/filter args override them. Verified end-to-end by
  the app smoke test: `-c:v libx265 -crf 18 -preset ultrafast -pix_fmt
  yuv420p10le` → ffprobe confirms HEVC + 10-bit output. Default output stays
  x264 `veryfast` (overridable via `SENMEI_X264_PRESET`).
- **Pipeline-stack Inspector (2026-08-17)** — Inspector's flat accordion list is
  replaced by a **dynamic layer stack** (order top→bottom = execution order):
  add steps via a "+ Add step" menu, remove (✕), enable/disable (checkbox),
  reorder (▲/▼). Step types: `interpolation`, `upscale`, `denoise`, `deblur`,
  `deduplication`, `resize`, `output` — the **not-yet-implemented** ones
  (denoise/deblur/dedup) are **disabled in the add menu** ("Soon"). `output` is
  a regular step addable anywhere (multi-output design: each carries a `name`
  label + video/audio codec + subtitle mode; the backend renders the last
  active one for now). `ProjectSettings` schema changed
  (`stepsEnabled`/`upscaleModel`/`scale` → ordered `steps: Vec<PipelineStep>`
  with a typed `StepParams`); bindings regenerated via the specta export test.
  Frontend holds `steps[]` in App state, persists per project, and `startRender`
  derives scale/model/resize/fps from the **enabled** steps. Model select
  auto-fills the first loadable upscaler (ShuffleCugan).
- **UX feedback batch (2026-08-17)** — projects are deletable (🗑 on the
  project screen, `delete_project` command, confirm dialog); videos can be
  removed from the library (✕ per row); **cancel render** (TopBar ■ + Queue
  tab) via a shared `AtomicBool` checked between frames — partial output is
  deleted on abort; Monitor gains a **Compare (side-by-side)** mode for the
  source/result frames plus an auto-switch to the Result view when a render
  finishes. Preview extraction now uses the **resolved ffmpeg** (portable
  fallback) instead of bare `ffmpeg`. Tile size raised 256 → 512 to cut GPU
  sync overhead (better GPU utilization at 1080p). `dev:release` script added
  (`cargo tauri dev --release`) — debug builds render 10–50× slower.
- **Prototype polish (2026-08-17)** — per-project persistence extended: selected
  model/scale, imported videos and output folder are saved in `project.json`
  (`ProjectSettings`). **ShuffleCugan** is the default upscaler (converted f16
  `.bpk`; license flagged "prototype opt-in" pending author clarification).
  Output folder is pickable (Media Library 📁) and used as the render save
  default. Queue tab shows the active render + finished output. Monitor gains
  Original/Result tabs (previews the rendered file) + an in-view render progress
  overlay. Language switch removed from the top bar (Settings only).
- **Preview prototype (2026-08-17)** — working Monitor: new `probe_video` +
  `read_frame` commands (`senmei_media::extract_frame`: ffmpeg `-ss pos -i …
  -frames:v 1 -c:v mjpeg -`, base64 JPEG over IPC) drive a canvas `<img>`
  preview with a **timeline scrubber** (debounced seek) + play/pause. Inspector
  gains a **Download weights** button (`download_model` now reachable from the
  UI). Render now honors `stepsEnabled` (default-on; toggling a step off
  disables it). End-to-end proof: ignored pipeline test
  `burn_engine_upscales_real_model` runs decode → real `real-cugan-x2` burn
  Vulkan fp16 (tiled) → encode → 320×240.
- **Engine switch v3 (decision, 2026-08-17)** — ncnn removed; inference = **burn
  (`burn-wgpu`) on the Vulkan backend, fp16**, CPU fallback. Deleted
  `crates/senmei-ncnn` (C++ shim) and `NcnnEngine`; dropped the `ncnn` registry
  field. Replaced `xz2` with `liblzma` in `senmei-media` (resolves the
  `links="lzma"` conflict with `cubecl-cpu`/`tracel-llvm-bundler`). Added
  **`BurnEngine`** (feature `senmei-ml/burn`, wired into `senmei-app`): loads f16
  `.bpk` burnpacks via `BurnpackStore` and runs the clean **`UpCunet2x`** arch
  (port from `~/github/rust-sr-bench`, verified) on `Vulkan<f16>`. Registry
  schema: `ncnn` → `weights` + `download_url`/`sha256`; `models/metadata.json`
  re-catalogued from VSGAN/TAS hosts. Archs ported: **`upcunet2x`**,
  **`upcunet2x-fast`** (ShuffleCugan) and **`realesrgan`** (RRDBNet, scale 2/4
  via `Option` conv_up2, `num_block` from metadata) — `real-cugan-x2` + 3×
  Real-ESRGAN are `loadable`; SCUNet / Real-PLKSr / Anime1080Fixer (license
  verify) and RIFE (+ 2-input API) still pending. `BurnEngine` dispatches on
  `ModelRef::arch`. See `docs/models.md` + `docs/benchmarks.md`.
- **Burn re-benchmark (2026-08-17)** — re-tested burn with the **real** Real-CUGAN
  upcunet (`up2x-no-denoise.pth` via `burn-store::PytorchStore`) instead of the
  3-conv toy. All outputs numerically verified against the torch reference.
  Findings: **burn-ROCm f32** = 1119/2197 ms @720p/1080p and fp16/bf16 are
  **impossible on RDNA4** (cubecl-hip uses CDNA-only WMMA kernels → `LLVM ERROR`);
  **burn-Vulkan fp16** runs the real model at **136/302 ms** (720p/1080p) —
  *faster than ncnn* (249/398 ms) — and the **ShuffleCugan** variant at
  **46/103 ms**. Vulkan f32 1080p crashes on a `burn-fusion` bug. This **revises
  the 2026-08-16 "burn set aside" verdict** (it was a toy on the wrong backend);
  burn is re-opened as a candidate, but adoption must weigh the ~800-crate /
  1.6 GB build, the fusion bug, and the f32→f16 load workflow. Engine stays
  **NCNN/Vulkan** until a maintainer decision. Details: `docs/benchmarks.md`;
  repo: `~/github/rust-sr-bench`.
- **Candle-ROCm evaluation (2026-08-17)** — tried the `xmiksay/feat/rocm-backend`
  candle fork (local `~/github/candle`, branch `test/xmiksay-rocm`; rocBLAS GEMM
  + im2col conv) via a feature-gated `candle` bin in `~/github/rust-sr-bench`.
  Numerically correct (HIP vs CPU ~1e-5), but f32 convs always materialize the
  im2col matrix → memory cliff from ~640p (multi-GB buffers crash the desktop on
  shared-display GPUs; SD/FLUX VAE decode OOMs at 1024²); f16 scales linearly
  but stays ~6× slower than burn-Vulkan fp16 (290 vs 46 ms @720p ShuffleCugan);
  the ShuffleCugan port additionally OOMs at any size (fork conv bug).
  **Not pursued — burn stays the candidate** (Vulkan fp16). Abandoned work
  remains feature-gated/uncommitted in `rust-sr-bench`.
- **Weights workflow (2026-08-17)** — `senmei-ml` gains a feature-gated
  `senmei-ml-convert` bin: loads a torch `.pth` (f32, Vulkan, upcunet key
  remap) and saves the arch as an f16 `.bpk` burnpack (`HalfPrecisionAdapter`).
  Proven end-to-end on the real `up2x-no-denoise.pth` (→ 2.5 MB `.bpk`); an
  ignored GPU test loads the `.bpk` through `BurnEngine` and infers 32×32 →
  64×64. New `download_model` Tauri command: downloads the `.pth`
  (`download_to_temp`, sha256-verified when pinned) and converts it to the
  `.bpk` in-app. Removed dead `extract_zip` from `senmei-media`.
- **Archs (2026-08-17)** — ported **`UpCunet2xFast`** (ShuffleCugan, from
  `rust-sr-bench`) and **`RrdbNet`** (Real-ESRGAN, BSD-3 reference) into
  `senmei-ml::burn`; `BurnEngine` now dispatches on `ModelRef::arch`
  (`upcunet2x` / `upcunet2x-fast` / `realesrgan`). `RrdbNet` uses burn's
  `Vec<Rrdb>` (torch `body.0…`) and `Option<Conv2d>` (`conv_up2` only at
  scale 4). Real-ESRGAN models flipped `loadable`; RRDBNet numerical
  verification vs torch is the next step (rust-sr-bench harness).
- **M6 (foundation, 2026-08-17)** — new `crates/senmei-ncnn` C++ shim (bindgen + cc): `build.rs` builds NCNN `20260526` from `third_party/ncnn` (Vulkan + CPU, auto-cloned if missing; dir is gitignored) and exposes a safe Rust `Engine` (load `.param`/`.bin`, planar NCHW infer). `NcnnEngine` in `senmei-ml` is now real (was a stub). Verified with the Real-CUGAN `up2x-no-denoise` model — its upcunet crops a fixed border (`out = 2·h − 72`), which the shim/engine faithfully returns; border-aware tiling is a follow-up. `metadata.json` pins the real asset name `up2x-no-denoise`. Build deps: `cmake`, `g++`, Vulkan.
- ~~**Inference engine switch v2 (decision, 2026-08-16)**~~ — **superseded 2026-08-17 (v3: burn/Vulkan)** — after benchmarking on the target AMD RX 9070 (RDNA4/`gfx1201`), the engine was **NCNN/Vulkan** via C++ shim (`cxx`/bindgen) with **CPU fallback**. **candle dropped** (no ROCm backend; per-model Rust ports). **burn set aside** (fusion/JIT immature for SR). Model format was ncnn `.param`/`.bin` (community ports) — **no safetensors graph loading, no conversion, no Python, no Rust arch ports**. The per-model cost is finding a permissively-licensed NCNN port. Evidence in `docs/benchmarks.md`: ncnn 1080p x2 = 398 ms vs torch-ROCm 7153 ms (pathological) + tile OOM/hard-fault on RDNA4. Obsolete vs v1: `CandleEngine`, `.safetensors` loading, "port each arch to Rust" plan. Registry schema: `torch` field → `ncnn` (see §6.4).
- ~~**NCNN-only code switch (2026-08-16)**~~ — **superseded 2026-08-17 (v3)** — removed the `torch` feature, `tch` dep, `TorchEngine`, and the `torch`/`download_url`/`sha256` `ModelMetadata` fields. `engine_for_model` mapped only `.param`/`.bin` → `NcnnEngine`; `Registry::resolve` pointed at the `.param` (the `.bin` sat alongside). Registry = 7 NCNN models (`rife-4.26`, Real-ESRGAN ×3, Real-CUGAN up2x, SwinIR x2/x4) — `loadable: false` until the C++ shim lands (M6). Dropped the `download_model` command + `senmei_media::download_model`, the `scripts/convert_*.py` pipeline, and local `models/*.pt`. `Backend` = `Cpu | Vulkan`. Bridge bindings regenerated. Exact NCNN asset filenames still need pinning in M7.
- **Download-on-demand (decision, 2026-08-16)** — model weights are **not bundled or redistributed**. The app downloads `.param`/`.bin` from a pinned upstream URL on first use (M7). Keeps the runtime small and sidesteps redistribution-license questions for models whose ports lack a clear license; `metadata.json` records license + source for transparency.
- **Libtorch downloader/UI cleanup (TODO, 2026-08-16)** — the libtorch provisioning path still exists (`senmei-media/src/libtorch.rs`, `get_libtorch_status`/`download_libtorch`, `useLibtorch`, SettingsPage inference section, i18n strings) and contradicts the engine switch. Remove it in a follow-up; the Settings inference section should later show the NCNN backend instead.
- ~~**Inference engine switch (decision, 2026-08-16)**~~ — **superseded by v2** (candle dropped after benchmarks; engine = NCNN/Vulkan) — libtorch/`tch`/TorchScript is **dropped**. `senmei-ml` moves to **candle** (CPU/CUDA/Metal) + **NCNN/Vulkan** (no ONNX, no TorchScript). Models are **downloaded** as `.safetensors` from pinned HF repos (Koharu-style `model_repository!` pattern, repo + commit SHA) — **no conversion, no Python**. Each architecture is **ported to Rust** (`candle-nn`) once; that is the main per-model cost. Consequence: the ROCm/AMD-Linux accelerated path is dropped (AMD → NCNN/Vulkan). Obsolete: the `torch` feature, `tch` dep, `TorchEngine`, `scripts/convert_*.py`, and the existing `.pt` files (models re-fetched as `.safetensors`).
- **M0 done** — workspace, 5 crates, `InferenceEngine` trait + engine stubs + model registry, Tauri shell (frameless), React UI, `models/metadata.json`, LICENSE (MIT/Apache), crates.io names secured.
- **M1 done** — FFmpeg passthrough: `senmei-media` decoder/encoder (`rawvideo` pipe), `senmei-pipeline` (`Step`, `Passthrough`, `Pipeline::run`), `render` command + progress channel.
- **UI deviations from §3.1** — Settings panel uses **2 top-level tabs** (`Settings` / `Advanced`) with **accordions** inside, instead of the 6 tabs from the plan. Steps: Interpolate, Decompress, Denoise, Deblur, Upscale, Deduplication, Resize, Output Resize (Settings) + Video Encoder, Audio Encoder, Backend (Advanced).
- **Project flow** — start screen with new/open project; projects stored as directories under `~/.local/share/senmei/projects/` (+ JSON index for browsed folders).
- **Settings page** — dedicated page (not modal) with section sidebar; **Appearance** section holds Language (EN/DE) + Theme (light/dark/system), persisted in `~/.local/share/senmei/settings.json`. Extensible for more settings.
- **i18n** — English default + German; switch in the top bar and Settings.
- **Window controls** — minimize/maximize/close on both the project start screen and the main window (frameless).
- **Theme** — light/dark applied across all components via Tailwind `dark:` classes; `system` follows `prefers-color-scheme`.
- **UI fix** — top bar has `z-50` so menu dropdowns render above the live monitor (stacking-context issue with `backdrop-blur`).
- **Window controls fix** — Tauri v2 ACL: `minimize`/`toggleMaximize`/`close` need explicit `core:window:allow-*` permissions in `capabilities/default.json` (not part of `core:default`). **Window dragging** needs `core:window:allow-start-dragging` (also not in `core:default`) — added.
- ~~**libtorch provisioning (decision)**~~ — **superseded 2026-08-16** (libtorch dropped; see engine switch) — libtorch is **not** bundled; downloaded at first run via Settings → Inference: backend auto-detected (CUDA via `nvidia-smi`, else ROCm via `/dev/kfd`, else CPU) and the matching pytorch.org archive is fetched into `~/.local/share/senmei/libtorch/`. **Version:** `tch 0.24` / `torch-sys 0.24` expect **libtorch 2.11.0** (URLs pinned to 2.11.0: CPU / cu126 / **rocm7.1**; newer archives use the `libtorch-shared-with-deps` filename — no `cxx11-abi`). **Note:** `tch` links libtorch at **build time**, so after the download build with `LIBTORCH=~/.local/share/senmei/libtorch cargo build --features senmei-ml/torch`. Runtime dynamic-load is not supported by `tch`.
- ~~**ROCm not a system dependency (decision)**~~ — **superseded 2026-08-16** (ROCm path dropped with libtorch) — the libtorch ROCm archive **bundles its own ROCm runtime libs** (`libamdhip64.so`, `libMIOpen.so`, `librocblas.so`, … resolve inside `libtorch/lib/`, verified via `ldd`). End users therefore need **no system ROCm install**; only a Linux kernel with `amdgpu` + KFD (`/dev/kfd`) and an AMD GPU. Backend detection uses `/dev/kfd`, not an installed ROCm version. At M8 (packaging) ship the bundled libtorch and document „AMD GPU + `/dev/kfd`" as the requirement.
- **FFmpeg sourcing (decision)** — prefer **system FFmpeg** (Linux: x264/x265/NVENC/VAAPI present). If missing/too old: download **portable FFmpeg** (BtbN GPL builds) into `~/.local/share/senmei/bin/` with progress UI (RVE-style). `get_ffmpeg_status` + `download_ffmpeg` commands; no bundling in installer (GPL binary is a separate process, does not affect MIT/Apache code). macOS download TBD at M8. Resolution order (used by status AND the decode/encode pipeline): valid `SENMEI_FFMPEG` env → system `ffmpeg` → portable. `SENMEI_FORCE_FFMPEG_MISSING=1` simulates a missing FFmpeg for testing the download flow.
- **Webview decision (revision)** — CEF dropped; use Tauri platform webview. In-app preview is **frame-based** (FFmpeg decode → canvas) so it is codec-agnostic (incl. H.265); audio via `<audio>` (AAC/Opus). Final output via FFmpeg (x264/x265).
- **tauri-specta** — typed bindings replace the hand-written bridge: `collect_commands!` + `#[specta::specta]`, `bindings.ts` generated (camelCase, Throw errors), bridge re-exports. `export_ts_bindings` test regenerates.
- **Logging** — `log`/`env_logger` initialized in `senmei`; logs in commands/media/pipeline (render, download, ffmpeg install).
- **Tests** — registry (`from_json`, `load_dir`), encoder capability parsing, settings roundtrip/defaults, project dir creation, passthrough, ffmpeg probe (9 tests).
- **Download integrity** — SHA-256 of the portable FFmpeg archive is verified against a pinned constant (`FFMPEG_SHA256`); BtbN provides no stable tag/checksum, so update the constant on bump.
- **De-mocked UI** — render button wired (output dialog + progress in status bar); TopBar/StatusBar/Monitor driven by real state (files, health, FFmpeg version); Inspector model selects populated from the real model registry via `list_models`.
- **UI kit** — `packages/ui` provides theme-aware `Button` (primary/secondary/ghost) and `Chip`; used in ProjectScreen/SettingsPage; added to Tailwind content.
- **M2 (partial, upscaling)** — `senmei-ml`: **tiling** (tile/stitch, overlap, tested) + **reference bilinear scaler** (tested). `senmei-pipeline`: **`Upscale` step** (Frame↔Tensor, engine or reference fallback, tested). `render` accepts `scale`; UI has 2x/3x/4x control (Inspector) and progress; **upscaling works end-to-end** via the reference scaler without ML. **`TorchEngine`** (real `tch`/libtorch) is implemented behind the `torch` feature — **requires a full libtorch install (headers) + a TorchScript model**; not compiled/verified here (local libtorch is runtime-only). Enable with `--features senmei-ml/torch` + `LIBTORCH=<full-libtorch>`.
- **M2 (tiled inference)** — **`infer_tiled`** in `senmei-ml` wraps an engine so large inputs are split into overlapping tiles, inferred per tile, and stitched (overlap-averaged, canvas scaled by the engine's per-tile scale). Used by `Upscale` with a default `tile_size` of 256 when the engine advertises `tiles`. **Tests** (4): identity reconstruction, scaled output dims, skip-tiling on small input, whole-image path for engines without tiling.
- **M2 (engine selection)** — **`engine_for_model`** picks an engine by weight-file format (`.pt` → `TorchEngine`, `.param`/`.bin` → `NcnnEngine`, else error). **`Registry::resolve(id, dir)`** maps a model id to a `ModelRef` pointing at its torch weight file. The `render` command now takes an optional `model_id`: the Inspector's Upscale model select passes it through, so a real engine is loaded and handed to the `Upscale` step (reference scaler remains the fallback when no model is selected). **Tests** (2): factory-by-format, registry-resolves-model-ref.
- **M2 (model download)** — **`senmei_media::download_model`** (reusing the shared `downloader`) fetches a model weight file into `models/`, verifies SHA-256 against `metadata.json`'s `sha256` (temp-file-then-rename, so a mismatch never leaves a corrupt weight). `ModelMetadata` gains `download_url` + `sha256` fields. New `download_model` command + Inspector "Download weights" button per downloadable model (`useModel` hook). Real-ESRGAN `realesrgan-x4plus` has a pinned URL + checksum. **Note:** the official Real-ESRGAN release is a `.pth` state dict — a one-time conversion to TorchScript `.pt` (per PLAN §6) is still required before `TorchEngine` can load it. **Tests** (1): checksum match/mismatch.
- **M2 (resize + encoder dims)** — new **`Resize` step** (planar-RGB bilinear by a **scale factor**, tested: grow/shrink/noop/color). `Pipeline::run` now opens the encoder with the **first processed frame's dims** instead of the decoder's, so any size-changing step (upscale/resize) produces a correctly-sized output. `render` takes optional `resize` + `output_resize` (`f32` factor, applied before/after upscale); Inspector Resize/Output Resize accordions get a single factor input (empty = off), replacing the "— M2" placeholders. If a selected model can't be loaded (missing weights/unsupported format), render logs a warning and falls back to the **reference scaler** instead of aborting. **Tests** (2 e2e): 160x120 → upscale x2 → 320x240; 160x120 → resize 0.5 → 80x60.
- **M2 (review fixes)** — **Frame↔Tensor layout fix**: FFmpeg frames are packed `rgb24`, but `frame_to_tensor`/`tensor_to_frame` did a linear copy into planar NCHW, scrambling every upscaled/resized frame's pixels (dims-only tests hid it). Now de-interleaved/interleaved correctly. **Scale enforcement**: an engine's fixed upscale factor (e.g. x4) is now resized back to the requested scale, so the UI scale choice is authoritative. **`loadable` flag** on `ModelMetadata`: `realesrgan-x4plus` is marked `loadable: false` (`.pth` state dict awaiting TorchScript conversion), so render no longer auto-downloads unusable weights. Non-torch `TorchEngine` stub now reports `Cpu` (was `Cuda`), and `Decoder.total_frames` guards against 0 (progress NaN). **Tests** added: frame↔tensor pixel roundtrip, upscale x1 pixel preservation, engine-scale enforcement.
- **Project settings persistence** — the Inspector's per-step enabled toggles are persisted in **`<project>/project.json`** (`steps_enabled` map), loaded when a project opens and saved on every change (`load_project_settings` / `save_project_settings` commands, tested roundtrip). The Accordion `enabled` state is now controlled from App. Expanding an accordion enables the step; **collapsing leaves it unchanged** (fix: previously every row click also enabled the step).
- **Dev workflow (decision)** — `bun run dev` runs **`cargo tauri dev`** (Koharu-style): Tauri CLI starts Vite (`beforeDevCommand`) and `cargo run`s the app with hot-reload. `tauri.conf.json` (+ `capabilities/`, `icons/`, `build.rs`) live in the **bin crate `crates/senmei`**, not `senmei-app`, because `tauri::generate_context!`/`tauri_build` read the config relative to the crate's manifest dir; `senmei-app` stays a pure lib (`specta_builder` + commands). Root `package.json`: `dev` → `cargo tauri dev`, `ui:dev` → frontend only. **Note:** `beforeDevCommand` runs from the **repo root** (verified), so paths are `packages/app`, not `../`. `default-run = "senmei"` makes `cargo tauri dev` pick the right bin.
- **Dependency security** — **JS:** bumped `vite` `5.4.x → 7.3.x` to clear 4 Dependabot alerts (vite `fs.deny` bypass + `.map` path traversal + launch-editor NTLMv2, and esbuild dev-server — esbuild is only patched ≥0.25, hence Vite 7). `bun audit` reports no vulnerabilities. **Rust:** one open Dependabot alert on `glib 0.18.5` (`VariantStrIter` unsoundness, fixed only in ≥0.20) is an **upstream blocker** — `gtk 0.18.2` (last GTK3 binding) pins `glib ^0.18`, and `tauri 2.11.5` is the latest; no patched 0.18.x exists, so it is unresolvable by `cargo update` until the Tauri/gtk-rs stack moves to glib 0.20. **Accepted risk** (dismissed on GitHub as "Risk is tolerable to this project"): the vulnerable `VariantStrIter` API is never exercised by the app.
- **M3 (interpolation, partial)** — `senmei_ml::interpolate` provides `mean_abs_diff`/`is_scene_cut`/`blend`; a stateful pipeline **`Interpolator`** emits `factor-1` blended intermediates between consecutive frames, or **duplicates across scene cuts** (threshold 0.25 mean-abs-diff). `Pipeline::run` accepts an optional interpolator and scales the encoder **fps** and progress total by the factor; frame↔tensor conversion moved to a shared `frame` module. `render` takes `fps_multiplier`; the Inspector fps buttons (2x/3x/4x, toggleable) drive it, and the interpolate model select no longer writes to the upscale model. **RIFE TorchScript inference is still pending** — the reference path is linear blending (rife-4.26 has no downloadable `.pt` yet). **Tests**: ml blend/scene-cut (4), interpolator factor/scene-cut (4), e2e fps doubling (1).
- **Real-ESRGAN TorchScript conversion** — `scripts/convert_realesrgan.py` converts official Real-ESRGAN RRDBNet checkpoints into loadable TorchScript; it auto-detects `num_block` and the input layout (classic 3-channel, or `pixel_unshuffle` for the x2 model) and traces two 2× nearest upsamples. Registry has three loadable upscalers: `realesrgan-x4plus` (4×), `realesrgan-x4plus-anime` (6B, 4×), `realesrgan-x2plus` (2×). Verified via the ignored `torch_loads_realesrgan_models` test (loads in `TorchEngine`, 64→64·scale). The `.pt`s are **not committed** (`models/*.pt` ignored) and `download_url`/`sha256` are dropped (they pointed at unloadable `.pth`s). Requires `torch` + `libtorch` to build/run with the `torch` feature.
- ~~**Model ingest (decision)**~~ — **superseded 2026-08-16** (models are now downloaded as ncnn `.param`/`.bin`, not converted to TorchScript) — raw `.pth` state dicts are converted **once** to TorchScript, maintainer-side (`scripts/convert_realesrgan.py`, needs Python + torch). End users never convert: the finished `.pt`s are **bundled/downloaded at M8** (like libtorch provisioning). `spandrel` (MIT) can later broaden the converter to more architectures; each adopted model must itself carry a permissive license (BSD/MIT/Apache).
- ~~**More upscalers (spandrel)**~~ — **superseded 2026-08-16** (conversion pipeline dropped; models downloaded as ncnn `.param`/`.bin`) — `scripts/convert_spandrel.py` converts permissively-licensed checkpoints (`.pth`/`.safetensors`) to TorchScript via spandrel (MIT): it retargets window-attention models (SwinIR/HAT) to the runtime tile size (256), re-derives their attention masks, and verifies trace==eager. Registered 4 new loadable upscalers: `real-cugan-x2` (MIT, anime 2×), `swinir-x2` (Apache-2.0, classical 2×), `swinir-x4` (Apache-2.0, real-world 4×), `hat-x4` (Apache-2.0, real-world 4×). **Tiling fix:** traced window-attention transformers are resolution-locked, so `infer_tiled` now pads small inputs to a full tile and `tile()` edge-aligns the last tile — every tile is exactly `tile_size`; padded borders are cropped from the output. `.pt`s are not committed; verified via the ignored `torch_loads_upscaler_models` test.
