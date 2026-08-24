//! Transport-agnostic backend contract for the Senmei UI.
//!
//! The UI calls `Backend.*` only — never Tauri or fetch directly. Implementations:
//! - `tauri.ts` — wraps `@senmei/bridge` (Tauri IPC)
//! - `http.ts`  — talks to `senmei-server --http` (REST + polling)
//! - `mock.ts`  — in-memory demo data (dev only, `VITE_SENMEI_MOCK=1`)
//!
//! Progress is callback-based (not Tauri `Channel`); each implementation maps
//! it to its transport (Channel vs polling).

import type {
  BackendInfo,
  DownloadProgress,
  FfmpegStatus,
  HardwareSnapshot,
  LogEntry,
  ModelFileInfo,
  ModelMetadata,
  ProjectEntry,
  ProjectSettings,
  RenderConfig,
  RenderProgress,
  Settings,
  VideoInfo,
} from "@senmei/bridge";

export type {
  BackendInfo,
  DownloadProgress,
  FfmpegStatus,
  HardwareSnapshot,
  LogEntry,
  ModelMetadata,
  ProjectEntry,
  ProjectSettings,
  RenderConfig,
  RenderProgress,
  Settings,
  VideoInfo,
} from "@senmei/bridge";

/// A URL usable as an `<img>`/`<video>` `src` (Tauri: `asset://`, HTTP: data).
export type FrameSource = string;

/// A decoded preview frame: raw RGB24 pixels + dimensions, for direct canvas
/// rendering via `ImageData` (no `<img>`/PNG round-trip). Tauri delivers the
/// bytes as an `ArrayBuffer` (raw channel), HTTP as base64 (decoded here).
export interface RawFrame {
  width: number;
  height: number;
  /// Raw RGB24 pixels.
  data: Uint8Array;
}

/// Register a drag-and-drop handler; returns an unregister function.
export type DropHandler = (paths: string[]) => void;
/// Register a log-event listener; returns an unregister function.
export type LogListener = (entry: LogEntry) => void;

export interface Backend {
  // Status / info
  healthCheck(): Promise<string>;
  backendInfo(): Promise<BackendInfo>;
  getFfmpegStatus(): Promise<FfmpegStatus>;
  /// Live hardware usage; `null` when unavailable on the transport.
  hardwareStatus(): Promise<HardwareSnapshot | null>;
  getLogs(): Promise<LogEntry[]>;
  /// Empty the backend log buffer (Logs panel Clear).
  clearLogs(): Promise<void>;
  /// Subscribe to live log events (Tauri IPC); web returns a no-op.
  onLog(listener: LogListener): () => void;

  // Media
  probeVideo(input: string): Promise<VideoInfo>;
  /// Decode a preview frame at `positionMs` (raw RGB24, base64).
  readFrame(input: string, positionMs: number, projectDir?: string | null): Promise<RawFrame>;
  /// Native-playable URL for a video file; `null` when the transport can't
  /// stream it (web falls back to FFmpeg-decoded frames).
  nativeVideoUrl(input: string): FrameSource | null;
  /// Fullscreen the OS window (Full Video Mode); no-op on transports without a
  /// window (headless HTTP / mock).
  setWindowFullscreen(fullscreen: boolean): Promise<void>;

  // Settings
  getSettings(): Promise<Settings>;
  saveSettings(settings: Settings): Promise<void>;

  // Models
  listModels(): Promise<ModelMetadata[]>;
  /// Download + convert a model; resolves to the `.bpk` path on success.
  downloadModel(modelId: string, onProgress: (p: DownloadProgress) => void): Promise<string>;
  /// Installed weight files with size + sha256 verification (model manager).
  modelFiles(): Promise<ModelFileInfo[]>;
  /// Delete a model's weight files (model manager).
  deleteModelFile(id: string): Promise<void>;

  // Projects
  listProjects(): Promise<ProjectEntry[]>;
  createProject(name: string): Promise<string>;
  deleteProject(path: string): Promise<void>;
  openProject(archive: string): Promise<string>;
  exportProject(src: string, dest: string): Promise<void>;
  /// Package logs + system info into a `.tar.xz` (diagnose export).
  exportDiagnostics(dest: string): Promise<void>;
  importFolder(dir: string): Promise<string[]>;
  /// Recursively collect all videos under `dir` (batch folder processing).
  scanFolder(dir: string): Promise<string[]>;
  /// Probe content and suggest a default pipeline (JSON string: anime + steps).
  suggestPipeline(input: string): Promise<string>;
  /// Persist the batch queue state (JSON string) so a crash doesn't lose it.
  saveBatchQueue(state: string): Promise<void>;
  /// Load the persisted batch queue state, if any.
  loadBatchQueue(): Promise<string | null>;
  /// Drop the persisted batch queue state.
  clearBatchQueue(): Promise<void>;
  loadProjectSettings(path: string): Promise<ProjectSettings>;
  saveProjectSettings(path: string, settings: ProjectSettings): Promise<void>;

  // File access (native dialog in Tauri; path input/prompt in web)
  pickVideoFiles(): Promise<string[]>;
  pickFolder(title?: string): Promise<string | null>;
  pickSaveFile(defaultName: string, extensions: string[]): Promise<string | null>;
  /// Pick a single file with the given filters (project archives).
  pickFile(filters: { name: string; extensions: string[] }[], title?: string): Promise<string | null>;

  // Audio (native rodio in Tauri; web: the `<video>` element plays sound)
  extractAudio(input: string, projectDir?: string | null): Promise<string>;
  audioLoad(path: string): Promise<void>;
  audioPlay(): Promise<void>;
  audioPause(): Promise<void>;
  audioClear(): Promise<void>;
  audioSeek(positionMs: number): Promise<void>;
  audioSetVolume(volume: number): Promise<void>;

  // Render
  /// Render one file; `onProgress` fires on updates, resolves to the output
  /// path on success.
  render(
    input: string,
    output: string,
    config: RenderConfig,
    onProgress: (p: RenderProgress) => void,
  ): Promise<string>;
  cancelRender(): Promise<void>;
  pauseRender(paused: boolean): Promise<void>;
  uniquePath(path: string): Promise<string>;
  pruneSamples(dir: string, keep: number): Promise<void>;

  /// Download the bundled FFmpeg build (Tauri); web: no-op.
  downloadFfmpeg(onProgress: (p: DownloadProgress) => void): Promise<void>;

  // Misc
  openExternal(url: string): Promise<void>;
  onFileDrop(handler: DropHandler): () => void;
}
