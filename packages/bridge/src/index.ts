import type { Channel } from "@tauri-apps/api/core";
import { commands } from "./bindings";
import type { Settings, RenderProgress, DownloadProgress, ProjectSettings, RenderConfig } from "./bindings";

export type {
  Settings,
  ProjectEntry,
  ProjectSettings,
  PipelineStep,
  StepParams,
  RenderProgress,
  DownloadProgress,
  ModelMetadata,
  ModelKind,
  VideoInfo,
  FfmpegInfo as FfmpegStatus,
  RenderConfig,
  FilterParams,
  LogEntry,
  EngineBackend,
  BackendInfo,
  HardwareSnapshot,
} from "./bindings";

export const healthCheck = () => commands.healthCheck();

export const render = (
  input: string,
  output: string,
  config: RenderConfig,
  onProgress: Channel<RenderProgress>,
) => commands.render(input, output, config, onProgress);

export const importFolder = (dir: string) => commands.importFolder(dir);

export const getSettings = () => commands.getSettings();
export const saveSettings = (settings: Settings) => commands.saveSettings(settings);
export const backendInfo = () => commands.backendInfo();
export const hardwareStatus = () => commands.hardwareStatus();

export const listProjects = () => commands.listProjects();
export const createProject = (name: string) => commands.createProject(name);
export const deleteProject = (path: string) => commands.deleteProject(path);
export const exportProject = (src: string, dest: string) => commands.exportProject(src, dest);
export const exportDiagnostics = (dest: string) => commands.exportDiagnostics(dest);
export const openProject = (file: string) => commands.openProject(file);

export const loadProjectSettings = (path: string) => commands.loadProjectSettings(path);
export const saveProjectSettings = (path: string, settings: ProjectSettings) =>
  commands.saveProjectSettings(path, settings);

export const getFfmpegStatus = () => commands.getFfmpegStatus();
export const downloadFfmpeg = (onProgress: Channel<DownloadProgress>) =>
  commands.downloadFfmpeg(onProgress);

export const listModels = () => commands.listModels();

export const downloadModel = (modelId: string, onProgress: Channel<DownloadProgress>) =>
  commands.downloadModel(modelId, onProgress);

export const probeVideo = (input: string) => commands.probeVideo(input);
export const readFrame = (input: string, positionMs: number | null, projectDir: string | null = null) =>
  commands.readFrame(input, positionMs, projectDir);

export const extractAudio = (input: string, projectDir: string | null = null) =>
  commands.extractAudio(input, projectDir);

export const audioLoad = (path: string) => commands.audioLoad(path);
export const audioPlay = () => commands.audioPlay();
export const audioPause = () => commands.audioPause();
export const audioClear = () => commands.audioClear();
export const audioSeek = (positionMs: number) => commands.audioSeek(positionMs);
export const audioSetVolume = (volume: number) => commands.audioSetVolume(volume);

export const cancelRender = () => commands.cancelRender();
export const pruneSamples = (dir: string, keep: number) => commands.pruneSamples(dir, keep);

export const pauseRender = (paused: boolean) => commands.pauseRender(paused);

export const uniquePath = (path: string) => commands.uniquePath(path);

export const getLogs = () => commands.getLogs();

export const clearLogs = () => commands.clearLogs();
