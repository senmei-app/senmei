import type { Channel } from "@tauri-apps/api/core";
import { commands } from "./bindings";
import type { Settings, RenderProgress, DownloadProgress, ProjectSettings } from "./bindings";

export type {
  Settings,
  ProjectEntry,
  ProjectSettings,
  RenderProgress,
  DownloadProgress,
  ModelMetadata,
  ModelKind,
  LibTorchInfo,
  LibTorchBackend,
  FfmpegInfo as FfmpegStatus,
} from "./bindings";

export const healthCheck = () => commands.healthCheck();

export const render = (
  input: string,
  output: string,
  scale: number | null,
  modelId: string | null,
  resize: number | null,
  outputResize: number | null,
  onProgress: Channel<RenderProgress>,
) => commands.render(input, output, scale, modelId, resize, outputResize, onProgress);

export const importFolder = (dir: string) => commands.importFolder(dir);

export const getSettings = () => commands.getSettings();
export const saveSettings = (settings: Settings) => commands.saveSettings(settings);

export const listProjects = () => commands.listProjects();
export const createProject = (name: string) => commands.createProject(name);
export const rememberProject = (path: string) => commands.rememberProject(path);

export const loadProjectSettings = (path: string) => commands.loadProjectSettings(path);
export const saveProjectSettings = (path: string, settings: ProjectSettings) =>
  commands.saveProjectSettings(path, settings);

export const getFfmpegStatus = () => commands.getFfmpegStatus();
export const downloadFfmpeg = (onProgress: Channel<DownloadProgress>) =>
  commands.downloadFfmpeg(onProgress);

export const listModels = () => commands.listModels();

export const getLibtorchStatus = () => commands.getLibtorchStatus();
export const downloadLibtorch = (onProgress: Channel<DownloadProgress>) =>
  commands.downloadLibtorch(onProgress);

export const downloadModel = (
  modelId: string,
  onProgress: Channel<DownloadProgress>,
) => commands.downloadModel(modelId, onProgress);
