import type { Channel } from "@tauri-apps/api/core";
import { commands } from "./bindings";
import type { Settings, RenderProgress, DownloadProgress } from "./bindings";

export type {
  Settings,
  ProjectEntry,
  RenderProgress,
  DownloadProgress,
  ModelMetadata,
  ModelKind,
  FfmpegInfo as FfmpegStatus,
} from "./bindings";

export const healthCheck = () => commands.healthCheck();

export const render = (
  input: string,
  output: string,
  onProgress: Channel<RenderProgress>,
) => commands.render(input, output, onProgress);

export const importFolder = (dir: string) => commands.importFolder(dir);

export const getSettings = () => commands.getSettings();
export const saveSettings = (settings: Settings) => commands.saveSettings(settings);

export const listProjects = () => commands.listProjects();
export const createProject = (name: string) => commands.createProject(name);
export const rememberProject = (path: string) => commands.rememberProject(path);

export const getFfmpegStatus = () => commands.getFfmpegStatus();
export const downloadFfmpeg = (onProgress: Channel<DownloadProgress>) =>
  commands.downloadFfmpeg(onProgress);

export const listModels = () => commands.listModels();
