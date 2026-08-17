import type { Channel } from "@tauri-apps/api/core";
import { commands } from "./bindings";
import type { Settings, RenderProgress, DownloadProgress, ProjectSettings } from "./bindings";

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
} from "./bindings";

export const healthCheck = () => commands.healthCheck();

export const render = (
  input: string,
  output: string,
  scale: number | null,
  modelId: string | null,
  resize: number | null,
  outputResize: number | null,
  fpsMultiplier: number | null,
  interpModel: string | null,
  ffmpegArgs: string | null,
  onProgress: Channel<RenderProgress>,
) =>
  commands.render(
    input,
    output,
    scale,
    modelId,
    resize,
    outputResize,
    fpsMultiplier,
    interpModel,
    ffmpegArgs,
    onProgress,
  );

export const importFolder = (dir: string) => commands.importFolder(dir);

export const getSettings = () => commands.getSettings();
export const saveSettings = (settings: Settings) => commands.saveSettings(settings);

export const listProjects = () => commands.listProjects();
export const createProject = (name: string) => commands.createProject(name);
export const deleteProject = (path: string) => commands.deleteProject(path);
export const rememberProject = (path: string) => commands.rememberProject(path);

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
export const readFrame = (input: string, positionMs: number | null) =>
  commands.readFrame(input, positionMs);

export const cancelRender = () => commands.cancelRender();

export const pauseRender = (paused: boolean) => commands.pauseRender(paused);

export const uniquePath = (path: string) => commands.uniquePath(path);
