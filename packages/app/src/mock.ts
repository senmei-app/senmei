// In-memory demo backend (dev only, selected via `VITE_SENMEI_MOCK=1`) so the
// UI is testable in a plain browser without a running `senmei-server`.
// Mutable arrays keep a single shared instance across reloads.

import type {
  DownloadProgress,
  ModelMetadata,
  ProjectEntry,
  RenderProgress,
  VideoInfo,
} from "@senmei/bridge";
import type { Backend, RawFrame } from "./backend/types";

export const demoSettings = { language: "en", theme: "dark" };

const demoProjects: ProjectEntry[] = [
  { name: "Demo: Quanzhi Fashi", path: "/demo/quanzhi-fashi" },
  { name: "Demo: One Punch Man", path: "/demo/opm" },
  { name: "Demo: Frieren", path: "/demo/frieren" },
];

const demoModels: ModelMetadata[] = [
  {
    id: "fallin-soft",
    kind: "upscale",
    scale: 2,
    arch: "fallin-cugan",
    family: "real-cugan",
    weights: ["2x_Fallin_soft_renarchi_fp16.f16.bpk"],
    loadable: true,
    license: "CC-BY-4.0",
    source_url: "demo",
    download_url: "demo",
  },
  {
    id: "fallin-strong",
    kind: "upscale",
    scale: 2,
    arch: "fallin-cugan",
    family: "real-cugan",
    weights: ["2x_Fallin_strong_renarchi_fp16.f16.bpk"],
    loadable: true,
    license: "CC-BY-4.0",
    source_url: "demo",
    download_url: "demo",
  },
  {
    id: "4x-alchemy",
    kind: "upscale",
    scale: 4,
    arch: "real-plksr",
    family: "real-plksr",
    weights: ["4x_Alchemy.pth.f16.bpk"],
    loadable: true,
    license: "CC-BY-4.0",
    source_url: "demo",
    download_url: "demo",
  },
  {
    id: "real-cugan-x2",
    kind: "upscale",
    scale: 2,
    arch: "upcunet2x",
    family: "real-cugan",
    weights: ["up2x-no-denoise.pth.f16.bpk"],
    loadable: true,
    license: "demo",
    source_url: "demo",
    download_url: "demo",
  },
  {
    id: "realesrgan-animevideo-x4",
    kind: "upscale",
    scale: 4,
    arch: "realesrgan",
    family: "real-esrgan",
    loadable: false,
    license: "demo",
    source_url: "demo",
    download_url: "demo",
  },
  {
    id: "rife-v4.6",
    kind: "interpolate",
    scale: 1,
    arch: "rife46",
    family: "rife",
    loadable: true,
    license: "demo",
    source_url: "demo",
    download_url: "demo",
  },
];

const demoVideos = [
  "/demo/Quanzhi Fashi (Staffel 3) Folge 7.mp4",
  "/demo/Quanzhi Fashi (Staffel 3) Folge 8.mp4",
];

function demoProbe(): VideoInfo {
  return {
    width: 1920,
    height: 1080,
    fps: 23.976,
    duration: 1440,
    rotation: 0,
    colorTransfer: null,
    colorPrimaries: null,
  };
}

function demoFrame(): RawFrame {
  // Solid indigo 32x16 raw RGB24 so the preview shows a color in mock/dev.
  const w = 32;
  const h = 16;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < rgb.length; i += 3) {
    rgb[i] = 79;
    rgb[i + 1] = 70;
    rgb[i + 2] = 229;
  }
  return { width: w, height: h, data: rgb };
}

let demoRenderTimer: ReturnType<typeof setInterval> | null = null;

function startDemoRender(onProgress: (p: RenderProgress) => void): Promise<string> {
  return new Promise((resolve) => {
    let frames = 0;
    const total = 3000;
    onProgress({ framesProcessed: 0, totalFrames: total, steps: [] });
    demoRenderTimer = setInterval(() => {
      frames += 25;
      onProgress({ framesProcessed: frames, totalFrames: total, steps: [] });
      if (frames >= total) {
        if (demoRenderTimer) clearInterval(demoRenderTimer);
        demoRenderTimer = null;
        resolve("/demo/output.mp4");
      }
    }, 60);
  });
}

function stopDemoRender() {
  if (demoRenderTimer) {
    clearInterval(demoRenderTimer);
    demoRenderTimer = null;
  }
}

function demoDownloadModel(onProgress: (p: DownloadProgress) => void): Promise<string> {
  return new Promise((resolve) => {
    onProgress({ downloaded: 0, total: 100 });
    setTimeout(() => {
      onProgress({ downloaded: 100, total: 100 });
      resolve("/demo/model.bpk");
    }, 1500);
  });
}

const projectSettings = new Map<string, unknown>();

export const mockBackend: Backend = {
  async healthCheck() {
    return "ok";
  },

  async backendInfo() {
    return {
      vulkanCompiled: true,
      libtorchCompiled: false,
      libtorchVersion: null,
      cudaAvailable: false,
      cudaDeviceCount: 0,
    };
  },

  async getFfmpegStatus() {
    return { found: true, path: "/usr/bin/ffmpeg", version: "demo", encoders: [], decoders: [] };
  },

  async hardwareStatus() {
    return null;
  },

  async getLogs() {
    return [];
  },

  async clearLogs() {},

  onLog() {
    return () => {};
  },

  async listModels() {
    return demoModels;
  },

  async modelFiles() {
    return [];
  },

  async deleteModelFile() {},

  async probeVideo(_input) {
    return demoProbe();
  },

  async readFrame() {
    return demoFrame();
  },

  nativeVideoUrl() {
    return null;
  },

  async setWindowFullscreen() {},

  async getSettings() {
    return demoSettings;
  },

  async saveSettings() {},

  async downloadModel(_id, onProgress) {
    return demoDownloadModel(onProgress);
  },

  async listProjects() {
    return demoProjects;
  },

  async createProject(name) {
    const path = `/demo/${name.toLowerCase().replace(/\s+/g, "-")}`;
    if (!demoProjects.some((p) => p.path === path)) demoProjects.push({ name, path });
    return path;
  },

  async deleteProject(path) {
    const i = demoProjects.findIndex((p) => p.path === path);
    if (i >= 0) demoProjects.splice(i, 1);
  },

  async openProject(archive) {
    return archive.replace(/\.tar\.xz$/i, "");
  },

  async exportProject() {},

  async exportDiagnostics() {},

  async importFolder() {
    return demoVideos;
  },

  async scanFolder() {
    return demoVideos;
  },

  async suggestPipeline() {
    return JSON.stringify({
      anime: true,
      steps: [
        { stepType: "interpolation", params: { fpsMultiplier: 2, modelId: "rife-v4.6" } },
        { stepType: "upscale", params: { scale: 4, modelId: "realesrgan-animevideo-x4" } },
        { stepType: "output", params: {} },
      ],
    });
  },

  async saveBatchQueue() {},
  async loadBatchQueue() {
    return null;
  },
  async clearBatchQueue() {},

  async loadProjectSettings(path) {
    return (projectSettings.get(path) ?? { steps: [], files: [], outputDir: null }) as never;
  },

  async saveProjectSettings(path, settings) {
    projectSettings.set(path, settings);
  },

  async pickVideoFiles() {
    return demoVideos;
  },

  async pickFolder() {
    return "/demo/output";
  },

  async pickSaveFile() {
    return "/demo/project.tar.xz";
  },

  async pickFile() {
    return "/demo/project.tar.xz";
  },

  async audioLoad() {},
  async audioPlay() {},
  async audioPause() {},
  async audioClear() {},
  async audioSeek() {},
  async audioSetVolume() {},

  async render(_input, output, _config, onProgress) {
    return startDemoRender(onProgress).then(() => output);
  },

  async cancelRender() {
    stopDemoRender();
  },

  async pauseRender() {},
  async uniquePath(path) {
    return path;
  },
  async pruneSamples() {},

  async downloadFfmpeg() {},

  async openExternal() {},

  onFileDrop(handler) {
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      handler(demoVideos);
    };
    const onOver = (e: DragEvent) => e.preventDefault();
    document.addEventListener("dragover", onOver);
    document.addEventListener("drop", onDrop);
    return () => {
      document.removeEventListener("dragover", onOver);
      document.removeEventListener("drop", onDrop);
    };
  },
};
