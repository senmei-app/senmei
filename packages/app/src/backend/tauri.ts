//! Tauri backend: wraps `@senmei/bridge` (Tauri IPC). Only loaded when
//! `isTauri()` — keeps the Tauri surface out of the web bundle.

import { Channel, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import * as bridge from "@senmei/bridge";
import type { DownloadProgress, LogEntry, RenderProgress } from "@senmei/bridge";
import type { Backend, FrameSource, RawFrame } from "./types";

export const tauriBackend: Backend = {
  async healthCheck() {
    return bridge.healthCheck();
  },

  backendInfo() {
    return bridge.backendInfo();
  },

  getFfmpegStatus() {
    return bridge.getFfmpegStatus();
  },

  hardwareStatus() {
    return bridge.hardwareStatus();
  },

  getLogs() {
    return bridge.getLogs();
  },

  clearLogs() {
    return bridge.clearLogs();
  },

  onLog(listener) {
    let un: (() => void) | undefined;
    listen<LogEntry>("log", (e) => listener(e.payload)).then((fn) => (un = fn));
    return () => un?.();
  },

  async listModels() {
    return bridge.listModels();
  },

  modelFiles() {
    return bridge.modelFiles();
  },

  async deleteModelFile(id) {
    await bridge.deleteModelFile(id);
  },

  async probeVideo(input) {
    return bridge.probeVideo(input);
  },

  async readFrame(input, positionMs, projectDir = null): Promise<RawFrame> {
    // bridge.readFrame delivers width/height on the meta channel (JSON) and
    // the raw RGB24 on the frame channel (ArrayBuffer) — no base64 over IPC.
    return new Promise<RawFrame>((resolve, reject) => {
      let meta: { width: number; height: number } | null = null;
      const onMeta = new Channel<{ width: number; height: number }>((m) => {
        meta = m;
      });
      // Specta types the frame payload `number[]`, but Tauri delivers a raw
      // ArrayBuffer — `any` keeps the wrapper the single cast site.
      const onFrame = new Channel<any>((buf: ArrayBuffer) => {
        if (meta) {
          resolve({ width: meta.width, height: meta.height, data: new Uint8Array(buf) });
        }
      });
      bridge.readFrame(input, positionMs, projectDir, onMeta, onFrame).catch(reject);
    });
  },

  nativeVideoUrl(input): FrameSource | null {
    return convertFileSrc(input);
  },

  async setWindowFullscreen(fullscreen: boolean): Promise<void> {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().setFullscreen(fullscreen);
  },

  getSettings() {
    return bridge.getSettings();
  },

  async saveSettings(settings) {
    await bridge.saveSettings(settings);
  },

  async downloadModel(modelId, onProgress) {
    const ch = new Channel<DownloadProgress>();
    ch.onmessage = onProgress;
    return bridge.downloadModel(modelId, ch);
  },

  listProjects() {
    return bridge.listProjects();
  },

  createProject(name) {
    return bridge.createProject(name);
  },

  async deleteProject(path) {
    await bridge.deleteProject(path);
  },

  openProject(archive) {
    return bridge.openProject(archive);
  },

  async exportProject(src, dest) {
    await bridge.exportProject(src, dest);
  },

  async exportDiagnostics(dest) {
    await bridge.exportDiagnostics(dest);
  },

  importFolder(dir) {
    return bridge.importFolder(dir);
  },
  scanFolder(dir) {
    return bridge.scanFolder(dir);
  },
  suggestPipeline(input) {
    return bridge.suggestPipeline(input);
  },
  async saveBatchQueue(state) {
    await bridge.saveBatchQueue(state);
  },
  loadBatchQueue() {
    return bridge.loadBatchQueue();
  },
  async clearBatchQueue() {
    await bridge.clearBatchQueue();
  },

  loadProjectSettings(path) {
    return bridge.loadProjectSettings(path);
  },

  async saveProjectSettings(path, settings) {
    await bridge.saveProjectSettings(path, settings);
  },

  async pickVideoFiles(): Promise<string[]> {
    const picked = await open({
      multiple: true,
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "mov", "webm", "avi", "m4v"] }],
    });
    if (!picked) return [];
    return Array.isArray(picked) ? picked : [picked];
  },

  async pickFolder(title?: string): Promise<string | null> {
    const dir = await open({ directory: true, title });
    return typeof dir === "string" ? dir : null;
  },

  async pickSaveFile(defaultName, extensions): Promise<string | null> {
    const dest = await save({
      defaultPath: defaultName,
      filters: [{ name: "Senmei project", extensions }],
    });
    return dest ?? null;
  },

  async pickFile(filters, title?): Promise<string | null> {
    const picked = await open({ multiple: false, title, filters });
    return typeof picked === "string" ? picked : null;
  },

  async extractAudio(input, projectDir = null) {
    return bridge.extractAudio(input, projectDir);
  },

  async audioLoad(path) {
    await bridge.audioLoad(path);
  },
  async audioPlay() {
    await bridge.audioPlay();
  },
  async audioPause() {
    await bridge.audioPause();
  },
  async audioClear() {
    await bridge.audioClear();
  },
  async audioSeek(positionMs) {
    await bridge.audioSeek(positionMs);
  },
  async audioSetVolume(volume) {
    await bridge.audioSetVolume(volume);
  },

  async render(input, output, config, onProgress) {
    const ch = new Channel<RenderProgress>();
    ch.onmessage = onProgress;
    await bridge.render(input, output, config, ch);
    return output;
  },

  async cancelRender() {
    await bridge.cancelRender();
  },

  pauseRender(paused) {
    return bridge.pauseRender(paused);
  },

  uniquePath(path) {
    return bridge.uniquePath(path);
  },

  async pruneSamples(dir, keep) {
    await bridge.pruneSamples(dir, keep);
  },

  async downloadFfmpeg(onProgress) {
    const ch = new Channel<DownloadProgress>();
    ch.onmessage = onProgress;
    await bridge.downloadFfmpeg(ch);
  },

  async openExternal(url) {
    await openUrl(url);
  },

  onFileDrop(handler) {
    let un: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        if (event.payload.type === "drop") handler(event.payload.paths);
      })
      .then((fn) => (un = fn));
    return () => un?.();
  },
};
