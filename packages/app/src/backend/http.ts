//! HTTP backend: talks to `senmei-server --http` (REST + status polling).
//! Base URL: `VITE_SENMEI_API` (dev: http://127.0.0.1:8765) or same-origin
//! when the UI is served by the server itself.

import type {
  BackendInfo,
  FfmpegStatus,
  HardwareSnapshot,
  LogEntry,
  ModelFileInfo,
  ModelMetadata,
  ProjectEntry,
  ProjectSettings,
  Settings,
  StepTimingInfo,
  VideoInfo,
} from "@senmei/bridge";
import type { Backend, RawFrame } from "./types";
import { openPathDialog } from "./pathDialog";

const base = () => (import.meta.env.VITE_SENMEI_API as string | undefined) ?? "";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Race a promise against a deadline so a hung server can't block forever.
function withTimeout<T>(p: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(message)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      }
    );
  });
}

// Web audio plays a server-transcoded track (browsers can't decode every
// container); one shared element mirrors the rodio surface.
let audioEl: HTMLAudioElement | null = null;
function audioElement(): HTMLAudioElement {
  if (!audioEl) {
    audioEl = new Audio();
    audioEl.preload = "auto";
    audioEl.style.display = "none";
    document.body.appendChild(audioEl); // DOM-attached so it can't be GC'd
  }
  return audioEl;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base()}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  const text = await res.text();
  let json: { error?: string } & T;
  try {
    json = text ? JSON.parse(text) : ({} as T);
  } catch {
    json = { error: text } as unknown as { error?: string } & T;
  }
  if (!res.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
  return json;
}

// Web-only persistence for settings/projects (the server has no storage yet).
const SETTINGS_KEY = "senmei.settings";
const PROJECTS_KEY = "senmei.projects";

export const httpBackend: Backend = {
  async healthCheck() {
    const res = await fetch(`${base()}/api/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.text();
  },

  backendInfo(): Promise<BackendInfo> {
    return api<BackendInfo>("/api/backend-info");
  },

  getFfmpegStatus(): Promise<FfmpegStatus> {
    return api<FfmpegStatus>("/api/ffmpeg");
  },

  async hardwareStatus(): Promise<HardwareSnapshot | null> {
    return null; // live GPU/CPU usage is Tauri-only for now
  },

  async getLogs(): Promise<LogEntry[]> {
    return api<LogEntry[]>("/api/logs");
  },

  async clearLogs() {
    await api<{ ok: boolean }>("/api/logs/clear", { method: "POST" });
  },

  // No push channel over HTTP — poll the buffer and deliver appended entries.
  onLog(listener: (entry: LogEntry) => void): () => void {
    let lastCount = 0;
    let timer: number | undefined;
    const poll = async (seed = false) => {
      let logs: LogEntry[] = [];
      try {
        logs = await api<LogEntry[]>("/api/logs");
      } catch {
        return;
      }
      if (seed) {
        lastCount = logs.length; // existing entries come via getLogs()
        return;
      }
      if (logs.length < lastCount) lastCount = 0; // buffer was cleared
      for (let i = lastCount; i < logs.length; i++) listener(logs[i]);
      lastCount = logs.length;
    };
    void poll(true).then(() => {
      timer = window.setInterval(() => poll(), 1000);
    });
    return () => {
      if (timer) window.clearInterval(timer);
    };
  },

  async listModels() {
    return api<ModelMetadata[]>("/api/models");
  },

  async modelFiles(): Promise<ModelFileInfo[]> {
    return []; // file sizes/verify aren't exposed over HTTP yet
  },

  async deleteModelFile() {},

  async probeVideo(input) {
    return api<VideoInfo>("/api/probe", { method: "POST", body: JSON.stringify({ input }) });
  },

  async thumbnail(input) {
    const j = await api<{ data: string; info: VideoInfo }>("/api/thumbnail", {
      method: "POST",
      body: JSON.stringify({ input, maxW: 160 }),
    });
    return j;
  },

  async readFrame(input, positionMs): Promise<RawFrame> {
    const res = await fetch(`${base()}/api/frame`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input, positionMs }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j && j.error) msg = j.error;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(msg);
    }
    // Raw RGB24 body; width/height ride in headers so the payload stays binary
    // (ArrayBuffer — same shape the Tauri channel delivers).
    const width = Number(res.headers.get("x-frame-width"));
    const height = Number(res.headers.get("x-frame-height"));
    const data = new Uint8Array(await res.arrayBuffer());
    return { width, height, data };
  },

  nativeVideoUrl(input) {
    // Range-stream the source so the browser <video> plays video+audio;
    // unsupported codecs fall back to FFmpeg frames.
    return `${base()}/api/stream?path=${encodeURIComponent(input)}`;
  },

  async setWindowFullscreen() {},

  async getSettings(): Promise<Settings> {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? (JSON.parse(raw) as Settings) : { language: "en", theme: "dark" };
  },

  async saveSettings(settings) {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  },

  async downloadModel(modelId, _onProgress) {
    const res = await api<{ bpk?: string }>("/api/download-model", {
      method: "POST",
      body: JSON.stringify({ modelId }),
    });
    if (!res.bpk) throw new Error("download failed");
    return res.bpk;
  },

  async listProjects(): Promise<ProjectEntry[]> {
    return JSON.parse(localStorage.getItem(PROJECTS_KEY) ?? "[]");
  },

  async createProject(name): Promise<string> {
    const slug = name.toLowerCase().replace(/\s+/g, "-");
    const path = `/projects/${slug}`;
    const projects = await this.listProjects();
    if (!projects.some((p) => p.path === path)) {
      projects.push({ name, path });
      localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects));
    }
    return path;
  },

  async deleteProject(path) {
    const projects = (await this.listProjects()).filter((p) => p.path !== path);
    localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects));
  },

  async openProject(archive) {
    // Imported archives aren't supported over HTTP yet; treat the path as a project dir.
    const path = archive.replace(/\.tar\.xz$/i, "");
    const projects = await this.listProjects();
    if (!projects.some((p) => p.path === path)) {
      projects.push({ name: path.split("/").pop() ?? path, path });
      localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects));
    }
    return path;
  },

  async exportProject(_src, _dest) {
    throw new Error("project export is not available over HTTP yet");
  },

  async exportDiagnostics(_dest) {
    throw new Error("diagnostics export is not available over HTTP yet");
  },

  async importFolder(dir) {
    const info = await this.probeVideo(dir);
    return info ? [dir] : [];
  },

  async scanFolder(dir) {
    return api<string[]>("/api/scan-folder", {
      method: "POST",
      body: JSON.stringify({ dir }),
    });
  },

  async suggestPipeline(_input) {
    throw new Error("suggest pipeline is not available over HTTP yet");
  },

  async saveBatchQueue() {}, // best-effort: queue resume is a desktop feature
  async loadBatchQueue() {
    return null;
  },
  async clearBatchQueue() {},

  async loadProjectSettings(path): Promise<ProjectSettings> {
    const raw = localStorage.getItem(`${PROJECTS_KEY}.${path}`);
    return raw ? JSON.parse(raw) : { steps: [], files: [], outputDir: null };
  },

  async saveProjectSettings(path, settings) {
    localStorage.setItem(`${PROJECTS_KEY}.${path}`, JSON.stringify(settings));
  },

  async pickVideoFiles(): Promise<string[]> {
    // No native picker over HTTP: enter server-side paths in the path dialog.
    const input = await openPathDialog({
      title: "Add videos",
      placeholder: "/path/to/video1.mp4, /path/to/video2.mp4",
      multiple: true,
    });
    return (input ?? "")
      .split(",")
      .map((p) => p.trim())
      .filter(Boolean);
  },

  async pickFolder(title?): Promise<string | null> {
    return openPathDialog({ title: title ?? "Choose folder", placeholder: "/path/to/folder" });
  },

  async pickSaveFile(defaultName): Promise<string | null> {
    return openPathDialog({ title: "Save as", default: defaultName, placeholder: "/path/to/file" });
  },

  async pickFile(_filters, title?): Promise<string | null> {
    return openPathDialog({ title: title ?? "Choose file", placeholder: "/path/to/file" });
  },

  async audioLoad(input, positionMs): Promise<void> {
    const el = audioElement();
    el.src = `${base()}/api/audio?path=${encodeURIComponent(input)}`;
    el.load();
    // Seek once playable — a pre-metadata seek aborts the pending (slow) transcode.
    await new Promise<void>((resolve) => {
      const onReady = () => {
        el.removeEventListener("canplay", onReady);
        if (el.readyState >= 1) el.currentTime = positionMs / 1000;
        resolve();
      };
      el.addEventListener("canplay", onReady);
      window.setTimeout(() => {
        el.removeEventListener("canplay", onReady);
        resolve();
      }, 8000);
    });
  },
  async audioPlay() {
    audioElement().play().catch(() => {});
  },
  async audioPause() {
    audioElement().pause();
  },
  async audioClear() {
    const el = audioElement();
    el.pause();
    el.removeAttribute("src");
    el.load();
  },
  async audioSeek(positionMs) {
    const el = audioElement();
    // A pre-metadata seek aborts the pending load (audioLoad sets the start).
    if (el.readyState >= 1) el.currentTime = positionMs / 1000;
  },
  async audioSetVolume(volume) {
    audioElement().volume = volume;
  },

  async render(input, output, config, onProgress) {
    try {
      await api("/api/render", {
        method: "POST",
        body: JSON.stringify({ ...config, input, output }),
      });
    } catch (e) {
      // A render is already running (stale frontend after a reload): join it
      // instead of failing or spamming 400s.
      if (!String(e).includes("already running")) throw e;
    }
    // Poll the shared render status until done. Each poll races a watchdog so
    // a hung server can't leave the UI stuck in "rendering" forever.
    for (;;) {
      await sleep(500);
      const st = await withTimeout(
        api<{
          state: string;
          framesProcessed?: number;
          totalFrames?: number;
          error?: string | null;
          steps?: StepTimingInfo[];
        }>("/api/render/status"),
        60_000,
        "render status timed out (server unresponsive)"
      );
      if (st.framesProcessed != null || st.steps) {
        onProgress({
          framesProcessed: st.framesProcessed ?? 0,
          totalFrames: st.totalFrames ?? 0,
          steps: st.steps ?? [],
        });
      }
      if (st.state === "done") return output;
      if (st.state === "failed") throw new Error(st.error ?? "render failed");
    }
  },

  async cancelRender() {
    await api("/api/render/cancel", { method: "POST", body: "{}" });
  },

  async pauseRender() {},
  async uniquePath(path) {
    return path; // server has no collision-avoiding path helper yet
  },
  async pruneSamples() {},

  async downloadFfmpeg() {},

  async openExternal(url) {
    window.open(url, "_blank", "noopener");
  },

  onFileDrop(handler) {
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const paths = Array.from(e.dataTransfer?.files ?? []).map((f) => f.name);
      handler(paths);
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
