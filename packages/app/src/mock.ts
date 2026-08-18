import type {
  DownloadProgress,
  ModelMetadata,
  ProjectEntry,
  RenderProgress,
  VideoInfo,
} from "@senmei/bridge";

// In-memory demo backend so the UI is fully testable in a plain browser,
// where the Tauri IPC (and therefore the real Rust commands) is unavailable.

export const demoSettings = { language: "en", theme: "dark" };

export const demoProjects: ProjectEntry[] = [
  { name: "Demo: Quanzhi Fashi", path: "/demo/quanzhi-fashi" },
  { name: "Demo: One Punch Man", path: "/demo/opm" },
  { name: "Demo: Frieren", path: "/demo/frieren" },
];

export const demoModels: ModelMetadata[] = [
  {
    id: "fallin-soft",
    kind: "upscale",
    scale: 2,
    arch: "fallin-cugan",
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
    loadable: true,
    license: "demo",
    source_url: "demo",
    download_url: "demo",
  },
];

export const demoVideos = [
  "/demo/Quanzhi Fashi (Staffel 3) Folge 7.mp4",
  "/demo/Quanzhi Fashi (Staffel 3) Folge 8.mp4",
];

export function demoProbe(): VideoInfo {
  return { width: 1920, height: 1080, fps: 23.976, duration: 1440, rotation: 0 };
}

// 320x180 indigo frame (generated with ffmpeg) so the preview shows an image.
const DEMO_FRAME_B64 =
  "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYyLjI4LjEwMgD/2wBDAAgEBAQEBAUFBQUFBQYGBgYGBgYGBgYGBgYHBwcIC" +
  "AgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/xABNAAEBAAAAAAAAAAAAAAAAAAAABwEBAQEAAAAAAAAAAAAAAA" +
  "AAAAUGEAEAAAAAAAAAAAAAAAAAAAAAEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAtAFAAwEiAAIRAAMRAP/aAAwDAQACEQMRAD8" +
  "AkgDaLYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD" +
  "/9k=";

export function demoFrame(): string {
  return DEMO_FRAME_B64;
}

let demoRenderTimer: ReturnType<typeof setInterval> | null = null;

export function startDemoRender(onProgress: (p: RenderProgress) => void): Promise<string> {
  return new Promise((resolve) => {
    let frames = 0;
    const total = 3000;
    onProgress({ framesProcessed: 0, totalFrames: total });
    demoRenderTimer = setInterval(() => {
      frames += 25;
      onProgress({ framesProcessed: frames, totalFrames: total });
      if (frames >= total) {
        if (demoRenderTimer) clearInterval(demoRenderTimer);
        demoRenderTimer = null;
        resolve("/demo/output.mp4");
      }
    }, 60);
  });
}

export function stopDemoRender() {
  if (demoRenderTimer) {
    clearInterval(demoRenderTimer);
    demoRenderTimer = null;
  }
}

export function demoDownloadModel(onProgress: (p: DownloadProgress) => void): Promise<string> {
  return new Promise((resolve) => {
    onProgress({ downloaded: 0, total: 100 });
    setTimeout(() => {
      onProgress({ downloaded: 100, total: 100 });
      resolve("/demo/model.bpk");
    }, 1500);
  });
}
