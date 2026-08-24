// Processing-stack step model. Order top→bottom = execution order.
// The persisted shape is the Rust backend's PipelineStep (see bindings.ts).

import type { PipelineStep, RenderProgress, StepParams } from "@senmei/bridge";

export type { PipelineStep, StepParams } from "@senmei/bridge";

export type BatchStatus = "queued" | "rendering" | "done" | "failed" | "cancelled";

export interface BatchJob {
  input: string;
  output: string;
  status: BatchStatus;
  progress: RenderProgress | null;
  error?: string;
}

/** Persisted batch queue state (crash-safe resume). */
export interface BatchQueueState {
  inputs: string[];
  done: string[];
  updatedAt: number;
}

export type StepType =
  | "interpolation"
  | "decompress"
  | "upscale"
  | "denoise"
  | "deblur"
  | "deduplication"
  | "filter"
  | "resize"
  | "output";

export const STEP_META: Record<StepType, { icon: string; labelKey: string; implemented: boolean }> = {
  interpolation: { icon: "⚡", labelKey: "tab.interpolate", implemented: true },
  decompress: { icon: "🧼", labelKey: "tab.decompress", implemented: true },
  upscale: { icon: "🔍", labelKey: "tab.upscale", implemented: true },
  denoise: { icon: "🧹", labelKey: "tab.denoise", implemented: true },
  deblur: { icon: "✨", labelKey: "tab.deblur", implemented: true },
  deduplication: { icon: "🎞️", labelKey: "tab.dedup", implemented: true },
  filter: { icon: "🎛️", labelKey: "tab.filter", implemented: true },
  resize: { icon: "📐", labelKey: "tab.resize", implemented: true },
  output: { icon: "📦", labelKey: "tab.output", implemented: true },
};

/** Encoder quality profiles (RVE-style): each sets crf + preset as a bundle.
 * At a fixed CRF the preset only trades bitrate for speed — same visible
 * quality. Tempo-safe by default (High/Medium = veryfast) so the encoder
 * stays hidden behind the upscale step; slow presets at 4× make the encode
 * the bottleneck (measured ~730 ms/frame on libx265 2304×1728 10-bit).
 * Lossless/Very High stay slow for offline masters. */
export const QUALITY_PRESETS: Record<string, { crf: number; preset: string }> = {
  Lossless: { crf: 0, preset: "slow" },
  "Very High": { crf: 12, preset: "medium" },
  High: { crf: 16, preset: "veryfast" },
  Medium: { crf: 20, preset: "veryfast" },
  Low: { crf: 24, preset: "fast" },
};

export function qualityKey(params: StepParams | undefined): string {
  const crf = params?.crf ?? 20;
  const preset = params?.preset ?? "veryfast";
  return (
    Object.keys(QUALITY_PRESETS).find((k) => {
      const p = QUALITY_PRESETS[k];
      return p.crf === crf && p.preset === preset;
    }) ?? "Custom"
  );
}

export const STEP_ORDER: StepType[] = [
  "interpolation",
  "decompress",
  "upscale",
  "denoise",
  "deblur",
  "deduplication",
  "filter",
  "resize",
  "output",
];

const DEFAULTS: Record<StepType, StepParams> = {
  interpolation: { modelId: null, fpsMultiplier: 2 },
  decompress: { modelId: null },
  upscale: { modelId: null, scale: 2 },
  denoise: { radius: 1, modelId: null },
  deblur: { amount: 0.5, modelId: null },
  deduplication: { threshold: 0.02 },
  filter: { filter: "hue=h=0" },
  resize: { factor: "" },
  output: {
    label: "",
    container: "mkv",
    outputMode: "input",
    outputFolder: "",
    videoCodec: "H.264",
    audioCodec: "Passthrough",
    subtitleMode: "None",
    ffmpegArgs: "",
    crf: 20,
    preset: "veryfast",
    pixFmt: "yuv420p",
    tune: "",
    encoderBackend: "auto",
    encodeDevice: "auto",
    quality: "Medium",
    colorPrimaries: "",
    colorTransfer: "",
    colorMatrix: "",
    tonemap: "auto",
  },
};

export function newStepId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `step-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function isStepType(v: string): v is StepType {
  return (STEP_ORDER as string[]).includes(v);
}

export function createStep(type: StepType): PipelineStep {
  return { id: newStepId(), stepType: type, enabled: true, params: { ...DEFAULTS[type] } };
}

export function defaultSteps(): PipelineStep[] {
  return [createStep("interpolation"), createStep("upscale")];
}

/** Drop persisted steps whose type is unknown (schema drift guard). */
export function normalizeSteps(steps: PipelineStep[]): PipelineStep[] {
  return steps.filter((s) => isStepType(s.stepType));
}

// LGPL-safe encoder mapping (matches the backend's policy): libx264/libx265
// are GPL and absent from the pinned BtbN LGPL builds, so H.264/H.265 map to
// libopenh264/libkvazaar (both BSD, shipped in the LGPL builds).
const CODEC_MAP: Record<string, string> = {
  "H.264": "libopenh264",
  "H.265": "libkvazaar",
  AV1: "libsvtav1",
  VP9: "libvpx-vp9",
};

const AUDIO_MAP: Record<string, string> = {
  Passthrough: "copy",
  AAC: "aac",
  Opus: "libopus",
  FLAC: "flac",
};

function splitArgs(s: string): string[] {
  const out: string[] = [];
  let cur = "";
  let q = false;
  for (const c of s) {
    if (c === '"') q = !q;
    else if ((c === " " || c === "\t") && !q) {
      if (cur) {
        out.push(cur);
        cur = "";
      }
    } else cur += c;
  }
  if (cur) out.push(cur);
  return out;
}

/**
 * Merge the structured encoder params with the raw custom field. The custom
 * field wins for any flag it defines (e.g. `-tune grain`); the structured
 * dropdown values fill the rest (e.g. `-pix_fmt`).
 */
export function buildEncoderArgs(params: StepParams | undefined, custom: string): string[] {
  const structured: string[] = [];
  const vc = params?.videoCodec;
  const codec = vc && CODEC_MAP[vc] ? CODEC_MAP[vc] : null;
  if (codec) structured.push("-c:v", codec);
  // CRF/preset only where the codec supports them: svtav1/vpx take a CRF,
  // kvazaar takes a preset (quality-based rate control), openh264 is
  // bitrate-only (ABR) and gets its `-b:v` from the backend.
  if (codec === "libsvtav1" || codec === "libvpx-vp9") {
    if (params?.crf != null) structured.push("-crf", String(params.crf));
    if (params?.preset) structured.push("-preset", params.preset);
  } else if (codec === "libkvazaar") {
    if (params?.preset) structured.push("-preset", params.preset);
    // CRF flows through so the backend can map it to `-qp` for a VA-API
    // encode (the hardware quality knob); kvazaar/x265 accept it directly.
    if (params?.crf != null) structured.push("-crf", String(params.crf));
  }
  if (params?.pixFmt) structured.push("-pix_fmt", params.pixFmt);
  if (params?.tune) structured.push("-tune", params.tune);
  // Encoder backend preference (auto = HW first with software fallback). The
  // backend strips this sentinel before ffmpeg sees it.
  const encBackend = params?.encoderBackend ?? "auto";
  if (encBackend !== "auto") structured.push("-senmei_encoder", encBackend);
  // Encode device: iGPU offload (encode on the iGPU while the discrete GPU
  // runs inference). The backend strips this sentinel.
  const encDevice = params?.encodeDevice ?? "auto";
  if (encDevice !== "auto") structured.push("-senmei_vaapi", encDevice);
  if (params?.colorPrimaries) structured.push("-color_primaries", params.colorPrimaries);
  if (params?.colorTransfer) structured.push("-color_trc", params.colorTransfer);
  if (params?.colorMatrix) structured.push("-colorspace", params.colorMatrix);
  const ac = params?.audioCodec;
  if (ac === "None") structured.push("-an");
  else if (ac && AUDIO_MAP[ac]) structured.push("-c:a", AUDIO_MAP[ac]);
  if (params?.subtitleMode === "Copy") structured.push("-c:s", "copy");

  const customTokens = custom.trim() ? splitArgs(custom) : [];
  const customFlags = new Set(customTokens.filter((t) => t.startsWith("-")));
  const merged = [...customTokens];
  for (let i = 0; i + 1 < structured.length; i += 2) {
    if (!customFlags.has(structured[i])) merged.push(structured[i], structured[i + 1]);
  }
  return merged;
}
