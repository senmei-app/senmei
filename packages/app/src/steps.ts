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

export type StepType =
  | "interpolation"
  | "upscale"
  | "denoise"
  | "deblur"
  | "deduplication"
  | "resize"
  | "output";

export const STEP_META: Record<StepType, { icon: string; labelKey: string; implemented: boolean }> = {
  interpolation: { icon: "⚡", labelKey: "tab.interpolate", implemented: true },
  upscale: { icon: "🔍", labelKey: "tab.upscale", implemented: true },
  denoise: { icon: "🧹", labelKey: "tab.denoise", implemented: false },
  deblur: { icon: "✨", labelKey: "tab.deblur", implemented: false },
  deduplication: { icon: "🎞️", labelKey: "tab.dedup", implemented: false },
  resize: { icon: "📐", labelKey: "tab.resize", implemented: true },
  output: { icon: "📦", labelKey: "tab.output", implemented: true },
};

export const STEP_ORDER: StepType[] = [
  "interpolation",
  "upscale",
  "denoise",
  "deblur",
  "deduplication",
  "resize",
  "output",
];

const DEFAULTS: Record<StepType, StepParams> = {
  interpolation: { modelId: null, fpsMultiplier: 2 },
  upscale: { modelId: null, scale: 2 },
  denoise: {},
  deblur: {},
  deduplication: {},
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
    preset: "medium",
    pixFmt: "yuv420p",
    tune: "",
  },
};

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `step-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function isStepType(v: string): v is StepType {
  return (STEP_ORDER as string[]).includes(v);
}

export function createStep(type: StepType): PipelineStep {
  return { id: newId(), stepType: type, enabled: true, params: { ...DEFAULTS[type] } };
}

export function defaultSteps(): PipelineStep[] {
  return [createStep("interpolation"), createStep("upscale")];
}

/** Drop persisted steps whose type is unknown (schema drift guard). */
export function normalizeSteps(steps: PipelineStep[]): PipelineStep[] {
  return steps.filter((s) => isStepType(s.stepType));
}

const CODEC_MAP: Record<string, string> = {
  "H.264": "libx264",
  "H.265": "libx265",
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
export function buildEncoderArgs(params: StepParams | undefined, custom: string): string {
  const structured: string[] = [];
  const vc = params?.videoCodec;
  if (vc && CODEC_MAP[vc]) structured.push("-c:v", CODEC_MAP[vc]);
  if (params?.crf != null) structured.push("-crf", String(params.crf));
  if (params?.preset) structured.push("-preset", params.preset);
  if (params?.pixFmt) structured.push("-pix_fmt", params.pixFmt);
  if (params?.tune) structured.push("-tune", params.tune);
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
  return merged.join(" ");
}
