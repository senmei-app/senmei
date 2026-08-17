// Processing-stack step model. Order top→bottom = execution order.
// The persisted shape is the Rust backend's PipelineStep (see bindings.ts).

import type { PipelineStep, StepParams } from "@senmei/bridge";

export type { PipelineStep, StepParams } from "@senmei/bridge";

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
  output: { name: "Output", videoCodec: "H.264", audioCodec: "Passthrough", subtitleMode: "None" },
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
