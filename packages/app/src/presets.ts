// Named step-chain templates ("art presets"), persisted in localStorage so
// they work on every transport (Tauri / web / mock).

import type { PipelineStep } from "./steps";

export interface PipelinePreset {
  name: string;
  steps: PipelineStep[];
}

const KEY = "senmei.presets";

export function loadPresets(): PipelinePreset[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persist(presets: PipelinePreset[]) {
  localStorage.setItem(KEY, JSON.stringify(presets));
}

/// Save the current steps under `name`, replacing an existing template.
export function savePreset(name: string, steps: PipelineStep[]) {
  const presets = loadPresets().filter((p) => p.name !== name);
  presets.push({ name, steps: JSON.parse(JSON.stringify(steps)) });
  persist(presets);
}

export function deletePreset(name: string) {
  persist(loadPresets().filter((p) => p.name !== name));
}
