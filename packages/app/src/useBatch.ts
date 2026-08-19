// Batch render orchestration: one render per file, sequentially. A single
// file is just a batch of one. Errors mark the job failed and continue;
// cancel stops after the current file; pause freezes the running file.

import { useState } from "react";
import { isTauri, Channel } from "@tauri-apps/api/core";
import {
  cancelRender,
  pauseRender,
  uniquePath,
  render,
  pruneSamples,
  type RenderConfig,
  type RenderProgress,
} from "@senmei/bridge";
import { buildEncoderArgs, type BatchJob, type PipelineStep } from "./steps";
import { basename, dirname, joinPath } from "./paths";
import { startDemoRender, stopDemoRender } from "./mock";

function fmtTs(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h${m}m${sec}s`;
  if (m) return `${m}m${sec}s`;
  return `${sec}s`;
}

// Show only the actionable tail of a render error in the status bar; the full
// message stays in the job tooltip and the Logs panel.
function shortReason(msg: string): string {
  const last = msg.split("\n").filter(Boolean).pop() ?? msg;
  return last.length > 160 ? `${last.slice(0, 160)}…` : last;
}

export interface UseBatchDeps {
  files: string[];
  selected: string[];
  steps: PipelineStep[];
  outputDir: string | null;
  projectDir: string | null;
  onError: (msg: string) => void;
}

export function useBatch({ files, selected, steps, outputDir, projectDir, onError }: UseBatchDeps) {
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [rendering, setRendering] = useState(false);
  const [paused, setPaused] = useState(false);
  const [progress, setProgress] = useState<RenderProgress | null>(null);
  const [renderedFile, setRenderedFile] = useState<string | null>(null);

  const desiredPath = (
    input: string,
    lastOut?: PipelineStep,
    up?: PipelineStep,
    range?: { inMs: number; outMs: number } | null,
    pd: string | null = projectDir,
  ): string => {
    const container = lastOut?.params?.container || "mkv";
    const outMode = lastOut?.params?.outputMode ?? "input";
    const customFolder = lastOut?.params?.outputFolder ?? "";
    const targetDir =
      outMode === "global" ? outputDir : outMode === "custom" ? customFolder || null : null;
    const label = lastOut?.params?.label?.trim();
    const marker = label || "senmei";
    const info = up?.params?.modelId && up.params?.scale ? `_${up.params.modelId}_x${up.params.scale}` : "";
    // Sample renders are scratch/preview files: keep them out of the output
    // folder root (in the project's `sample/` folder) and tag them with their
    // time range so repeated samples don't differ only by a collision counter.
    const isSample = !!(range && range.outMs > range.inMs);
    const rangeTag = isSample && range ? `_${fmtTs(range.inMs)}-${fmtTs(range.outMs)}` : "";
    const name =
      basename(input)
        ?.replace(/\.[^.]+$/, `_${marker}${info}${rangeTag}.${container}`) ??
      `output_${marker}${info}${rangeTag}.${container}`;
    const dir = targetDir ?? dirname(input);
    if (isSample) return joinPath(pd ?? dir, "sample", name);
    return joinPath(dir, name);
  };

  const startBatch = async (
    onlySelected = false,
    range?: { inMs: number; outMs: number } | null,
    explicit: string[] | null = null,
  ) => {
    const inputs = explicit ?? (onlySelected ? files.filter((f) => selected.includes(f)) : files);
    if (!inputs.length || rendering) return;
    const outs = steps.filter((s) => s.enabled && s.stepType === "output");
    const lastOut = outs.length ? outs[outs.length - 1] : undefined;
    const enabled = steps.filter((s) => s.enabled);
    const interp = enabled.find((s) => s.stepType === "interpolation");
    const up = enabled.find((s) => s.stepType === "upscale");
    const res = enabled.find((s) => s.stepType === "resize");
    const dn = enabled.find((s) => s.stepType === "denoise");
    const db = enabled.find((s) => s.stepType === "deblur");
    const dd = enabled.find((s) => s.stepType === "deduplication");
    const outScale = up ? (up.params?.scale ?? null) : null;
    const outModel = up ? (up.params?.modelId ?? null) : null;
    const outOutputResize = res ? toFactor(res.params?.factor ?? "") : null;
    const outFps = interp ? (interp.params?.fpsMultiplier ?? null) : null;
    const outInterpModel = interp ? (interp.params?.modelId ?? null) : null;
    const outFfmpegArgs = buildEncoderArgs(lastOut?.params, lastOut?.params?.ffmpegArgs ?? "");
    const outTonemap = lastOut?.params?.tonemap ?? null;
    const outFilter = {
      denoiseRadius: dn ? (dn.params?.radius ?? null) : null,
      denoiseModelId: dn ? (dn.params?.modelId ?? null) : null,
      deblurAmount: db ? (db.params?.amount ?? null) : null,
      deblurModelId: db ? (db.params?.modelId ?? null) : null,
      dedupThreshold: dd ? (dd.params?.threshold ?? null) : null,
    };
    const config: RenderConfig = {
      scale: outScale,
      modelId: outModel,
      resize: null,
      filter: outFilter,
      outputResize: outOutputResize,
      fpsMultiplier: outFps,
      interpModel: outInterpModel,
      ffmpegArgs: outFfmpegArgs,
      tonemap: outTonemap,
      startMs: range?.inMs ?? null,
      endMs: range?.outMs ?? null,
    };

    const initial: BatchJob[] = inputs.map((f) => ({
      input: f,
      output: desiredPath(f, lastOut, up, range),
      status: "queued",
      progress: null,
    }));
    setJobs(initial);
    setRendering(true);
    setPaused(false);
    setRenderedFile(null);

    const patch = (i: number, p: Partial<BatchJob>) =>
      setJobs((prev) => prev.map((j, k) => (k === i ? { ...j, ...p } : j)));

    try {
      for (let i = 0; i < initial.length; i++) {
        let output = initial[i].output;
        if (isTauri()) {
          try {
            output = await uniquePath(output); // collision -> _2, _3, …
          } catch {
            // keep the intended path if resolution fails
          }
        }
        patch(i, { output, status: "rendering", progress: null });
        try {
          if (isTauri()) {
            const ch = new Channel<RenderProgress>();
            ch.onmessage = (p) => {
              patch(i, { progress: p });
              setProgress(p);
            };
            await render(initial[i].input, output, config, ch);
          } else {
            await startDemoRender((p) => {
              patch(i, { progress: p });
              setProgress(p);
            });
          }
          patch(i, { status: "done" });
          setRenderedFile(output);
          if (range) {
            // Sample renders live in the project's sample/ folder: keep only the newest.
            void pruneSamples(dirname(output), 5);
          }
        } catch (e) {
          const msg = String(e);
          if (msg.toLowerCase().includes("cancelled")) {
            patch(i, { status: "cancelled" });
            setJobs((prev) => prev.map((j, k) => (k > i ? { ...j, status: "cancelled" as const } : j)));
            break; // stop the batch
          }
          patch(i, { status: "failed", error: msg });
          if (isTauri()) onError(`render failed: ${shortReason(msg)}`);
          // continue with the next file
        }
      }
    } finally {
      setRendering(false);
      setPaused(false);
      setProgress(null);
    }
  };

  const cancel = () => {
    if (!isTauri()) stopDemoRender();
    setRendering(false);
    setPaused(false);
    setJobs((prev) =>
      prev.map((j) =>
        j.status === "queued" || j.status === "rendering" ? { ...j, status: "cancelled" as const } : j,
      ),
    );
    void cancelRender();
  };

  const togglePause = () => {
    if (!isTauri()) {
      setPaused((p) => !p);
      return;
    }
    setPaused((p) => {
      void pauseRender(!p);
      return !p;
    });
  };

  return {
    jobs,
    setJobs,
    rendering,
    setRendering,
    paused,
    setPaused,
    progress,
    setProgress,
    renderedFile,
    setRenderedFile,
    startBatch,
    cancel,
    togglePause,
  };
}

function toFactor(v: string): number | null {
  const f = Number(v);
  return f > 0 ? f : null;
}
