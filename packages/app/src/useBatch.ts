// Batch render orchestration: one render per file, sequentially. A single
// file is just a batch of one. Errors mark the job failed and continue;
// cancel stops after the current file; pause freezes the running file.

import { useEffect, useRef, useState } from "react";
import type { RenderConfig, RenderProgress, StepTimingInfo } from "@senmei/bridge";
import { backend, isWeb } from "./backend";
import { buildEncoderArgs, type BatchJob, type BatchQueueState, type PipelineStep } from "./steps";
import { basename, dirname, joinPath } from "./paths";

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
  // Mirrors `rendering` for callers holding a stale closure (e.g. the global
  // hotkey handler): the start guard reads the ref so a second start can't slip
  // through while a render is already running.
  const renderingRef = useRef(false);
  useEffect(() => {
    renderingRef.current = rendering;
  }, [rendering]);
  const [paused, setPaused] = useState(false);
  const [progress, setProgress] = useState<RenderProgress | null>(null);
  const [timings, setTimings] = useState<StepTimingInfo[]>([]);
  const [renderedFile, setRenderedFileState] = useState<string | null>(null);
  // A/B compare: keep the previous render result so two pipelines can be
  // compared side by side (render once, tweak, render again).
  const [prevRenderedFile, setPrevRenderedFile] = useState<string | null>(null);
  const renderedRef = useRef<string | null>(null);
  // Input of the last completed render; A/B keeps its pair when the same
  // single input is rendered again (model A → B).
  const lastInputRef = useRef<string | null>(null);

  // Setting the result to null (file switch) also clears the A/B pair.
  const setRenderedFile = (v: string | null) => {
    if (v === null) {
      renderedRef.current = null;
      setPrevRenderedFile(null);
    }
    setRenderedFileState(v);
  };
  // Crash-safe queue resume: the batch is persisted on start and pruned as
  // jobs finish, so a restart can re-enqueue the files that never completed.
  const [savedQueue, setSavedQueue] = useState<BatchQueueState | null>(null);
  const queueRef = useRef<BatchQueueState | null>(null);

  const persistQueue = (q: BatchQueueState | null) => {
    void backend().then((b) => {
      if (q) void b.saveBatchQueue(JSON.stringify(q)).catch(() => {});
      else void b.clearBatchQueue().catch(() => {});
    });
  };

  useEffect(() => {
    let on = true;
    void backend()
      .then((b) => b.loadBatchQueue())
      .then((raw) => {
        if (!on || !raw) return;
        try {
          const q = JSON.parse(raw) as BatchQueueState;
          if (q.inputs?.length) {
            queueRef.current = q;
            setSavedQueue(q);
          }
        } catch {
          // corrupt state: ignore
        }
      })
      .catch(() => {});
    return () => {
      on = false;
    };
  }, []);

  const markDone = (input: string) => {
    const qi = queueRef.current;
    if (!qi) return;
    const qn = { ...qi, done: [...qi.done, input], updatedAt: Date.now() };
    if (qn.done.length >= qn.inputs.length) {
      queueRef.current = null;
      setSavedQueue(null);
      persistQueue(null);
    } else {
      queueRef.current = qn;
      setSavedQueue(qn);
      persistQueue(qn);
    }
  };

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
    // Sample renders live in a `sample/` folder. In Tauri that's under the
    // project; in the web mode the project dir is browser-only (no server
    // equivalent), so write next to the input instead.
    if (isSample) return joinPath(isWeb() ? dir : (pd ?? dir), "sample", name);
    return joinPath(dir, name);
  };

  const startBatch = async (
    onlySelected = false,
    range?: { inMs: number; outMs: number } | null,
    explicit: string[] | null = null,
  ) => {
    const inputs = explicit ?? (onlySelected ? files.filter((f) => selected.includes(f)) : files);
    if (!inputs.length || renderingRef.current) return;
    renderingRef.current = true; // block stale-closure re-entry immediately
    const outs = steps.filter((s) => s.enabled && s.stepType === "output");
    const lastOut = outs.length ? outs[outs.length - 1] : undefined;
    const enabled = steps.filter((s) => s.enabled);
    const interp = enabled.find((s) => s.stepType === "interpolation");
    const dc = enabled.find((s) => s.stepType === "decompress");
    const up = enabled.find((s) => s.stepType === "upscale");
    const res = enabled.find((s) => s.stepType === "resize");
    const dn = enabled.find((s) => s.stepType === "denoise");
    const db = enabled.find((s) => s.stepType === "deblur");
    const dd = enabled.find((s) => s.stepType === "deduplication");
    const fl = enabled.find((s) => s.stepType === "filter");
    const outScale = up ? (up.params?.scale ?? null) : null;
    const outModel = up ? (up.params?.modelId ?? null) : null;
    const outDecompressModel = dc ? (dc.params?.modelId ?? null) : null;
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
      ffmpegFilter: fl ? (fl.params?.filter?.trim() ? fl.params.filter : null) : null,
    };
    const config: RenderConfig = {
      scale: outScale,
      modelId: outModel,
      resize: null,
      filter: outFilter,
      decompressModelId: outDecompressModel,
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
    // Reset the result view but keep the A/B pair when re-rendering the same
    // single input (model A → B). A file switch or multi-file batch clears it.
    const keepPair = inputs.length === 1 && inputs[0] === lastInputRef.current;
    setRenderedFileState(null);
    if (!keepPair) {
      renderedRef.current = null;
      setPrevRenderedFile(null);
    }
    setTimings([]);
    const q0: BatchQueueState = {
      inputs: initial.map((j) => j.input),
      done: [],
      updatedAt: Date.now(),
    };
    queueRef.current = q0;
    setSavedQueue(q0);
    persistQueue(q0);

    const patch = (i: number, p: Partial<BatchJob>) =>
      setJobs((prev) => prev.map((j, k) => (k === i ? { ...j, ...p } : j)));

    try {
      const be = await backend();
      for (let i = 0; i < initial.length; i++) {
        let output = initial[i].output;
        try {
          output = await be.uniquePath(output); // collision -> _2, _3, …
        } catch {
          // keep the intended path if resolution fails
        }
        patch(i, { output, status: "rendering", progress: null });
        try {
          await be.render(initial[i].input, output, config, (p) => {
            patch(i, { progress: p });
            setProgress(p);
            if (p.steps.length) setTimings(p.steps);
          });
          patch(i, { status: "done" });
          markDone(initial[i].input);
          setPrevRenderedFile(renderedRef.current);
          renderedRef.current = output;
          setRenderedFile(output);
          lastInputRef.current = initial[i].input;
          if (range) {
            // Sample renders live in the project's sample/ folder: keep only the newest.
            void be.pruneSamples(dirname(output), 5);
          }
        } catch (e) {
          const msg = String(e);
          if (msg.toLowerCase().includes("cancelled")) {
            patch(i, { status: "cancelled" });
            setJobs((prev) => prev.map((j, k) => (k > i ? { ...j, status: "cancelled" as const } : j)));
            break; // stop the batch
          }
          patch(i, { status: "failed", error: msg });
          markDone(initial[i].input); // don't re-run known failures on resume
          onError(`render failed: ${shortReason(msg)}`);
          // continue with the next file
        }
      }
    } finally {
      renderingRef.current = false;
      setRendering(false);
      setPaused(false);
      setProgress(null);
    }
  };

  const cancel = () => {
    renderingRef.current = false;
    setRendering(false);
    setPaused(false);
    setJobs((prev) =>
      prev.map((j) =>
        j.status === "queued" || j.status === "rendering" ? { ...j, status: "cancelled" as const } : j,
      ),
    );
    void backend().then((b) => b.cancelRender());
  };

  const togglePause = () => {
    setPaused((p) => {
      void backend().then((b) => b.pauseRender(!p));
      return !p;
    });
  };

  const resumeQueue = () => {
    const q = queueRef.current;
    if (!q) return;
    const remaining = q.inputs.filter((f) => !q.done.includes(f));
    if (!remaining.length) {
      discardQueue();
      return;
    }
    void startBatch(false, null, remaining);
  };

  const discardQueue = () => {
    queueRef.current = null;
    setSavedQueue(null);
    persistQueue(null);
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
    timings,
    setTimings,
    renderedFile,
    setRenderedFile,
    prevRenderedFile,
    savedQueue,
    resumeQueue,
    discardQueue,
    startBatch,
    cancel,
    togglePause,
  };
}

function toFactor(v: string): number | null {
  const f = Number(v);
  return f > 0 ? f : null;
}
