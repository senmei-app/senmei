import { useEffect, useRef, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { probeVideo, readFrame, type RenderProgress, type VideoInfo } from "@senmei/bridge";
import { useI18n } from "../i18n";

const FRAME_STEP_MS = 250;

function fmt(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const cs = Math.floor((ms % 1000) / 10);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(sec)}.${pad(cs)}`;
}

export default function Monitor({
  file,
  renderedFile,
  rendering,
  progress,
}: {
  file?: string;
  renderedFile: string | null;
  rendering: boolean;
  progress: RenderProgress | null;
}) {
  const { t } = useI18n();
  const [mode, setMode] = useState<"source" | "result" | "compare">("source");
  const src = mode === "result" && renderedFile ? renderedFile : (file ?? null);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [posMs, setPosMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [frames, setFrames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounce = useRef<number | null>(null);
  const name = src ? src.split("/").pop() : null;

  const loadFrame = (ms: number) => {
    if (!isTauri()) return;
    const targets: string[] = [];
    if (mode === "compare") {
      if (file) targets.push(file);
      if (renderedFile) targets.push(renderedFile);
    } else if (src) {
      targets.push(src);
    }
    if (targets.length === 0) return;
    setLoading(true);
    targets.forEach((p) => {
      readFrame(p, ms)
        .then((b64) => {
          setFrames((prev) => ({ ...prev, [p]: `data:image/jpeg;base64,${b64}` }));
          setError(null);
        })
        .catch((e) => {
          console.error("readFrame failed:", e);
          setError(String(e));
        })
        .finally(() => setLoading(false));
    });
  };

  // Auto-switch to the Result view once a render completes.
  const prevRendered = useRef<string | null>(null);
  useEffect(() => {
    if (renderedFile && renderedFile !== prevRendered.current) setMode("result");
    prevRendered.current = renderedFile;
  }, [renderedFile]);

  useEffect(() => {
    setInfo(null);
    setPosMs(0);
    setPlaying(false);
    setFrames({});
    setError(null);
    if (!isTauri()) return;
    const probeTarget = src ?? file;
    if (probeTarget) {
      probeVideo(probeTarget)
        .then(setInfo)
        .catch((e) => {
          console.error("probeVideo failed:", e);
          setError(String(e));
        });
    }
    loadFrame(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, file, mode]);

  const onScrub = (ms: number) => {
    setPosMs(ms);
    if (debounce.current) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => loadFrame(ms), 120);
  };

  useEffect(() => {
    if (!playing || !info || (info.duration ?? 0) <= 0) return;
    const durMs = (info.duration ?? 0) * 1000;
    const id = window.setInterval(() => {
      setPosMs((p) => {
        const next = p + FRAME_STEP_MS;
        if (next >= durMs) {
          setPlaying(false);
          loadFrame(durMs);
          return durMs;
        }
        loadFrame(next);
        return next;
      });
    }, FRAME_STEP_MS + 80);
    return () => window.clearInterval(id);
  }, [playing, info]);

  const maxMs = info ? Math.max(1, (info.duration ?? 0) * 1000) : 1;
  const pct =
    rendering && progress && progress.totalFrames > 0
      ? Math.round((progress.framesProcessed / progress.totalFrames) * 100)
      : null;

  const tabCls = (active: boolean) =>
    active
      ? "rounded-md border border-indigo-500/40 bg-indigo-600/40 px-2 py-1 text-[10px] font-mono text-indigo-200 backdrop-blur"
      : "rounded-md bg-black/50 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur hover:bg-black/60";

  return (
    <main className="flex h-full flex-col bg-slate-100 p-4 dark:bg-slate-950">
      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-black shadow-2xl dark:border-slate-800">
        {mode === "compare" && file && renderedFile ? (
          <div className="flex h-full w-full">
            <div className="relative flex-1 overflow-hidden border-r border-slate-700/50">
              {frames[file] ? (
                <img src={frames[file]} alt="original" className="h-full w-full object-contain" />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
                  <span className="truncate px-4 font-mono text-sm text-slate-500">
                    {file.split("/").pop()}
                  </span>
                </div>
              )}
              <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                {t("monitor.original")}
              </span>
            </div>
            <div className="relative flex-1 overflow-hidden">
              {frames[renderedFile] ? (
                <img src={frames[renderedFile]} alt="result" className="h-full w-full object-contain" />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
                  <span className="truncate px-4 font-mono text-sm text-slate-500">
                    {renderedFile.split("/").pop()}
                  </span>
                </div>
              )}
              <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                {t("monitor.result")}
              </span>
            </div>
          </div>
        ) : src && frames[src] ? (
          <img src={frames[src]} alt="preview" className="max-h-full max-w-full object-contain" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-200/80 dark:bg-slate-900/80">
            <span className="truncate px-4 font-mono text-sm text-slate-500 dark:text-slate-500">
              {name ?? t("monitor.placeholder")}
            </span>
          </div>
        )}

        <div className="absolute top-3 left-3 flex space-x-2">
          <button onClick={() => setMode("source")} className={tabCls(mode === "source")}>
            {t("monitor.original")}
            {mode === "source" && info ? ` ${info.width}x${info.height}` : ""}
          </button>
          <button
            onClick={() => setMode("compare")}
            disabled={!renderedFile}
            className={tabCls(mode === "compare")}
          >
            {t("monitor.compare")}
          </button>
          <button
            onClick={() => setMode("result")}
            disabled={!renderedFile}
            className={tabCls(mode === "result")}
          >
            {t("monitor.result")}
          </button>
        </div>

        {loading && (
          <div className="absolute top-3 right-3 rounded-md bg-black/60 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur">
            …
          </div>
        )}

        {error && (
          <div className="absolute bottom-3 left-3 max-w-[80%] rounded-md bg-red-600/80 px-2 py-1 font-mono text-[10px] text-white backdrop-blur">
            {error}
          </div>
        )}

        {pct !== null && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-md bg-black/70 px-3 py-1.5 font-mono text-xs text-white backdrop-blur">
            {t("monitor.rendering")} {pct}%
            <div className="mt-1 h-1 w-40 overflow-hidden rounded-full bg-slate-700">
              <div className="h-full bg-indigo-400 transition-all" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white/60 p-3 backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/40">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setPlaying((p) => !p)}
              disabled={!info}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40"
            >
              {playing ? "⏸" : "▶"}
            </button>
            <span className="font-mono text-xs text-slate-600 dark:text-slate-300">
              {fmt(posMs)} / {info ? fmt((info.duration ?? 0) * 1000) : "00:00:00.00"}
            </span>
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={maxMs}
          step={50}
          value={Math.min(posMs, maxMs)}
          onChange={(e) => onScrub(Number(e.target.value))}
          disabled={!info}
          className="w-full cursor-ew-resize accent-indigo-600"
        />
      </div>
    </main>
  );
}
