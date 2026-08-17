import { useEffect, useRef, useState, type CSSProperties, type SyntheticEvent } from "react";
import { convertFileSrc, isTauri } from "@tauri-apps/api/core";
import { probeVideo, readFrame, type RenderProgress, type VideoInfo } from "@senmei/bridge";
import { demoFrame, demoProbe } from "../mock";
import { useI18n } from "../i18n";

const FRAME_STEP_MS = 100;

function fmt(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const cs = Math.floor((ms % 1000) / 10);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(sec)}.${pad(cs)}`;
}

// "55s", "10m", "1h", "1m30s" or a bare number (seconds).
function parseDuration(input: string): number | null {
  const s = input.trim().toLowerCase();
  if (!s) return null;
  if (/^\d+(\.\d+)?$/.test(s)) return Math.round(Number(s) * 1000);
  const re = /(\d+)([smh])/g;
  let ms = 0;
  let m: RegExpExecArray | null;
  let any = false;
  while ((m = re.exec(s)) !== null) {
    any = true;
    const v = Number(m[1]);
    ms += m[2] === "s" ? v * 1000 : m[2] === "m" ? v * 60000 : v * 3600000;
  }
  return any ? ms : null;
}

function fmtDuration(ms: number): string {
  if (ms % 60000 === 0) return `${Math.round(ms / 60000)}m`;
  if (ms % 1000 === 0) return `${Math.round(ms / 1000)}s`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function Monitor({
  file,
  renderedFile,
  rendering,
  progress,
  sampleInMs = 0,
  sampleOutMs = 0,
  onSampleChange,
  onRenderSample,
}: {
  file?: string;
  renderedFile: string | null;
  rendering: boolean;
  progress: RenderProgress | null;
  sampleInMs?: number;
  sampleOutMs?: number;
  onSampleChange?: (inMs: number, outMs: number) => void;
  onRenderSample?: () => void;
}) {
  const { t } = useI18n();
  const [mode, setMode] = useState<"source" | "result" | "compare">("source");
  // Demo mode has no real render output; simulate one per video so Compare /
  // Result are usable in the browser without running a fake render.
  const demoResult = !isTauri() && file ? file.replace(/\.[^.]+$/, ".senmei.mp4") : null;
  const effRendered = renderedFile ?? demoResult;
  const src = mode === "result" && effRendered ? effRendered : (file ?? null);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [posMs, setPosMs] = useState(0);
  const inMs = sampleInMs;
  const outMs = sampleOutMs;
  const [playing, setPlaying] = useState(false);
  const [customVal, setCustomVal] = useState("");
  const [sampleMenu, setSampleMenu] = useState(false);
  const [frames, setFrames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounce = useRef<number | null>(null);
  const sampleMenuRef = useRef<HTMLDivElement>(null);
  const posRef = useRef(0);
  const name = src ? src.split("/").pop() : null;

  // Native <video> for the source preview; fall back to FFmpeg-decoded frames
  // only when the webview cannot load/play the file.
  const [nativeFailed, setNativeFailed] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const nativeSrc = isTauri() && !nativeFailed && file && mode === "source" ? convertFileSrc(file) : null;

  const onVideoTime = (e: SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    setPosMs(v.currentTime * 1000);
    const endSec = outMs / 1000;
    if (endSec > 0 && v.currentTime >= endSec) v.currentTime = inMs / 1000; // loop within sample
  };

  const togglePlay = () => {
    if (!info) return;
    if (nativeSrc && videoRef.current) {
      const v = videoRef.current;
      if (v.paused) void v.play();
      else v.pause();
      return;
    }
    setPlaying((p) => !p);
  };

  const loadFrame = (ms: number): Promise<void> => {
    // In compare both sides show the same source moment: the original is
    // clamped to the sample in-point (the result has no frames before it) and
    // the result is read at `source - inMs` (its timeline starts at inMs).
    const targets: { path: string; ms: number }[] = [];
    if (mode === "compare") {
      const source = Math.max(ms, inMs);
      if (file) targets.push({ path: file, ms: source });
      if (effRendered) targets.push({ path: effRendered, ms: Math.max(0, source - inMs) });
    } else if (src) {
      targets.push({ path: src, ms });
    }
    if (targets.length === 0) return Promise.resolve();
    if (!isTauri()) {
      targets.forEach(({ path }) =>
        setFrames((prev) => ({ ...prev, [path]: `data:image/jpeg;base64,${demoFrame()}` })),
      );
      return Promise.resolve();
    }
    setLoading(true);
    return Promise.all(
      targets.map(({ path, ms: t }) =>
        readFrame(path, t)
          .then((b64) => {
            setFrames((prev) => ({ ...prev, [path]: `data:image/jpeg;base64,${b64}` }));
            setError(null);
          })
          .catch((e) => {
            console.error("readFrame failed:", e);
            setError(String(e));
          }),
      ),
    )
      .then(() => undefined)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    posRef.current = posMs;
  }, [posMs]);

  // Close the sample preset menu on outside click.
  useEffect(() => {
    if (!sampleMenu) return;
    const onDown = (e: MouseEvent) => {
      if (sampleMenuRef.current && !sampleMenuRef.current.contains(e.target as Node)) setSampleMenu(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [sampleMenu]);

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
    setNativeFailed(false);
    if (!isTauri()) {
      const probeTarget = src ?? file;
      if (probeTarget) {
        setInfo(demoProbe());
        onSampleChange?.(0, 10000);
      }
      loadFrame(0);
      return;
    }
    const probeTarget = src ?? file;
    if (probeTarget) {
      probeVideo(probeTarget)
        .then((i) => {
          setInfo(i);
          onSampleChange?.(0, Math.min(10000, (i.duration ?? 0) * 1000));
        })
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
    if (nativeSrc && videoRef.current) {
      videoRef.current.currentTime = ms / 1000;
      return;
    }
    if (debounce.current) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => loadFrame(ms), 120);
  };

  // Playback advances the time indicator 1:1 with wall clock. At most one
  // decode is in flight: frames load only on FRAME_STEP_MS boundaries and are
  // skipped if the decoder can't keep up, so requests never pile up.
  useEffect(() => {
    if (!playing || nativeSrc || !info || (info.duration ?? 0) <= 0) return;
    const durMs = (info.duration ?? 0) * 1000;
    const endMs = Math.max(inMs, Math.min(outMs || durMs, durMs));
    let last = performance.now();
    let busy = false;
    const id = window.setInterval(() => {
      const now = performance.now();
      const elapsed = now - last;
      last = now;
      const prev = posRef.current;
      let next = prev + elapsed;
      if (next >= endMs) {
        next = inMs; // loop the sample within in..out
        posRef.current = next;
        setPosMs(next);
        if (!busy) {
          busy = true;
          loadFrame(inMs).finally(() => {
            busy = false;
          });
        }
        return;
      }
      posRef.current = next;
      setPosMs(next);
      if (!busy && Math.floor(next / FRAME_STEP_MS) !== Math.floor(prev / FRAME_STEP_MS)) {
        busy = true;
        loadFrame(next).finally(() => {
          busy = false;
        });
      }
    }, 33);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, nativeSrc, info, inMs, outMs]);

  const maxMs = info ? Math.max(1, (info.duration ?? 0) * 1000) : 1;
  const scrubPct = maxMs > 0 ? Math.min(100, (posMs / maxMs) * 100) : 0;
  const inPct = maxMs > 0 ? Math.min(100, (inMs / maxMs) * 100) : 0;
  const outPct = maxMs > 0 ? Math.min(100, ((outMs || maxMs) / maxMs) * 100) : 100;

  // Sample-range presets relative to the current position.
  const applySample = (sec: number) => {
    if (!info) return;
    const durMs = (info.duration ?? 0) * 1000;
    const start = Math.min(posMs, durMs);
    onSampleChange?.(start, Math.min(start + sec * 1000, durMs));
  };
  const setFullRange = () => {
    if (!info) return;
    onSampleChange?.(0, (info.duration ?? 0) * 1000);
  };

  const presetOf = (): string => {
    if (!info) return "10s";
    const durMs = outMs - inMs;
    const totalMs = (info.duration ?? 0) * 1000;
    if (Math.abs(durMs - 10000) < 50) return "10s";
    if (Math.abs(durMs - 30000) < 50) return "30s";
    if (Math.abs(durMs - 60000) < 50) return "60s";
    if (outMs >= totalMs || Math.abs(durMs - totalMs) < 50) return "full";
    return "custom";
  };

  const applyCustom = () => {
    if (!info) return;
    const ms = parseDuration(customVal);
    if (ms === null) return;
    const durMs = (info.duration ?? 0) * 1000;
    const start = Math.min(posMs, durMs);
    onSampleChange?.(start, Math.min(start + ms, durMs));
    setSampleMenu(false);
  };
  const pct =
    rendering && progress && progress.totalFrames > 0
      ? Math.round((progress.framesProcessed / progress.totalFrames) * 100)
      : null;

  const renderStart = useRef<number | null>(null);
  useEffect(() => {
    renderStart.current = rendering ? (renderStart.current ?? Date.now()) : null;
  }, [rendering]);

  const fmtEta = (s: number): string => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  const stats = (() => {
    if (!rendering || !progress || progress.totalFrames <= 0 || renderStart.current === null) {
      return null;
    }
    const elapsed = Math.max(0.001, (Date.now() - renderStart.current) / 1000);
    const fps = progress.framesProcessed / elapsed;
    const remaining = fps > 0 ? (progress.totalFrames - progress.framesProcessed) / fps : 0;
    return { fps, eta: fmtEta(remaining) };
  })();

  const tabCls = (active: boolean) =>
    active
      ? "rounded-md border border-indigo-500/40 bg-indigo-600/40 px-2 py-1 text-[10px] font-mono text-indigo-200 backdrop-blur"
      : "rounded-md bg-black/50 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur hover:bg-black/60";

  return (
    <main className="flex h-full flex-col bg-slate-100 p-4 dark:bg-slate-950">
      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-black shadow-2xl dark:border-slate-800">
        {mode === "compare" && file && effRendered ? (
          <div className="flex h-full w-full">
            <div className="relative flex-1 overflow-hidden border-r border-slate-700/50">
              {frames[file] ? (
                <img src={frames[file]} alt="original" className="h-full w-full object-contain opacity-80" />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
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
              {frames[effRendered] ? (
                <img
                  src={frames[effRendered]}
                  alt="result"
                  className={"h-full w-full object-contain opacity-80" + (demoResult ? " saturate-150 brightness-105" : "")}
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
                  <span className="truncate px-4 font-mono text-sm text-slate-500">
                    {effRendered.split("/").pop()}
                  </span>
                </div>
              )}
              <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                {t("monitor.result")}
              </span>
            </div>
          </div>
        ) : nativeSrc ? (
          <video
            key={nativeSrc}
            ref={videoRef}
            src={nativeSrc}
            onError={() => setNativeFailed(true)}
            onLoadedMetadata={(e) => (e.currentTarget.currentTime = inMs / 1000)}
            onTimeUpdate={onVideoTime}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            className="max-h-full max-w-full object-contain opacity-80"
          />
        ) : src && frames[src] ? (
          <img
            src={frames[src]}
            alt="preview"
            className={"max-h-full max-w-full object-contain opacity-80" + (demoResult && mode === "result" ? " saturate-150 brightness-105" : "")}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
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
            disabled={!effRendered}
            className={tabCls(mode === "compare")}
          >
            {t("monitor.compare")}
          </button>
          <button
            onClick={() => setMode("result")}
            disabled={!effRendered}
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

        {pct !== null && progress && (
          <div className="absolute bottom-3 left-3 rounded-md bg-black/75 px-3 py-2 font-mono text-[11px] leading-5 text-white backdrop-blur">
            <div>{t("monitor.status")}: {t("queue.rendering")}</div>
            <div>{t("monitor.fps")}: {stats ? stats.fps.toFixed(1) : "–"}</div>
            <div>{t("monitor.eta")}: {stats ? stats.eta : "–"}</div>
            <div className="mt-1 h-1 w-40 overflow-hidden rounded-full bg-slate-700">
              <div className="h-full bg-indigo-400 transition-all" style={{ width: `${pct}%` }} />
            </div>
            <div className="mt-0.5 text-[10px] text-slate-400">
              {pct}% · {progress.framesProcessed}/{progress.totalFrames}
            </div>
          </div>
        )}
      </div>

      <div className="relative z-10 mt-4 rounded-xl border border-slate-200 bg-white/60 p-3 backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/40">
        <div className="mb-2 flex items-center">
          <div className="flex items-center space-x-2">
            <button
              onClick={togglePlay}
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
        <div className="mb-2 flex items-center space-x-1">
          <span className="text-[10px] text-slate-400 dark:text-slate-500">{t("sample.range")}</span>
          <div className="flex divide-x divide-slate-200 overflow-hidden rounded-lg border border-slate-200 dark:divide-slate-700 dark:border-slate-700">
            {[
              { v: "10s", label: "10s", sec: 10 },
              { v: "30s", label: "30s", sec: 30 },
              { v: "60s", label: "60s", sec: 60 },
            ].map((o) => (
              <button
                key={o.v}
                onClick={() => applySample(o.sec)}
                disabled={!info}
                className={
                  "px-2.5 py-1 font-mono text-[10px] disabled:opacity-40 " +
                  (presetOf() === o.v
                    ? "bg-indigo-600/20 text-indigo-600 dark:text-indigo-300"
                    : "bg-white text-slate-600 hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800")
                }
              >
                {o.label}
              </button>
            ))}
            <button
              onClick={setFullRange}
              disabled={!info}
              className={
                "px-2.5 py-1 font-mono text-[10px] disabled:opacity-40 " +
                (presetOf() === "full"
                  ? "bg-indigo-600/20 text-indigo-600 dark:text-indigo-300"
                  : "bg-white text-slate-600 hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800")
              }
            >
              {t("sample.full")}
            </button>
          </div>
          <div className="relative flex items-center" ref={sampleMenuRef}>
            <button
              onClick={() => {
                setSampleMenu((m) => !m);
                setCustomVal(fmtDuration(outMs - inMs));
              }}
              disabled={!info}
              title={t("sample.custom")}
              className={
                "rounded-lg border px-2 py-1 text-[10px] leading-none disabled:opacity-40 " +
                (presetOf() === "custom"
                  ? "border-indigo-500/60 bg-indigo-600/20 text-indigo-600 dark:text-indigo-300"
                  : "border-slate-200 bg-white text-slate-500 hover:border-indigo-500/50 hover:text-indigo-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400")
              }
            >
              ▾
            </button>
            {sampleMenu && (
              <div className="absolute left-0 bottom-full z-30 mb-1 w-48 rounded-lg border border-slate-200 bg-white p-2 shadow-lg dark:border-slate-700 dark:bg-slate-900">
                <p className="mb-1 text-[9px] uppercase tracking-wider text-slate-400">{t("sample.custom")}</p>
                <div className="flex items-center space-x-1">
                  <input
                    value={customVal}
                    onChange={(e) => setCustomVal(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") applyCustom();
                      if (e.key === "Escape") setSampleMenu(false);
                    }}
                    placeholder={t("sample.customPlaceholder")}
                    autoFocus
                    className="w-full rounded-lg border border-slate-200 bg-white px-2 py-1 font-mono text-[10px] text-slate-700 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                  />
                  <button
                    onClick={applyCustom}
                    disabled={!info}
                    title={t("sample.apply")}
                    className="shrink-0 rounded-md bg-indigo-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
                  >
                    ✓
                  </button>
                </div>
              </div>
            )}
          </div>
          {presetOf() === "custom" && (
            <span className="font-mono text-[10px] text-indigo-500 dark:text-indigo-300">
              {fmtDuration(outMs - inMs)}
            </span>
          )}
          {onRenderSample && (
            <button
              onClick={onRenderSample}
              disabled={!info || outMs <= inMs}
              className="rounded-lg bg-indigo-600 px-3 py-1 font-mono text-[10px] font-medium text-white shadow-md shadow-indigo-600/30 transition hover:bg-indigo-500 active:scale-95 disabled:opacity-40"
            >
              {t("sample.render")}
            </button>
          )}
        </div>
        <div className="relative">
          <input
            type="range"
            min={0}
            max={maxMs}
            step={50}
            value={Math.min(posMs, maxMs)}
            onChange={(e) => onScrub(Number(e.target.value))}
            disabled={!info}
            className="scrubber relative z-10 w-full cursor-ew-resize"
            style={{ "--scrub-pct": `${scrubPct}%` } as CSSProperties}
          />
          {info && (
            <div
              className="pointer-events-none absolute top-1/2 z-0 h-1.5 -translate-y-1/2 rounded-full bg-indigo-500/25 dark:bg-indigo-500/30"
              style={{ left: `${inPct}%`, width: `${Math.max(0, outPct - inPct)}%` }}
            />
          )}
        </div>
        {info && (
          <div className="mt-1 flex justify-between font-mono text-[9px] text-slate-400 dark:text-slate-500">
            <span>
              {t("timeline.in")} {fmt(inMs)}
            </span>
            <span>
              {t("timeline.out")} {fmt(outMs || (info.duration ?? 0) * 1000)}
            </span>
          </div>
        )}
      </div>
    </main>
  );
}
