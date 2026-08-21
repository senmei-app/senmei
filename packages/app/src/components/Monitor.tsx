import { useEffect, useRef, useState, type SyntheticEvent } from "react";
import type { RenderProgress, StepTimingInfo, VideoInfo } from "@senmei/bridge";
import { backend, type Backend } from "../backend";
import { useI18n } from "../i18n";
import { comboFromEvent } from "../hotkeys";
import { basename } from "../paths";
import { fmt, fmtDuration, parseDuration, snapFrame } from "./monitor/format";
import Benchmark from "./monitor/Benchmark";
import CompareView from "./monitor/CompareView";
import ModeTabs from "./monitor/ModeTabs";
import Timeline from "./monitor/Timeline";

export default function Monitor({
  file,
  renderedFile,
  prevRenderedFile,
  rendering,
  progress,
  timings = [],
  sampleInMs = 0,
  sampleOutMs = 0,
  projectDir,
  onSampleChange,
  onRenderSample,
  toggleFullscreenSignal = 0,
  togglePlayHotkey = "Space",
}: {
  file?: string;
  renderedFile: string | null;
  rendering: boolean;
  progress: RenderProgress | null;
  /** Per-step timing from the last finished render (FPS benchmark). */
  timings?: StepTimingInfo[];
  /** Previous render result, kept for A/B compare. */
  prevRenderedFile?: string | null;
  sampleInMs?: number;
  sampleOutMs?: number;
  projectDir?: string | null;
  onSampleChange?: (inMs: number, outMs: number) => void;
  onRenderSample?: () => void;
  toggleFullscreenSignal?: number;
  togglePlayHotkey?: string;
}) {
  const { t } = useI18n();
  // Native fullscreen on the monitor element itself (WebKit fullscreen API):
  // the same video/frame instance stays mounted, so playback continues and no
  // second decoder runs underneath. Esc exits fullscreen natively.
  const rootRef = useRef<HTMLElement | null>(null);
  const [isFull, setIsFull] = useState(false);
  const toggleFullscreen = () => {
    if (document.fullscreenElement) {
      setIsFull(false);
      void document.exitFullscreen();
    } else {
      void rootRef.current?.requestFullscreen().then(() => setIsFull(true)).catch(() => {});
    }
  };
  // View menu "Full Video Mode" toggles fullscreen on each signal.
  useEffect(() => {
    if (toggleFullscreenSignal > 0) toggleFullscreen();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toggleFullscreenSignal]);
  // Keep the ✕/hint state in sync with native exits (Esc).
  useEffect(() => {
    const onFs = () => setIsFull(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const [mode, setMode] = useState<"source" | "result" | "compare" | "ab">("source");
  // Source shows the whole source timeline; result/compare show only the
  // sample window (the rendered result spans exactly in..out).
  const tlSource = mode === "source";
  const effRendered = renderedFile;
  const src = mode === "result" && effRendered ? effRendered : (file ?? null);
  // A/B + compare render side-by-side panes; the single-view fallback
  // (video/frame/placeholder) must not also render underneath.
  const showingCompare =
    (mode === "ab" && prevRenderedFile && effRendered) ||
    (mode === "compare" && file && effRendered);
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
  const frameBustRef = useRef(0);
  const sampleMenuRef = useRef<HTMLDivElement>(null);
  const posRef = useRef(0);
  const name = src ? basename(src) : null;

  // Native <video> for the source preview; fall back to FFmpeg-decoded frames
  // only when the webview cannot load/play the file. The URL is only set after
  // probeVideo succeeds — that command grants the file into the asset:// scope
  // (allow_file), so the <video> never races a not-yet-allowed path.
  const [nativeFailed, setNativeFailed] = useState(false);
  const [nativeUrl, setNativeUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const nativeSrc = !nativeFailed && file && mode === "source" ? nativeUrl : null;

  // Sound comes from the backend (rodio): WebKitGTK can't play media over
  // Tauri's asset:// scheme, so audio is decoded/played natively and driven
  // over IPC. The native <video> stays muted (its audio would double up).
  const [audioReady, setAudioReady] = useState(false);
  // The resolved backend; re-render once available so nativeSrc is non-null.
  const beRef = useRef<Backend | null>(null);
  const [beReady, setBeReady] = useState(false);
  useEffect(() => {
    let on = true;
    backend().then((b) => {
      if (!on) return;
      beRef.current = b;
      setBeReady(true);
    });
    return () => {
      on = false;
    };
  }, []);
  const be = () => beRef.current;

  // A stale extraction after a file switch must not load the old track.
  const audioFileRef = useRef<string | null>(null);
  useEffect(() => {
    // Drop the previous track so a stale sink can't play during extraction.
    void be()?.audioClear().catch(() => {});
    setAudioReady(false);
    if (!be() || !file) return;
    audioFileRef.current = file;
    be()!
      .extractAudio(file, projectDir ?? null)
      .then((p) => {
        if (audioFileRef.current !== file) return;
        return be()!
          .audioLoad(p)
          .then(() => setAudioReady(true))
          .catch((e) => console.error("audio load failed:", e));
      })
      .catch((e) => console.error("preview extractAudio failed:", e));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, beReady]);

  // Audio arrives async (extraction takes seconds); if playback already
  // started, join it at the current position instead of staying silent.
  useEffect(() => {
    if (audioReady && playing) {
      void be()?.audioSeek(posRef.current).catch(() => {});
      void be()?.audioPlay().catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioReady]);

  // Preview volume.
  const [volume, setVolume] = useState(() => {
    const saved = Number(localStorage.getItem("senmei.volume"));
    return Number.isFinite(saved) ? Math.min(1, Math.max(0, saved)) : 1;
  });
  const changeVolume = (v: number) => {
    setVolume(v);
    localStorage.setItem("senmei.volume", String(v));
  };
  useEffect(() => {
    void be()?.audioSetVolume(volume).catch(() => {});
  }, [volume]);

  const onVideoTime = (e: SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    setPosMs(v.currentTime * 1000);
    if (v.paused) return; // don't loop while scrubbing
    // Only result/compare loop the sample window; the original plays to the end.
    if (mode === "source") return;
    const endSec = outMs / 1000;
    if (endSec > 0 && v.currentTime >= endSec) v.currentTime = inMs / 1000; // loop within sample
  };

  const togglePlay = () => {
    if (!info) return;
    if (nativeSrc && videoRef.current) {
      const v = videoRef.current;
      if (v.paused) {
        void v.play();
        void be()?.audioPlay().catch(() => {});
      } else {
        v.pause();
        void be()?.audioPause().catch(() => {});
      }
      return;
    }
    // Frame-fallback: rodio carries the sound, the timer drives frames.
    setPlaying((p) => {
      if (!p) void be()?.audioPlay().catch(() => {});
      else void be()?.audioPause().catch(() => {});
      return !p;
    });
  };

  // togglePlayHotkey (default Space) toggles play/pause (ignored while typing
  // or on a focused button).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          t.tagName === "BUTTON" ||
          t.isContentEditable)
      ) {
        return;
      }
      if (comboFromEvent(e) !== togglePlayHotkey) return;
      e.preventDefault();
      togglePlay();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info, nativeSrc, inMs, outMs, togglePlayHotkey]);

  const loadFrame = (ms: number): Promise<void> => {
    // In compare both sides show the same source moment: the original is
    // clamped to the sample in-point (the result has no frames before it) and
    // the result is read at `source - inMs` (its timeline starts at inMs).
    const targets: { path: string; ms: number }[] = [];
    if (mode === "compare") {
      const source = Math.max(ms, inMs);
      if (file) targets.push({ path: file, ms: source });
      if (effRendered) targets.push({ path: effRendered, ms: Math.max(0, source - inMs) });
    } else if (mode === "ab") {
      if (prevRenderedFile) targets.push({ path: prevRenderedFile, ms: Math.max(0, ms - inMs) });
      if (effRendered) targets.push({ path: effRendered, ms: Math.max(0, ms - inMs) });
    } else if (mode === "result" && effRendered) {
      targets.push({ path: effRendered, ms: Math.max(0, ms - inMs) });
    } else if (src) {
      targets.push({ path: src, ms });
    }
    if (targets.length === 0) return Promise.resolve();
    setLoading(true);
    const b = be();
    if (!b) return Promise.resolve();
    return Promise.all(
      targets.map(({ path, ms: t }) =>
        b
          .readFrame(path, t, projectDir ?? null)
          .then((src) => ({ path, src }))
          .catch((e) => {
            console.error("readFrame failed:", e);
            setError(String(e));
            return null;
          }),
      ),
    )
      .then((results) => {
        // Update every side together so compare never shows one ahead of the
        // other (the result decode is slower than the source decode). Frames
        // reuse a stable file per source; a query forces the webview to
        // re-fetch (asset:// URLs cache otherwise). data: URIs are always fresh.
        const updates: Record<string, string> = {};
        for (const r of results) {
          if (r) {
            updates[r.path] = r.src.startsWith("data:") ? r.src : r.src + "?v=" + ++frameBustRef.current;
          }
        }
        if (Object.keys(updates).length) {
          setFrames((prev) => ({ ...prev, ...updates }));
          setError(null);
        }
      })
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

  // A new video starts at 0; switching views keeps the position. The result
  // view clamps to the sample in-point so it shows the rendered moment instead
  // of jumping to 0, and the sample range is only (re)initialised per file.
  const prevFile = useRef<string | null>(null);
  useEffect(() => {
    let on = true;
    const fileChanged = file !== prevFile.current;
    prevFile.current = file ?? null;
    const next =
      fileChanged ? 0 : mode === "result" || mode === "compare" ? Math.max(posMs, inMs) : posMs;
    setInfo(null);
    setPosMs(next);
    setPlaying(false);
    void be()?.audioPause().catch(() => {});
    setFrames({});
    setError(null);
    setNativeFailed(false);
    setNativeUrl(null);
    const b = be();
    const probeTarget = file;
    if (probeTarget && b) {
      b.probeVideo(probeTarget)
        .then((i) => {
          if (!on) return;
          setInfo(i);
          setNativeUrl(b.nativeVideoUrl(probeTarget));
          if (fileChanged) onSampleChange?.(0, snapFrame(Math.min(10000, (i.duration ?? 0) * 1000), i.fps ?? 0));
        })
        .catch((e) => {
          if (!on) return;
          console.error("probeVideo failed:", e);
          setError(String(e));
        });
    }
    loadFrame(next);
    return () => {
      on = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, file, mode, beReady]);

  const onScrub = (ms: number) => {
    setPosMs(ms);
    // A scrub outside the sample window repositions the window to start at the
    // playhead, so "Render Sample" clips from where you're looking.
    if (ms < inMs || ms >= outMs) {
      const dur = outMs > inMs ? outMs - inMs : 10000;
      const fps = info?.fps ?? 0;
      onSampleChange?.(snapFrame(ms, fps), snapFrame(ms + dur, fps));
    }
    void be()?.audioSeek(ms).catch(() => {});
    if (nativeSrc && videoRef.current) {
      videoRef.current.currentTime = ms / 1000;
      return;
    }
    if (debounce.current) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => loadFrame(ms), 120);
  };

  // Playback advances the time indicator 1:1 with wall clock. At most one
  // decode is in flight: frames load on source-frame boundaries (no fixed
  // 100ms) and are skipped if the decoder can't keep up, so requests never
  // pile up and playback runs at the source rate (~24-30 fps).
  useEffect(() => {
    if (!playing || nativeSrc || !info || (info.duration ?? 0) <= 0) return;
    const durMs = (info.duration ?? 0) * 1000;
    // Original plays the whole file; result/compare loop the sample window.
    const endMs = mode === "source" ? durMs : Math.max(inMs, Math.min(outMs || durMs, durMs));
    const stepMs = info.fps && info.fps > 0 ? Math.max(33, Math.round(1000 / info.fps)) : 100;
    let last = performance.now();
    let busy = false;
    const id = window.setInterval(() => {
      const now = performance.now();
      const elapsed = now - last;
      last = now;
      const prev = posRef.current;
      let next = prev + elapsed;
      if (next >= endMs) {
        if (mode === "source") {
          // End of the video: stop instead of looping the sample window.
          setPlaying(false);
          void be()?.audioPause().catch(() => {});
          return;
        }
        next = inMs; // loop the sample within in..out
        posRef.current = next;
        setPosMs(next);
        void be()?.audioSeek(inMs).catch(() => {});
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
      if (!busy && Math.floor(next / stepMs) !== Math.floor(prev / stepMs)) {
        busy = true;
        loadFrame(next).finally(() => {
          busy = false;
        });
      }
    }, 33);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, nativeSrc, info, inMs, outMs]);

  const tlMin = tlSource ? 0 : inMs;
  const tlMax = tlSource
    ? info
      ? Math.max(1, (info.duration ?? 0) * 1000)
      : 1
    : Math.max(inMs + 1, outMs);
  const tlSpan = tlMax - tlMin;
  const tlPos = posMs;
  const scrubPct = tlSpan > 0 ? Math.min(100, Math.max(0, ((posMs - tlMin) / tlSpan) * 100)) : 0;
  const inPct = tlSource ? Math.min(100, Math.max(0, ((inMs - tlMin) / tlSpan) * 100)) : 0;
  const outPct = tlSource ? Math.min(100, Math.max(0, ((outMs || tlMax) / tlSpan) * 100)) : 100;

  // Sample-range presets relative to the current position.
  const applySample = (sec: number) => {
    if (!info) return;
    const durMs = (info.duration ?? 0) * 1000;
    const fps = info.fps ?? 0;
    const start = snapFrame(Math.min(posMs, durMs), fps);
    onSampleChange?.(start, snapFrame(Math.min(start + sec * 1000, durMs), fps));
  };
  const setFullRange = () => {
    if (!info) return;
    onSampleChange?.(0, snapFrame((info.duration ?? 0) * 1000, info.fps ?? 0));
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
    const fps = info.fps ?? 0;
    const start = snapFrame(Math.min(posMs, durMs), fps);
    onSampleChange?.(start, snapFrame(Math.min(start + ms, durMs), fps));
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

  return (
    <main ref={rootRef} className={"relative flex h-full flex-col bg-slate-100 dark:bg-slate-950" + (isFull ? "" : " p-4")}>
      <div
        onDoubleClick={toggleFullscreen}
        className={
          "relative flex flex-1 items-center justify-center overflow-hidden bg-black" +
          (isFull ? "" : " rounded-2xl border border-slate-200 shadow-2xl dark:border-slate-800")
        }
      >
        <CompareView
          mode={mode}
          file={file}
          effRendered={effRendered}
          prevRenderedFile={prevRenderedFile}
          frames={frames}
        />
        {!showingCompare &&
          (nativeSrc ? (
            <video
              key={nativeSrc}
              ref={(el) => {
                videoRef.current = el;
                if (el) el.volume = volume;
              }}
              src={nativeSrc}
              muted
              onError={() => setNativeFailed(true)}
              onLoadedMetadata={(e) => (e.currentTarget.currentTime = inMs / 1000)}
              onTimeUpdate={onVideoTime}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              className="h-full w-full object-contain opacity-80"
            />
          ) : src && frames[src] ? (
            <img
              src={frames[src]}
              alt="preview"
              className="h-full w-full object-contain opacity-80"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
              <span className="truncate px-4 font-mono text-sm text-slate-500 dark:text-slate-500">
                {name ?? t("monitor.placeholder")}
              </span>
            </div>
          ))}

        <ModeTabs
          mode={mode}
          onMode={setMode}
          effRendered={effRendered}
          prevRenderedFile={prevRenderedFile}
          info={info}
        />

        {loading && (
          <div className="absolute top-3 right-3 rounded-md bg-black/60 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur">
            …
          </div>
        )}

        {isFull && (
          <button
            onClick={toggleFullscreen}
            title={t("monitor.exitFull")}
            className="absolute top-10 right-3 z-10 rounded-md bg-black/60 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur hover:bg-black/80"
          >
            ✕
          </button>
        )}

        {isFull && (
          <div className="absolute inset-x-0 bottom-0 z-10 flex items-center gap-3 bg-gradient-to-t from-black/80 to-transparent px-4 pt-10 pb-3">
            <button
              onClick={togglePlay}
              disabled={!info}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/15 text-white hover:bg-white/25 disabled:opacity-40"
            >
              {playing ? "⏸" : "▶"}
            </button>
            <span className="shrink-0 font-mono text-xs text-slate-200">
              {fmt(tlPos)} / {fmt(tlMax)}
            </span>
            <div className="relative flex-1">
              <div className="pointer-events-none absolute top-1/2 h-1.5 w-full -translate-y-1/2 rounded-full bg-white/25" />
              <div
                className="pointer-events-none absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-indigo-400"
                style={{ width: `${scrubPct}%` }}
              />
              <input
                type="range"
                min={tlMin}
                max={tlMax}
                step={50}
                value={Math.min(Math.max(tlPos, tlMin), tlMax)}
                onChange={(e) => onScrub(Number(e.target.value))}
                disabled={!info}
                className="scrubber relative z-10 w-full cursor-ew-resize"
              />
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <span className="text-xs leading-none">{volume === 0 ? "🔇" : volume < 0.5 ? "🔉" : "🔊"}</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                onChange={(e) => changeVolume(Number(e.target.value))}
                className="h-1 w-16 cursor-pointer accent-indigo-500"
              />
            </div>
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

      <div className={"relative z-10 mt-4 rounded-xl border border-slate-200 bg-white/60 p-3 backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/40" + (isFull ? " hidden" : "")}>
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
              {fmt(tlPos)} / {fmt(tlMax)}
            </span>
            <div className="flex items-center space-x-1">
              <span className="text-xs leading-none" title={t("monitor.volume")}>
                {volume === 0 ? "🔇" : volume < 0.5 ? "🔉" : "🔊"}
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                onChange={(e) => changeVolume(Number(e.target.value))}
                className="h-1 w-16 cursor-pointer accent-indigo-500"
              />
            </div>
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
        <Timeline
          tlMin={tlMin}
          tlMax={tlMax}
          tlPos={tlPos}
          scrubPct={scrubPct}
          inPct={inPct}
          outPct={outPct}
          onScrub={onScrub}
          disabled={!info}
        />
        <Benchmark timings={timings} />
        {info && (
          <div className="mt-1 flex justify-between font-mono text-[9px] text-slate-400 dark:text-slate-500">
            <span>
              {t("timeline.in")} {fmt(inMs)}
            </span>
            <span>
              {t("timeline.out")} {fmt(outMs || tlMax)}
            </span>
          </div>
        )}
      </div>
    </main>
  );
}
