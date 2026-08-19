import { useEffect, useRef, useState, type SyntheticEvent } from "react";
import { convertFileSrc, isTauri } from "@tauri-apps/api/core";
import { extractAudio, probeVideo, readFrame, type RenderProgress, type VideoInfo } from "@senmei/bridge";
import { loadDemo } from "../demo";
import { useI18n } from "../i18n";
import { comboFromEvent } from "../hotkeys";
import { basename } from "../paths";

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

// Round a time to the nearest frame boundary (whole ms) so a rendered sample
// starts on the exact source frame that contains it (keeps compare in lockstep).
function snapFrame(ms: number, fps: number): number {
  const frameMs = fps > 0 ? 1000 / fps : 0;
  return frameMs > 0 ? Math.round(Math.round(ms / frameMs) * frameMs) : Math.round(ms);
}

export default function Monitor({
  file,
  renderedFile,
  rendering,
  progress,
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

  const [mode, setMode] = useState<"source" | "result" | "compare">("source");
  // Source shows the whole source timeline; result/compare show only the
  // sample window (the rendered result spans exactly in..out).
  const tlSource = mode === "source";
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
  const name = src ? basename(src) : null;

  // Native <video> for the source preview; fall back to FFmpeg-decoded frames
  // only when the webview cannot load/play the file.
  const [nativeFailed, setNativeFailed] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const nativeSrc = isTauri() && !nativeFailed && file && mode === "source" ? convertFileSrc(file) : null;

  // Sound always comes from an FFmpeg-extracted AAC track: the webview can't
  // decode every audio codec (e.g. AC3 in anime files). The native <video> is
  // muted while this track is present so the two don't double up.
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setAudioUrl(null);
    if (!isTauri() || !file) return;
    extractAudio(file, projectDir ?? null)
      .then((p) => setAudioUrl(convertFileSrc(p)))
      .catch(() => setAudioUrl(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  // Preview volume shared by the extracted <audio> and the native <video>.
  const [volume, setVolume] = useState(() => {
    const saved = Number(localStorage.getItem("senmei.volume"));
    return Number.isFinite(saved) ? Math.min(1, Math.max(0, saved)) : 1;
  });
  const changeVolume = (v: number) => {
    setVolume(v);
    localStorage.setItem("senmei.volume", String(v));
  };
  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
    if (videoRef.current) videoRef.current.volume = volume;
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
    const a = audioRef.current;
    const audioActive = !!a && !!audioUrl;
    if (nativeSrc && videoRef.current) {
      const v = videoRef.current;
      const start = audioActive ? a!.paused : v.paused;
      if (audioActive) {
        if (a!.paused) void a!.play();
        else a!.pause();
        // muted video still needs play() to advance its timeline
        if (start) void v.play();
        else v.pause();
      } else if (v.paused) void v.play();
      else v.pause();
      return;
    }
    // Frame-fallback: the audio element carries the sound, the timer drives frames.
    if (audioActive) {
      if (a!.paused) void a!.play();
      else a!.pause();
    }
    setPlaying((p) => !p);
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
    } else if (mode === "result" && effRendered) {
      targets.push({ path: effRendered, ms: Math.max(0, ms - inMs) });
    } else if (src) {
      targets.push({ path: src, ms });
    }
    if (targets.length === 0) return Promise.resolve();
    if (!isTauri()) {
      loadDemo().then(({ demoFrame }) =>
        targets.forEach(({ path }) =>
          setFrames((prev) => ({ ...prev, [path]: `data:image/jpeg;base64,${demoFrame()}` })),
        ),
      );
      return Promise.resolve();
    }
    setLoading(true);
    return Promise.all(
      targets.map(({ path, ms: t }) =>
        readFrame(path, t, projectDir ?? null)
          .then((filePath) => ({ path, filePath }))
          .catch((e) => {
            console.error("readFrame failed:", e);
            setError(String(e));
            return null;
          }),
      ),
    )
      .then((results) => {
        // Update every side together so compare never shows one ahead of the
        // other (the result decode is slower than the source decode).
        const updates: Record<string, string> = {};
        for (const r of results) {
          if (r) updates[r.path] = convertFileSrc(r.filePath);
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
    const fileChanged = file !== prevFile.current;
    prevFile.current = file ?? null;
    const next =
      fileChanged ? 0 : mode === "result" || mode === "compare" ? Math.max(posMs, inMs) : posMs;
    setInfo(null);
    setPosMs(next);
    setPlaying(false);
    if (audioRef.current) audioRef.current.pause();
    setFrames({});
    setError(null);
    setNativeFailed(false);
    if (!isTauri()) {
      const probeTarget = file;
      if (probeTarget) {
        loadDemo().then(({ demoProbe }) => setInfo(demoProbe()));
        if (fileChanged) onSampleChange?.(0, 10000);
      }
      loadFrame(next);
      return;
    }
    const probeTarget = file;
    if (probeTarget) {
      probeVideo(probeTarget)
        .then((i) => {
          setInfo(i);
          if (fileChanged) onSampleChange?.(0, snapFrame(Math.min(10000, (i.duration ?? 0) * 1000), i.fps ?? 0));
        })
        .catch((e) => {
          console.error("probeVideo failed:", e);
          setError(String(e));
        });
    }
    loadFrame(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, file, mode]);

  const onScrub = (ms: number) => {
    setPosMs(ms);
    // A scrub outside the sample window repositions the window to start at the
    // playhead, so "Render Sample" clips from where you're looking.
    if (ms < inMs || ms >= outMs) {
      const dur = outMs > inMs ? outMs - inMs : 10000;
      const fps = info?.fps ?? 0;
      onSampleChange?.(snapFrame(ms, fps), snapFrame(ms + dur, fps));
    }
    if (audioRef.current) audioRef.current.currentTime = ms / 1000;
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
          if (audioRef.current) audioRef.current.pause();
          return;
        }
        next = inMs; // loop the sample within in..out
        posRef.current = next;
        setPosMs(next);
        if (audioRef.current) audioRef.current.currentTime = inMs / 1000;
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

  const tabCls = (active: boolean) =>
    active
      ? "rounded-md border border-indigo-500/40 bg-indigo-600/40 px-2 py-1 text-[10px] font-mono text-indigo-200 backdrop-blur"
      : "rounded-md bg-black/50 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur hover:bg-black/60";

  return (
    <main ref={rootRef} className={"flex h-full flex-col bg-slate-100 dark:bg-slate-950" + (isFull ? "" : " p-4")}>
      <div
        onDoubleClick={toggleFullscreen}
        className={
          "relative flex flex-1 items-center justify-center overflow-hidden bg-black" +
          (isFull ? "" : " rounded-2xl border border-slate-200 shadow-2xl dark:border-slate-800")
        }
      >
        {mode === "compare" && file && effRendered ? (
          <div className="flex h-full w-full">
            <div className="relative flex-1 overflow-hidden border-r border-slate-700/50">
              {frames[file] ? (
                <img src={frames[file]} alt="original" className="h-full w-full object-contain opacity-80" />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
                  <span className="truncate px-4 font-mono text-sm text-slate-500">
                    {basename(file)}
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
                    {basename(effRendered)}
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
            ref={(el) => {
              videoRef.current = el;
              if (el) el.volume = volume;
            }}
            src={nativeSrc}
            muted={!!audioUrl}
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
            className={"h-full w-full object-contain opacity-80" + (demoResult && mode === "result" ? " saturate-150 brightness-105" : "")}
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

        {isFull && (
          <button
            onClick={toggleFullscreen}
            title={t("monitor.exitFull")}
            className="absolute top-10 right-3 z-10 rounded-md bg-black/60 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur hover:bg-black/80"
          >
            ✕
          </button>
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
        <div className="relative">
          <div className="pointer-events-none absolute top-1/2 z-0 h-1.5 w-full -translate-y-1/2 rounded-full bg-slate-300 dark:bg-slate-700" />
          <div
            className="pointer-events-none absolute top-1/2 z-0 h-1.5 -translate-y-1/2 rounded-full bg-indigo-300 dark:bg-indigo-400/60"
            style={{ width: `${scrubPct}%` }}
          />
          {info && (
            <div
              className="pointer-events-none absolute top-1/2 z-0 h-1.5 -translate-y-1/2 rounded-full bg-indigo-600 ring-1 ring-indigo-400 dark:bg-indigo-500 dark:ring-indigo-300"
              style={{ left: `${inPct}%`, width: `${Math.max(0, outPct - inPct)}%` }}
            />
          )}
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
      {audioUrl && (
        /* WebKitGTK won't play media with display:none; keep it rendered
           but off-screen so the AAC track actually produces sound. */
        <audio
          ref={(el) => {
            audioRef.current = el;
            if (el) el.volume = volume;
          }}
          src={audioUrl}
          preload="auto"
          className="pointer-events-none absolute -left-[9999px] h-px w-px"
        />
      )}
    </main>
  );
}
