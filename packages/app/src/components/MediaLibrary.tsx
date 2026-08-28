import { useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { Check, Copy, Film, FolderOpen, Hourglass, Pause, Play, Square, Upload, X } from "lucide-react";
import type { VideoInfo } from "@senmei/bridge";
import { backend } from "../backend";
import { useI18n } from "../i18n";
import { basename } from "../paths";
import type { BatchJob, BatchStatus } from "../steps";

const STATUS_ICON: Record<BatchStatus, { icon: LucideIcon; color: string; labelKey: string }> = {
  queued: { icon: Hourglass, color: "text-slate-400", labelKey: "queue.queued" },
  rendering: { icon: Play, color: "text-indigo-500 dark:text-indigo-400", labelKey: "queue.rendering" },
  done: { icon: Check, color: "text-emerald-500", labelKey: "queue.done" },
  failed: { icon: X, color: "text-rose-500", labelKey: "queue.failed" },
  cancelled: { icon: Square, color: "text-slate-400", labelKey: "queue.cancelled" },
};

/// "h264" → "H.264", "hevc" → "H.265", else upper-cased.
function codecLabel(c: string | null | undefined): string {
  if (!c) return "";
  const m: Record<string, string> = { h264: "H.264", hevc: "H.265", av1: "AV1", vp9: "VP9" };
  return m[c.toLowerCase()] ?? c.toUpperCase();
}

/// "1920×1080 · H.264" (or just "video" when the probe failed).
function tileMeta(info: VideoInfo): string {
  const codec = codecLabel(info.videoCodec);
  return `${info.width}×${info.height}${codec ? ` · ${codec}` : ""}`;
}

export default function MediaLibrary({
  files,
  hotkeys,
  onOpen,
  onRemoveFile,
  outputDir,
  onPickOutputDir,
  rendering,
  paused,
  onStartRender,
  onTogglePause,
  onCancel,
  jobs,
  selected,
  onSelect,
  multiSelect,
  onMultiSelectChange,
  view,
  onViewChange,
  savedQueue,
  onResumeQueue,
  onDiscardQueue,
}: {
  files: string[];
  hotkeys: Record<string, string>;
  onOpen: () => void;
  onRemoveFile: (path: string) => void;
  outputDir: string | null;
  onPickOutputDir: () => void;
  onStartRender: () => void;
  rendering: boolean;
  paused: boolean;
  onTogglePause: () => void;
  onCancel: () => void;
  jobs: BatchJob[];
  selected: string[];
  onSelect: (path: string, toggle: boolean) => void;
  multiSelect: boolean;
  onMultiSelectChange: (v: boolean) => void;
  view: "library" | "queue";
  savedQueue?: { inputs: string[]; done: string[] } | null;
  onResumeQueue?: () => void;
  onDiscardQueue?: () => void;
  onViewChange: (v: "library" | "queue") => void;
}) {
  const { t } = useI18n();

  const doneCount = jobs.filter((j) => j.status !== "queued" && j.status !== "rendering").length;

  // Per-file thumbnails + size/codec, fetched lazily once per file.
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [metas, setMetas] = useState<Record<string, string>>({});
  const pendingRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    let on = true;
    void backend()
      .catch(() => null)
      .then(async (be) => {
        if (!be) return;
        const missing = files.filter((f) => !thumbs[f] && !pendingRef.current.has(f));
        if (!missing.length) return;
        missing.forEach((f) => pendingRef.current.add(f));
        await Promise.all(
          missing.map(async (f) => {
            try {
              const [thumb, info] = await Promise.all([
                be.thumbnail(f),
                be.probeVideo(f).catch(() => null),
              ]);
              if (!on) return;
              if (thumb) setThumbs((m) => ({ ...m, [f]: thumb }));
              if (info) setMetas((m) => ({ ...m, [f]: tileMeta(info) }));
            } catch {
              // keep the placeholder tile
            }
          }),
        );
      });
    return () => {
      on = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  return (
    <aside className="flex h-full flex-col border-r border-slate-200 bg-slate-100/70 p-3 dark:border-slate-800/80 dark:bg-slate-900/30">
      <div className="mb-3 flex items-center justify-between px-1">
        <div className="flex gap-1">
          <button
            onClick={() => onViewChange("library")}
            title={`${t("media.tab.library")} (${hotkeys.viewLibrary})`}
            className={
              view === "library"
                ? "rounded-md border border-indigo-500/40 bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-600 dark:text-indigo-300"
                : "rounded-md bg-slate-200 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
            }
          >
            {t("media.tab.library")}
          </button>
          <button
            onClick={() => onViewChange("queue")}
            title={`${t("media.tab.queue")} (${hotkeys.viewQueue})`}
            className={
              view === "queue"
                ? "rounded-md border border-indigo-500/40 bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-600 dark:text-indigo-300"
                : "rounded-md bg-slate-200 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
            }
          >
            {t("media.tab.queue")}
          </button>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={() => onMultiSelectChange(!multiSelect)}
            title={`${t("media.multiSelect")} (${hotkeys.toggleMultiSelect})`}
            aria-label={t("media.multiSelect")}
            className={
              "flex h-[26px] w-7 items-center justify-center rounded-md " +
              (multiSelect
                ? "border border-indigo-500/40 bg-indigo-600/30 text-indigo-600 dark:text-indigo-300"
                : "bg-slate-200 text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700")
            }
          >
            <Copy className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onOpen}
            title={`${t("media.addVideos")} (${hotkeys.openFile})`}
            aria-label={t("media.addVideos")}
            className="flex h-[26px] w-7 items-center justify-center rounded-md bg-indigo-600 text-white hover:bg-indigo-500"
          >
            <Upload className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {savedQueue && !rendering && (
        <div className="mb-2 rounded-lg border border-indigo-500/40 bg-indigo-600/10 px-2 py-1.5">
          <div className="flex items-center justify-between gap-2 text-[11px]">
            <span className="text-indigo-600 dark:text-indigo-300">
              {t("queue.resumeBatch")} ({savedQueue.inputs.length - savedQueue.done.length})
            </span>
            <div className="flex shrink-0 gap-1">
              <button
                onClick={onResumeQueue}
                className="rounded-md bg-indigo-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-indigo-500"
              >
                {t("queue.resumeGo")}
              </button>
              <button
                onClick={onDiscardQueue}
                title={t("queue.resumeDiscard")}
                aria-label={t("queue.resumeDiscard")}
                className="rounded-md bg-slate-200 px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {view === "library" ? (
          files.length === 0 ? (
            <div
              onClick={onOpen}
              className="flex h-full cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-200/50 p-4 text-center transition hover:border-indigo-500/50 hover:bg-slate-200 dark:border-slate-700/80 dark:bg-slate-900/40 dark:hover:bg-slate-900/80"
            >
              <div className="mb-2 rounded-full bg-indigo-500/10 p-2 text-indigo-500 dark:text-indigo-400">
                <Upload className="h-5 w-5" />
              </div>
              <p className="text-xs font-medium text-slate-700 dark:text-slate-300">{t("media.drop")}</p>
              <p className="mt-1 text-[11px] text-slate-500">{t("media.formats")}</p>
            </div>
          ) : (
            <div className="space-y-2">
              {files.map((path) => {
                const isSel = selected.includes(path);
                return (
                <div
                  key={path}
                  onClick={(e) => onSelect(path, multiSelect || e.ctrlKey || e.metaKey)}
                  className={
                    "group flex cursor-pointer items-center space-x-3 rounded-lg border p-2 transition " +
                    (isSel
                      ? "border-indigo-500 bg-indigo-500/20 ring-1 ring-indigo-400"
                      : "border-indigo-500/30 bg-indigo-500/10 hover:bg-indigo-500/15")
                  }
                >
                  <div className="relative h-10 w-14 shrink-0 overflow-hidden rounded bg-slate-300 dark:bg-slate-800">
                    {thumbs[path] ? (
                      <img src={thumbs[path]} alt="" className="h-full w-full object-cover" />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-slate-400 dark:text-slate-500">
                        <Film className="h-5 w-5" />
                      </div>
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-slate-900 dark:text-slate-100">
                      {basename(path)}
                    </p>
                    <div className="mt-0.5 truncate text-[11px] text-slate-500 dark:text-slate-400">
                      {metas[path] ?? t("media.type.video")}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveFile(path);
                    }}
                    title={t("media.remove")}
                    aria-label={t("media.remove")}
                    className="rounded-md p-1 text-slate-300 transition hover:bg-red-500/10 hover:text-red-500 dark:text-slate-500 dark:hover:text-red-400"
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeWidth="2" d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                </div>
                );
              })}
            </div>
          )
        ) : (
          <div className="space-y-2 p-1">
            {rendering && (
              <div className="flex items-center justify-between rounded-lg border border-indigo-500/40 bg-indigo-500/10 p-2">
                <p className="text-[11px] font-medium text-indigo-600 dark:text-indigo-300">
                  {paused ? t("queue.paused") : t("queue.rendering")} · {doneCount}/{jobs.length}
                </p>
                <div className="flex space-x-1">
                  <button
                    onClick={onTogglePause}
                    className="flex items-center gap-1 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-600 hover:bg-amber-500/20 dark:text-amber-400"
                  >
                    {paused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
                    {paused ? t("queue.resume") : t("queue.pause")}
                  </button>
                  <button
                    onClick={onCancel}
                    className="rounded-md border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-[11px] font-medium text-red-500 hover:bg-red-500/20 dark:text-red-400"
                  >
                    {t("queue.cancel")}
                  </button>
                </div>
              </div>
            )}
            {jobs.length === 0 ? (
              <div className="flex h-32 items-center justify-center text-xs text-slate-500">
                {t("queue.empty")}
              </div>
            ) : (
              <div className="space-y-1.5">
                {jobs.map((j) => {
                  const meta = STATUS_ICON[j.status];
                  const pct =
                    j.status === "rendering" && j.progress && j.progress.totalFrames > 0
                      ? Math.round((j.progress.framesProcessed / j.progress.totalFrames) * 100)
                      : 0;
                  return (
                    <div key={j.input} className="rounded-lg border border-slate-200 p-2 dark:border-slate-800">
                      <div className="flex items-center justify-between space-x-2">
                        <p className="min-w-0 truncate text-[11px] font-medium text-slate-800 dark:text-slate-200">
                          {basename(j.input)}
                        </p>
                        <span title={t(meta.labelKey)} className={`shrink-0 ${meta.color}`}>
                          <meta.icon className="h-4 w-4" />
                        </span>
                      </div>
                      {j.status === "rendering" && (
                        <>
                          <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                            <div className="h-full bg-indigo-500 transition-all" style={{ width: `${pct}%` }} />
                          </div>
                          {j.progress && (
                            <p className="mt-0.5 font-mono text-[11px] text-slate-500">
                              {j.progress.framesProcessed} / {j.progress.totalFrames}
                            </p>
                          )}
                        </>
                      )}
                      {j.status === "done" && (
                        <p className="mt-0.5 truncate font-mono text-[11px] text-emerald-600 dark:text-emerald-400">
                          {basename(j.output)}
                        </p>
                      )}
                      {j.status === "failed" && j.error && (
                        <p className="mt-0.5 truncate text-[11px] text-rose-500" title={j.error}>
                          {j.error}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mt-3 border-t border-slate-200 pt-3 dark:border-slate-800/80">
        <button
          onClick={onStartRender}
          disabled={files.length === 0 || rendering}
          className="mb-3 flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40"
        >
          <Play className="h-4 w-4" /> {t("render.start")}
        </button>
        <label className="mb-1 block text-[11px] text-slate-500 dark:text-slate-400">{t("tab.output")}</label>
        <div className="flex items-center space-x-2">
          <div
            title={outputDir ?? undefined}
            className="flex-1 truncate rounded-md border border-slate-300 bg-white px-2 py-1.5 text-[11px] text-slate-700 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-300"
          >
            {outputDir ?? t("output.path")}
          </div>
          <button
            onClick={onPickOutputDir}
            className="rounded-md bg-slate-200 p-1.5 text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
            title={t("output.pick")}
            aria-label={t("output.pick")}
          >
            <FolderOpen className="h-4 w-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}
