import { useState } from "react";
import type { RenderProgress } from "@senmei/bridge";
import { useI18n } from "../i18n";

export default function MediaLibrary({
  files,
  onOpen,
  onRemoveFile,
  outputDir,
  onPickOutputDir,
  rendering,
  onCancel,
  progress,
  renderedFile,
}: {
  files: string[];
  onOpen: () => void;
  onRemoveFile: (path: string) => void;
  outputDir: string | null;
  onPickOutputDir: () => void;
  rendering: boolean;
  onCancel: () => void;
  progress: RenderProgress | null;
  renderedFile: string | null;
}) {
  const { t } = useI18n();
  const [view, setView] = useState<"library" | "queue">("library");

  const pct =
    rendering && progress && progress.totalFrames > 0
      ? Math.round((progress.framesProcessed / progress.totalFrames) * 100)
      : null;

  return (
    <aside className="flex h-full flex-col border-r border-slate-200 bg-slate-100/70 p-3 dark:border-slate-800/80 dark:bg-slate-900/30">
      <div className="mb-3 flex items-center justify-between px-1">
        <div className="flex gap-1">
          <button
            onClick={() => setView("library")}
            className={
              view === "library"
                ? "rounded-md border border-indigo-500/40 bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-600 dark:text-indigo-300"
                : "rounded-md bg-slate-200 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
            }
          >
            {t("media.tab.library")}
          </button>
          <button
            onClick={() => setView("queue")}
            className={
              view === "queue"
                ? "rounded-md border border-indigo-500/40 bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-600 dark:text-indigo-300"
                : "rounded-md bg-slate-200 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
            }
          >
            {t("media.tab.queue")}
          </button>
        </div>
        <button className="rounded-md bg-slate-200 p-1 text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700">+</button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {view === "library" ? (
          <>
            <div
              onClick={onOpen}
              className="mb-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-200/50 p-4 text-center transition hover:border-indigo-500/50 hover:bg-slate-200 dark:border-slate-700/80 dark:bg-slate-900/40 dark:hover:bg-slate-900/80"
            >
              <div className="mb-2 rounded-full bg-indigo-500/10 p-2 text-indigo-500 dark:text-indigo-400">
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <p className="text-xs font-medium text-slate-700 dark:text-slate-300">{t("media.drop")}</p>
              <p className="mt-1 text-[10px] text-slate-500">{t("media.formats")}</p>
            </div>

            <div className="space-y-2">
              {files.map((path) => (
                <div
                  key={path}
                  className="group flex cursor-pointer items-center space-x-3 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-2 transition"
                >
                  <div className="relative h-10 w-14 shrink-0 overflow-hidden rounded bg-slate-300 dark:bg-slate-800">
                    <div className="absolute inset-0 flex items-center justify-center bg-slate-200/60 text-[9px] text-slate-500 dark:bg-slate-700/50 dark:text-slate-300">
                      THUMB
                    </div>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-slate-900 dark:text-slate-100">
                      {path.split("/").pop()}
                    </p>
                    <div className="mt-0.5 text-[10px] text-slate-500 dark:text-slate-400">video</div>
                  </div>
                  <button
                    onClick={() => onRemoveFile(path)}
                    title={t("media.remove")}
                    className="rounded-md p-1 text-slate-300 transition hover:bg-red-500/10 hover:text-red-500 dark:text-slate-500 dark:hover:text-red-400"
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeWidth="2" d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="space-y-2 p-1">
            {rendering && (
              <div className="rounded-lg border border-indigo-500/40 bg-indigo-500/10 p-2">
                <div className="flex items-center justify-between">
                  <p className="text-[11px] font-medium text-indigo-600 dark:text-indigo-300">
                    {t("queue.rendering")}
                  </p>
                  <button
                    onClick={onCancel}
                    className="rounded-md border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-[10px] font-medium text-red-500 hover:bg-red-500/20 dark:text-red-400"
                  >
                    {t("queue.cancel")}
                  </button>
                </div>
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <div
                    className="h-full bg-indigo-500 transition-all"
                    style={{ width: `${pct ?? 0}%` }}
                  />
                </div>
                {progress && (
                  <p className="mt-1 font-mono text-[10px] text-slate-500">
                    {progress.framesProcessed} / {progress.totalFrames}
                  </p>
                )}
              </div>
            )}
            {renderedFile && (
              <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-2">
                <p className="text-[11px] font-medium text-emerald-600 dark:text-emerald-300">
                  {t("queue.done")}
                </p>
                <p className="mt-0.5 truncate font-mono text-[10px] text-slate-500">
                  {renderedFile.split("/").pop()}
                </p>
              </div>
            )}
            {!rendering && !renderedFile && (
              <div className="flex h-32 items-center justify-center text-xs text-slate-500">
                {t("queue.empty")}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mt-3 border-t border-slate-200 pt-3 dark:border-slate-800/80">
        <label className="mb-1 block text-[10px] text-slate-500 dark:text-slate-400">{t("tab.output")}</label>
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
          >
            📁
          </button>
        </div>
      </div>
    </aside>
  );
}
