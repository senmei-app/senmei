import { useState } from "react";
import { useI18n } from "../i18n";

export default function MediaLibrary() {
  const { t } = useI18n();
  const [view, setView] = useState<"library" | "queue">("library");

  return (
    <aside className="flex h-full flex-col border-r border-slate-800/80 bg-slate-900/30 p-3">
      <div className="mb-3 flex items-center justify-between px-1">
        <div className="flex gap-1">
          <button
            onClick={() => setView("library")}
            className={
              view === "library"
                ? "rounded-md bg-indigo-600/30 border border-indigo-500/40 px-2 py-1 text-[11px] text-indigo-300"
                : "rounded-md bg-slate-800 px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-700"
            }
          >
            {t("media.tab.library")}
          </button>
          <button
            onClick={() => setView("queue")}
            className={
              view === "queue"
                ? "rounded-md bg-indigo-600/30 border border-indigo-500/40 px-2 py-1 text-[11px] text-indigo-300"
                : "rounded-md bg-slate-800 px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-700"
            }
          >
            {t("media.tab.queue")}
          </button>
        </div>
        <button className="rounded-md bg-slate-800 p-1 text-slate-300 hover:bg-slate-700">+</button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {view === "library" ? (
          <>
            <div className="mb-4 flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-700/80 bg-slate-900/40 p-4 text-center transition hover:border-indigo-500/50 hover:bg-slate-900/80 cursor-pointer">
              <div className="mb-2 rounded-full bg-indigo-500/10 p-2 text-indigo-400">
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <p className="text-xs font-medium text-slate-300">{t("media.drop")}</p>
              <p className="mt-1 text-[10px] text-slate-500">{t("media.formats")}</p>
            </div>

            <div className="space-y-2">
              <div className="group flex cursor-pointer items-center space-x-3 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-2 transition">
                <div className="h-10 w-14 shrink-0 overflow-hidden rounded bg-slate-800 relative">
                  <div className="absolute inset-0 bg-slate-700/50 flex items-center justify-center text-[9px] text-slate-300">
                    THUMB
                  </div>
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium text-slate-100">jujutsu_kaisen_op.mkv</p>
                  <div className="mt-0.5 flex items-center space-x-1.5 text-[10px] text-slate-400">
                    <span className="rounded bg-slate-800 px-1">1080p</span>
                    <span>•</span>
                    <span>24 fps</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-500">
            {t("queue.empty")}
          </div>
        )}
      </div>

      <div className="mt-3 border-t border-slate-800/80 pt-3">
        <label className="block text-[10px] text-slate-400 mb-1">{t("tab.output")}</label>
        <div className="flex items-center space-x-2">
          <div className="flex-1 truncate rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-[11px] text-slate-300">
            {t("output.path")}
          </div>
          <button className="rounded-md bg-slate-800 p-1.5 text-slate-300 hover:bg-slate-700">📁</button>
        </div>
      </div>
    </aside>
  );
}
