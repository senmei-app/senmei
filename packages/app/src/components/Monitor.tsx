import { useI18n } from "../i18n";

export default function Monitor() {
  const { t } = useI18n();

  return (
    <main className="flex h-full flex-col bg-slate-100 p-4 dark:bg-slate-950">
      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-black shadow-2xl dark:border-slate-800">
        <div className="absolute inset-0 flex items-center justify-center bg-slate-200/80 dark:bg-slate-900/80">
          <span className="font-mono text-sm text-slate-500 dark:text-slate-600">{t("monitor.placeholder")}</span>
        </div>

        <div className="absolute inset-y-0 left-1/2 w-0.5 bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.8)]">
          <div className="absolute top-1/2 -left-3 -translate-y-1/2 flex h-6 w-6 items-center justify-center rounded-full bg-indigo-600 text-[10px] text-white shadow-md">
            ↔
          </div>
        </div>

        <div className="absolute top-3 left-3 flex space-x-2">
          <span className="rounded-md bg-black/60 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur">
            {t("monitor.original")}
          </span>
          <span className="rounded-md border border-indigo-500/40 bg-indigo-950/80 px-2 py-1 text-[10px] font-mono text-indigo-300 backdrop-blur">
            {t("monitor.senmei")}
          </span>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white/60 p-3 backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/40">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white hover:bg-indigo-500">
              ▶
            </button>
            <span className="font-mono text-xs text-slate-600 dark:text-slate-300">00:00:12.40 / 00:02:15.00</span>
          </div>
          <div className="flex items-center space-x-1">
            <button className="rounded-md bg-slate-200 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700">
              {t("timeline.sample10")}
            </button>
            <button className="rounded-md border border-indigo-500/40 bg-indigo-600/30 px-2 py-1 text-[11px] text-indigo-600 dark:text-indigo-300">
              {t("timeline.sample15")}
            </button>
            <button className="rounded-md bg-slate-200 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700">
              {t("timeline.sample30")}
            </button>
          </div>
        </div>

        <div className="relative h-3 w-full rounded-full bg-slate-200 dark:bg-slate-800">
          <div className="absolute left-1/4 right-1/2 h-full rounded-full border-x-2 border-indigo-400 bg-indigo-500/40">
            <div className="absolute -left-1.5 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full border-2 border-slate-100 bg-indigo-400 cursor-ew-resize dark:border-slate-950"></div>
            <div className="absolute -right-1.5 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full border-2 border-slate-100 bg-indigo-400 cursor-ew-resize dark:border-slate-950"></div>
          </div>
        </div>
        <div className="mt-1.5 flex justify-between font-mono text-[10px] text-slate-500">
          <span>{t("timeline.in")}: 00:00:12.40</span>
          <span>{t("timeline.out")}: 00:00:27.40</span>
        </div>
      </div>
    </main>
  );
}
