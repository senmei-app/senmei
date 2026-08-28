import type { StepTimingInfo } from "@senmei/bridge";

import { useI18n } from "../../i18n";

/// Per-step FPS benchmark after a finished render.
export default function Benchmark({ timings }: { timings: StepTimingInfo[] }) {
  const { t } = useI18n();
  if (timings.length === 0) return null;

  return (
    <div className="mt-2 border-t border-slate-200 pt-2 dark:border-slate-800">
      <div className="mb-1 font-mono text-[11px] uppercase tracking-wider text-slate-400 dark:text-slate-500">
        {t("monitor.benchmark")}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-3">
        {timings.map((s) => (
          <div key={s.name} className="flex items-baseline justify-between gap-2 font-mono text-[11px]">
            <span className="truncate text-slate-600 dark:text-slate-300">{s.name}</span>
            <span className="shrink-0 text-slate-400 dark:text-slate-500">
              {s.msPerFrame != null ? `${s.msPerFrame.toFixed(1)} ms/f` : "–"} ·{" "}
              {s.fps != null ? `${s.fps.toFixed(1)} fps` : "–"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
