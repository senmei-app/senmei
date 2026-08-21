import { useI18n } from "../../i18n";

export type MonitorMode = "source" | "result" | "compare" | "ab";

/// Source / compare / A/B / result view switcher (overlay top-left).
export default function ModeTabs({
  mode,
  onMode,
  effRendered,
  prevRenderedFile,
  info,
}: {
  mode: MonitorMode;
  onMode: (m: MonitorMode) => void;
  effRendered: string | null;
  prevRenderedFile?: string | null;
  info: { width?: number; height?: number } | null;
}) {
  const { t } = useI18n();
  const tabCls = (active: boolean) =>
    active
      ? "rounded-md border border-indigo-500/40 bg-indigo-600/40 px-2 py-1 text-[10px] font-mono text-indigo-200 backdrop-blur"
      : "rounded-md bg-black/50 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur hover:bg-black/60";

  return (
    <div className="absolute top-3 left-3 flex space-x-2">
      <button onClick={() => onMode("source")} className={tabCls(mode === "source")}>
        {t("monitor.original")}
        {mode === "source" && info ? ` ${info.width}x${info.height}` : ""}
      </button>
      <button
        onClick={() => onMode("compare")}
        disabled={!effRendered}
        className={tabCls(mode === "compare")}
      >
        {t("monitor.compare")}
      </button>
      <button
        onClick={() => onMode("ab")}
        disabled={!effRendered || !prevRenderedFile}
        className={tabCls(mode === "ab")}
        title={t("monitor.ab")}
      >
        A/B
      </button>
      <button
        onClick={() => onMode("result")}
        disabled={!effRendered}
        className={tabCls(mode === "result")}
      >
        {t("monitor.result")}
      </button>
    </div>
  );
}
