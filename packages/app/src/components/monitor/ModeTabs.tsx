import { useI18n } from "../../i18n";

export type MonitorMode = "source" | "result" | "compare" | "ab";

/// Source / compare / A/B / result view switcher (overlay top-left).
export default function ModeTabs({
  mode,
  onMode,
  effRendered,
  prevRenderedFile,
  info,
  modeSourceHotkey,
  modeResultHotkey,
  modeCompareHotkey,
  modeABHotkey,
}: {
  mode: MonitorMode;
  onMode: (m: MonitorMode) => void;
  effRendered: string | null;
  prevRenderedFile?: string | null;
  info: { width?: number; height?: number } | null;
  modeSourceHotkey?: string;
  modeResultHotkey?: string;
  modeCompareHotkey?: string;
  modeABHotkey?: string;
}) {
  const { t } = useI18n();
  const tabCls = (active: boolean) =>
    active
      ? "rounded-md border border-indigo-500/40 bg-indigo-600/60 px-2 py-1 text-[11px] font-mono text-white backdrop-blur"
      : "rounded-md bg-black/60 px-2 py-1 text-[11px] font-mono text-slate-300 backdrop-blur hover:bg-black/70";

  return (
    <div className="absolute top-3 left-3 flex space-x-2">
      <button
        onClick={() => onMode("source")}
        title={`${t("monitor.original")} (${modeSourceHotkey})`}
        className={tabCls(mode === "source")}
      >
        {t("monitor.original")}
        {mode === "source" && info ? ` ${info.width}x${info.height}` : ""}
      </button>
      <button
        onClick={() => onMode("compare")}
        disabled={!effRendered}
        title={effRendered ? `${t("monitor.compare")} (${modeCompareHotkey})` : t("monitor.needRender")}
        className={tabCls(mode === "compare")}
      >
        {t("monitor.compare")}
      </button>
      <button
        onClick={() => onMode("ab")}
        disabled={!effRendered || !prevRenderedFile}
        title={effRendered && prevRenderedFile ? `${t("monitor.ab")} (${modeABHotkey})` : t("monitor.needSecondRender")}
        className={tabCls(mode === "ab")}
      >
        A/B
      </button>
      <button
        onClick={() => onMode("result")}
        disabled={!effRendered}
        title={effRendered ? `${t("monitor.result")} (${modeResultHotkey})` : t("monitor.needRender")}
        className={tabCls(mode === "result")}
      >
        {t("monitor.result")}
      </button>
    </div>
  );
}
