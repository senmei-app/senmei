import { useState } from "react";
import { useI18n } from "../i18n";
import type { PipelineStep } from "../steps";
import Inspector from "./Inspector";
import LogsPanel from "./LogsPanel";

// Right-hand panel with a tab bar switching between the processing stack and
// the live system log.
export default function RightPanel({
  steps,
  outputDir,
  onChange,
  onSuggest,
}: {
  steps: PipelineStep[];
  outputDir?: string | null;
  onChange: (steps: PipelineStep[]) => void;
  onSuggest?: () => void;
}) {
  const { t } = useI18n();
  const [tab, setTab] = useState<"stack" | "logs">("stack");
  const tabCls = (active: boolean) =>
    active
      ? "rounded-md bg-indigo-600 px-3 py-1 text-[11px] font-medium text-white"
      : "rounded-md px-3 py-1 text-[11px] text-slate-600 hover:bg-slate-200 dark:text-slate-400 dark:hover:bg-slate-800";

  return (
    <div className="flex h-full flex-col">
      <div className="flex gap-1 border-b border-slate-200 p-2 dark:border-slate-800/80">
        <button onClick={() => setTab("stack")} className={tabCls(tab === "stack")}>
          {t("stack.title")}
        </button>
        <button onClick={() => setTab("logs")} className={tabCls(tab === "logs")}>
          {t("logs.title")}
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        {tab === "stack" ? (
          <Inspector steps={steps} outputDir={outputDir} onChange={onChange} onSuggest={onSuggest} />
        ) : (
          <LogsPanel />
        )}
      </div>
    </div>
  );
}
