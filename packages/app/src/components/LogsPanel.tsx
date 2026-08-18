import { useEffect, useRef, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getLogs, type LogEntry } from "@senmei/bridge";
import { useI18n } from "../i18n";

const LEVEL_SEV: Record<string, number> = { ERROR: 4, WARN: 3, INFO: 2, DEBUG: 1, TRACE: 0 };
const LEVEL_CLS: Record<string, string> = {
  ERROR: "text-rose-500",
  WARN: "text-amber-500",
  INFO: "text-sky-500",
  DEBUG: "text-slate-400",
  TRACE: "text-slate-500",
};
const MAX_ENTRIES = 2000;

function fmtTime(ts: number): string {
  const d = new Date(ts);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export default function LogsPanel() {
  const { t } = useI18n();
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [minLevel, setMinLevel] = useState<string>("INFO");
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isTauri()) {
      setEntries([{ level: "INFO", message: t("logs.demoHint"), timestamp: Date.now() }]);
      return;
    }
    let un: UnlistenFn | undefined;
    getLogs()
      .then((existing) => setEntries(existing))
      .catch(() => {});
    listen<LogEntry>("log", (e) => {
      setEntries((prev) => (prev.length >= MAX_ENTRIES ? [...prev.slice(-MAX_ENTRIES), e.payload] : [...prev, e.payload]));
    })
      .then((fn) => (un = fn))
      .catch(() => {});
    return () => {
      un?.();
    };
  }, [t]);

  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [entries]);

  const minSev = LEVEL_SEV[minLevel] ?? 0;
  const shown = entries.filter((e) => (LEVEL_SEV[e.level] ?? 0) >= minSev);
  const levels = ["ALL", "ERROR", "WARN", "INFO"] as const;
  const chipCls = (active: boolean) =>
    active
      ? "rounded-md bg-indigo-600 px-2 py-0.5 text-[10px] font-medium text-white"
      : "rounded-md bg-slate-200 px-2 py-0.5 text-[10px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700";

  return (
    <div className="flex h-full w-full flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-4 py-2 dark:border-slate-800/80">
        <div className="flex gap-1">
          {levels.map((l) => (
            <button key={l} onClick={() => setMinLevel(l)} className={chipCls(minLevel === l)}>
              {l}
            </button>
          ))}
        </div>
        <button
          onClick={() => setEntries([])}
          className="rounded-md px-2 py-0.5 text-[10px] text-slate-500 transition hover:bg-slate-200 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-300"
        >
          {t("logs.clear")}
        </button>
      </div>
      <div ref={boxRef} className="min-h-0 flex-1 select-text overflow-y-auto px-3 py-2 font-mono text-[10px] leading-relaxed">
        {shown.length === 0 && <p className="text-slate-400">{t("logs.empty")}</p>}
        {shown.map((e, i) => (
          <div key={i} className="flex gap-2">
            <span className="shrink-0 text-slate-400">{fmtTime(e.timestamp)}</span>
            <span className={`w-12 shrink-0 ${LEVEL_CLS[e.level] ?? ""}`}>{e.level}</span>
            <span className="break-words whitespace-pre-wrap text-slate-700 dark:text-slate-300">{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
