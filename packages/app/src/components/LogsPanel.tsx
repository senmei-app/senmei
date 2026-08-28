import { useEffect, useRef, useState } from "react";
import type { LogEntry } from "@senmei/bridge";
import { backend } from "../backend";
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
  const [q, setQ] = useState("");
  const boxRef = useRef<HTMLDivElement | null>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    let un: (() => void) | undefined;
    backend()
      .then((b) => {
        b.getLogs()
          .then((existing) => setEntries(existing))
          .catch(() => {});
        un = b.onLog((e) => {
          setEntries((prev) =>
            prev.length >= MAX_ENTRIES ? [...prev.slice(-MAX_ENTRIES), e] : [...prev, e],
          );
        });
      })
      .catch(() => {});
    return () => {
      un?.();
    };
  }, [t]);

  // Follow new entries only while the user is pinned to the bottom; once they
  // scroll up, leave the viewport alone until they scroll back down.
  const onScroll = () => {
    const box = boxRef.current;
    if (!box) return;
    stickToBottom.current = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  };
  useEffect(() => {
    if (stickToBottom.current && boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [entries]);

  const minSev = LEVEL_SEV[minLevel] ?? 0;
  const shown = entries.filter(
    (e) => (LEVEL_SEV[e.level] ?? 0) >= minSev && e.message.toLowerCase().includes(q.trim().toLowerCase()),
  );
  const [copied, setCopied] = useState(false);
  const dragStart = useRef<{ x: number; y: number } | null>(null);

  // Copy all shown entries (HH:MM:SS LEVEL message) as one text blob.
  const copyAll = () => {
    const text = shown.map((e) => `${fmtTime(e.timestamp)} ${e.level} ${e.message}`).join("\n");
    if (!text) return;
    const done = () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    };
    const fallback = () => {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
      } catch {
        /* noop */
      }
      document.body.removeChild(ta);
      done();
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(fallback);
    } else {
      fallback();
    }
  };

  // Select the whole log body on a plain click (drag still selects a range).
  const selectAll = () => {
    const box = boxRef.current;
    if (!box) return;
    const sel = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(box);
    sel?.removeAllRanges();
    sel?.addRange(range);
  };

  const levels = ["ALL", "ERROR", "WARN", "INFO"] as const;
  const chipCls = (active: boolean) =>
    active
      ? "rounded-md bg-indigo-600 px-2 py-0.5 text-[11px] font-medium text-white"
      : "rounded-md bg-slate-200 px-2 py-0.5 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700";

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
        <div className="flex items-center gap-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={t("logs.search")}
            className="w-40 rounded-md border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-600 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
          />
          <button
            onClick={copyAll}
            className="rounded-md px-2 py-0.5 text-[11px] text-slate-500 transition hover:bg-slate-200 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-300"
          >
            {copied ? t("logs.copied") : t("logs.copy")}
          </button>
          <button
            onClick={() => {
              setEntries([]);
              // Also empty the backend buffer so re-mounts don't reload old logs.
              backend().then((b) => b.clearLogs().catch(() => {}));
            }}
            className="rounded-md px-2 py-0.5 text-[11px] text-slate-500 transition hover:bg-slate-200 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-300"
          >
            {t("logs.clear")}
          </button>
        </div>
      </div>
      <div
        ref={boxRef}
        onScroll={onScroll}
        onMouseDown={(e) => (dragStart.current = { x: e.clientX, y: e.clientY })}
        onMouseUp={(e) => {
          const d = dragStart.current;
          dragStart.current = null;
          if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) <= 5) selectAll();
        }}
        className="min-h-0 flex-1 select-text overflow-y-auto px-3 py-2 font-mono text-[11px] leading-relaxed"
      >
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
