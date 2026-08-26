import { useI18n } from "../../i18n";
import { basename } from "../../paths";
import type { RawFrame } from "../../backend/types";
import FrameCanvas from "./FrameCanvas";

/// Side-by-side A/B or original↔result comparison. Renders nothing when the
/// mode doesn't apply (source/result single-view handled by Monitor).
export default function CompareView({
  mode,
  file,
  effRendered,
  prevRenderedFile,
  frames,
}: {
  mode: string;
  file?: string;
  effRendered: string | null;
  prevRenderedFile?: string | null;
  frames: Record<string, RawFrame>;
}) {
  const { t } = useI18n();

  if (mode === "ab" && prevRenderedFile && effRendered) {
    return (
      <div className="flex h-full w-full">
        <div className="relative flex-1 overflow-hidden border-r border-slate-700/50">
          {frames[prevRenderedFile] ? (
            <div className="flex h-full w-full items-center justify-center">
              <FrameCanvas frame={frames[prevRenderedFile]} className="opacity-80" />
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
              <span className="truncate px-4 font-mono text-sm text-slate-500">
                {basename(prevRenderedFile)}
              </span>
            </div>
          )}
          <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-sky-300">
            A
          </span>
        </div>
        <div className="relative flex-1 overflow-hidden">
          {frames[effRendered] ? (
            <div className="flex h-full w-full items-center justify-center">
              <FrameCanvas frame={frames[effRendered]} className="opacity-80" />
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
              <span className="truncate px-4 font-mono text-sm text-slate-500">
                {basename(effRendered)}
              </span>
            </div>
          )}
          <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
            B
          </span>
        </div>
      </div>
    );
  }

  if (mode === "compare" && file && effRendered) {
    return (
      <div className="flex h-full w-full">
        <div className="relative flex-1 overflow-hidden border-r border-slate-700/50">
          {frames[file] ? (
            <div className="flex h-full w-full items-center justify-center">
              <FrameCanvas frame={frames[file]} className="opacity-80" />
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
              <span className="truncate px-4 font-mono text-sm text-slate-500">
                {basename(file)}
              </span>
            </div>
          )}
          <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
            {t("monitor.original")}
          </span>
        </div>
        <div className="relative flex-1 overflow-hidden">
          {frames[effRendered] ? (
            <div className="flex h-full w-full items-center justify-center">
              <FrameCanvas frame={frames[effRendered]} className="opacity-80" />
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-200/70 dark:bg-slate-900/70 grayscale">
              <span className="truncate px-4 font-mono text-sm text-slate-500">
                {basename(effRendered)}
              </span>
            </div>
          )}
          <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
            {t("monitor.result")}
          </span>
        </div>
      </div>
    );
  }

  return null;
}
