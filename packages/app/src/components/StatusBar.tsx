import type { RenderProgress } from "@senmei/bridge";
import { useI18n } from "../i18n";
import { useFfmpeg } from "../useFfmpeg";

export default function StatusBar({
  health,
  fileCount,
  progress,
  rendering,
}: {
  health: string;
  fileCount: number;
  progress: RenderProgress | null;
  rendering: boolean;
}) {
  const { t } = useI18n();
  const { status } = useFfmpeg();

  const pct =
    progress && progress.totalFrames > 0
      ? Math.round((progress.framesProcessed / progress.totalFrames) * 100)
      : 0;

  return (
    <footer className="flex h-7 items-center justify-between border-t border-slate-200 bg-white/80 px-3 text-[11px] text-slate-500 dark:border-slate-800/80 dark:bg-slate-950">
      <div className="flex items-center space-x-3">
        <span className={health === "ok" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-500"}>
          {health === "ok" ? t("status.ready") : health}
        </span>
        <span>•</span>
        <span>FFmpeg {status?.version ?? "—"}</span>
        <span>•</span>
        <span>
          {fileCount} {fileCount === 1 ? t("status.file") : t("status.files")}
        </span>
      </div>
      {rendering && progress && (
        <div className="flex items-center gap-2">
          <span>{t("status.rendering")}</span>
          <span>{pct}%</span>
          <div className="h-1.5 w-24 rounded-full bg-slate-200 dark:bg-slate-800">
            <div className="h-full rounded-full bg-indigo-500" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
    </footer>
  );
}
