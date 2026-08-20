import type { HardwareSnapshot, RenderProgress } from "@senmei/bridge";
import { useI18n } from "../i18n";
import { useFfmpeg } from "../useFfmpeg";

const fmtGb = (bytes: number | null | undefined) =>
  bytes == null ? "—" : `${(bytes / 1024 ** 3).toFixed(1)}G`;

export default function StatusBar({
  health,
  fileCount,
  progress,
  rendering,
  hardware,
  onSettings,
}: {
  health: string;
  fileCount: number;
  progress: RenderProgress | null;
  rendering: boolean;
  hardware: HardwareSnapshot | null;
  onSettings: () => void;
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
        <button
          onClick={onSettings}
          title={t("menu.settings")}
          className="rounded p-0.5 text-slate-500 transition hover:bg-slate-200 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeWidth="2"
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
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
      <div className="flex items-center space-x-3">
        {rendering && progress && (
          <div className="flex items-center gap-2">
            <span>{t("status.rendering")}</span>
            <span>{pct}%</span>
            <div className="h-1.5 w-24 rounded-full bg-slate-200 dark:bg-slate-800">
              <div className="h-full rounded-full bg-indigo-500" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}
        {hardware && (
          <div className="flex items-center space-x-3">
            {hardware.gpuName && (
              <span>
                {hardware.gpuName}{" "}
                {hardware.gpuUtilizationPercent != null
                  ? `${Math.round(hardware.gpuUtilizationPercent)}%`
                  : "—"}{" "}
                {hardware.gpuMemoryUsedBytes != null
                  ? fmtGb(hardware.gpuMemoryUsedBytes)
                  : "—"}
                /{fmtGb(hardware.gpuMemoryTotalBytes)}
              </span>
            )}
            <span>CPU {Math.round((hardware.cpuUsage ?? 0) * 100)}%</span>
            <span>
              RAM {fmtGb(hardware.memoryUsedBytes)}/{fmtGb(hardware.memoryTotalBytes)}
            </span>
          </div>
        )}
        <span
          title={`build ${__BUILD_HASH__}`}
          className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[9px] text-slate-500 dark:bg-slate-800 dark:text-slate-400"
        >
          v{__APP_VERSION__}-{__BUILD_HASH__}
        </span>
      </div>
    </footer>
  );
}
