import { useI18n } from "../i18n";
import { useFfmpeg } from "../useFfmpeg";

export default function FfmpegIndicator() {
  const { t } = useI18n();
  const { status, downloading, pct, error, download } = useFfmpeg();

  if (!status || status.found) return null;

  return (
    <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2">
      <div className="flex items-center gap-3 rounded-xl border border-amber-500/30 bg-white/95 px-4 py-2.5 text-xs shadow-xl backdrop-blur dark:bg-slate-900/95">
        <span className="h-2 w-2 shrink-0 rounded-full bg-amber-400" />
        <span className="text-amber-600 dark:text-amber-400">{t("settings.ffmpeg.notFound")}</span>
        <button
          onClick={download}
          disabled={downloading}
          className="rounded-md bg-amber-500/15 px-2 py-1 font-medium text-amber-700 hover:bg-amber-500/25 disabled:opacity-50 dark:text-amber-300"
        >
          {downloading ? `${pct}%` : t("settings.ffmpeg.download")}
        </button>
        {error && <span className="max-w-[200px] truncate text-[10px] text-rose-500">{error}</span>}
      </div>
    </div>
  );
}
