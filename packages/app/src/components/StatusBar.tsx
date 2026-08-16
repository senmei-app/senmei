import { useI18n } from "../i18n";

export default function StatusBar({ health }: { health: string }) {
  const { t } = useI18n();

  return (
    <footer className="flex h-7 items-center justify-between border-t border-slate-200 bg-white/80 px-3 text-[11px] text-slate-500 dark:border-slate-800/80 dark:bg-slate-950">
      <div className="flex items-center space-x-3">
        <span className="flex items-center space-x-1.5 text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>{t("status.cuda")}</span>
        </span>
        <span>•</span>
        <span>{t("status.backend")}</span>
      </div>
      <span>{health === "ok" ? t("status.ready") : `health: ${health}`}</span>
    </footer>
  );
}
