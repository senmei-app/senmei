import { useState } from "react";
import { useI18n } from "../i18n";
import type { UpdateInfo } from "../updater";
import { downloadAndRelaunch } from "../updater";

export default function UpdateDialog({
  update,
  onClose,
}: {
  update: UpdateInfo;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInstall = async () => {
    setInstalling(true);
    setError(null);
    try {
      await downloadAndRelaunch();
    } catch (e) {
      setError(String(e));
      setInstalling(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-80 rounded-xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-700 dark:bg-slate-900"
      >
        <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
          {t("update.title")}
        </h2>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          {t("update.available")} <span className="font-mono font-medium text-slate-800 dark:text-slate-200">v{update.version}</span>
        </p>

        {update.body && (
          <div className="mt-3 max-h-40 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-600 dark:border-slate-800 dark:bg-slate-950/50 dark:text-slate-400">
            {update.body}
          </div>
        )}

        {error && (
          <p className="mt-2 text-[11px] text-red-500">{error}</p>
        )}

        <div className="mt-4 flex justify-end space-x-2">
          <button
            onClick={onClose}
            disabled={installing}
            className="rounded-md bg-slate-200 px-3 py-1.5 text-[11px] font-medium text-slate-700 transition hover:bg-slate-300 disabled:opacity-50 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            {t("update.later")}
          </button>
          <button
            onClick={handleInstall}
            disabled={installing}
            className="rounded-md bg-indigo-600 px-3 py-1.5 text-[11px] font-medium text-white transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {installing ? t("update.installing") : t("update.install")}
          </button>
        </div>
      </div>
    </div>
  );
}
