import { useI18n } from "../i18n";

export default function AboutDialog({
  onClose,
  onGithub,
}: {
  onClose: () => void;
  onGithub: () => void;
}) {
  const { t } = useI18n();

  const rows: [string, string][] = [
    [t("about.version"), `v${__APP_VERSION__}-${__BUILD_HASH__}`],
    [t("about.engine"), t("about.engineValue")],
    [t("about.license"), "MIT OR Apache-2.0"],
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-80 rounded-xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-700 dark:bg-slate-900"
      >
        <div className="flex flex-col items-center text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-600 text-2xl font-bold text-white shadow-lg shadow-indigo-500/30">
            鮮
          </div>
          <h2 className="mt-3 text-base font-bold text-slate-900 dark:text-slate-100">Senmei</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">{t("project.subtitle")}</p>
        </div>

        <div className="mt-4 space-y-1.5 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950/50">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-[11px]">
              <span className="text-slate-500 dark:text-slate-400">{k}</span>
              <span className="font-mono text-slate-800 dark:text-slate-200">{v}</span>
            </div>
          ))}
        </div>

        <p className="mt-3 text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
          {t("about.description")}
        </p>

        <div className="mt-4 flex justify-end space-x-2">
          <button
            onClick={onGithub}
            className="rounded-md bg-slate-200 px-3 py-1.5 text-[11px] font-medium text-slate-700 transition hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            {t("menu.github")}
          </button>
          <button
            onClick={onClose}
            className="rounded-md bg-indigo-600 px-3 py-1.5 text-[11px] font-medium text-white transition hover:bg-indigo-500"
          >
            {t("about.close")}
          </button>
        </div>
      </div>
    </div>
  );
}
