import { useEffect, useState } from "react";
import {
  getPathDialog,
  subscribePathDialog,
  submitPath,
  type PathDialogOptions,
} from "../backend/pathDialog";
import { useI18n } from "../i18n";

/// Modal for entering server-side paths in the web backend. Rendered once at
/// the app root; `http.ts` opens it via `openPathDialog` (no native picker).
export default function PathDialog() {
  const { t } = useI18n();
  const [opts, setOpts] = useState<PathDialogOptions | null>(null);
  const [value, setValue] = useState("");

  useEffect(() => {
    const un = subscribePathDialog(() => {
      const o = getPathDialog();
      setOpts(o);
      setValue(o?.default ?? "");
    });
    return un;
  }, []);

  if (!opts) return null;

  const ok = () => submitPath(value.trim() || null);
  const cancel = () => submitPath(null);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={cancel}>
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-96 rounded-xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-700 dark:bg-slate-900"
      >
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{opts.title}</h2>
        {opts.multiple && (
          <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
            {t("pathDialog.multipleHint")}
          </p>
        )}
        <input
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ok()}
          placeholder={opts.placeholder ?? "/path/to/file"}
          className="mt-3 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
        />
        <div className="mt-4 flex justify-end space-x-2">
          <button
            onClick={cancel}
            className="rounded-md bg-slate-200 px-3 py-1.5 text-[11px] font-medium text-slate-700 transition hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            {t("about.close")}
          </button>
          <button
            onClick={ok}
            className="rounded-md bg-indigo-600 px-3 py-1.5 text-[11px] font-medium text-white transition hover:bg-indigo-500"
          >
            {t("pathDialog.open")}
          </button>
        </div>
      </div>
    </div>
  );
}
