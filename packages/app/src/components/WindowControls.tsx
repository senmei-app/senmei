import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

export default function WindowControls() {
  const minimize = () => {
    if (isTauri()) void getCurrentWindow().minimize();
  };
  const toggleMax = () => {
    if (isTauri()) void getCurrentWindow().toggleMaximize();
  };
  const close = () => {
    if (isTauri()) void getCurrentWindow().close();
  };

  return (
    <div className="flex items-center space-x-3 text-slate-500">
      <button onClick={minimize} className="hover:text-slate-900 dark:hover:text-slate-200 transition">
        ─
      </button>
      <button onClick={toggleMax} className="hover:text-slate-900 dark:hover:text-slate-200 transition">
        ▢
      </button>
      <button onClick={close} className="hover:text-rose-400 transition">
        ✕
      </button>
    </div>
  );
}
