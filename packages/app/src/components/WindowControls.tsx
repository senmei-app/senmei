import { useEffect, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

export default function WindowControls() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    if (!isTauri()) return;
    const win = getCurrentWindow();
    void win.isMaximized().then(setMaximized);
    // Maximize/restore resizes the window; sync the icon from the event.
    const unlisten = win.onResized(() => {
      void win.isMaximized().then(setMaximized);
    });
    return () => {
      void unlisten.then((fn) => fn());
    };
  }, []);

  const minimize = () => {
    if (isTauri()) void getCurrentWindow().minimize();
  };
  const toggleMax = () => {
    if (isTauri()) {
      void getCurrentWindow().toggleMaximize();
      setMaximized((m) => !m); // optimistic; onResized reconciles
    }
  };
  const close = () => {
    if (isTauri()) void getCurrentWindow().close();
  };

  return (
    <div className="flex items-center space-x-3 text-slate-500">
      <button
        onClick={minimize}
        title="Minimize"
        aria-label="Minimize"
        className="flex h-4 w-4 items-end justify-center hover:text-slate-900 dark:hover:text-slate-200 transition"
      >
        ─
      </button>
      <button
        onClick={toggleMax}
        title={maximized ? "Restore" : "Maximize"}
        aria-label={maximized ? "Restore" : "Maximize"}
        className="flex h-4 w-4 items-center justify-center hover:text-slate-900 dark:hover:text-slate-200 transition"
      >
        {maximized ? (
          <span className="relative block h-3.5 w-3.5">
            <span className="absolute left-0 top-0 h-2.5 w-2.5 border border-current" />
            <span className="absolute bottom-0 right-0 h-2.5 w-2.5 border border-current bg-white dark:bg-slate-900" />
          </span>
        ) : (
          <span className="block h-3 w-3 border border-current" />
        )}
      </button>
      <button
        onClick={close}
        title="Close"
        aria-label="Close"
        className="hover:text-rose-400 transition"
      >
        ✕
      </button>
    </div>
  );
}
