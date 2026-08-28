import { useEffect, useState } from "react";
import { Minus, Square, X } from "lucide-react";
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
        className="flex h-4 w-4 items-center justify-center hover:text-slate-900 dark:hover:text-slate-200 transition"
      >
        <Minus className="h-4 w-4" />
      </button>
      <button
        onClick={toggleMax}
        title={maximized ? "Restore" : "Maximize"}
        aria-label={maximized ? "Restore" : "Maximize"}
        className="flex h-4 w-4 items-center justify-center hover:text-slate-900 dark:hover:text-slate-200 transition"
      >
        <Square className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={close}
        title="Close"
        aria-label="Close"
        className="hover:text-rose-400 transition"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
