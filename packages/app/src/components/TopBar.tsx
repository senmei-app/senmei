import { useI18n } from "../i18n";
import MenuBar from "./MenuBar";
import WindowControls from "./WindowControls";

export default function TopBar({
  file,
  projectName,
  rendering,
  onImportFile,
  onImportFolder,
  onStartRender,
  onCancelRender,
  onCloseProject,
  onSettings,
  onGithub,
}: {
  file?: string;
  projectName?: string;
  rendering?: boolean;
  onImportFile: () => void;
  onImportFolder: () => void;
  onStartRender: () => void;
  onCancelRender: () => void;
  onCloseProject: () => void;
  onSettings: () => void;
  onGithub: () => void;
}) {
  const { t } = useI18n();

  return (
    <header className="relative z-50 flex h-12 w-full items-center gap-4 border-b border-slate-200 bg-white/90 px-4 text-xs backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
      <div data-tauri-drag-region className="flex items-center space-x-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 font-bold text-sm text-white shadow-lg shadow-indigo-500/30">
          鮮
        </div>
        <span className="font-bold tracking-wide text-slate-900 dark:text-slate-100 text-sm">Senmei</span>
        <span className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[9px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">
          v0.1.0
        </span>
      </div>

      <MenuBar
        onImportFile={onImportFile}
        onImportFolder={onImportFolder}
        onCloseProject={onCloseProject}
        onSettings={onSettings}
        onGithub={onGithub}
      />

      <div data-tauri-drag-region className="flex-1 self-stretch" />

      {projectName && (
        <div
          data-tauri-drag-region
          className="pointer-events-none absolute left-1/2 -translate-x-1/2 max-w-[320px] truncate text-sm font-medium text-slate-600 dark:text-slate-300"
        >
          {projectName}
        </div>
      )}

      <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-slate-100 px-3 py-1 font-mono text-[11px] text-slate-700 dark:border-slate-800/80 dark:bg-slate-950/80 dark:text-slate-300">
        <span className={"h-2 w-2 rounded-full " + (file ? "bg-emerald-400 animate-pulse" : "bg-slate-400")}></span>
        <span className="truncate max-w-[220px]">{file ? file.split("/").pop() : t("topbar.noFile")}</span>
      </div>

      <div className="flex items-center space-x-2">
        <button
          onClick={onStartRender}
          disabled={!file || rendering}
          className="flex items-center space-x-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-3.5 py-1.5 font-medium text-white shadow-md shadow-indigo-600/30 transition active:scale-95 disabled:opacity-40"
        >
          <span>▶</span>
          <span>{t("render.start")}</span>
        </button>
        <button
          onClick={onCancelRender}
          disabled={!rendering}
          title={t("queue.cancel")}
          className="flex items-center space-x-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3.5 py-1.5 font-medium text-red-500 transition hover:bg-red-500/20 active:scale-95 disabled:opacity-30 dark:text-red-400"
        >
          <span>■</span>
          <span>{t("queue.cancel")}</span>
        </button>
        <div className="border-l border-slate-200 pl-3 dark:border-slate-800">
          <WindowControls />
        </div>
      </div>
    </header>
  );
}
