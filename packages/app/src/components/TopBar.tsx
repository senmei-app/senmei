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
  onCloseProject,
  onExportProject,
  onSettings,
  onGithub,
  onAbout,
  onSelectAll,
  onDeleteSelected,
  onAddAllToQueue,
  onAddSelectedToQueue,
  onProcessSelected,
  onProcessAll,
  onToggleFullscreen,
}: {
  file?: string;
  projectName?: string;
  rendering?: boolean;
  onImportFile: () => void;
  onImportFolder: () => void;
  onStartRender: () => void;
  onCloseProject: () => void;
  onExportProject: () => void;
  onSettings: () => void;
  onGithub: () => void;
  onAbout: () => void;
  onSelectAll: () => void;
  onDeleteSelected: () => void;
  onAddAllToQueue: () => void;
  onAddSelectedToQueue: () => void;
  onProcessSelected: () => void;
  onProcessAll: () => void;
  onToggleFullscreen: () => void;
}) {
  const { t } = useI18n();

  return (
    <header className="relative z-50 flex h-12 w-full items-center gap-4 border-b border-slate-200 bg-white/90 px-4 text-xs backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
      <div data-tauri-drag-region className="flex items-center space-x-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 font-bold text-sm text-white shadow-lg shadow-indigo-500/30">
          鮮
        </div>
      </div>

      <MenuBar
        onImportFile={onImportFile}
        onImportFolder={onImportFolder}
        onCloseProject={onCloseProject}
        onExportProject={onExportProject}
        onSettings={onSettings}
        onGithub={onGithub}
        onAbout={onAbout}
        onSelectAll={onSelectAll}
        onDeleteSelected={onDeleteSelected}
        onAddAllToQueue={onAddAllToQueue}
        onAddSelectedToQueue={onAddSelectedToQueue}
        onProcessSelected={onProcessSelected}
        onProcessAll={onProcessAll}
        onToggleFullscreen={onToggleFullscreen}
      />

      <div data-tauri-drag-region className="flex-1 self-stretch" />

      <div
        data-tauri-drag-region
        className="pointer-events-none absolute left-1/2 -translate-x-1/2 max-w-[400px] truncate text-sm font-medium text-slate-600 dark:text-slate-300"
        title={file ?? undefined}
      >
        {projectName ? `${projectName} / ${file ? file.split("/").pop() : t("topbar.noFile")}` : (file ? file.split("/").pop() : "")}
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
          onClick={onSettings}
          title={t("menu.settings")}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition hover:bg-slate-200 hover:text-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeWidth="2"
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
        <div className="border-l border-slate-200 pl-3 dark:border-slate-800">
          <WindowControls />
        </div>
      </div>
    </header>
  );
}
