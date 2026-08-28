import { useI18n } from "../i18n";
import MenuBar from "./MenuBar";
import WindowControls from "./WindowControls";
import { basename } from "../paths";

export default function TopBar({
  file,
  projectName,
  hotkeys,
  onImportFile,
  onImportFolder,
  onBatchFolder,
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
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  hasFiles,
  hasSelection,
}: {
  file?: string;
  projectName?: string;
  hotkeys: Record<string, string>;
  onImportFile: () => void;
  onImportFolder: () => void;
  onBatchFolder: () => void;
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
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  hasFiles: boolean;
  hasSelection: boolean;
}) {
  const { t } = useI18n();

  return (
    <header className="relative z-50 flex h-10 w-full items-center gap-4 border-b border-slate-200 bg-white/90 px-4 text-xs backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
      <div data-tauri-drag-region className="flex items-center space-x-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 font-bold text-sm text-white">
          鮮
        </div>
      </div>

      <MenuBar
        hotkeys={hotkeys}
        onImportFile={onImportFile}
        onImportFolder={onImportFolder}
        onBatchFolder={onBatchFolder}
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
        onUndo={onUndo}
        onRedo={onRedo}
        canUndo={canUndo}
        canRedo={canRedo}
        hasFiles={hasFiles}
        hasSelection={hasSelection}
      />

      <div data-tauri-drag-region className="flex-1 self-stretch" />

      <div
        data-tauri-drag-region
        className="pointer-events-none absolute left-1/2 -translate-x-1/2 max-w-[400px] truncate text-sm font-medium text-slate-600 dark:text-slate-300"
        title={file ?? undefined}
      >
        {projectName ? `${projectName} / ${file ? basename(file) : t("topbar.noFile")}` : (file ? basename(file) : "")}
      </div>

      <WindowControls />
    </header>
  );
}
