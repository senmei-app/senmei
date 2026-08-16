import { useI18n } from "../i18n";
import MenuBar from "./MenuBar";
import WindowControls from "./WindowControls";

export default function TopBar({
  file,
  rendering,
  onImportFile,
  onImportFolder,
  onStartRender,
  onCloseProject,
  onSettings,
  onGithub,
}: {
  file?: string;
  rendering?: boolean;
  onImportFile: () => void;
  onImportFolder: () => void;
  onStartRender: () => void;
  onCloseProject: () => void;
  onSettings: () => void;
  onGithub: () => void;
}) {
  const { lang, setLang, t } = useI18n();

  return (
    <header className="relative z-50 flex h-12 w-full items-center gap-4 border-b border-slate-200 bg-white/90 px-4 text-xs backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
      <div data-tauri-drag-region className="flex items-center space-x-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 font-bold text-sm text-white shadow-lg shadow-indigo-500/30">
          鮮
        </div>
        <span className="font-bold tracking-wide text-slate-900 dark:text-slate-100 text-sm">Senmei</span>
      </div>

      <MenuBar
        onImportFile={onImportFile}
        onImportFolder={onImportFolder}
        onCloseProject={onCloseProject}
        onSettings={onSettings}
        onGithub={onGithub}
      />

      <div data-tauri-drag-region className="flex-1 self-stretch" />

      <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-slate-100 px-3 py-1 font-mono text-[11px] text-slate-700 dark:border-slate-800/80 dark:bg-slate-950/80 dark:text-slate-300">
        <span className={"h-2 w-2 rounded-full " + (file ? "bg-emerald-400 animate-pulse" : "bg-slate-400")}></span>
        <span className="truncate max-w-[220px]">{file ? file.split("/").pop() : t("topbar.noFile")}</span>
      </div>

      <div className="flex items-center space-x-3">
        <button
          onClick={onStartRender}
          disabled={!file || rendering}
          className="flex items-center space-x-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-3.5 py-1.5 font-medium text-white shadow-md shadow-indigo-600/30 transition active:scale-95 disabled:opacity-40"
        >
          <span>{rendering ? "…" : "▶"}</span>
          <span>{rendering ? t("topbar.rendering") : t("render.start")}</span>
        </button>

        <div className="flex items-center space-x-1 text-[10px] font-medium">
          <button
            onClick={() => setLang("en")}
            className={lang === "en" ? "text-indigo-600 dark:text-indigo-300" : "text-slate-500 hover:text-slate-900 dark:text-slate-500 dark:hover:text-slate-300 transition"}
          >
            EN
          </button>
          <span className="text-slate-400 dark:text-slate-600">/</span>
          <button
            onClick={() => setLang("de")}
            className={lang === "de" ? "text-indigo-600 dark:text-indigo-300" : "text-slate-500 hover:text-slate-900 dark:text-slate-500 dark:hover:text-slate-300 transition"}
          >
            DE
          </button>
        </div>

        <div className="border-l border-slate-200 pl-3 dark:border-slate-800">
          <WindowControls />
        </div>
      </div>
    </header>
  );
}
