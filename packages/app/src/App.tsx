import { useEffect, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { isTauri, Channel } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  createProject,
  getSettings,
  healthCheck,
  importFolder,
  listProjects,
  render,
  saveSettings,
  type ProjectEntry,
  type RenderProgress,
} from "@senmei/bridge";
import { I18nProvider, type Lang } from "./i18n";
import TopBar from "./components/TopBar";
import MediaLibrary from "./components/MediaLibrary";
import Monitor from "./components/Monitor";
import Inspector from "./components/Inspector";
import StatusBar from "./components/StatusBar";
import ProjectScreen from "./components/ProjectScreen";
import SettingsPage from "./components/SettingsPage";

const VIDEO_EXTS = ["mp4", "mkv", "mov", "webm", "avi", "m4v"];

export default function App() {
  const [health, setHealth] = useState("…");
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [lang, setLang] = useState<Lang>("en");
  const [theme, setTheme] = useState<string>("dark");
  const [systemDark, setSystemDark] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [rendering, setRendering] = useState(false);
  const [progress, setProgress] = useState<RenderProgress | null>(null);
  const [scale, setScale] = useState(2);
  const [modelId, setModelId] = useState<string | null>(null);

  const currentFile = files[0];

  const resolvedTheme = theme === "system" ? (systemDark ? "dark" : "light") : theme;

  const reloadProjects = async () => {
    if (!isTauri()) return;
    const list = await listProjects();
    setProjects(list);
  };

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth("n/a"));
    void getSettings()
      .then((s) => {
        setLang((s.language as Lang) || "en");
        setTheme(s.theme || "dark");
      })
      .catch(() => {});
    void reloadProjects();

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemDark(mq.matches);
    const handler = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const changeLang = (l: Lang) => {
    setLang(l);
    void saveSettings({ language: l, theme });
  };

  const changeTheme = (t: string) => {
    setTheme(t);
    void saveSettings({ language: lang, theme: t });
  };

  const openFiles = async () => {
    if (!isTauri()) return;
    const selected = await open({
      multiple: true,
      filters: [{ name: "Video", extensions: VIDEO_EXTS }],
    });
    if (!selected) return;
    const list = Array.isArray(selected) ? selected : [selected];
    setFiles((prev) => [...prev, ...list]);
  };

  const importFolderFiles = async () => {
    if (!isTauri()) return;
    const dir = await open({ directory: true, title: "Import videos from folder" });
    if (typeof dir !== "string") return;
    const found = await importFolder(dir);
    setFiles((prev) => [...prev, ...found]);
  };

  const handleCreateProject = async (name: string) => {
    if (!isTauri()) return;
    const dir = await createProject(name);
    setProjectDir(dir);
    setFiles([]);
  };

  const handleOpenProject = (path: string) => {
    setProjectDir(path);
    setFiles([]);
  };

  const browseProject = async () => {
    if (!isTauri()) return;
    const dir = await open({ directory: true, title: "Open project folder" });
    if (typeof dir === "string") {
      setProjectDir(dir);
      setFiles([]);
    }
  };

  const closeProject = () => {
    setProjectDir(null);
    setFiles([]);
    void reloadProjects();
  };

  const openGithub = () => {
    if (isTauri()) void openUrl("https://github.com/senmei-app/senmei");
  };

  const startRender = async () => {
    if (!isTauri() || !currentFile || rendering) return;
    const output = await save({
      defaultPath: currentFile.replace(/\.[^.]+$/, "_senmei.mp4"),
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "webm"] }],
    });
    if (typeof output !== "string") return;
    setRendering(true);
    setProgress(null);
    const ch = new Channel<RenderProgress>();
    ch.onmessage = setProgress;
    try {
      await render(currentFile, output, scale, modelId, ch);
    } catch (e) {
      setHealth(`render failed: ${e}`);
    } finally {
      setRendering(false);
    }
  };

  return (
    <I18nProvider lang={lang} setLang={changeLang}>
      <div className={resolvedTheme === "dark" ? "dark" : ""}>
        {settingsOpen ? (
          <SettingsPage
            language={lang}
            theme={theme}
            onLanguageChange={changeLang}
            onThemeChange={changeTheme}
            onBack={() => setSettingsOpen(false)}
          />
        ) : !projectDir ? (
          <ProjectScreen
            projects={projects}
            onCreate={handleCreateProject}
            onOpen={handleOpenProject}
            onBrowse={browseProject}
          />
        ) : (
          <div className="flex h-screen w-full flex-col bg-slate-100 font-sans text-slate-900 dark:bg-slate-950 dark:text-slate-200 select-none antialiased">
            <TopBar
              file={currentFile}
              rendering={rendering}
              onImportFile={openFiles}
              onImportFolder={importFolderFiles}
              onStartRender={startRender}
              onCloseProject={closeProject}
              onSettings={() => setSettingsOpen(true)}
              onGithub={openGithub}
            />
            <PanelGroup direction="horizontal" className="flex flex-1 overflow-hidden">
              <Panel defaultSize={20} minSize={14}>
                <MediaLibrary files={files} onOpen={openFiles} />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={55} minSize={35}>
                <Monitor file={currentFile} />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={25} minSize={18}>
                <Inspector
                  scale={scale}
                  onScaleChange={setScale}
                  onModelChange={setModelId}
                />
              </Panel>
            </PanelGroup>
            <StatusBar health={health} fileCount={files.length} progress={progress} rendering={rendering} />
          </div>
        )}
      </div>
    </I18nProvider>
  );
}
