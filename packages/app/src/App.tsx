import { useEffect, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { isTauri, Channel } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  cancelRender,
  createProject,
  deleteProject,
  getSettings,
  healthCheck,
  importFolder,
  listProjects,
  loadProjectSettings,
  render,
  saveProjectSettings,
  saveSettings,
  type ProjectEntry,
  type ProjectSettings,
  type RenderProgress,
} from "@senmei/bridge";
import { I18nProvider, type Lang } from "./i18n";
import {
  demoProjects,
  demoVideos,
  startDemoRender,
  stopDemoRender,
} from "./mock";
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
  const [resizeFactor, setResizeFactor] = useState("");
  const [outputResizeFactor, setOutputResizeFactor] = useState("");
  const [fpsMultiplier, setFpsMultiplier] = useState<number | null>(null);
  const [stepsEnabled, setStepsEnabled] = useState<Record<string, boolean>>({});
  const [hydrated, setHydrated] = useState(false);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [renderedFile, setRenderedFile] = useState<string | null>(null);

  const currentFile = files[0];

  const toFactor = (v: string): number | null => {
    const f = Number(v);
    return f > 0 ? f : null;
  };

  const resolvedTheme = theme === "system" ? (systemDark ? "dark" : "light") : theme;

  const reloadProjects = async () => {
    if (!isTauri()) {
      setProjects([...demoProjects]);
      return;
    }
    const list = await listProjects();
    setProjects(list);
  };

  useEffect(() => {
    if (!isTauri()) {
      setHealth("demo (browser)");
    } else {
      healthCheck()
        .then(setHealth)
        .catch(() => setHealth("n/a"));
      void getSettings()
        .then((s) => {
          setLang((s.language as Lang) || "en");
          setTheme(s.theme || "dark");
        })
        .catch(() => {});
    }
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

  // Load per-project settings when a project opens; save on any change.
  useEffect(() => {
    if (!projectDir || !isTauri()) {
      setHydrated(false);
      return;
    }
    setHydrated(false);
    loadProjectSettings(projectDir)
      .then((s: ProjectSettings) => {
        setStepsEnabled(s.stepsEnabled ?? {});
        if (s.upscaleModel) setModelId(s.upscaleModel);
        if (s.scale) setScale(s.scale);
        if (s.files && s.files.length > 0) setFiles(s.files);
        if (s.outputDir) setOutputDir(s.outputDir);
        setRenderedFile(null);
        setHydrated(true);
      })
      .catch(() => setHydrated(true));
  }, [projectDir]);

  useEffect(() => {
    if (!projectDir || !isTauri() || !hydrated) return;
    void saveProjectSettings(projectDir, {
      stepsEnabled,
      upscaleModel: modelId,
      scale,
      files,
      outputDir,
    }).catch(() => {});
  }, [projectDir, hydrated, stepsEnabled, modelId, scale, files, outputDir]);

  const changeTheme = (t: string) => {
    setTheme(t);
    void saveSettings({ language: lang, theme: t });
  };

  const openFiles = async () => {
    if (!isTauri()) {
      setFiles((prev) => [...prev, ...demoVideos]);
      return;
    }
    const selected = await open({
      multiple: true,
      filters: [{ name: "Video", extensions: VIDEO_EXTS }],
    });
    if (!selected) return;
    const list = Array.isArray(selected) ? selected : [selected];
    setFiles((prev) => [...prev, ...list]);
  };

  const importFolderFiles = async () => {
    if (!isTauri()) {
      setFiles((prev) => [...prev, ...demoVideos]);
      return;
    }
    const dir = await open({ directory: true, title: "Import videos from folder" });
    if (typeof dir !== "string") return;
    const found = await importFolder(dir);
    setFiles((prev) => [...prev, ...found]);
  };

  const handleCreateProject = async (name: string) => {
    if (!isTauri()) {
      const p: ProjectEntry = { name, path: `/demo/${name.toLowerCase().replace(/\s+/g, "-")}` };
      demoProjects.push(p);
      setProjectDir(p.path);
      setFiles([]);
      setRenderedFile(null);
      setOutputDir(null);
      return;
    }
    const dir = await createProject(name);
    setProjectDir(dir);
    setFiles([]);
    setRenderedFile(null);
    setOutputDir(null);
  };

  const handleOpenProject = (path: string) => {
    setProjectDir(path);
    setFiles([]);
    setRenderedFile(null);
    setOutputDir(null);
    if (!isTauri()) setFiles([...demoVideos]);
  };

  const browseProject = async () => {
    if (!isTauri()) {
      handleOpenProject("/demo/quanzhi-fashi");
      return;
    }
    const dir = await open({ directory: true, title: "Open project folder" });
    if (typeof dir === "string") {
      setProjectDir(dir);
      setFiles([]);
      setRenderedFile(null);
      setOutputDir(null);
    }
  };

  const closeProject = () => {
    setProjectDir(null);
    setFiles([]);
    setRenderedFile(null);
    setOutputDir(null);
    void reloadProjects();
  };

  const pickOutputDir = async () => {
    if (!isTauri()) {
      setOutputDir("/demo/output");
      return;
    }
    const dir = await open({ directory: true, title: "Output folder" });
    if (typeof dir === "string") setOutputDir(dir);
  };

  const removeFile = (path: string) => {
    setFiles((prev) => prev.filter((f) => f !== path));
    if (currentFile === path) setRenderedFile(null);
  };

  const handleCancelRender = () => {
    if (!isTauri()) {
      stopDemoRender();
      setRendering(false);
      return;
    }
    void cancelRender();
  };

  const handleDeleteProject = async (path: string) => {
    try {
      if (!isTauri()) {
        const i = demoProjects.findIndex((p) => p.path === path);
        if (i >= 0) demoProjects.splice(i, 1);
      } else {
        await deleteProject(path);
      }
      await reloadProjects();
    } catch (e) {
      setHealth(`delete failed: ${e}`);
    }
  };

  const openGithub = () => {
    if (isTauri()) void openUrl("https://github.com/senmei-app/senmei");
  };

  const startRender = async () => {
    if (!currentFile || rendering) return;
    if (!isTauri()) {
      setRendering(true);
      setProgress(null);
      setRenderedFile(null);
      try {
        const out = await startDemoRender(setProgress);
        setRenderedFile(out);
      } finally {
        setRendering(false);
      }
      return;
    }
    const base = currentFile.split("/").pop()?.replace(/\.[^.]+$/, "_senmei.mp4") ?? "output_senmei.mp4";
    const defaultPath = outputDir ? `${outputDir}/${base}` : currentFile.replace(/\.[^.]+$/, "_senmei.mp4");
    const output = await save({
      defaultPath,
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "webm"] }],
    });
    if (typeof output !== "string") return;
    setRendering(true);
    setProgress(null);
    setRenderedFile(null);
    const ch = new Channel<RenderProgress>();
    ch.onmessage = setProgress;
    const enabled = (id: string) => stepsEnabled[id] !== false;
    const outScale = enabled("upscale") ? scale : null;
    const outModel = enabled("upscale") ? modelId : null;
    const outResize = enabled("resize") ? toFactor(resizeFactor) : null;
    const outOutputResize = enabled("output_resize") ? toFactor(outputResizeFactor) : null;
    const outFps = enabled("interpolate") ? fpsMultiplier : null;
    try {
      await render(
        currentFile,
        output,
        outScale,
        outModel,
        outResize,
        outOutputResize,
        outFps,
        ch,
      );
      setRenderedFile(output);
    } catch (e) {
      const msg = String(e);
      if (!msg.toLowerCase().includes("cancelled")) setHealth(`render failed: ${e}`);
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
            onDelete={handleDeleteProject}
          />
        ) : (
          <div className="flex h-screen w-full flex-col bg-slate-100 font-sans text-slate-900 dark:bg-slate-950 dark:text-slate-200 select-none antialiased">
            <TopBar
              file={currentFile}
              projectName={projectDir ? projectDir.split("/").pop() : undefined}
              rendering={rendering}
              onImportFile={openFiles}
              onImportFolder={importFolderFiles}
              onStartRender={startRender}
              onCancelRender={handleCancelRender}
              onCloseProject={closeProject}
              onSettings={() => setSettingsOpen(true)}
              onGithub={openGithub}
            />
            <PanelGroup direction="horizontal" className="flex flex-1 overflow-hidden">
              <Panel defaultSize={20} minSize={14}>
                <MediaLibrary
                  files={files}
                  onOpen={openFiles}
                  onRemoveFile={removeFile}
                  outputDir={outputDir}
                  onPickOutputDir={pickOutputDir}
                  rendering={rendering}
                  onCancel={handleCancelRender}
                  progress={progress}
                  renderedFile={renderedFile}
                />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={55} minSize={35}>
                <Monitor
                  file={currentFile}
                  renderedFile={renderedFile}
                  rendering={rendering}
                  progress={progress}
                />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={25} minSize={18}>
                <Inspector
                  modelId={modelId}
                  scale={scale}
                  onScaleChange={setScale}
                  onModelChange={setModelId}
                  resizeFactor={resizeFactor}
                  onResizeFactorChange={setResizeFactor}
                  outputResizeFactor={outputResizeFactor}
                  onOutputResizeFactorChange={setOutputResizeFactor}
                  fpsMultiplier={fpsMultiplier}
                  onFpsChange={setFpsMultiplier}
                  stepsEnabled={stepsEnabled}
                  onToggleStep={(id, enabled) =>
                    setStepsEnabled((prev) => ({ ...prev, [id]: enabled }))
                  }
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
