import { useEffect, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  createProject,
  deleteProject,
  exportProject,
  getSettings,
  healthCheck,
  importFolder,
  listProjects,
  loadProjectSettings,
  openProject,
  saveProjectSettings,
  saveSettings,
  type ProjectEntry,
  type ProjectSettings,
} from "@senmei/bridge";
import { I18nProvider, type Lang } from "./i18n";
import { defaultSteps, normalizeSteps, type PipelineStep } from "./steps";
import { defaultHotkey, comboFromEvent, resolveHotkeys } from "./hotkeys";
import { basename } from "./paths";
import { demoProjects, demoVideos } from "./mock";
import { useBatch } from "./useBatch";
import TopBar from "./components/TopBar";
import MediaLibrary from "./components/MediaLibrary";
import Monitor from "./components/Monitor";
import RightPanel from "./components/RightPanel";
import StatusBar from "./components/StatusBar";
import ProjectScreen from "./components/ProjectScreen";
import SettingsPage from "./components/SettingsPage";
import AboutDialog from "./components/AboutDialog";

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
  const [aboutOpen, setAboutOpen] = useState(false);
  const [multiSelect, setMultiSelect] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [mediaView, setMediaView] = useState<"library" | "queue">("library");
  const [steps, setSteps] = useState<PipelineStep[]>(defaultSteps);
  const [hydrated, setHydrated] = useState(false);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [sampleRange, setSampleRange] = useState<{ inMs: number; outMs: number } | null>(null);
  const [fullscreenSignal, setFullscreenSignal] = useState(0);
  const [hotkeyOverrides, setHotkeyOverrides] = useState<Record<string, string>>({});

  const currentFile = files[0];

  const batch = useBatch({ files, selected, steps, outputDir, projectDir, onError: setHealth });

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
          setHotkeyOverrides(s.hotkeys ?? {});
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
        if (s.steps && s.steps.length > 0) setSteps(normalizeSteps(s.steps));
        if (s.files && s.files.length > 0) setFiles(s.files);
        if (s.outputDir) setOutputDir(s.outputDir);
        batch.setRenderedFile(null);
        setHydrated(true);
      })
      .catch(() => setHydrated(true));
  }, [projectDir]);

  useEffect(() => {
    if (!projectDir || !isTauri() || !hydrated) return;
    void saveProjectSettings(projectDir, {
      steps,
      files,
      outputDir,
    }).catch(() => {});
  }, [projectDir, hydrated, steps, files, outputDir]);

  const changeTheme = (t: string) => {
    setTheme(t);
    void saveSettings({ language: lang, theme: t });
  };

  // Persist a hotkey override; resetting to the default drops the entry.
  const changeHotkey = (id: string, combo: string) => {
    setHotkeyOverrides((prev) => {
      const next = { ...prev };
      if (combo === defaultHotkey(id)) delete next[id];
      else next[id] = combo;
      void saveSettings({ language: lang, theme, hotkeys: Object.keys(next).length ? next : null });
      return next;
    });
  };

  // App-wide video drag & drop (not just onto the drop box). Tauri reports
  // absolute paths via onDragDropEvent; the browser demo falls back to HTML5
  // drops (names only).
  const addDropped = (paths: string[]) => {
    const videos = paths.filter((p) => VIDEO_EXTS.some((e) => p.toLowerCase().endsWith(`.${e}`)));
    if (videos.length) setFiles((prev) => [...prev, ...videos]);
  };

  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        if (event.payload.type === "drop") addDropped(event.payload.paths);
      })
      .then((fn) => (unlisten = fn))
      .catch(() => {});
    return () => unlisten?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (isTauri()) return;
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const paths = Array.from(e.dataTransfer?.files ?? []).map((f) => `/demo/${f.name}`);
      addDropped(paths);
    };
    const onOver = (e: DragEvent) => e.preventDefault();
    document.addEventListener("dragover", onOver);
    document.addEventListener("drop", onDrop);
    return () => {
      document.removeEventListener("dragover", onOver);
      document.removeEventListener("drop", onDrop);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      batch.setRenderedFile(null);
      setOutputDir(null);
      return;
    }
    const dir = await createProject(name);
    setProjectDir(dir);
    setFiles([]);
    batch.setRenderedFile(null);
    setOutputDir(null);
  };

  const handleOpenProject = (path: string) => {
    setProjectDir(path);
    setFiles([]);
    batch.setRenderedFile(null);
    setOutputDir(null);
    if (!isTauri()) setFiles([...demoVideos]);
  };

  // Open an exported project archive (.tar.xz); import it into the app
  // storage and switch to it.
  const browseProject = async () => {
    if (!isTauri()) {
      handleOpenProject("/demo/quanzhi-fashi");
      return;
    }
    const file = await open({
      multiple: false,
      title: "Open project",
      filters: [{ name: "Senmei project", extensions: ["tar.xz", "xz"] }],
    });
    if (typeof file !== "string") return;
    try {
      const dir = await openProject(file);
      setProjectDir(dir);
      setFiles([]);
      batch.setRenderedFile(null);
      setOutputDir(null);
    } catch (e) {
      setHealth(`open project failed: ${e}`);
    }
  };

  const closeProject = () => {
    setProjectDir(null);
    setFiles([]);
    batch.setRenderedFile(null);
    setOutputDir(null);
    void reloadProjects();
  };

  const handleExportProject = async () => {
    if (!projectDir) return;
    if (!isTauri()) {
      setHealth("export not available in the browser demo");
      return;
    }
    const base = basename(projectDir) || "project";
    const dest = await save({
      defaultPath: `${base}.tar.xz`,
      filters: [{ name: "Senmei project", extensions: ["tar.xz"] }],
    });
    if (!dest) return;
    try {
      await exportProject(projectDir, dest);
      setHealth(`project exported to ${dest}`);
    } catch (e) {
      setHealth(`export failed: ${e}`);
    }
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
    if (currentFile === path) batch.setRenderedFile(null);
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

  // Plain click selects only that file; toggle (multi-select mode or Ctrl/Cmd) adds/removes.
  const selectFile = (path: string, toggle: boolean) =>
    setSelected((prev) =>
      toggle
        ? prev.includes(path)
          ? prev.filter((p) => p !== path)
          : [...prev, path]
        : [path],
    );
  const selectAll = () => setSelected(files);
  const deleteSelected = () => {
    setFiles((prev) => prev.filter((f) => !selected.includes(f)));
    setSelected([]);
  };

  // Hotkeys (configurable in Settings). Ctrl/Cmd are interchangeable.
  useEffect(() => {
    if (!projectDir) return;
    const hotkeys = resolveHotkeys(hotkeyOverrides);
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      const combo = comboFromEvent(e);
      if (!combo) return;
      switch (combo) {
        case hotkeys.selectAll:
          e.preventDefault();
          selectAll();
          break;
        case hotkeys.deleteSelected:
          if (target?.tagName !== "BUTTON") {
            e.preventDefault();
            deleteSelected();
          }
          break;
        case hotkeys.render:
          e.preventDefault();
          void batch.startBatch();
          break;
        case hotkeys.openFile:
          e.preventDefault();
          void openFiles();
          break;
        case hotkeys.exportProject:
          e.preventDefault();
          void handleExportProject();
          break;
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files, selected, projectDir, hotkeyOverrides]);

  return (
    <I18nProvider lang={lang} setLang={changeLang}>
      <div className={resolvedTheme === "dark" ? "dark" : ""}>
        {settingsOpen ? (
          <SettingsPage
            language={lang}
            theme={theme}
            hotkeys={resolveHotkeys(hotkeyOverrides)}
            onLanguageChange={changeLang}
            onThemeChange={changeTheme}
            onHotkeyChange={changeHotkey}
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
              projectName={projectDir ? basename(projectDir) : undefined}
              rendering={batch.rendering}
              onImportFile={openFiles}
              onImportFolder={importFolderFiles}
              onStartRender={() => batch.startBatch()}
              onCloseProject={closeProject}
              onExportProject={handleExportProject}
              onSettings={() => setSettingsOpen(true)}
              onGithub={openGithub}
              onAbout={() => setAboutOpen(true)}
              onSelectAll={selectAll}
              onDeleteSelected={deleteSelected}
              onAddAllToQueue={() => {
                selectAll();
                setMediaView("queue");
              }}
              onAddSelectedToQueue={() => setMediaView("queue")}
              onProcessSelected={() => batch.startBatch(true)}
              onProcessAll={() => batch.startBatch(false)}
              onToggleFullscreen={() => setFullscreenSignal((n) => n + 1)}
            />
            <PanelGroup direction="horizontal" className="flex flex-1 overflow-hidden">
              <Panel defaultSize={20} minSize={14}>
                <MediaLibrary
                  files={files}
                  onOpen={openFiles}
                  onRemoveFile={removeFile}
                  outputDir={outputDir}
                  onPickOutputDir={pickOutputDir}
                  rendering={batch.rendering}
                  paused={batch.paused}
                  onTogglePause={batch.togglePause}
                  onCancel={batch.cancel}
                  jobs={batch.jobs}
                  selected={selected}
                  onSelect={selectFile}
                  multiSelect={multiSelect}
                  onMultiSelectChange={setMultiSelect}
                  view={mediaView}
                  onViewChange={setMediaView}
                />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={55} minSize={35}>
                <Monitor
                  file={currentFile}
                  renderedFile={batch.renderedFile}
                  rendering={batch.rendering}
                  progress={batch.progress}
                  projectDir={projectDir}
                  sampleInMs={sampleRange?.inMs ?? 0}
                  sampleOutMs={sampleRange?.outMs ?? 0}
                  onSampleChange={(inMs, outMs) => setSampleRange({ inMs, outMs })}
                  onRenderSample={() => currentFile && void batch.startBatch(false, sampleRange, [currentFile])}
                  toggleFullscreenSignal={fullscreenSignal}
                  togglePlayHotkey={resolveHotkeys(hotkeyOverrides).togglePlay}
                />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={25} minSize={18}>
                <RightPanel steps={steps} outputDir={outputDir} onChange={setSteps} />
              </Panel>
            </PanelGroup>
            <StatusBar
              health={health}
              fileCount={files.length}
              progress={batch.progress}
              rendering={batch.rendering}
              onSettings={() => setSettingsOpen(true)}
            />
          </div>
        )}
        {aboutOpen && <AboutDialog onClose={() => setAboutOpen(false)} onGithub={openGithub} />}
      </div>
    </I18nProvider>
  );
}
