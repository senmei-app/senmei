import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type {
  BackendInfo,
  EngineBackend,
  HardwareSnapshot,
  ProjectEntry,
  ProjectSettings,
  Settings,
} from "@senmei/bridge";
import { backend as getBackend } from "./backend";
import { I18nProvider, type Lang } from "./i18n";
import { defaultSteps, isStepType, newStepId, normalizeSteps, type PipelineStep, type StepParams, type StepType } from "./steps";
import { defaultHotkey, comboFromEvent, resolveHotkeys } from "./hotkeys";
import { basename } from "./paths";
import { useBatch } from "./useBatch";
import TopBar from "./components/TopBar";
import MediaLibrary from "./components/MediaLibrary";
import Monitor from "./components/Monitor";
import OnboardingWizard from "./components/OnboardingWizard";
import RightPanel from "./components/RightPanel";
import StatusBar from "./components/StatusBar";
import ProjectScreen from "./components/ProjectScreen";
import SettingsPage from "./components/SettingsPage";
import AboutDialog from "./components/AboutDialog";
import UpdateDialog from "./components/UpdateDialog";
import PathDialog from "./components/PathDialog";
import { checkForUpdates, type UpdateInfo } from "./updater";

const VIDEO_EXTS = ["mp4", "mkv", "mov", "webm", "avi", "m4v"];

export default function App() {
  const [health, setHealth] = useState("…");
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [lang, setLang] = useState<Lang>("en");
  const [theme, setTheme] = useState<string>("dark");
  const [tileSize, setTileSize] = useState<number>(640);
  const [gpuIndex, setGpuIndex] = useState<number>(0);
  const [backend, setBackend] = useState<EngineBackend>("auto");
  const [backendInfoState, setBackendInfoState] = useState<BackendInfo | null>(null);
  const [hardware, setHardware] = useState<HardwareSnapshot | null>(null);
  const [systemDark, setSystemDark] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [multiSelect, setMultiSelect] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [mediaView, setMediaView] = useState<"library" | "queue">("library");
  // Undo/redo history for the step stack — every structural edit commits. A
  // pure reducer keeps commit/undo/redo atomic (no shared mutable refs, so two
  // rapid commits can't drop an undo step) and inside React's render cycle.
  type HistoryState = { present: PipelineStep[]; past: PipelineStep[][]; future: PipelineStep[][] };
  type HistoryAction =
    | { type: "commit"; next: PipelineStep[] }
    | { type: "reset"; next: PipelineStep[] }
    | { type: "undo" }
    | { type: "redo" };
  const historyReducer = (s: HistoryState, a: HistoryAction): HistoryState => {
    switch (a.type) {
      case "commit":
        return { present: a.next, past: [...s.past, s.present].slice(-100), future: [] };
      case "reset":
        return { present: a.next, past: [], future: [] };
      case "undo": {
        if (s.past.length === 0) return s;
        const prev = s.past[s.past.length - 1];
        return { present: prev, past: s.past.slice(0, -1), future: [s.present, ...s.future] };
      }
      case "redo": {
        if (s.future.length === 0) return s;
        const [next, ...rest] = s.future;
        return { present: next, past: [...s.past, s.present], future: rest };
      }
    }
  };
  const [history, dispatchHistory] = useReducer(historyReducer, {
    present: defaultSteps(),
    past: [],
    future: [],
  });
  const steps = history.present;
  const canUndo = history.past.length > 0;
  const canRedo = history.future.length > 0;
  const commitSteps = useCallback(
    (next: PipelineStep[]) => dispatchHistory({ type: "commit", next }),
    [],
  );
  const resetHistory = (next: PipelineStep[]) => dispatchHistory({ type: "reset", next });
  const undo = useCallback(() => dispatchHistory({ type: "undo" }), []);
  const redo = useCallback(() => dispatchHistory({ type: "redo" }), []);
  const [onboarded, setOnboarded] = useState(() => localStorage.getItem("senmei.onboarded") === "1");
  const [hydrated, setHydrated] = useState(false);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [sampleRange, setSampleRange] = useState<{ inMs: number; outMs: number } | null>(null);
  const [fullVideo, setFullVideo] = useState(false);
  const [hotkeyOverrides, setHotkeyOverrides] = useState<Record<string, string>>({});
  const resolvedHotkeys = useMemo(() => resolveHotkeys(hotkeyOverrides), [hotkeyOverrides]);

  const [currentFile, setCurrentFile] = useState<string | null>(null);
  // Keep the preview on a valid file: drop removed/cleared paths, default to the
  // first when the current one is gone (new imports preview the first file).
  useEffect(() => {
    setCurrentFile((cur) => (cur && files.includes(cur) ? cur : (files[0] ?? null)));
  }, [files]);

  const batch = useBatch({ files, selected, steps, outputDir, projectDir, onError: setHealth });

  const resolvedTheme = theme === "system" ? (systemDark ? "dark" : "light") : theme;

  const reloadProjects = async () => {
    const list = await (await getBackend()).listProjects();
    setProjects(list);
  };

  useEffect(() => {
    (async () => {
      const be = await getBackend();
      be.healthCheck()
        .then(setHealth)
        .catch(() => setHealth("n/a"));
      void be
        .getSettings()
        .then((s) => {
          setLang((s.language as Lang) || "en");
          setTheme(s.theme || "dark");
          setHotkeyOverrides(s.hotkeys ?? {});
          setTileSize(s.tileSize ?? 640);
          setGpuIndex(s.gpuIndex ?? 0);
          setBackend(s.backend ?? "auto");
        })
        .catch(() => {});
      be.backendInfo()
        .then(setBackendInfoState)
        .catch(() => {});
      void reloadProjects();
    })();

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemDark(mq.matches);
    const handler = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Check for app updates once on startup.
  useEffect(() => {
    checkForUpdates().then((u) => { if (u) setUpdateInfo(u); });
  }, []);

  // Poll hardware usage (GPU/CPU/RAM) once per second.
  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      getBackend()
        .then((b) => b.hardwareStatus())
        .then((snapshot) => {
          if (!cancelled) setHardware(snapshot);
        })
        .catch(() => {});
    poll();
    const id = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Persist the current settings merged with `partial`.
  const persistSettings = (partial: Partial<Settings>) => {
    void getBackend().then((b) =>
      b.saveSettings({
        language: lang,
        theme,
        hotkeys: Object.keys(hotkeyOverrides).length ? hotkeyOverrides : null,
        tileSize,
        gpuIndex,
        backend,
        ...partial,
      }),
    );
  };

  const changeLang = (l: Lang) => {
    setLang(l);
    persistSettings({ language: l });
  };

  const changeBackend = (b: EngineBackend) => {
    setBackend(b);
    persistSettings({ backend: b });
  };

  // Load per-project settings when a project opens; save on any change.
  useEffect(() => {
    if (!projectDir) {
      setHydrated(false);
      return;
    }
    setHydrated(false);
    getBackend()
      .then((b) => b.loadProjectSettings(projectDir))
      .then((s: ProjectSettings) => {
        resetHistory(s.steps && s.steps.length > 0 ? normalizeSteps(s.steps) : defaultSteps());
        if (s.files && s.files.length > 0) setFiles(s.files);
        if (s.outputDir) setOutputDir(s.outputDir);
        batch.setRenderedFile(null);
        setHydrated(true);
      })
      .catch(() => setHydrated(true));
  }, [projectDir]);

  useEffect(() => {
    if (!projectDir || !hydrated) return;
    void getBackend().then((b) =>
      b.saveProjectSettings(projectDir, { steps, files, outputDir }).catch(() => {}),
    );
  }, [projectDir, hydrated, steps, files, outputDir]);

  const changeTheme = (t: string) => {
    setTheme(t);
    persistSettings({ theme: t });
  };

  const changeTileSize = (n: number) => {
    setTileSize(n);
    persistSettings({ tileSize: n });
  };

  const changeGpuIndex = (n: number) => {
    setGpuIndex(n);
    persistSettings({ gpuIndex: n });
  };

  // Persist a hotkey override; resetting to the default drops the entry.
  const changeHotkey = (id: string, combo: string) => {
    setHotkeyOverrides((prev) => {
      const next = { ...prev };
      if (combo === defaultHotkey(id)) delete next[id];
      else next[id] = combo;
      persistSettings({ hotkeys: Object.keys(next).length ? next : null });
      return next;
    });
  };

  // App-wide video drag & drop (not just onto the drop box). The backend
  // registers the transport's drop source (Tauri webview / HTML5 in web).
  const addDropped = (paths: string[]) => {
    const videos = paths.filter((p) => VIDEO_EXTS.some((e) => p.toLowerCase().endsWith(`.${e}`)));
    if (videos.length) setFiles((prev) => [...prev, ...videos]);
  };

  useEffect(() => {
    let un: (() => void) | undefined;
    getBackend().then((b) => {
      un = b.onFileDrop(addDropped);
    });
    return () => un?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openFiles = async () => {
    const list = await (await getBackend()).pickVideoFiles();
    setFiles((prev) => [...prev, ...list]);
  };

  const importFolderFiles = async () => {
    const be = await getBackend();
    const dir = await be.pickFolder("Import videos from folder");
    if (!dir) return;
    const found = await be.importFolder(dir);
    setFiles((prev) => [...prev, ...found]);
  };

  const processFolderFiles = async () => {
    const be = await getBackend();
    const dir = await be.pickFolder("Process videos in folder (incl. subfolders)");
    if (!dir) return;
    const found = await be.scanFolder(dir);
    if (!found.length) {
      setHealth("no videos found in folder");
      return;
    }
    setFiles((prev) => [...prev, ...found]);
    void batch.startBatch(false, null, found);
  };

  // Content-aware defaults: probe the current file (anime vs live-action,
  // resolution, fps) and populate the pipeline with a suggested step chain.
  const suggestPipeline = async () => {
    if (!currentFile) return;
    const be = await getBackend();
    try {
      const raw = await be.suggestPipeline(currentFile);
      const sug = JSON.parse(raw) as {
        anime: boolean;
        steps: { stepType: string; params?: StepParams }[];
      };
      const mapped = sug.steps
        .filter((s) => isStepType(s.stepType))
        .map((s) => ({
          id: newStepId(),
          stepType: s.stepType as StepType,
          enabled: true,
          params: s.params ?? {},
        }));
      if (!mapped.length) {
        setHealth("suggest produced no steps");
        return;
      }
      resetHistory(mapped);
      setHealth(`Suggested pipeline: ${sug.anime ? "anime" : "live-action"} · ${mapped.length} steps`);
    } catch (e) {
      setHealth(`suggest failed: ${e}`);
    }
  };

  const handleCreateProject = async (name: string) => {
    const dir = await (await getBackend()).createProject(name);
    setProjectDir(dir);
    setFiles([]);
    batch.setRenderedFile(null);
    setOutputDir(null);
  };

  const handleOpenProject = async (path: string) => {
    setProjectDir(path);
    setFiles([]);
    batch.setRenderedFile(null);
    setOutputDir(null);
  };

  // Open an exported project archive (.tar.xz); import it into the app
  // storage and switch to it.
  const browseProject = async () => {
    const be = await getBackend();
    const file = await be.pickFile(
      [{ name: "Senmei project", extensions: ["tar.xz", "xz"] }],
      "Open project",
    );
    if (!file) return;
    try {
      const dir = await be.openProject(file);
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
    const be = await getBackend();
    const base = basename(projectDir) || "project";
    const dest = await be.pickSaveFile(`${base}.tar.xz`, ["tar.xz"]);
    if (!dest) return;
    try {
      await be.exportProject(projectDir, dest);
      setHealth(`project exported to ${dest}`);
    } catch (e) {
      setHealth(`export failed: ${e}`);
    }
  };

  const pickOutputDir = async () => {
    const dir = await (await getBackend()).pickFolder("Output folder");
    if (dir) setOutputDir(dir);
  };

  const removeFile = (path: string) => {
    setFiles((prev) => prev.filter((f) => f !== path));
    if (currentFile === path) batch.setRenderedFile(null);
  };

  const handleDeleteProject = async (path: string) => {
    try {
      await (await getBackend()).deleteProject(path);
      await reloadProjects();
    } catch (e) {
      setHealth(`delete failed: ${e}`);
    }
  };

  const openGithub = () => {
    void getBackend().then((b) => b.openExternal("https://github.com/senmei-app/senmei"));
  };

  // Plain click selects only that file; toggle (multi-select mode or Ctrl/Cmd) adds/removes.
  const selectFile = (path: string, toggle: boolean) => {
    if (toggle) {
      setSelected((prev) =>
        prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path],
      );
      return;
    }
    // Plain click selects AND previews the file.
    setSelected([path]);
    setCurrentFile(path);
  };
  const selectAll = () => setSelected(files);
  const deleteSelected = () => {
    setFiles((prev) => prev.filter((f) => !selected.includes(f)));
    setSelected([]);
  };

  // Full Video Mode: fullscreen the OS window + show only the monitor. Window
  // fullscreen keeps the <video> in the DOM (smooth) instead of webkit2gtk's
  // native media fullscreen (separate layer, uncontrolled dblclick). The toggle
  // is pure; the sync effect below applies it to the OS window.
  const batchRef = useRef(batch);
  batchRef.current = batch;
  const renderSample = useCallback(() => {
    if (currentFile) void batchRef.current.startBatch(false, sampleRange, [currentFile]);
  }, [currentFile, sampleRange]);

  const toggleFullVideo = () => {
    setFullVideo((v) => !v);
  };

  // Keep the OS window fullscreen in sync with Full Video Mode.
  useEffect(() => {
    void getBackend()
      .then((b) => b.setWindowFullscreen(fullVideo))
      .catch((e) => console.error("setWindowFullscreen failed:", e));
  }, [fullVideo]);

  // Escape exits Full Video Mode (OS window fullscreen has no native Escape;
  // requestFullscreen previously exited natively).
  useEffect(() => {
    if (!fullVideo) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (e.key === "Escape") {
        e.preventDefault();
        setFullVideo(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [fullVideo]);

  // Hotkeys (configurable in Settings). Ctrl/Cmd are interchangeable.
  useEffect(() => {
    if (!projectDir) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      const combo = comboFromEvent(e);
      if (!combo) return;
      switch (combo) {
        case resolvedHotkeys.selectAll:
          e.preventDefault();
          selectAll();
          break;
        case resolvedHotkeys.deleteSelected:
          if (target?.tagName !== "BUTTON") {
            e.preventDefault();
            deleteSelected();
          }
          break;
        case resolvedHotkeys.render:
          e.preventDefault();
          void batchRef.current.startBatch();
          break;
        case resolvedHotkeys.openFile:
          e.preventDefault();
          void openFiles();
          break;
        case resolvedHotkeys.exportProject:
          e.preventDefault();
          void handleExportProject();
          break;
        case resolvedHotkeys.toggleFullscreen:
          e.preventDefault();
          toggleFullVideo();
          break;
        case resolvedHotkeys.renderSample:
          e.preventDefault();
          renderSample();
          break;
        case resolvedHotkeys.toggleMultiSelect:
          e.preventDefault();
          setMultiSelect((v) => !v);
          break;
        case resolvedHotkeys.undo:
          e.preventDefault();
          undo();
          break;
        case resolvedHotkeys.redo:
          e.preventDefault();
          redo();
          break;
        case resolvedHotkeys.viewLibrary:
          e.preventDefault();
          setMediaView("library");
          break;
        case resolvedHotkeys.viewQueue:
          e.preventDefault();
          setMediaView("queue");
          break;
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files, selected, projectDir, hotkeyOverrides, renderSample, undo, redo]);

  const monitorEl = useMemo(() => (
    <Monitor
      file={currentFile ?? undefined}
      renderedFile={batch.renderedFile}
      prevRenderedFile={batch.prevRenderedFile}
      rendering={batch.rendering}
      progress={batch.progress}
      timings={batch.timings}
      steps={steps}
      sampleInMs={sampleRange?.inMs ?? 0}
      sampleOutMs={sampleRange?.outMs ?? 0}
      onSampleChange={(inMs, outMs) => setSampleRange({ inMs, outMs })}
      onRenderSample={renderSample}
      fullVideo={fullVideo}
      onToggleFullVideo={toggleFullVideo}
      toggleMetaHotkey={resolvedHotkeys.toggleMeta}
      renderSampleHotkey={resolvedHotkeys.renderSample}
      togglePlayHotkey={resolvedHotkeys.togglePlay}
      muteHotkey={resolvedHotkeys.mute}
      volumeUpHotkey={resolvedHotkeys.volumeUp}
      volumeDownHotkey={resolvedHotkeys.volumeDown}
      seekBackHotkey={resolvedHotkeys.seekBack}
      seekForwardHotkey={resolvedHotkeys.seekForward}
      modeSourceHotkey={resolvedHotkeys.modeSource}
      modeResultHotkey={resolvedHotkeys.modeResult}
      modeCompareHotkey={resolvedHotkeys.modeCompare}
      modeABHotkey={resolvedHotkeys.modeAB}
    />
  ), [currentFile, batch.renderedFile, batch.prevRenderedFile, batch.rendering, batch.progress, batch.timings, steps, sampleRange, renderSample, fullVideo, toggleFullVideo, resolvedHotkeys]);

  return (
    <I18nProvider lang={lang} setLang={changeLang}>
      <div className={resolvedTheme === "dark" ? "dark" : ""}>
        <OnboardingWizard
          open={!onboarded}
          onDone={() => {
            localStorage.setItem("senmei.onboarded", "1");
            setOnboarded(true);
          }}
        />
        {settingsOpen ? (
          <SettingsPage
            language={lang}
            theme={theme}
            tileSize={tileSize}
            gpuIndex={gpuIndex}
            backend={backend}
            backendInfo={backendInfoState}
            hardware={hardware}
            hotkeys={resolvedHotkeys}
            onLanguageChange={changeLang}
            onThemeChange={changeTheme}
            onTileSizeChange={changeTileSize}
            onGpuIndexChange={changeGpuIndex}
            onBackendChange={changeBackend}
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
            {/* The monitor panel stays in place (no remount → the <video> keeps
                its position); Full Video Mode covers the viewport with it. */}
            <TopBar
              file={currentFile ?? undefined}
              projectName={projectDir ? basename(projectDir) : undefined}
              hotkeys={resolvedHotkeys}
              onImportFile={openFiles}
              onImportFolder={importFolderFiles}
              onBatchFolder={processFolderFiles}
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
              onToggleFullscreen={toggleFullVideo}
              onUndo={undo}
              onRedo={redo}
              canUndo={canUndo}
              canRedo={canRedo}
              hasFiles={files.length > 0}
              hasSelection={selected.length > 0}
            />
            <PanelGroup direction="horizontal" className="relative flex flex-1 overflow-hidden">
              <Panel defaultSize={20} minSize={14}>
                <MediaLibrary
                  files={files}
                  hotkeys={resolvedHotkeys}
                  onOpen={openFiles}
                  onRemoveFile={removeFile}
                  outputDir={outputDir}
                  onPickOutputDir={pickOutputDir}
                  onStartRender={() => batch.startBatch()}
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
                  savedQueue={batch.savedQueue}
                  onResumeQueue={batch.resumeQueue}
                  onDiscardQueue={batch.discardQueue}
                />
              </Panel>
              <PanelResizeHandle className="w-1.5 cursor-col-resize bg-slate-200 transition-colors hover:bg-indigo-300/70 active:bg-indigo-400/80 dark:bg-slate-800/80 dark:hover:bg-indigo-400/40" />
              <Panel
                defaultSize={55}
                minSize={35}
                className={fullVideo ? "fixed inset-0 z-[60]" : undefined}
              >
                {monitorEl}
              </Panel>
              <PanelResizeHandle className="w-1.5 cursor-col-resize bg-slate-200 transition-colors hover:bg-indigo-300/70 active:bg-indigo-400/80 dark:bg-slate-800/80 dark:hover:bg-indigo-400/40" />
              <Panel defaultSize={25} minSize={18}>
                <RightPanel steps={steps} outputDir={outputDir} onChange={commitSteps} onSuggest={suggestPipeline} />
              </Panel>
            </PanelGroup>
            <StatusBar
              health={health}
              fileCount={files.length}
              progress={batch.progress}
              rendering={batch.rendering}
              hardware={hardware}
              onSettings={() => setSettingsOpen(true)}
            />
          </div>
        )}
        {aboutOpen && <AboutDialog onClose={() => setAboutOpen(false)} onGithub={openGithub} />}
        {updateInfo && <UpdateDialog update={updateInfo} onClose={() => setUpdateInfo(null)} />}
        <PathDialog />
      </div>
    </I18nProvider>
  );
}
