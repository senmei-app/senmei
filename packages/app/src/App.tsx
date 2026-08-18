import { useEffect, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { isTauri, Channel } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  cancelRender,
  pauseRender,
  uniquePath,
  createProject,
  deleteProject,
  exportProject,
  getSettings,
  healthCheck,
  importFolder,
  listProjects,
  loadProjectSettings,
  openProject,
  render,
  pruneSamples,
  saveProjectSettings,
  saveSettings,
  type ProjectEntry,
  type ProjectSettings,
  type RenderConfig,
  type RenderProgress,
} from "@senmei/bridge";
import { I18nProvider, type Lang } from "./i18n";
import { buildEncoderArgs, defaultSteps, normalizeSteps, type BatchJob, type PipelineStep } from "./steps";
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
  const [rendering, setRendering] = useState(false);
  const [paused, setPaused] = useState(false);
  const [progress, setProgress] = useState<RenderProgress | null>(null);
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [mediaView, setMediaView] = useState<"library" | "queue">("library");
  const [steps, setSteps] = useState<PipelineStep[]>(defaultSteps);
  const [hydrated, setHydrated] = useState(false);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [renderedFile, setRenderedFile] = useState<string | null>(null);
  const [sampleRange, setSampleRange] = useState<{ inMs: number; outMs: number } | null>(null);

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
        if (s.steps && s.steps.length > 0) setSteps(normalizeSteps(s.steps));
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
      steps,
      files,
      outputDir,
    }).catch(() => {});
  }, [projectDir, hydrated, steps, files, outputDir]);

  const changeTheme = (t: string) => {
    setTheme(t);
    void saveSettings({ language: lang, theme: t });
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
      setRenderedFile(null);
      setOutputDir(null);
    } catch (e) {
      setHealth(`open project failed: ${e}`);
    }
  };

  const closeProject = () => {
    setProjectDir(null);
    setFiles([]);
    setRenderedFile(null);
    setOutputDir(null);
    void reloadProjects();
  };

  const handleExportProject = async () => {
    if (!projectDir) return;
    if (!isTauri()) {
      setHealth("export not available in the browser demo");
      return;
    }
    const base = projectDir.split("/").pop() ?? "project";
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
    if (currentFile === path) setRenderedFile(null);
  };

  const handleCancelRender = () => {
    if (!isTauri()) {
      stopDemoRender();
    }
    setRendering(false);
    setPaused(false);
    setJobs((prev) =>
      prev.map((j) =>
        j.status === "queued" || j.status === "rendering" ? { ...j, status: "cancelled" as const } : j,
      ),
    );
    void cancelRender();
  };

  const handleTogglePause = () => {
    if (!isTauri()) {
      setPaused((p) => !p);
      return;
    }
    setPaused((p) => {
      void pauseRender(!p);
      return !p;
    });
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

  const fmtTs = (ms: number): string => {
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h) return `${h}h${m}m${sec}s`;
    if (m) return `${m}m${sec}s`;
    return `${sec}s`;
  };

  const desiredPath = (
    input: string,
    lastOut?: PipelineStep,
    up?: PipelineStep,
    range?: { inMs: number; outMs: number } | null,
    projectDir?: string | null,
  ): string => {
    const container = lastOut?.params?.container || "mkv";
    const outMode = lastOut?.params?.outputMode ?? "input";
    const customFolder = lastOut?.params?.outputFolder ?? "";
    const targetDir =
      outMode === "global" ? outputDir : outMode === "custom" ? customFolder || null : null;
    const label = lastOut?.params?.label?.trim();
    const marker = label || "senmei";
    const info = up?.params?.modelId && up.params?.scale ? `_${up.params.modelId}_x${up.params.scale}` : "";
    // Sample renders are scratch/preview files: keep them out of the output
    // folder root (in the project's `sample/` folder) and tag them with their
    // time range so repeated samples don't differ only by a collision counter.
    const isSample = !!(range && range.outMs > range.inMs);
    const rangeTag = isSample && range ? `_${fmtTs(range.inMs)}-${fmtTs(range.outMs)}` : "";
    const name =
      input
        .split("/")
        .pop()
        ?.replace(/\.[^.]+$/, `_${marker}${info}${rangeTag}.${container}`) ??
      `output_${marker}${info}${rangeTag}.${container}`;
    const dir = targetDir ?? input.split("/").slice(0, -1).join("/");
    if (isSample) return [projectDir ?? dir, "sample", name].join("/");
    return [dir, name].join("/");
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

  // Hotkeys: Ctrl/Cmd+A select all, Delete removes selected, Ctrl/Cmd+R renders,
  // Ctrl/Cmd+O imports a file, Ctrl/Cmd+E exports the project.
  useEffect(() => {
    if (!projectDir) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a") {
        e.preventDefault();
        selectAll();
      } else if (e.key === "Delete" && target?.tagName !== "BUTTON") {
        e.preventDefault();
        deleteSelected();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "r") {
        e.preventDefault();
        void startBatch();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "o") {
        e.preventDefault();
        void openFiles();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "e") {
        e.preventDefault();
        void handleExportProject();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files, selected, projectDir]);

  // Batch render: one render per file, sequentially. A single file is just a
  // batch of one. Errors mark the job failed and continue; cancel stops after
  // the current file; pause freezes the running file.
  const startBatch = async (onlySelected = false, range?: { inMs: number; outMs: number } | null) => {
    const inputs = onlySelected ? files.filter((f) => selected.includes(f)) : files;
    if (!inputs.length || rendering) return;
    const outs = steps.filter((s) => s.enabled && s.stepType === "output");
    const lastOut = outs.length ? outs[outs.length - 1] : undefined;
    const enabled = steps.filter((s) => s.enabled);
    const interp = enabled.find((s) => s.stepType === "interpolation");
    const up = enabled.find((s) => s.stepType === "upscale");
    const res = enabled.find((s) => s.stepType === "resize");
    const dn = enabled.find((s) => s.stepType === "denoise");
    const db = enabled.find((s) => s.stepType === "deblur");
    const dd = enabled.find((s) => s.stepType === "deduplication");
    const outScale = up ? (up.params?.scale ?? null) : null;
    const outModel = up ? (up.params?.modelId ?? null) : null;
    const outOutputResize = res ? toFactor(res.params?.factor ?? "") : null;
    const outFps = interp ? (interp.params?.fpsMultiplier ?? null) : null;
    const outInterpModel = interp ? (interp.params?.modelId ?? null) : null;
    const outFfmpegArgs = buildEncoderArgs(lastOut?.params, lastOut?.params?.ffmpegArgs ?? "");
    const outFilter = {
      denoiseRadius: dn ? (dn.params?.radius ?? null) : null,
      deblurAmount: db ? (db.params?.amount ?? null) : null,
      dedupThreshold: dd ? (dd.params?.threshold ?? null) : null,
    };
    const config: RenderConfig = {
      scale: outScale,
      modelId: outModel,
      resize: null,
      filter: outFilter,
      outputResize: outOutputResize,
      fpsMultiplier: outFps,
      interpModel: outInterpModel,
      ffmpegArgs: outFfmpegArgs,
      startMs: range?.inMs ?? null,
      endMs: range?.outMs ?? null,
    };

    const initial: BatchJob[] = inputs.map((f) => ({
      input: f,
      output: desiredPath(f, lastOut, up, range, projectDir),
      status: "queued",
      progress: null,
    }));
    setJobs(initial);
    setRendering(true);
    setPaused(false);
    setRenderedFile(null);

    const patch = (i: number, p: Partial<BatchJob>) =>
      setJobs((prev) => prev.map((j, k) => (k === i ? { ...j, ...p } : j)));

    try {
      for (let i = 0; i < initial.length; i++) {
        let output = initial[i].output;
        if (isTauri()) {
          try {
            output = await uniquePath(output); // collision -> _2, _3, …
          } catch {
            // keep the intended path if resolution fails
          }
        }
        patch(i, { output, status: "rendering", progress: null });
        try {
          if (isTauri()) {
            const ch = new Channel<RenderProgress>();
            ch.onmessage = (p) => {
              patch(i, { progress: p });
              setProgress(p);
            };
            await render(initial[i].input, output, config, ch);
          } else {
            await startDemoRender((p) => {
              patch(i, { progress: p });
              setProgress(p);
            });
          }
          patch(i, { status: "done" });
          setRenderedFile(output);
          if (range) {
            // Sample renders live in the project's sample/ folder: keep only the newest.
            void pruneSamples(output.split("/").slice(0, -1).join("/"), 5);
          }
        } catch (e) {
          const msg = String(e);
          if (msg.toLowerCase().includes("cancelled")) {
            patch(i, { status: "cancelled" });
            setJobs((prev) => prev.map((j, k) => (k > i ? { ...j, status: "cancelled" as const } : j)));
            break; // stop the batch
          }
          patch(i, { status: "failed", error: msg });
          if (isTauri()) setHealth(`render failed: ${msg}`);
          // continue with the next file
        }
      }
    } finally {
      setRendering(false);
      setPaused(false);
      setProgress(null);
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
              onStartRender={() => startBatch()}
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
              onProcessSelected={() => startBatch(true)}
              onProcessAll={() => startBatch(false)}
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
                  paused={paused}
                  onTogglePause={handleTogglePause}
                  onCancel={handleCancelRender}
                  jobs={jobs}
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
                  renderedFile={renderedFile}
                  rendering={rendering}
                  progress={progress}
                  projectDir={projectDir}
                  sampleInMs={sampleRange?.inMs ?? 0}
                  sampleOutMs={sampleRange?.outMs ?? 0}
                  onSampleChange={(inMs, outMs) => setSampleRange({ inMs, outMs })}
                  onRenderSample={() => void startBatch(false, sampleRange)}
                />
              </Panel>
              <PanelResizeHandle className="w-px bg-slate-200 dark:bg-slate-800/80" />
              <Panel defaultSize={25} minSize={18}>
                <Inspector steps={steps} outputDir={outputDir} onChange={setSteps} />
              </Panel>
            </PanelGroup>
            <StatusBar
              health={health}
              fileCount={files.length}
              progress={progress}
              rendering={rendering}
              onSettings={() => setSettingsOpen(true)}
            />
          </div>
        )}
      </div>
      {aboutOpen && <AboutDialog onClose={() => setAboutOpen(false)} onGithub={openGithub} />}
    </I18nProvider>
  );
}
