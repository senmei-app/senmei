import { useEffect, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  createProject,
  healthCheck,
  importFolder,
  listProjects,
  type ProjectEntry,
} from "@senmei/bridge";
import { I18nProvider } from "./i18n";
import TopBar from "./components/TopBar";
import MediaLibrary from "./components/MediaLibrary";
import Monitor from "./components/Monitor";
import Inspector from "./components/Inspector";
import StatusBar from "./components/StatusBar";
import ProjectScreen from "./components/ProjectScreen";

const VIDEO_EXTS = ["mp4", "mkv", "mov", "webm", "avi", "m4v"];

export default function App() {
  const [health, setHealth] = useState("…");
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [files, setFiles] = useState<string[]>([]);

  const reloadProjects = async () => {
    if (!isTauri()) return;
    const list = await listProjects();
    setProjects(list);
  };

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth("n/a"));
    void reloadProjects();
  }, []);

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

  if (!projectDir) {
    return (
      <I18nProvider>
        <ProjectScreen
          projects={projects}
          onCreate={handleCreateProject}
          onOpen={handleOpenProject}
          onBrowse={browseProject}
        />
      </I18nProvider>
    );
  }

  return (
    <I18nProvider>
      <div className="flex h-screen w-full flex-col bg-slate-950 font-sans text-slate-200 select-none antialiased">
        <TopBar
          onImportFile={openFiles}
          onImportFolder={importFolderFiles}
          onCloseProject={closeProject}
          onGithub={openGithub}
        />
        <PanelGroup direction="horizontal" className="flex flex-1 overflow-hidden">
          <Panel defaultSize={20} minSize={14}>
            <MediaLibrary files={files} onOpen={openFiles} />
          </Panel>
          <PanelResizeHandle className="w-px bg-slate-800/80" />
          <Panel defaultSize={55} minSize={35}>
            <Monitor />
          </Panel>
          <PanelResizeHandle className="w-px bg-slate-800/80" />
          <Panel defaultSize={25} minSize={18}>
            <Inspector />
          </Panel>
        </PanelGroup>
        <StatusBar health={health} />
      </div>
    </I18nProvider>
  );
}
