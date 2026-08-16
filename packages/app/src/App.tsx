import { useEffect, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { healthCheck } from "@senmei/bridge";
import TopBar from "./components/TopBar";
import MediaLibrary from "./components/MediaLibrary";
import Monitor from "./components/Monitor";
import Inspector from "./components/Inspector";
import StatusBar from "./components/StatusBar";

export default function App() {
  const [health, setHealth] = useState("…");

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth("n/a"));
  }, []);

  return (
    <div className="flex h-screen w-full flex-col bg-slate-950 font-sans text-slate-200 select-none antialiased">
      <TopBar />
      <PanelGroup direction="horizontal" className="flex flex-1 overflow-hidden">
        <Panel defaultSize={20} minSize={14}>
          <MediaLibrary />
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
  );
}
