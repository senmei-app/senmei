import { useEffect, useState, type ReactNode } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Zap } from "lucide-react";
import { healthCheck } from "@senmei/bridge";
import { Button } from "@senmei/ui";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-zinc-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-400">
        {title}
      </div>
      <div className="flex-1 overflow-auto p-3 text-sm text-zinc-300">{children}</div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState("…");

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth("n/a"));
  }, []);

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <span className="flex items-center gap-2 text-sm font-semibold">
          <Zap className="h-4 w-4 text-yellow-400" />
          Senmei
        </span>
        <div className="flex items-center gap-3">
          <span className="text-xs text-zinc-500">health: {health}</span>
          <Button>Render</Button>
        </div>
      </header>
      <PanelGroup direction="horizontal" className="flex-1">
        <Panel defaultSize={18} minSize={10}>
          <Section title="Input">file browser · models · queue</Section>
        </Panel>
        <PanelResizeHandle className="w-px bg-zinc-800" />
        <Panel defaultSize={58} minSize={30}>
          <Section title="Monitor">live preview · before/after</Section>
        </Panel>
        <PanelResizeHandle className="w-px bg-zinc-800" />
        <Panel defaultSize={24} minSize={14}>
          <Section title="Settings">model · interpolate · upscale · ffmpeg</Section>
        </Panel>
      </PanelGroup>
      <footer className="border-t border-zinc-800 px-4 py-2 text-xs text-zinc-500">
        timeline placeholder: in/out · 10s · 15s · 30s · 60s
      </footer>
    </div>
  );
}
