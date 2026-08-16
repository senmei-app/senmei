export default function StatusBar({ health }: { health: string }) {
  return (
    <footer className="flex h-7 items-center justify-between border-t border-slate-800/80 bg-slate-950 px-3 text-[11px] text-slate-500">
      <div className="flex items-center space-x-3">
        <span className="flex items-center space-x-1.5 text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>CUDA: RTX 4080 (16GB VRAM)</span>
        </span>
        <span>•</span>
        <span>Backend: libtorch</span>
      </div>
      <span>{health === "ok" ? "Bereit" : `health: ${health}`}</span>
    </footer>
  );
}
