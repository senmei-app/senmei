export default function Monitor() {
  return (
    <main className="flex h-full flex-col bg-slate-950 p-4">
      <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-2xl border border-slate-800 bg-black shadow-2xl">
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80">
          <span className="text-slate-600 font-mono text-sm">[ Live Monitor / Split-View Canvas ]</span>
        </div>

        <div className="absolute inset-y-0 left-1/2 w-0.5 bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.8)]">
          <div className="absolute top-1/2 -left-3 -translate-y-1/2 flex h-6 w-6 items-center justify-center rounded-full bg-indigo-600 text-[10px] text-white shadow-md">
            ↔
          </div>
        </div>

        <div className="absolute top-3 left-3 flex space-x-2">
          <span className="rounded-md bg-black/60 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur">
            Original: 1920x1080
          </span>
          <span className="rounded-md bg-indigo-950/80 border border-indigo-500/40 px-2 py-1 text-[10px] font-mono text-indigo-300 backdrop-blur">
            Senmei: 3840x2160 (60fps)
          </span>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-slate-800/80 bg-slate-900/40 p-3 backdrop-blur">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <button className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white hover:bg-indigo-500">
              ▶
            </button>
            <span className="font-mono text-xs text-slate-300">00:00:12.40 / 00:02:15.00</span>
          </div>
          <div className="flex items-center space-x-1">
            <button className="rounded-md bg-slate-800 px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-700">
              10s
            </button>
            <button className="rounded-md bg-indigo-600/30 border border-indigo-500/40 px-2 py-1 text-[11px] text-indigo-300">
              15s Sample
            </button>
            <button className="rounded-md bg-slate-800 px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-700">
              30s
            </button>
          </div>
        </div>

        <div className="relative h-3 w-full rounded-full bg-slate-800">
          <div className="absolute left-1/4 right-1/2 h-full rounded-full bg-indigo-500/40 border-x-2 border-indigo-400"></div>
        </div>
      </div>
    </main>
  );
}
