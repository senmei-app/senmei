export default function Inspector() {
  return (
    <aside className="h-full w-full overflow-y-auto border-l border-slate-800/80 bg-slate-900/30 p-4">
      <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Inspector Settings
      </h2>

      <div className="space-y-3">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="text-indigo-400">⚡</span>
              <span className="text-xs font-medium text-slate-200">Frame Interpolation</span>
            </div>
            <input type="checkbox" defaultChecked className="accent-indigo-500" />
          </div>
          <div className="space-y-2 text-xs">
            <div>
              <label className="block text-[10px] text-slate-400 mb-1">Modell</label>
              <select className="w-full rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-slate-200">
                <option>RIFE v4.26 (Heavy)</option>
                <option>GMFSS Fortuna</option>
              </select>
            </div>
            <div>
              <label className="block text-[10px] text-slate-400 mb-1">FPS Multiplikator</label>
              <div className="flex space-x-2">
                <button className="flex-1 rounded-md bg-indigo-600 py-1 text-center font-medium text-white">
                  2x (48fps)
                </button>
                <button className="flex-1 rounded-md bg-slate-800 py-1 text-center text-slate-400 hover:bg-slate-700">
                  3x
                </button>
                <button className="flex-1 rounded-md bg-slate-800 py-1 text-center text-slate-400 hover:bg-slate-700">
                  4x
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="text-indigo-400">🔍</span>
              <span className="text-xs font-medium text-slate-200">AI Upscaling</span>
            </div>
            <input type="checkbox" defaultChecked className="accent-indigo-500" />
          </div>
          <div className="space-y-2 text-xs">
            <div>
              <label className="block text-[10px] text-slate-400 mb-1">Modell</label>
              <select className="w-full rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-slate-200">
                <option>SPAN (Fast Anime)</option>
                <option>Real-ESRGAN x4+</option>
              </select>
            </div>
          </div>
        </div>

        <button className="w-full rounded-xl bg-indigo-600 py-2.5 font-medium text-xs text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 transition">
          Sample vorschauen (15s)
        </button>
      </div>
    </aside>
  );
}
