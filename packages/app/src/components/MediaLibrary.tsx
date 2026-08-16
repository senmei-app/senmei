export default function MediaLibrary() {
  return (
    <aside className="flex h-full flex-col border-r border-slate-800/80 bg-slate-900/30 p-3">
      <div className="mb-3 flex items-center justify-between px-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Medien-Bibliothek
        </span>
        <button className="rounded-md bg-slate-800 p-1 text-slate-300 hover:bg-slate-700">+</button>
      </div>

      <div className="mb-4 flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-700/80 bg-slate-900/40 p-4 text-center transition hover:border-indigo-500/50 hover:bg-slate-900/80 cursor-pointer">
        <div className="mb-2 rounded-full bg-indigo-500/10 p-2 text-indigo-400">
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <p className="text-xs font-medium text-slate-300">Video hier ablegen</p>
        <p className="mt-1 text-[10px] text-slate-500">MP4, MKV, MOV bis 4K</p>
      </div>

      <div className="space-y-2 overflow-y-auto">
        <div className="group flex cursor-pointer items-center space-x-3 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-2 transition">
          <div className="h-10 w-14 shrink-0 overflow-hidden rounded bg-slate-800 relative">
            <div className="absolute inset-0 bg-slate-700/50 flex items-center justify-center text-[9px] text-slate-300">
              THUMB
            </div>
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium text-slate-100">jujutsu_kaisen_op.mkv</p>
            <div className="mt-0.5 flex items-center space-x-1.5 text-[10px] text-slate-400">
              <span className="rounded bg-slate-800 px-1">1080p</span>
              <span>•</span>
              <span>24 fps</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
