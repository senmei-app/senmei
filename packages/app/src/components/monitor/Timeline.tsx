/// Bottom scrubber with the sample-window in/out highlight.
export default function Timeline({
  tlMin,
  tlMax,
  tlPos,
  scrubPct,
  inPct,
  outPct,
  onScrub,
  disabled,
}: {
  tlMin: number;
  tlMax: number;
  tlPos: number;
  scrubPct: number;
  inPct: number;
  outPct: number;
  onScrub: (ms: number) => void;
  disabled: boolean;
}) {
  return (
    <div className="relative">
      <div className="pointer-events-none absolute top-1/2 z-0 h-1.5 w-full -translate-y-1/2 rounded-full bg-slate-300 dark:bg-slate-700" />
      <div
        className="pointer-events-none absolute top-1/2 z-0 h-1.5 -translate-y-1/2 rounded-full bg-indigo-300 dark:bg-indigo-400/60"
        style={{ width: `${scrubPct}%` }}
      />
      {!disabled && (
        <div
          className="pointer-events-none absolute top-1/2 z-0 h-1.5 -translate-y-1/2 rounded-full bg-indigo-600 ring-1 ring-indigo-400 dark:bg-indigo-500 dark:ring-indigo-300"
          style={{ left: `${inPct}%`, width: `${Math.max(0, outPct - inPct)}%` }}
        />
      )}
      <input
        type="range"
        min={tlMin}
        max={tlMax}
        step={50}
        value={Math.min(Math.max(tlPos, tlMin), tlMax)}
        onChange={(e) => onScrub(Number(e.target.value))}
        disabled={disabled}
        className="scrubber relative z-10 w-full cursor-ew-resize"
      />
    </div>
  );
}
