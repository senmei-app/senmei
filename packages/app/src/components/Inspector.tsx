import { useEffect, useRef, useState, type ReactNode } from "react";
import { isTauri, Channel } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { downloadModel, listModels, type DownloadProgress, type ModelMetadata } from "@senmei/bridge";
import { demoDownloadModel, demoModels } from "../mock";
import { useI18n } from "../i18n";
import { QUALITY_PRESETS, STEP_META, STEP_ORDER, buildEncoderArgs, createStep, qualityKey, type PipelineStep, type StepType } from "../steps";

const inputCls =
  "w-full rounded-lg border border-slate-300 bg-white p-1.5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200";
const segBtn = (active: boolean) =>
  active
    ? "flex-1 rounded-md bg-indigo-600 py-1 text-center font-medium text-white"
    : "flex-1 rounded-md bg-slate-200 py-1 text-center text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700";

let demoFolderN = 0;

export default function Inspector({
  steps,
  outputDir,
  onChange,
}: {
  steps: PipelineStep[];
  outputDir?: string | null;
  onChange: (steps: PipelineStep[]) => void;
}) {
  const { t } = useI18n();
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [expanded, setExpanded] = useState<string | null>(steps[0]?.id ?? null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [overIndex, setOverIndex] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const dragIndexRef = useRef<number | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [folderMenu, setFolderMenu] = useState<string | null>(null);
  const folderMenuRef = useRef<HTMLDivElement>(null);
  const [recentFolders, setRecentFolders] = useState<string[]>([]);

  // Close the add-step menu on outside click without swallowing the click,
  // so the clicked step still expands below.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [menuOpen]);

  // Close the folder quick-select menu on outside click.
  useEffect(() => {
    if (!folderMenu) return;
    const onDown = (e: MouseEvent) => {
      if (folderMenuRef.current && !folderMenuRef.current.contains(e.target as Node)) setFolderMenu(null);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [folderMenu]);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [dlPct, setDlPct] = useState(0);

  useEffect(() => {
    if (!isTauri()) {
      setModels(demoModels);
      return;
    }
    listModels().then(setModels).catch(() => {});
  }, []);

  // Fill default models once the catalog loads (only when a step has none yet).
  useEffect(() => {
    if (models.length === 0) return;
    let changed = false;
    const next = steps.map((s) => {
      if (s.params?.modelId) return s;
      if (s.stepType === "upscale") {
        const m = models.find((x) => x.kind === "upscale" && x.loadable);
        if (m) {
          changed = true;
          return { ...s, params: { ...s.params, modelId: m.id, scale: s.params?.scale ?? m.scale ?? 2 } };
        }
      } else if (s.stepType === "interpolation") {
        const m = models.find((x) => x.kind === "interpolate");
        if (m) {
          changed = true;
          return { ...s, params: { ...s.params, modelId: m.id } };
        }
      }
      return s;
    });
    if (changed) onChange(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);

  const interpolateModels = models.filter((m) => m.kind === "interpolate");
  const upscaleModels = models.filter((m) => m.kind === "upscale");

  const update = (id: string, patch: Partial<PipelineStep>) =>
    onChange(steps.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  const updateParams = (id: string, params: Partial<PipelineStep["params"]>) =>
    onChange(steps.map((s) => (s.id === id ? { ...s, params: { ...s.params, ...params } } : s)));

  const addStep = (type: StepType) => {
    const step = createStep(type);
    onChange([...steps, step]);
    setExpanded(step.id);
    setMenuOpen(false);
  };

  const removeStep = (id: string) => {
    const next = steps.filter((s) => s.id !== id);
    onChange(next);
    if (expanded === id) setExpanded(next[0]?.id ?? null);
  };

  const toggleStep = (id: string) => {
    const s = steps.find((x) => x.id === id);
    if (s) update(id, { enabled: !s.enabled });
  };

  const dragStartRef = useRef<{ x: number; y: number; index: number } | null>(null);
  const didDragRef = useRef(false);

  // Pointer-based drag on the whole step header: WebKitGTK handles HTML5 DnD
  // unreliably and this avoids the huge drag ghost. A small movement threshold
  // separates a click (expand) from a drag (reorder).
  useEffect(() => {
    const cardAt = (x: number, y: number): number | null => {
      const el = document.elementFromPoint(x, y);
      const card = el?.closest<HTMLElement>("[data-step-index]");
      return card ? Number(card.dataset.stepIndex) : null;
    };
    const onMove = (e: MouseEvent) => {
      const start = dragStartRef.current;
      if (!start) return;
      if (!dragging) {
        if (Math.hypot(e.clientX - start.x, e.clientY - start.y) < 4) return;
        dragIndexRef.current = start.index;
        setDragIndex(start.index);
        setDragging(true);
      } else {
        const i = cardAt(e.clientX, e.clientY);
        if (i !== null && i !== overIndex) setOverIndex(i);
      }
    };
    const onUp = (e: MouseEvent) => {
      if (dragging) {
        const target = cardAt(e.clientX, e.clientY);
        const from = dragIndexRef.current;
        if (from !== null && target !== null && target !== from) {
          const next = [...steps];
          const [moved] = next.splice(from, 1);
          next.splice(target, 0, moved);
          onChange(next);
        }
        didDragRef.current = true; // suppress the header click following a drag
        setTimeout(() => {
          didDragRef.current = false;
        }, 0);
      }
      dragStartRef.current = null;
      dragIndexRef.current = null;
      setDragIndex(null);
      setOverIndex(null);
      setDragging(false);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging, overIndex, steps, onChange]);

  const handleHeaderMouseDown = (i: number) => (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    dragStartRef.current = { x: e.clientX, y: e.clientY, index: i };
  };

  const handleHeaderClick = (id: string) => () => {
    if (didDragRef.current) {
      didDragRef.current = false;
      return;
    }
    setExpanded((cur) => (cur === id ? null : id));
  };

  const rememberFolder = (folder: string) =>
    setRecentFolders((r) => [folder, ...r.filter((x) => x !== folder)].slice(0, 6));

  const pickOutputFolder = async (id: string) => {
    let dir: string | null = null;
    if (!isTauri()) {
      // Browser demo: no native picker; prompt may be blocked (headless), so fall back to a demo path.
      try {
        dir = window.prompt(t("output.pick"));
      } catch {
        dir = null;
      }
      if (!dir) dir = `/demo/output${demoFolderN++ || ""}`;
    } else {
      const picked = await open({ directory: true });
      dir = typeof picked === "string" ? picked : null;
    }
    if (!dir) return;
    updateParams(id, { outputMode: "custom", outputFolder: dir });
    rememberFolder(dir);
    setFolderMenu(null);
  };

  const setFolderMode = (id: string, mode: "input" | "global" | "custom", folder?: string) => {
    updateParams(id, folder === undefined ? { outputMode: mode } : { outputMode: mode, outputFolder: folder });
    if (mode === "custom" && folder) rememberFolder(folder);
    setFolderMenu(null);
  };

  const downloadWeights = (modelId: string) => {
    if (downloading) return;
    setDownloading(modelId);
    setDlPct(0);
    if (!isTauri()) {
      demoDownloadModel((p) => setDlPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0))
        .catch((e) => console.error(e))
        .finally(() => setDownloading(null));
      return;
    }
    const ch = new Channel<DownloadProgress>();
    ch.onmessage = (p) => setDlPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0);
    downloadModel(modelId, ch)
      .catch((e) => console.error(e))
      .finally(() => setDownloading(null));
  };

  const modelSelect = (models: ModelMetadata[], value: string | null | undefined, onValue: (id: string) => void) => (
    <select value={value ?? ""} onChange={(e) => onValue(e.target.value)} className={inputCls}>
      <option value="">—</option>
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.id} {(m.scale ?? 1) > 1 ? `x${m.scale}` : ""}
        </option>
      ))}
    </select>
  );

  const segButtons = (options: number[], value: number | null | undefined, onValue: (v: number | null) => void) => (
    <div className="flex space-x-2">
      {options.map((o) => (
        <button key={o} onClick={() => onValue(value === o ? null : o)} className={segBtn(value === o)}>
          {o}x
        </button>
      ))}
    </div>
  );

  const field = (label: string, children: ReactNode) => (
    <div>
      <label className="mb-1 block text-[10px] text-slate-500 dark:text-slate-400">{label}</label>
      {children}
    </div>
  );

  const renderParams = (s: PipelineStep) => {
    switch (s.stepType) {
      case "interpolation":
        return (
          <>
            {field(
              t("fi.model"),
              modelSelect(interpolateModels, s.params?.modelId, (id) => updateParams(s.id, { modelId: id })),
            )}
            {field(
              t("fi.fps"),
              segButtons([2, 3, 4], s.params?.fpsMultiplier, (v) => updateParams(s.id, { fpsMultiplier: v })),
            )}
          </>
        );
      case "upscale": {
        const m = upscaleModels.find((x) => x.id === s.params?.modelId);
        return (
          <>
            {field(
              t("up.model"),
              modelSelect(upscaleModels, s.params?.modelId, (id) =>
                updateParams(s.id, {
                  modelId: id,
                  scale: s.params?.scale ?? upscaleModels.find((x) => x.id === id)?.scale ?? 2,
                }),
              ),
            )}
            {m?.loadable && (
              <button
                onClick={() => downloadWeights(m.id)}
                disabled={!!downloading}
                className="w-full rounded-md border border-indigo-500/40 bg-indigo-600/20 py-1 text-[11px] font-medium text-indigo-600 hover:bg-indigo-600/30 disabled:opacity-40 dark:text-indigo-300"
              >
                {downloading === m.id ? `${t("up.download")} … ${dlPct}%` : t("up.download")}
              </button>
            )}
            {field(
              t("up.scale"),
              segButtons([2, 3, 4], s.params?.scale, (v) => updateParams(s.id, { scale: v ?? 2 })),
            )}
          </>
        );
      }
      case "resize":
        return field(
          t("resize.factor"),
          <input
            type="text"
            inputMode="decimal"
            value={s.params?.factor ?? ""}
            placeholder="1.0"
            onChange={(e) => updateParams(s.id, { factor: e.target.value.replace(",", ".") })}
            className={inputCls}
          />,
        );
      case "denoise":
        return field(
          t("denoise.radius"),
          <select
            value={s.params?.radius ?? 1}
            onChange={(e) => updateParams(s.id, { radius: Number(e.target.value) })}
            className={inputCls}
          >
            {[1, 2, 3, 4].map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>,
        );
      case "deblur":
        return field(
          t("deblur.amount"),
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={s.params?.amount ?? 0.5}
            onChange={(e) => updateParams(s.id, { amount: Number(e.target.value) })}
            className="w-full accent-indigo-500"
          />,
        );
      case "deduplication": {
        const threshold = s.params?.threshold ?? 0.02;
        const presets = [
          { label: t("dedup.off"), v: 0 },
          { label: t("dedup.standard"), v: 0.02 },
          { label: t("dedup.aggressive"), v: 0.04 },
        ];
        return (
          <>
            {field(
              t("dedup.mode"),
              <div className="flex space-x-2">
                {presets.map((p) => (
                  <button
                    key={p.label}
                    onClick={() => updateParams(s.id, { threshold: p.v })}
                    className={segBtn(threshold === p.v)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>,
            )}
            {field(
              t("dedup.threshold"),
              <div>
                <input
                  type="range"
                  min={0}
                  max={0.05}
                  step={0.002}
                  value={threshold}
                  onChange={(e) => updateParams(s.id, { threshold: Number(e.target.value) })}
                  className="w-full accent-indigo-500"
                />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500 dark:text-slate-400">
                  <span>0%</span>
                  <span>{(threshold * 100).toFixed(1)}%</span>
                  <span>5%</span>
                </div>
                <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">{t("dedup.hint")}</p>
              </div>,
            )}
          </>
        );
      }
      case "output": {
        const mode = s.params?.outputMode ?? "input";
        const quality = qualityKey(s.params);
        const previewArgs = buildEncoderArgs(s.params, s.params?.ffmpegArgs ?? "");
        const applyQuality = (q: string) => {
          const prof = QUALITY_PRESETS[q];
          if (prof) updateParams(s.id, { quality: q, crf: prof.crf, preset: prof.preset });
          else updateParams(s.id, { quality: "Custom" });
        };
        return (
          <>
            {field(
              t("output.label"),
              <input
                type="text"
                value={s.params?.label ?? ""}
                onChange={(e) => updateParams(s.id, { label: e.target.value })}
                className={inputCls}
              />,
            )}
            {field(
              t("output.format"),
              <select
                value={s.params?.container ?? "mkv"}
                onChange={(e) => updateParams(s.id, { container: e.target.value })}
                className={inputCls}
              >
                {["mp4", "mkv", "webm", "mov"].map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>,
            )}
            {field(
              t("output.folder"),
              <div className="relative flex items-center" ref={folderMenuRef}>
                <input
                  type="text"
                  readOnly
                  value={
                    mode === "input"
                      ? t("output.folder.input")
                      : mode === "global"
                        ? (outputDir ?? t("output.folder.global"))
                        : (s.params?.outputFolder || t("output.folder.choose"))
                  }
                  title={
                    mode === "input"
                      ? t("output.folder.input")
                      : mode === "global"
                        ? (outputDir ?? t("output.folder.global"))
                        : (s.params?.outputFolder || "")
                  }
                  onClick={() => pickOutputFolder(s.id)}
                  className="w-full cursor-pointer truncate rounded-lg border border-slate-300 bg-white py-1.5 pl-3 pr-16 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
                />
                <div className="absolute right-1 flex items-center space-x-1">
                  <button
                    title={t("output.folder.recent")}
                    onClick={() => setFolderMenu(folderMenu === s.id ? null : s.id)}
                    className="flex h-6 w-6 items-center justify-center rounded-md border border-slate-300 bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200"
                  >
                    ▾
                  </button>
                  <button
                    title={t("output.folder.browse")}
                    onClick={() => pickOutputFolder(s.id)}
                    className="flex h-6 w-7 items-center justify-center rounded-md bg-indigo-600 text-white shadow-sm shadow-indigo-600/30 hover:bg-indigo-500"
                  >
                    📂
                  </button>
                </div>
                {folderMenu === s.id && (
                  <div className="absolute right-0 top-full z-20 mt-1 w-64 rounded-lg border border-slate-300 bg-white py-1 shadow-lg dark:border-slate-700 dark:bg-slate-900">
                    {[
                      { mode: "input" as const, label: t("output.folder.input") },
                      { mode: "global" as const, label: t("output.folder.global") },
                    ].map((o) => (
                      <button
                        key={o.mode}
                        onClick={() => setFolderMode(s.id, o.mode)}
                        className="block w-full truncate px-3 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
                      >
                        {o.label}
                      </button>
                    ))}
                    {recentFolders.length > 0 && <div className="my-1 border-t border-slate-200 dark:border-slate-700" />}
                    {recentFolders.map((f) => (
                      <button
                        key={f}
                        onClick={() => setFolderMode(s.id, "custom", f)}
                        className="block w-full truncate px-3 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
                      >
                        {f}
                      </button>
                    ))}
                    <div className="my-1 border-t border-slate-200 dark:border-slate-700" />
                    <button
                      onClick={() => pickOutputFolder(s.id)}
                      className="block w-full truncate px-3 py-1.5 text-left text-indigo-600 hover:bg-slate-100 dark:text-indigo-400 dark:hover:bg-slate-800"
                    >
                      {t("output.folder.choose")}
                    </button>
                  </div>
                )}
              </div>,
            )}
            {field(
              t("output.quality"),
              <select value={quality} onChange={(e) => applyQuality(e.target.value)} className={inputCls}>
                {[...Object.keys(QUALITY_PRESETS), "Custom"].map((q) => (
                  <option key={q} value={q}>
                    {q}
                  </option>
                ))}
              </select>,
            )}
            {field(
              t("output.videoCodec"),
              <select
                value={s.params?.videoCodec ?? "H.264"}
                onChange={(e) => updateParams(s.id, { videoCodec: e.target.value })}
                className={inputCls}
              >
                <option>H.264</option>
                <option>H.265</option>
                <option>AV1</option>
                <option>VP9</option>
              </select>,
            )}
            {field(
              t("output.preset"),
              <select
                value={s.params?.preset ?? "medium"}
                onChange={(e) => updateParams(s.id, { preset: e.target.value })}
                className={inputCls}
              >
                {["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"].map((p) => (
                  <option key={p}>{p}</option>
                ))}
              </select>,
            )}
            {field(
              t("output.crf"),
              <input
                type="number"
                min={0}
                max={51}
                value={s.params?.crf ?? 20}
                onChange={(e) => updateParams(s.id, { crf: Number(e.target.value) })}
                className={inputCls}
              />,
            )}
            {field(
              t("output.pixFmt"),
              <select
                value={s.params?.pixFmt ?? "yuv420p"}
                onChange={(e) => updateParams(s.id, { pixFmt: e.target.value })}
                className={inputCls}
              >
                {["yuv420p", "yuv420p10le", "yuv444p", "yuv444p10le"].map((p) => (
                  <option key={p}>{p}</option>
                ))}
              </select>,
            )}
            {field(
              t("output.tune"),
              <select
                value={s.params?.tune ?? ""}
                onChange={(e) => updateParams(s.id, { tune: e.target.value })}
                className={inputCls}
              >
                <option value="">—</option>
                {["film", "animation", "grain", "fastdecode", "zerolatency"].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>,
            )}
            <div className="border-t border-slate-200 pt-2 dark:border-slate-700/60">
              <label className="mb-1 block text-[10px] font-semibold text-slate-500 dark:text-slate-400">
                {t("output.color")}
              </label>
              {field(
                t("output.colorPrimaries"),
                <select
                  value={s.params?.colorPrimaries ?? ""}
                  onChange={(e) => updateParams(s.id, { colorPrimaries: e.target.value })}
                  className={inputCls}
                >
                  <option value="">—</option>
                  {["bt709", "bt2020"].map((x) => (
                    <option key={x} value={x}>{x}</option>
                  ))}
                </select>,
              )}
              {field(
                t("output.colorTransfer"),
                <select
                  value={s.params?.colorTransfer ?? ""}
                  onChange={(e) => updateParams(s.id, { colorTransfer: e.target.value })}
                  className={inputCls}
                >
                  <option value="">—</option>
                  {["bt709", "smpte2084", "arib-std-b67", "gamma22"].map((x) => (
                    <option key={x} value={x}>{x}</option>
                  ))}
                </select>,
              )}
              {field(
                t("output.colorMatrix"),
                <select
                  value={s.params?.colorMatrix ?? ""}
                  onChange={(e) => updateParams(s.id, { colorMatrix: e.target.value })}
                  className={inputCls}
                >
                  <option value="">—</option>
                  {["bt709", "bt2020nc", "bt2020c"].map((x) => (
                    <option key={x} value={x}>{x}</option>
                  ))}
                </select>,
              )}
            </div>
            {field(
              t("output.audio"),
              <select
                value={s.params?.audioCodec ?? "Passthrough"}
                onChange={(e) => updateParams(s.id, { audioCodec: e.target.value })}
                className={inputCls}
              >
                <option>Passthrough</option>
                <option>AAC</option>
                <option>Opus</option>
                <option>FLAC</option>
              </select>,
            )}
            {field(
              t("subtitle.mode"),
              <select
                value={s.params?.subtitleMode ?? "None"}
                onChange={(e) => updateParams(s.id, { subtitleMode: e.target.value })}
                className={inputCls}
              >
                <option>None</option>
                <option>Copy</option>
                <option>HardSub</option>
                <option>SoftSub</option>
              </select>,
            )}
            {field(
              t("output.ffmpeg"),
              <textarea
                value={s.params?.ffmpegArgs ?? ""}
                rows={2}
                placeholder="-c:v libx265 -crf 18 -preset medium -pix_fmt yuv420p10le"
                onChange={(e) => updateParams(s.id, { ffmpegArgs: e.target.value })}
                className={`${inputCls} font-mono text-[10px]`}
              />,
            )}
            {field(
              t("output.preview"),
              <pre className="whitespace-pre-wrap break-all rounded-lg border border-slate-200 bg-slate-50 p-2 font-mono text-[9px] leading-4 text-slate-600 dark:border-slate-700 dark:bg-slate-950/80 dark:text-slate-400">
                {`ffmpeg -y -i input.mp4 ${previewArgs || "(defaults)"} output.${s.params?.container ?? "mkv"}`}
              </pre>,
            )}
          </>
        );
      }
      default:
        return <div className="text-xs text-slate-500">{t(`tab.${s.stepType}.empty`)}</div>;
    }
  };

  return (
    <aside className="h-full w-full overflow-y-auto border-l border-slate-200 bg-slate-100/70 p-4 dark:border-slate-800/80 dark:bg-slate-900/30">
      <div className="mb-4">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          {t("stack.title")}
        </h2>
        <p className="text-[10px] text-slate-500">{t("stack.subtitle")}</p>
      </div>

      {steps.length === 0 && (
        <div className="rounded-xl border border-dashed border-slate-300 p-4 text-center text-xs text-slate-500 dark:border-slate-700">
          {t("stack.empty")}
        </div>
      )}

      <div className="space-y-1">
        {steps.map((s, i) => {
          const meta = STEP_META[s.stepType as StepType];
          const isOpen = expanded === s.id;
          return (
            <div key={s.id}>
              <div
                data-step-index={i}
                className={
                  (s.enabled
                    ? "rounded-xl border border-indigo-500/40 bg-indigo-500/[0.06] dark:bg-indigo-500/10"
                    : "rounded-xl border border-slate-200 bg-white/60 opacity-60 dark:border-slate-800 dark:bg-slate-950/60") +
                  (dragIndex === i ? " opacity-40" : "") +
                  (overIndex === i && dragIndex !== null && overIndex !== dragIndex
                    ? " ring-2 ring-indigo-400"
                    : "")
                }
              >
                <div
                  className="flex cursor-pointer select-none items-center justify-between p-2.5"
                  onMouseDown={handleHeaderMouseDown(i)}
                  onClick={handleHeaderClick(s.id)}
                >
                  <div className="flex items-center space-x-1.5 text-left">
                    <span
                      title={t("stack.drag")}
                      className="cursor-grab select-none text-xs leading-none text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                    >
                      ≡
                    </span>
                    <span className="text-indigo-500 dark:text-indigo-400">{meta.icon}</span>
                    <div className="flex items-center space-x-1.5">
                      <span className="text-xs font-medium text-slate-800 dark:text-slate-200">
                        {i + 1}. {t(meta.labelKey)}
                        {s.stepType === "upscale" &&
                          (() => {
                            const m = models.find((x) => x.id === s.params?.modelId);
                            const sc = s.params?.scale;
                            return m && sc ? ` · ${m.id} ×${sc}` : "";
                          })()}
                        {s.stepType === "interpolation" && s.params?.fpsMultiplier
                          ? ` ×${s.params.fpsMultiplier}`
                          : ""}
                      </span>
                      {s.stepType === "output" && s.params?.label && (
                        <span className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[9px] text-indigo-600 dark:bg-slate-800 dark:text-indigo-400">
                          {s.params.label}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={s.enabled}
                      onClick={(e) => e.stopPropagation()}
                      onChange={() => toggleStep(s.id)}
                      className="h-[18px] w-[18px] cursor-pointer accent-indigo-500"
                    />
                    <button
                      title="remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeStep(s.id);
                      }}
                      className="text-xs font-bold text-slate-500 hover:text-rose-400"
                    >
                      ✕
                    </button>
                  </div>
                </div>
                {isOpen && (
                  <div className="space-y-2.5 border-t border-indigo-500/20 p-3 text-xs">{renderParams(s)}</div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="pt-3">
        <div ref={menuRef} className="relative z-20">
          {!menuOpen ? (
            <button
              onClick={() => setMenuOpen(true)}
              className="flex w-full items-center justify-center space-x-2 rounded-xl border border-dashed border-slate-300 bg-slate-200/50 py-2.5 text-xs font-medium text-slate-600 transition hover:border-indigo-500/50 hover:bg-slate-200 dark:border-slate-700/80 dark:bg-slate-900/40 dark:text-slate-400 dark:hover:border-indigo-500/50 dark:hover:bg-slate-900/80"
            >
              <span className="text-sm font-bold">+</span>
              <span>{t("stack.add")}</span>
            </button>
          ) : (
            <div className="w-full rounded-xl border border-dashed border-slate-300 bg-slate-200/50 p-1.5 dark:border-slate-700/80 dark:bg-slate-900/40">
              {STEP_ORDER.filter((type) => STEP_META[type].implemented).map((type) => {
                const m = STEP_META[type];
                return (
                  <button
                    key={type}
                    onClick={() => addStep(type)}
                    className="flex w-full items-center rounded-lg px-2.5 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800/80"
                  >
                    <span>
                      {m.icon} {t(m.labelKey)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

    </aside>
  );
}
