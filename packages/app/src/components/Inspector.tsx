import { useEffect, useRef, useState } from "react";
import type { ModelMetadata } from "@senmei/bridge";
import { backend } from "../backend";
import { useI18n } from "../i18n";
import { STEP_META, STEP_ORDER, createStep, type PipelineStep, type StepType } from "../steps";
import { deletePreset, loadPresets, savePreset, type PipelinePreset } from "../presets";
import StepEditor from "./StepEditor";

export default function Inspector({
  steps,
  outputDir,
  onChange,
  onSuggest,
}: {
  steps: PipelineStep[];
  outputDir?: string | null;
  onChange: (steps: PipelineStep[]) => void;
  onSuggest?: () => void;
}) {
  const { t } = useI18n();
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [expanded, setExpanded] = useState<string | null>(steps[0]?.id ?? null);  const [menuOpen, setMenuOpen] = useState(false);
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
  const [dlError, setDlError] = useState<string | null>(null);
  const [dlPct, setDlPct] = useState(0);
  const [presets, setPresets] = useState<PipelinePreset[]>([]);
  const [saveOpen, setSaveOpen] = useState(false);
  const [presetName, setPresetName] = useState("");

  useEffect(() => {
    setPresets(loadPresets());
  }, []);

  const applyPreset = (p: PipelinePreset) => {
    onChange(JSON.parse(JSON.stringify(p.steps)));
    setExpanded(p.steps[0]?.id ?? null);
  };

  const commitPreset = () => {
    const name = presetName.trim() || `Preset ${presets.length + 1}`;
    savePreset(name, steps);
    setPresets(loadPresets());
    setPresetName("");
    setSaveOpen(false);
  };

  const removePreset = (name: string) => {
    deletePreset(name);
    setPresets(loadPresets());
  };

  useEffect(() => {
    backend()
      .then((b) => b.listModels())
      .then(setModels)
      .catch(() => {});
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
      } else if (s.stepType === "denoise") {
        const m = models.find((x) => x.kind === "denoise" && x.loadable);
        if (m) {
          changed = true;
          return { ...s, params: { ...s.params, modelId: m.id } };
        }
      } else if (s.stepType === "deblur") {
        const m = models.find((x) => x.kind === "deblur" && x.loadable);
        if (m) {
          changed = true;
          return { ...s, params: { ...s.params, modelId: m.id } };
        }
      } else if (s.stepType === "decompress") {
        const m = models.find((x) => x.kind === "decompress" && x.loadable);
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
  const denoiseModels = models.filter((m) => m.kind === "denoise");
  const deblurModels = models.filter((m) => m.kind === "deblur");
  const decompressModels = models.filter((m) => m.kind === "decompress");

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
    const be = await backend();
    let dir: string | null = null;
    try {
      dir = await be.pickFolder(t("output.pick"));
    } catch {
      dir = null;
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
    setDlError(null);
    backend()
      .then((b) =>
        b.downloadModel(modelId, (p) =>
          setDlPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0),
        ),
      )
      .catch((e) => setDlError(String((e as Error)?.message ?? e)))
      .finally(() => setDownloading(null));
  };

  return (
    <aside className="h-full w-full overflow-y-auto border-l border-slate-200 bg-slate-100/70 p-4 dark:border-slate-800/80 dark:bg-slate-900/30">
      <div className="mb-3 rounded-xl border border-slate-200 bg-white/60 p-2 dark:border-slate-800 dark:bg-slate-900/60">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
            {t("presets.title")}
          </span>
          <div className="flex items-center gap-1">
            {onSuggest && (
              <button
                onClick={onSuggest}
                className="rounded-md px-2 py-0.5 text-[11px] text-emerald-600 hover:bg-emerald-500/10 dark:text-emerald-300"
              >
                ✨ {t("presets.suggest")}
              </button>
            )}
            <button
              onClick={() => setSaveOpen((o) => !o)}
              className="rounded-md px-2 py-0.5 text-[11px] text-indigo-600 hover:bg-indigo-500/10 dark:text-indigo-300"
            >
              {t("presets.saveCurrent")}
            </button>
          </div>
        </div>
        {saveOpen && (
          <div className="mt-1.5 flex gap-1">
            <input
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitPreset();
                if (e.key === "Escape") setSaveOpen(false);
              }}
              placeholder={t("presets.namePlaceholder")}
              className="min-w-0 flex-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-700 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
            />
            <button
              onClick={commitPreset}
              className="rounded-md bg-indigo-600 px-2.5 py-1 text-[11px] font-medium text-white"
            >
              {t("presets.save")}
            </button>
          </div>
        )}
        {presets.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {presets.map((p) => (
              <span
                key={p.name}
                className="inline-flex items-center gap-1 rounded-md bg-slate-200/70 px-2 py-0.5 text-[11px] text-slate-700 dark:bg-slate-800 dark:text-slate-300"
              >
                <button
                  onClick={() => applyPreset(p)}
                  className="hover:text-indigo-600 dark:hover:text-indigo-300"
                >
                  {p.name}
                </button>
                <button
                  onClick={() => removePreset(p.name)}
                  title={t("presets.delete")}
                  className="text-slate-400 hover:text-rose-500"
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
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
                        {s.stepType === "decompress" &&
                          (() => {
                            const m = models.find((x) => x.id === s.params?.modelId);
                            return m ? ` · ${m.id}` : "";
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
                  <div className="space-y-2.5 border-t border-indigo-500/20 p-3 text-xs">
                    <StepEditor
                      step={s}
                      outputDir={outputDir}
                      interpolateModels={interpolateModels}
                      upscaleModels={upscaleModels}
                      denoiseModels={denoiseModels}
                      deblurModels={deblurModels}
                      decompressModels={decompressModels}
                      downloading={downloading}
                      dlPct={dlPct}
                      dlError={dlError}
                      folderMenu={folderMenu}
                      setFolderMenu={setFolderMenu}
                      folderMenuRef={folderMenuRef}
                      recentFolders={recentFolders}
                      updateParams={updateParams}
                      pickOutputFolder={pickOutputFolder}
                      setFolderMode={setFolderMode}
                      downloadWeights={downloadWeights}
                    />
                  </div>
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
