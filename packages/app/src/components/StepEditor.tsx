// Per-step parameter editor (the body of an expanded step card in the
// Inspector). Split out of Inspector.tsx to keep that file small.

import type { ReactNode, RefObject } from "react";
import type { ModelMetadata } from "@senmei/bridge";
import { useI18n } from "../i18n";
import {
  QUALITY_PRESETS,
  buildEncoderArgs,
  qualityKey,
  type PipelineStep,
  type StepParams,
} from "../steps";

const inputCls =
  "w-full rounded-lg border border-slate-300 bg-white p-1.5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200";
const segBtn = (active: boolean) =>
  active
    ? "flex-1 rounded-md bg-indigo-600 py-1 text-center font-medium text-white"
    : "flex-1 rounded-md bg-slate-200 py-1 text-center text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700";

export interface StepEditorProps {
  step: PipelineStep;
  outputDir?: string | null;
  interpolateModels: ModelMetadata[];
  upscaleModels: ModelMetadata[];
  denoiseModels: ModelMetadata[];
  deblurModels: ModelMetadata[];
  decompressModels: ModelMetadata[];
  downloading: string | null;
  dlPct: number;
  dlError: string | null;
  folderMenu: string | null;
  setFolderMenu: (id: string | null) => void;
  folderMenuRef: RefObject<HTMLDivElement>;
  recentFolders: string[];
  updateParams: (id: string, params: Partial<StepParams>) => void;
  pickOutputFolder: (id: string) => void;
  setFolderMode: (id: string, mode: "input" | "global" | "custom", folder?: string) => void;
  downloadWeights: (modelId: string) => void;
}

export default function StepEditor(props: StepEditorProps) {
  const { t } = useI18n();
  const {
    step: s,
    outputDir,
    interpolateModels,
    upscaleModels,
    denoiseModels,
    deblurModels,
    decompressModels,
    downloading,
    dlPct,
    dlError,
    folderMenu,
    setFolderMenu,
    folderMenuRef,
    recentFolders,
    updateParams,
    pickOutputFolder,
    setFolderMode,
    downloadWeights,
  } = props;

  const field = (label: string, children: ReactNode) => (
    <div>
      <label className="mb-1 block text-[10px] text-slate-500 dark:text-slate-400">{label}</label>
      {children}
    </div>
  );

  // Usable (loadable) models first, then family → scale → id.
  const sortModels = (a: ModelMetadata, b: ModelMetadata) =>
    Number(b.loadable) - Number(a.loadable) ||
    (a.family ?? "").localeCompare(b.family ?? "") ||
    (a.scale ?? 1) - (b.scale ?? 1) ||
    a.id.localeCompare(b.id);

  const modelSelect = (models: ModelMetadata[], value: string | null | undefined, onValue: (id: string) => void) => (
    <select value={value ?? ""} onChange={(e) => onValue(e.target.value)} className={inputCls}>
      <option value="">—</option>
      {[...models].sort(sortModels).map((m) => (
        <option key={m.id} value={m.id} disabled={!m.loadable}>
          {m.family ? `${m.family} · ` : ""}
          {m.id} {(m.scale ?? 1) > 1 ? `x${m.scale}` : ""}
          {!m.loadable ? ` (${t("up.notLoadable")})` : ""}
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

  switch (s.stepType) {
    case "interpolation": {
      const m = interpolateModels.find((x) => x.id === s.params?.modelId);
      return (
        <>
          {field(
            t("fi.model"),
            modelSelect(interpolateModels, s.params?.modelId, (id) => {
              const sel = interpolateModels.find((x) => x.id === id);
              updateParams(s.id, { modelId: id });
              if (sel && sel.loadable && !sel.downloaded && sel.download_url) downloadWeights(id);
            }),
          )}
          {m?.loadable && !m.downloaded && m.download_url && (
            <button
              onClick={() => downloadWeights(m.id)}
              disabled={!!downloading}
              className="w-full rounded-md border border-indigo-500/40 bg-indigo-600/20 py-1 text-[11px] font-medium text-indigo-600 hover:bg-indigo-600/30 disabled:opacity-40 dark:text-indigo-300"
            >
              {downloading === m.id ? `${t("up.download")} … ${dlPct}%` : t("up.download")}
            </button>
          )}
          {dlError && <p className="text-[10px] text-rose-500">{dlError}</p>}
          {field(
            t("fi.fps"),
            segButtons([2, 3, 4], s.params?.fpsMultiplier, (v) => updateParams(s.id, { fpsMultiplier: v })),
          )}
        </>
      );
    }
    case "upscale": {
      const m = upscaleModels.find((x) => x.id === s.params?.modelId);
      return (
        <>
          {field(
            t("up.model"),
            modelSelect(upscaleModels, s.params?.modelId, (id) => {
              const sel = upscaleModels.find((x) => x.id === id);
              updateParams(s.id, {
                modelId: id,
                scale: s.params?.scale ?? sel?.scale ?? 2,
              });
              if (sel && sel.loadable && !sel.downloaded && sel.download_url) downloadWeights(id);
            }),
          )}
          {m?.loadable && !m.downloaded && m.download_url && (
            <button
              onClick={() => downloadWeights(m.id)}
              disabled={!!downloading}
              className="w-full rounded-md border border-indigo-500/40 bg-indigo-600/20 py-1 text-[11px] font-medium text-indigo-600 hover:bg-indigo-600/30 disabled:opacity-40 dark:text-indigo-300"
            >
              {downloading === m.id ? `${t("up.download")} … ${dlPct}%` : t("up.download")}
            </button>
          )}
          {dlError && <p className="text-[10px] text-rose-500">{dlError}</p>}
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
    case "denoise": {
      const m = denoiseModels.find((x) => x.id === s.params?.modelId);
      return (
        <>
          {field(
            t("fi.model"),
            modelSelect(denoiseModels, s.params?.modelId, (id) => {
              const sel = denoiseModels.find((x) => x.id === id);
              updateParams(s.id, { modelId: id });
              if (sel && sel.loadable && !sel.downloaded && sel.download_url) downloadWeights(id);
            }),
          )}
          {m?.loadable && !m.downloaded && m.download_url && (
            <button
              onClick={() => downloadWeights(m.id)}
              disabled={!!downloading}
              className="w-full rounded-md border border-indigo-500/40 bg-indigo-600/20 py-1 text-[11px] font-medium text-indigo-600 hover:bg-indigo-600/30 disabled:opacity-40 dark:text-indigo-300"
            >
              {downloading === m.id ? `${t("up.download")} … ${dlPct}%` : t("up.download")}
            </button>
          )}
          {dlError && <p className="text-[10px] text-rose-500">{dlError}</p>}
          {field(
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
          )}
        </>
      );
    }
    case "decompress": {
      const m = decompressModels.find((x) => x.id === s.params?.modelId);
      return (
        <>
          {field(
            t("fi.model"),
            modelSelect(decompressModels, s.params?.modelId, (id) => {
              const sel = decompressModels.find((x) => x.id === id);
              updateParams(s.id, { modelId: id });
              if (sel && sel.loadable && !sel.downloaded && sel.download_url) downloadWeights(id);
            }),
          )}
          {m?.loadable && !m.downloaded && m.download_url && (
            <button
              onClick={() => downloadWeights(m.id)}
              disabled={!!downloading}
              className="w-full rounded-md border border-indigo-500/40 bg-indigo-600/20 py-1 text-[11px] font-medium text-indigo-600 hover:bg-indigo-600/30 disabled:opacity-40 dark:text-indigo-300"
            >
              {downloading === m.id ? `${t("up.download")} … ${dlPct}%` : t("up.download")}
            </button>
          )}
          {dlError && <p className="text-[10px] text-rose-500">{dlError}</p>}
        </>
      );
    }
    case "deblur": {
      const m = deblurModels.find((x) => x.id === s.params?.modelId);
      return (
        <>
          {field(
            t("fi.model"),
            modelSelect(deblurModels, s.params?.modelId, (id) => {
              const sel = deblurModels.find((x) => x.id === id);
              updateParams(s.id, { modelId: id });
              if (sel && sel.loadable && !sel.downloaded && sel.download_url) downloadWeights(id);
            }),
          )}
          {m?.loadable && !m.downloaded && m.download_url && (
            <button
              onClick={() => downloadWeights(m.id)}
              disabled={!!downloading}
              className="w-full rounded-md border border-indigo-500/40 bg-indigo-600/20 py-1 text-[11px] font-medium text-indigo-600 hover:bg-indigo-600/30 disabled:opacity-40 dark:text-indigo-300"
            >
              {downloading === m.id ? `${t("up.download")} … ${dlPct}%` : t("up.download")}
            </button>
          )}
          {dlError && <p className="text-[10px] text-rose-500">{dlError}</p>}
          {field(
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
          )}
        </>
      );
    }
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
    case "filter":
      return (
        <>
          {field(
            t("filter.graph"),
            <input
              type="text"
              value={s.params?.filter ?? ""}
              placeholder="hue=h=10,unsharp"
              onChange={(e) => updateParams(s.id, { filter: e.target.value })}
              className={inputCls}
            />,
          )}
          <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">{t("filter.hint")}</p>
        </>
      );
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
            {field(
              t("output.tonemap"),
              <select
                value={s.params?.tonemap ?? "auto"}
                onChange={(e) => updateParams(s.id, { tonemap: e.target.value })}
                className={inputCls}
              >
                {["auto", "always", "off"].map((x) => (
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
              {`ffmpeg -y -i input.mp4 ${previewArgs.join(" ") || "(defaults)"} output.${s.params?.container ?? "mkv"}`}
            </pre>,
          )}
        </>
      );
    }
    default:
      return <div className="text-xs text-slate-500">{t(`tab.${s.stepType}.empty`)}</div>;
  }
}
