import { useEffect, useState, type ReactNode } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { listModels, type ModelMetadata } from "@senmei/bridge";
import { useI18n } from "../i18n";

type Group = "settings" | "advanced";

function Accordion({
  icon,
  title,
  children,
  placeholder,
}: {
  icon: string;
  title: string;
  children?: ReactNode;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [enabled, setEnabled] = useState(false);

  return (
    <div
      className={
        enabled
          ? "rounded-xl border border-indigo-400/60 bg-indigo-50/60 dark:border-indigo-500/40 dark:bg-indigo-500/10"
          : "rounded-xl border border-slate-200 bg-white/70 dark:border-slate-800 dark:bg-slate-900/60"
      }
    >
      <div
        onClick={() => {
          setOpen((o) => !o);
          setEnabled(true);
        }}
        className="flex cursor-pointer select-none items-center justify-between p-3"
      >
        <div className="flex items-center space-x-2 text-left">
          <span className="text-indigo-500 dark:text-indigo-400">{icon}</span>
          <span className="text-xs font-medium text-slate-800 dark:text-slate-200">{title}</span>
        </div>
        <input
          type="checkbox"
          checked={enabled}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => setEnabled(e.target.checked)}
          className="h-[18px] w-[18px] cursor-pointer accent-indigo-500"
        />
      </div>
      {open && (
        <div className="border-t border-slate-200/60 p-3 dark:border-slate-800/60">
          {children ?? <div className="text-xs text-slate-500">{placeholder}</div>}
        </div>
      )}
    </div>
  );
}

export default function Inspector({
  scale,
  onScaleChange,
  onModelChange,
  resizeFactor,
  onResizeFactorChange,
  outputResizeFactor,
  onOutputResizeFactorChange,
}: {
  scale: number;
  onScaleChange: (scale: number) => void;
  onModelChange: (modelId: string | null) => void;
  resizeFactor: string;
  onResizeFactorChange: (v: string) => void;
  outputResizeFactor: string;
  onOutputResizeFactorChange: (v: string) => void;
}) {
  const { t } = useI18n();
  const [group, setGroup] = useState<Group>("settings");
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [interpolateModel, setInterpolateModel] = useState("");
  const [upscaleModel, setUpscaleModel] = useState("");

  useEffect(() => {
    if (!isTauri()) return;
    listModels()
      .then((list) => {
        setModels(list);
        const interp = list.find((m) => m.kind === "interpolate");
        if (interp) setInterpolateModel(interp.id);
        const up = list.find((m) => m.kind === "upscale");
        if (up) {
          setUpscaleModel(up.id);
          onModelChange(up.id);
          onScaleChange(up.scale ?? 1);
        }
      })
      .catch(() => {});
  }, []);

  const interpolateModels = models.filter((m) => m.kind === "interpolate");
  const upscaleModels = models.filter((m) => m.kind === "upscale");

  const modelSelect = (models: ModelMetadata[], value: string, onValue: (id: string) => void) => (
    <div className="space-y-1">
      <select
        value={value}
        onChange={(e) => {
          const id = e.target.value;
          onValue(id);
          const m = models.find((x) => x.id === id);
          onModelChange(m ? m.id : null);
          if (m) onScaleChange(m.scale ?? 1);
        }}
        className="w-full rounded-lg border border-slate-300 bg-white p-1.5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
      >
        <option value="">—</option>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.id} {(m.scale ?? 1) > 1 ? `x${m.scale}` : ""}
          </option>
        ))}
      </select>
    </div>
  );

  const factorField = (value: string, onValue: (v: string) => void) => (
    <div>
      <label className="mb-1 block text-[10px] text-slate-500 dark:text-slate-400">
        {t("resize.factor")}
      </label>
      <input
        type="number"
        min={0.1}
        step={0.1}
        value={value}
        placeholder="1.0"
        onChange={(e) => onValue(e.target.value)}
        className="w-full rounded-lg border border-slate-300 bg-white p-1.5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
      />
    </div>
  );

  return (
    <aside className="h-full w-full overflow-y-auto border-l border-slate-200 bg-slate-100/70 p-4 dark:border-slate-800/80 dark:bg-slate-900/30">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {t("inspector.title")}
      </h2>

      <div className="mb-4 flex gap-1">
        {(["settings", "advanced"] as Group[]).map((key) => (
          <button
            key={key}
            onClick={() => setGroup(key)}
            className={
              group === key
                ? "flex-1 rounded-md border border-indigo-500/40 bg-indigo-600/30 px-2 py-1.5 text-[11px] font-medium text-indigo-600 dark:text-indigo-300"
                : "flex-1 rounded-md bg-slate-200 px-2 py-1.5 text-[11px] text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
            }
          >
            {t(`group.${key}`)}
          </button>
        ))}
      </div>

      {group === "settings" && (
        <div className="space-y-3">
          <Accordion icon="⚡" title={t("tab.interpolate")}>
            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1">{t("fi.model")}</label>
                {modelSelect(interpolateModels, interpolateModel, setInterpolateModel)}
              </div>
              <div>
                <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1">{t("fi.fps")}</label>
                <div className="flex space-x-2">
                  <button className="flex-1 rounded-md bg-indigo-600 py-1 text-center font-medium text-white">
                    2x (48fps)
                  </button>
                  <button className="flex-1 rounded-md bg-slate-200 py-1 text-center text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700">
                    3x
                  </button>
                  <button className="flex-1 rounded-md bg-slate-200 py-1 text-center text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700">
                    4x
                  </button>
                </div>
              </div>
            </div>
          </Accordion>

          <Accordion icon="📦" title={t("tab.decompress")} placeholder={t("tab.decompress.empty")} />
          <Accordion icon="🧹" title={t("tab.denoise")} placeholder={t("tab.denoise.empty")} />
          <Accordion icon="✨" title={t("tab.deblur")} placeholder={t("tab.deblur.empty")} />

          <Accordion icon="🔍" title={t("tab.upscale")}>
            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1">{t("up.model")}</label>
                {modelSelect(upscaleModels, upscaleModel, setUpscaleModel)}
              </div>
              <div>
                <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1">{t("up.scale")}</label>
                <div className="flex space-x-2">
                  {[2, 3, 4].map((s) => (
                    <button
                      key={s}
                      onClick={() => onScaleChange(s)}
                      className={
                        scale === s
                          ? "flex-1 rounded-md bg-indigo-600 py-1 text-center font-medium text-white"
                          : "flex-1 rounded-md bg-slate-200 py-1 text-center text-slate-600 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
                      }
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Accordion>

          <Accordion icon="📑" title={t("tab.dedup")} placeholder={t("tab.dedup.empty")} />
          <Accordion icon="↔️" title={t("tab.resize")}>
            {factorField(resizeFactor, onResizeFactorChange)}
          </Accordion>
          <Accordion icon="⤴️" title={t("tab.output_resize")}>
            {factorField(outputResizeFactor, onOutputResizeFactorChange)}
          </Accordion>
        </div>
      )}

      {group === "advanced" && (
        <div className="space-y-3">
          <Accordion icon="🎬" title={t("tab.enc_video")} placeholder={t("tab.enc_video.empty")} />

          <Accordion icon="🎵" title={t("tab.enc_audio")}>
            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1">{t("enc_audio.codec")}</label>
                <select className="w-full rounded-lg border border-slate-300 bg-white p-1.5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200">
                  <option>Passthrough</option>
                  <option>AAC</option>
                  <option>Opus</option>
                  <option>FLAC</option>
                </select>
              </div>
            </div>
          </Accordion>

          <Accordion icon="💬" title={t("tab.subtitle")}>
            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1">{t("subtitle.mode")}</label>
                <select className="w-full rounded-lg border border-slate-300 bg-white p-1.5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200">
                  <option>None</option>
                  <option>Copy</option>
                  <option>HardSub</option>
                  <option>SoftSub</option>
                </select>
              </div>
            </div>
          </Accordion>

          <Accordion icon="⚙️" title={t("tab.backend")} placeholder={t("tab.backend.empty")} />
        </div>
      )}

      <button className="mt-3 w-full rounded-xl bg-indigo-600 py-2.5 font-medium text-xs text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 transition">
        {t("sample.preview")}
      </button>
    </aside>
  );
}
