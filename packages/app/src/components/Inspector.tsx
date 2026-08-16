import { useState, type ReactNode } from "react";
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
  const [enabled, setEnabled] = useState(true);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60">
      <div className="flex items-center justify-between p-3">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex items-center space-x-2 text-left"
        >
          <span className="text-indigo-400">{icon}</span>
          <span className="text-xs font-medium text-slate-200">{title}</span>
          <span className="text-xs text-slate-500">{open ? "−" : "+"}</span>
        </button>
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          className="accent-indigo-500"
        />
      </div>
      {open && (
        <div className="border-t border-slate-800/60 p-3">
          {children ?? <div className="text-xs text-slate-500">{placeholder}</div>}
        </div>
      )}
    </div>
  );
}

export default function Inspector() {
  const { t } = useI18n();
  const [group, setGroup] = useState<Group>("settings");

  return (
    <aside className="h-full w-full overflow-y-auto border-l border-slate-800/80 bg-slate-900/30 p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        {t("inspector.title")}
      </h2>

      <div className="mb-4 flex gap-1">
        {(["settings", "advanced"] as Group[]).map((key) => (
          <button
            key={key}
            onClick={() => setGroup(key)}
            className={
              group === key
                ? "flex-1 rounded-md bg-indigo-600/30 border border-indigo-500/40 px-2 py-1.5 text-[11px] font-medium text-indigo-300"
                : "flex-1 rounded-md bg-slate-800 px-2 py-1.5 text-[11px] text-slate-400 hover:bg-slate-700"
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
                <label className="block text-[10px] text-slate-400 mb-1">{t("fi.model")}</label>
                <select className="w-full rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-slate-200">
                  <option>RIFE v4.26 (Heavy)</option>
                  <option>GMFSS Fortuna</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-slate-400 mb-1">{t("fi.fps")}</label>
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
          </Accordion>

          <Accordion icon="📦" title={t("tab.decompress")} placeholder={t("tab.decompress.empty")} />
          <Accordion icon="🧹" title={t("tab.denoise")} placeholder={t("tab.denoise.empty")} />
          <Accordion icon="✨" title={t("tab.deblur")} placeholder={t("tab.deblur.empty")} />

          <Accordion icon="🔍" title={t("tab.upscale")}>
            <div className="space-y-2 text-xs">
              <div>
                <label className="block text-[10px] text-slate-400 mb-1">{t("up.model")}</label>
                <select className="w-full rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-slate-200">
                  <option>SPAN (Fast Anime)</option>
                  <option>Real-ESRGAN x4+</option>
                </select>
              </div>
            </div>
          </Accordion>

          <Accordion icon="📑" title={t("tab.dedup")} placeholder={t("tab.dedup.empty")} />
          <Accordion icon="↔️" title={t("tab.resize")} placeholder={t("tab.resize.empty")} />
          <Accordion icon="⤴️" title={t("tab.output_resize")} placeholder={t("tab.output_resize.empty")} />
        </div>
      )}

      {group === "advanced" && (
        <div className="space-y-3">
          <Accordion icon="🎬" title={t("tab.enc_video")} placeholder={t("tab.enc_video.empty")} />
          <Accordion icon="🎵" title={t("tab.enc_audio")} placeholder={t("tab.enc_audio.empty")} />
          <Accordion icon="⚙️" title={t("tab.backend")} placeholder={t("tab.backend.empty")} />
        </div>
      )}

      <button className="mt-3 w-full rounded-xl bg-indigo-600 py-2.5 font-medium text-xs text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 transition">
        {t("sample.preview")}
      </button>
    </aside>
  );
}
