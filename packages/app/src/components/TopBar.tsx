import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useI18n } from "../i18n";

export default function TopBar() {
  const { lang, setLang, t } = useI18n();

  const minimize = () => {
    if (isTauri()) void getCurrentWindow().minimize();
  };
  const toggleMax = () => {
    if (isTauri()) void getCurrentWindow().toggleMaximize();
  };
  const close = () => {
    if (isTauri()) void getCurrentWindow().close();
  };

  return (
    <header data-tauri-drag-region className="flex h-12 w-full items-center justify-between border-b border-slate-800/80 bg-slate-900/90 px-4 text-xs backdrop-blur-md">
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 font-bold text-sm text-white shadow-lg shadow-indigo-500/30">
            鮮
          </div>
          <span className="font-bold tracking-wide text-slate-100 text-sm">Senmei</span>
        </div>

        <nav className="flex items-center space-x-4 text-slate-400 font-medium">
          <button className="hover:text-slate-100 transition">{t("menu.file")}</button>
          <button className="hover:text-slate-100 transition">{t("menu.edit")}</button>
          <button className="hover:text-slate-100 transition">{t("menu.process")}</button>
          <button className="hover:text-slate-100 transition">{t("menu.view")}</button>
          <button className="hover:text-slate-100 transition">{t("menu.help")}</button>
        </nav>
      </div>

      <div className="flex items-center space-x-2 rounded-full bg-slate-950/80 border border-slate-800/80 px-3 py-1 text-slate-300 font-mono text-[11px]">
        <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span className="truncate max-w-[220px]">jujutsu_kaisen_op.mkv</span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-400">1080p @ 24fps</span>
      </div>

      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-1.5 rounded-lg bg-slate-800/80 border border-slate-700/50 px-2.5 py-1 text-slate-300 text-[11px]">
          <span className="text-indigo-400">⚡</span>
          <span className="font-medium">RIFE v4.26 + SPAN</span>
        </div>

        <button className="flex items-center space-x-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-3.5 py-1.5 font-medium text-white shadow-md shadow-indigo-600/30 transition active:scale-95">
          <span>▶</span>
          <span>{t("render.start")}</span>
        </button>

        <div className="flex items-center space-x-1 text-[10px] font-medium">
          <button
            onClick={() => setLang("en")}
            className={lang === "en" ? "text-indigo-300" : "text-slate-500 hover:text-slate-300 transition"}
          >
            EN
          </button>
          <span className="text-slate-600">/</span>
          <button
            onClick={() => setLang("de")}
            className={lang === "de" ? "text-indigo-300" : "text-slate-500 hover:text-slate-300 transition"}
          >
            DE
          </button>
        </div>

        <div className="flex items-center space-x-3 border-l border-slate-800 pl-3 text-slate-500">
          <button onClick={minimize} className="hover:text-slate-200 transition">
            ─
          </button>
          <button onClick={toggleMax} className="hover:text-slate-200 transition">
            ▢
          </button>
          <button onClick={close} className="hover:text-rose-400 transition">
            ✕
          </button>
        </div>
      </div>
    </header>
  );
}
