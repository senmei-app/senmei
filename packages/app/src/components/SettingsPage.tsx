import { useState } from "react";
import { useI18n, type Lang } from "../i18n";
import WindowControls from "./WindowControls";

type Theme = "light" | "dark" | "system";
type Section = "appearance";

export default function SettingsPage({
  language,
  theme,
  onLanguageChange,
  onThemeChange,
  onBack,
}: {
  language: string;
  theme: string;
  onLanguageChange: (lang: Lang) => void;
  onThemeChange: (theme: Theme) => void;
  onBack: () => void;
}) {
  const { t } = useI18n();
  const [section, setSection] = useState<Section>("appearance");

  const sections: { key: Section; label: string }[] = [
    { key: "appearance", label: t("settings.section.appearance") },
  ];

  return (
    <div className="flex h-screen w-full flex-col bg-slate-100 font-sans text-slate-900 select-none antialiased dark:bg-slate-950 dark:text-slate-200">
      <header className="flex h-12 w-full items-center justify-between border-b border-slate-200 bg-white/90 px-4 backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
        <div className="flex items-center space-x-3">
          <button
            onClick={onBack}
            className="flex items-center space-x-1.5 rounded-lg border border-slate-200 bg-slate-100 px-2.5 py-1.5 text-[11px] text-slate-700 hover:bg-slate-200 dark:border-slate-800 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            <span>←</span>
            <span>{t("settings.back")}</span>
          </button>
          <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            {t("settings.title")}
          </h1>
        </div>
        <WindowControls />
      </header>

      <div className="flex flex-1 overflow-hidden">
        <nav className="w-48 shrink-0 space-y-1 border-r border-slate-200 bg-white/60 p-2 dark:border-slate-800/80 dark:bg-slate-900/40">
          {sections.map((s) => (
            <button
              key={s.key}
              onClick={() => setSection(s.key)}
              className={
                section === s.key
                  ? "w-full rounded-lg bg-indigo-600/15 px-3 py-2 text-left text-xs font-medium text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300"
                  : "w-full rounded-lg px-3 py-2 text-left text-xs text-slate-600 hover:bg-slate-200/60 dark:text-slate-400 dark:hover:bg-slate-800/60"
              }
            >
              {s.label}
            </button>
          ))}
        </nav>

        <div className="flex-1 overflow-y-auto p-6">
          {section === "appearance" && (
            <div className="max-w-xl space-y-6">
              <div>
                <label className="mb-2 block text-xs font-medium text-slate-700 dark:text-slate-300">
                  {t("settings.language")}
                </label>
                <div className="flex gap-1">
                  {(["en", "de"] as Lang[]).map((l) => (
                    <button
                      key={l}
                      onClick={() => onLanguageChange(l)}
                      className={
                        language === l
                          ? "rounded-md bg-indigo-600 px-4 py-2 text-xs font-medium text-white"
                          : "rounded-md bg-slate-200 px-4 py-2 text-xs text-slate-700 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
                      }
                    >
                      {l.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="mb-2 block text-xs font-medium text-slate-700 dark:text-slate-300">
                  {t("settings.theme")}
                </label>
                <div className="flex gap-1">
                  {(["light", "dark", "system"] as Theme[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => onThemeChange(m)}
                      className={
                        theme === m
                          ? "rounded-md bg-indigo-600 px-4 py-2 text-xs font-medium text-white"
                          : "rounded-md bg-slate-200 px-4 py-2 text-xs text-slate-700 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
                      }
                    >
                      {t(`theme.${m}`)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
