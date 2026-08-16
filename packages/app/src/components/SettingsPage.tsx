import { useState } from "react";
import { Button } from "@senmei/ui";
import { useI18n, type Lang } from "../i18n";
import { useFfmpeg } from "../useFfmpeg";
import { useLibtorch } from "../useLibtorch";
import WindowControls from "./WindowControls";

type Theme = "light" | "dark" | "system";
type Section = "appearance" | "ffmpeg" | "inference";

const KEY_ENCODERS = [
  "libx264",
  "libx265",
  "libopenh264",
  "h264_nvenc",
  "hevc_nvenc",
  "av1_nvenc",
  "h264_vaapi",
  "hevc_vaapi",
  "av1_vaapi",
  "libsvtav1",
];

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
  const { status, downloading, pct, error, download } = useFfmpeg();
  const {
    status: libtorch,
    downloading: ltDownloading,
    pct: ltPct,
    error: ltError,
    download: startLibtorchDownload,
  } = useLibtorch();

  const sections: { key: Section; label: string }[] = [
    { key: "appearance", label: t("settings.section.appearance") },
    { key: "ffmpeg", label: t("settings.section.ffmpeg") },
    { key: "inference", label: t("settings.section.inference") },
  ];

  const encoders = status?.encoders ?? [];
  const backendLabel =
    libtorch?.backend === "rocm"
      ? t("settings.inference.backend.rocm")
      : libtorch?.backend === "cuda"
        ? t("settings.inference.backend.cuda")
        : t("settings.inference.backend.cpu");

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
        <div data-tauri-drag-region className="flex-1 self-stretch" />
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

          {section === "ffmpeg" && (
            <div className="max-w-xl space-y-6">
              <div>
                <label className="mb-2 block text-xs font-medium text-slate-700 dark:text-slate-300">
                  {t("settings.ffmpeg.status")}
                </label>
                {status?.found ? (
                  <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs dark:border-slate-800 dark:bg-slate-900">
                    <p className="text-slate-700 dark:text-slate-300">
                      {t("settings.ffmpeg.version")}: {status.version}
                    </p>
                    <p className="mt-1 truncate font-mono text-[11px] text-slate-500">{status.path}</p>
                  </div>
                ) : (
                  <p className="text-xs text-rose-500">{t("settings.ffmpeg.notFound")}</p>
                )}
              </div>

              <div>
                <label className="mb-2 block text-xs font-medium text-slate-700 dark:text-slate-300">
                  {t("settings.ffmpeg.encoders")}
                </label>
                <div className="flex flex-wrap gap-1">
                  {KEY_ENCODERS.map((e) => (
                    <span
                      key={e}
                      className={
                        encoders.includes(e)
                          ? "rounded-md bg-emerald-500/15 px-2 py-1 font-mono text-[11px] text-emerald-600 dark:text-emerald-400"
                          : "rounded-md bg-slate-200 px-2 py-1 font-mono text-[11px] text-slate-400 dark:bg-slate-800 dark:text-slate-600"
                      }
                    >
                      {e}
                    </span>
                  ))}
                </div>
                <p className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                  {t("settings.ffmpeg.available").replace("{count}", String(encoders.length))}
                </p>
              </div>

              {!status?.found && (
                <div className="space-y-2">
                  {error && <p className="text-xs text-rose-500">{error}</p>}
                  <Button onClick={download} disabled={downloading}>
                    {downloading
                      ? t("settings.ffmpeg.downloading").replace("{pct}", String(pct))
                      : t("settings.ffmpeg.download")}
                  </Button>
                </div>
              )}
            </div>
          )}

          {section === "inference" && (
            <div className="max-w-xl space-y-6">
              <div>
                <label className="mb-2 block text-xs font-medium text-slate-700 dark:text-slate-300">
                  {t("settings.inference.backend")}
                </label>
                <span className="rounded-md bg-indigo-600/15 px-2 py-1 font-mono text-[11px] text-indigo-600 dark:text-indigo-300">
                  {backendLabel}
                </span>
              </div>

              <div>
                <label className="mb-2 block text-xs font-medium text-slate-700 dark:text-slate-300">
                  {t("settings.section.inference")}
                </label>
                {libtorch?.downloaded ? (
                  <p className="text-xs text-emerald-600 dark:text-emerald-400">
                    {t("settings.inference.downloaded")}
                  </p>
                ) : (
                  <div className="space-y-2">
                    <p className="text-xs text-rose-500">{t("settings.inference.notDownloaded")}</p>
                    {ltError && <p className="text-xs text-rose-500">{ltError}</p>}
                    <Button onClick={startLibtorchDownload} disabled={ltDownloading}>
                      {ltDownloading
                        ? t("settings.inference.downloading").replace("{pct}", String(ltPct))
                        : t("settings.inference.download")}
                    </Button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
