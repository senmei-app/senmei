import { useEffect, useRef, useState } from "react";
import { Button, Select } from "@senmei/ui";
import type { BackendInfo, EngineBackend, HardwareSnapshot, ModelFileInfo } from "@senmei/bridge";
import { useI18n, type Lang } from "../i18n";
import { useFfmpeg } from "../useFfmpeg";
import { backend as getBackend } from "../backend";
import { HOTKEY_ACTIONS, comboFromEvent } from "../hotkeys";
import WindowControls from "./WindowControls";

type Theme = "light" | "dark" | "system";
type Section = "appearance" | "hotkeys" | "models" | "info";

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
  tileSize,
  gpuIndex,
  backend,
  backendInfo,
  hardware,
  hotkeys,
  onLanguageChange,
  onThemeChange,
  onTileSizeChange,
  onGpuIndexChange,
  onBackendChange,
  onHotkeyChange,
  onBack,
}: {
  language: string;
  theme: string;
  tileSize: number;
  gpuIndex: number;
  backend: EngineBackend;
  backendInfo: BackendInfo | null;
  hardware: HardwareSnapshot | null;
  hotkeys: Record<string, string>;
  onLanguageChange: (lang: Lang) => void;
  onThemeChange: (theme: Theme) => void;
  onTileSizeChange: (n: number) => void;
  onGpuIndexChange: (n: number) => void;
  onBackendChange: (b: EngineBackend) => void;
  onHotkeyChange: (id: string, combo: string) => void;
  onBack: () => void;
}) {
  const { t } = useI18n();
  const [section, setSection] = useState<Section>("appearance");
  const [recording, setRecording] = useState<string | null>(null);
  const { status, downloading, pct, error, download } = useFfmpeg();
  const [tileDraft, setTileDraft] = useState(String(tileSize));
  const [exporting, setExporting] = useState(false);
  const [diagMsg, setDiagMsg] = useState<string | null>(null);
  const [modelFiles, setModelFiles] = useState<ModelFileInfo[]>([]);
  // Catalog kind per model id (frontend-only; ModelFileInfo has no kind).
  const [kinds, setKinds] = useState<Record<string, string>>({});

  useEffect(() => {
    getBackend()
      .then((b) => b.modelFiles())
      .then(setModelFiles)
      .catch(() => {});
    getBackend()
      .then((b) => b.listModels())
      .then((ms) => setKinds(Object.fromEntries(ms.map((m) => [m.id, m.kind]))))
      .catch(() => {});
  }, []);

  const removeModel = async (id: string) => {
    if (!window.confirm(t("settings.models.deleteConfirm"))) return;
    const be = await getBackend();
    await be.deleteModelFile(id);
    setModelFiles(await be.modelFiles());
  };

  const fmtSize = (n: number) =>
    n >= 1 << 30
      ? `${(n / (1 << 30)).toFixed(1)} GiB`
      : n >= 1 << 20
        ? `${(n / (1 << 20)).toFixed(1)} MiB`
        : `${(n / 1024).toFixed(0)} KiB`;
  const commitTileSize = () => {
    const n = Math.round(Number(tileDraft));
    if (Number.isFinite(n)) {
      const clamped = Math.min(2048, Math.max(128, n));
      setTileDraft(String(clamped));
      onTileSizeChange(clamped);
    } else {
      setTileDraft(String(tileSize));
    }
  };

  const exportDiagnostics = async () => {
    const be = await getBackend();
    const dest = await be.pickSaveFile("senmei-diagnostics.tar.xz", ["tar.xz"]);
    if (!dest) return;
    setExporting(true);
    setDiagMsg(null);
    try {
      await be.exportDiagnostics(dest);
      setDiagMsg(t("settings.diagnostics.exported"));
    } catch (e) {
      setDiagMsg(String(e));
    } finally {
      setExporting(false);
    }
  };

  const onHotkeyChangeRef = useRef(onHotkeyChange);
  useEffect(() => {
    onHotkeyChangeRef.current = onHotkeyChange;
  }, [onHotkeyChange]);
  // Esc closes the page; while recording hotkeys, Esc only cancels the capture.
  const onBackRef = useRef(onBack);
  useEffect(() => {
    onBackRef.current = onBack;
  }, [onBack]);
  useEffect(() => {
    if (recording) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      onBackRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [recording]);
  // While recording, swallow the next key combo and bind it (Esc cancels).
  useEffect(() => {
    if (!recording) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") {
        setRecording(null);
        return;
      }
      const combo = comboFromEvent(e);
      if (combo) {
        onHotkeyChangeRef.current(recording, combo);
        setRecording(null);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [recording]);

  const sections: { key: Section; label: string }[] = [
    { key: "appearance", label: t("settings.section.appearance") },
    { key: "hotkeys", label: t("settings.section.hotkeys") },
    { key: "models", label: t("settings.section.models") },
    { key: "info", label: t("settings.section.info") },
  ];

  const encoders = status?.encoders ?? [];

  return (
    <div className="flex h-screen w-full flex-col bg-slate-100 font-sans text-slate-900 select-none antialiased dark:bg-slate-950 dark:text-slate-200">
      <header className="flex h-10 w-full items-center justify-between border-b border-slate-200 bg-white/90 px-4 backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
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
            <div className="max-w-xl space-y-3">
              {/* Language */}
              <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60">
                <span className="text-xs text-slate-700 dark:text-slate-300">{t("settings.language")}</span>
                <Select
                  value={language}
                  onChange={(v) => onLanguageChange(v as Lang)}
                  options={[
                    { value: "en", label: "English" },
                    { value: "de", label: "Deutsch" },
                    { value: "zh", label: "中文" },
                    { value: "ja", label: "日本語" },
                  ]}
                  className="w-[140px]"
                />
              </div>

              {/* Theme */}
              <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60">
                <span className="text-xs text-slate-700 dark:text-slate-300">{t("settings.theme")}</span>
                <Select
                  value={theme}
                  onChange={(v) => onThemeChange(v as Theme)}
                  options={[
                    { value: "light", label: t("theme.light") },
                    { value: "dark", label: t("theme.dark") },
                    { value: "system", label: t("theme.system") },
                  ]}
                  className="w-[140px]"
                />
              </div>

              {/* Tile size */}
              <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60">
                <span className="text-xs text-slate-700 dark:text-slate-300">{t("settings.tileSize")}</span>
                <div className="relative w-[140px]">
                  <input
                    type="number"
                    min={128}
                    max={2048}
                    step={64}
                    value={tileDraft}
                    onChange={(e) => setTileDraft(e.target.value)}
                    onBlur={commitTileSize}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") e.currentTarget.blur();
                    }}
                    className="no-spin w-full rounded-md border border-slate-200 bg-white py-1.5 pl-2.5 pr-7 text-left text-xs text-slate-700 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
                  />
                  <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-slate-400 dark:text-slate-500">px</span>
                </div>
              </div>

              {/* Backend */}
              <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60">
                <span className="text-xs text-slate-700 dark:text-slate-300">{t("settings.backend")}</span>
                <Select
                  value={backend}
                  onChange={(v) => onBackendChange(v as EngineBackend)}
                  options={[
                    { value: "auto", label: "Auto" },
                    { value: "vulkan", label: "Vulkan", disabled: backendInfo ? !backendInfo.vulkanCompiled : false },
                    { value: "libTorch", label: "LibTorch", disabled: backendInfo ? !(backendInfo.libtorchCompiled && backendInfo.cudaAvailable) : false },
                  ]}
                  className="w-[140px]"
                />
              </div>

              {/* GPU index */}
              <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60">
                <span className="text-xs text-slate-700 dark:text-slate-300">{t("settings.gpu")}</span>
                <Select
                  value={String(gpuIndex)}
                  onChange={(v) => onGpuIndexChange(Number(v))}
                  options={
                    (hardware?.gpus && hardware.gpus.length > 0)
                      ? hardware.gpus.map((g) => ({
                          value: String(g.index),
                          label: `${g.name}${g.vramTotalBytes ? ` (${Math.round(g.vramTotalBytes / 1024 / 1024 / 1024 * 10) / 10} GB)` : ""}`,
                        }))
                      : [{ value: "0", label: "GPU 0" }]
                  }
                  className="w-[140px]"
                />
              </div>
            </div>
          )}

          {section === "hotkeys" && (
            <div className="max-w-xl space-y-3">
              {(["global", "playback", "view", "media"] as const).map((g) => (
                <div key={g}>
                  <h3 className="mb-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300">
                    {t(`settings.hotkeys.group.${g}`)}
                  </h3>
                  <div className="space-y-1.5">
                    {HOTKEY_ACTIONS.filter((a) => a.group === g).map((a) => {
                      const active = recording === a.id;
                      const current = hotkeys[a.id];
                      const isDefault = current === a.default;
                      return (
                        <div
                          key={a.id}
                          className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60"
                        >
                          <span className="text-xs text-slate-700 dark:text-slate-300">{t(a.labelKey)}</span>
                          <div className="flex items-center gap-2">
                            {!isDefault && (
                              <button
                                onClick={() => onHotkeyChange(a.id, a.default)}
                                title={t("hotkeys.reset")}
                                className="rounded-md px-2 py-1 text-[11px] text-slate-400 transition hover:bg-slate-200 hover:text-slate-600 dark:hover:bg-slate-800 dark:hover:text-slate-300"
                              >
                                ↺
                              </button>
                            )}
                            <button
                              onClick={() => setRecording(active ? null : a.id)}
                              className={
                                active
                                  ? "min-w-[110px] rounded-md border border-indigo-500 bg-indigo-600/20 px-2.5 py-1 text-center font-mono text-[11px] text-indigo-500 dark:text-indigo-300"
                                  : "min-w-[110px] rounded-md border border-slate-200 bg-slate-100 px-2.5 py-1 text-center font-mono text-[11px] text-slate-700 transition hover:border-slate-300 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:border-slate-600"
                              }
                            >
                              {active ? t("hotkeys.press") : <kbd>{current}</kbd>}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
              <p className="pt-1 text-[11px] text-slate-500 dark:text-slate-400">{t("hotkeys.hint")}</p>
            </div>
          )}

          {section === "models" && (
            <div className="max-w-xl space-y-3">
              {modelFiles.length === 0 && (
                <p className="text-xs text-slate-500 dark:text-slate-400">{t("settings.models.empty")}</p>
              )}
              {(["upscale", "interpolate", "denoise", "deblur", "decompress", "other"] as const).map((kind) => {
                const items =
                  kind === "other"
                    ? modelFiles.filter(
                        (mf) =>
                          !(["upscale", "interpolate", "denoise", "deblur", "decompress"] as string[]).includes(
                            kinds[mf.id] ?? "other",
                          ),
                      )
                    : modelFiles.filter((mf) => kinds[mf.id] === kind);
                if (items.length === 0) return null;
                return (
                  <div key={kind}>
                    <h3 className="mb-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300">
                      {kind === "other" ? "Other" : t(`tab.${kind}`)}
                    </h3>
                    <div className="space-y-2">
                      {items.map((mf) => (
                        <div
                          key={mf.id}
                          className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60"
                        >
                          <div className="min-w-0">
                            <p className="truncate text-xs font-medium text-slate-800 dark:text-slate-200">{mf.id}</p>
                            <p className="truncate font-mono text-[11px] text-slate-500">
                              {mf.file} · {fmtSize(mf.size)}
                              {mf.verified ? " · ✓" : " · ✗"}
                            </p>
                          </div>
                          <button
                            onClick={() => removeModel(mf.id)}
                            className="shrink-0 rounded-md px-2 py-1 text-[11px] text-rose-500 hover:bg-rose-500/10"
                          >
                            {t("settings.models.delete")}
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {section === "info" && (
            <div className="max-w-xl space-y-6">
              <div className="rounded-xl border border-slate-200 bg-white/70 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/60">
                <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-200">
                  {t("settings.about.title")}
                </h3>
                <div className="space-y-1.5 text-xs text-slate-700 dark:text-slate-300">
                  <p>
                    {t("settings.about.version")}: v{__APP_VERSION__}-{__BUILD_HASH__}
                  </p>
                  <p>
                    {t("about.engine")}: {t("about.engineValue")}
                  </p>
                  <p>
                    {t("settings.about.platform")}: {navigator.userAgent.includes("Tauri") ? "Tauri" : "Web"} ·{" "}
                    {navigator.platform}
                  </p>
                  <p>
                    {t("settings.about.locale")}: {language}
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white/70 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/60">
                <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-200">
                  {t("settings.hardware.title")}
                </h3>
                <div className="space-y-1.5 text-xs text-slate-700 dark:text-slate-300">
                  <p>
                    {t("settings.hardware.gpu")}: {hardware?.gpuName ?? "—"}
                    {hardware?.gpuUtilizationPercent != null ? ` · ${Math.round(hardware.gpuUtilizationPercent)}%` : ""}
                  </p>
                  <p>
                    {t("settings.hardware.cpu")}: {hardware ? `${Math.round((hardware.cpuUsage ?? 0) * 100)}%` : "—"}
                  </p>
                  <p>
                    {t("settings.hardware.mem")}:{" "}
                    {hardware ? `${fmtSize(hardware.memoryUsedBytes)} / ${fmtSize(hardware.memoryTotalBytes)}` : "—"}
                  </p>
                  {hardware?.gpuMemoryUsedBytes != null && (
                    <p>
                      VRAM: {fmtSize(hardware.gpuMemoryUsedBytes)} / {fmtSize(hardware.gpuMemoryTotalBytes ?? 0)}
                    </p>
                  )}
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white/70 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/60">
                <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-200">
                  {t("settings.section.ffmpeg")}
                </h3>
                <div className="space-y-4">
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
              </div>

              <div className="rounded-xl border border-slate-200 bg-white/70 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/60">
                <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-200">
                  {t("settings.section.backend")}
                </h3>
                <div className="space-y-1.5 text-xs text-slate-700 dark:text-slate-300">
                  <p>
                    Vulkan: {backendInfo?.vulkanCompiled ? "✓" : "—"}
                  </p>
                  <p>
                    LibTorch:{" "}
                    {backendInfo?.libtorchCompiled
                      ? `✓ ${backendInfo.libtorchVersion ?? ""}`.trim()
                      : "—"}
                  </p>
                  {backendInfo?.libtorchCompiled && (
                    <p>
                      CUDA/ROCm:{" "}
                      {backendInfo.cudaAvailable
                        ? `${backendInfo.cudaDeviceCount} device${backendInfo.cudaDeviceCount === 1 ? "" : "s"}`
                        : "not available"}
                    </p>
                  )}
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white/70 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/60">
                <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-200">
                  {t("settings.section.diagnostics")}
                </h3>
                <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
                  {t("settings.diagnostics.hint")}
                </p>
                {diagMsg && (
                  <p className="mb-2 text-xs text-slate-600 dark:text-slate-300">{diagMsg}</p>
                )}
                <Button onClick={exportDiagnostics} disabled={exporting}>
                  {exporting ? "…" : t("settings.diagnostics.export")}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
