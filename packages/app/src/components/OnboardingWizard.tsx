import { useEffect, useState } from "react";
import type { BackendInfo, FfmpegStatus } from "@senmei/bridge";
import { backend } from "../backend";
import { useI18n } from "../i18n";

const STEPS = ["welcome", "ffmpeg", "engine", "done"];

export default function OnboardingWizard({ open, onDone }: { open: boolean; onDone: () => void }) {
  const { t } = useI18n();
  const [step, setStep] = useState(0);
  const [ffmpeg, setFfmpeg] = useState<FfmpegStatus | null>(null);
  const [ffmpegPct, setFfmpegPct] = useState<number | null>(null);
  const [ffmpegErr, setFfmpegErr] = useState<string | null>(null);
  const [info, setInfo] = useState<BackendInfo | null>(null);

  useEffect(() => {
    if (!open) return;
    setStep(0);
    void backend().then((b) => {
      void b.getFfmpegStatus().then(setFfmpeg).catch(() => {});
      void b.backendInfo().then(setInfo).catch(() => {});
    });
  }, [open]);

  if (!open) return null;

  const downloadFfmpeg = async () => {
    setFfmpegErr(null);
    setFfmpegPct(0);
    try {
      await (await backend()).downloadFfmpeg((p) =>
        setFfmpegPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0),
      );
      setFfmpeg(await (await backend()).getFfmpegStatus());
      setFfmpegPct(null);
    } catch (e) {
      setFfmpegErr(String((e as Error)?.message ?? e));
      setFfmpegPct(null);
    }
  };

  const canNext = step !== 1 || !!ffmpeg?.found;
  const isLast = step === STEPS.length - 1;

  const footer = (
    <div className="mt-4 flex items-center justify-between">
      <button
        onClick={() => setStep((s) => s - 1)}
        disabled={step === 0}
        className="rounded-lg px-3 py-1.5 text-[11px] text-slate-500 hover:bg-slate-200 disabled:opacity-30 dark:text-slate-400 dark:hover:bg-slate-800"
      >
        {t("onboard.back")}
      </button>
      <div className="flex gap-1">
        {STEPS.map((s, i) => (
          <span
            key={s}
            className={
              "h-1.5 w-4 rounded-full " + (i <= step ? "bg-indigo-500" : "bg-slate-300 dark:bg-slate-700")
            }
          />
        ))}
      </div>
      {isLast ? (
        <button
          onClick={onDone}
          className="rounded-lg bg-indigo-600 px-4 py-1.5 text-[11px] font-medium text-white hover:bg-indigo-500"
        >
          {t("onboard.start")}
        </button>
      ) : (
        <button
          onClick={() => setStep((s) => s + 1)}
          disabled={!canNext}
          className="rounded-lg bg-indigo-600 px-4 py-1.5 text-[11px] font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
        >
          {t("onboard.next")}
        </button>
      )}
    </div>
  );

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-[26rem] max-w-[90vw] rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 font-bold text-white">
            鮮
          </div>
          <div>
            <div className="text-sm font-semibold">Senmei</div>
            <div className="text-[11px] text-slate-400">setup</div>
          </div>
        </div>

        {step === 0 && (
          <div>
            <h2 className="mb-2 text-base font-semibold">{t("onboard.welcomeTitle")}</h2>
            <p className="text-xs leading-5 text-slate-500 dark:text-slate-400">
              {t("onboard.welcomeText")}
            </p>
          </div>
        )}

        {step === 1 && (
          <div>
            <h2 className="mb-2 text-base font-semibold">{t("onboard.ffmpegTitle")}</h2>
            {ffmpeg?.found ? (
              <p className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-600 dark:text-emerald-300">
                {t("onboard.ffmpegOk")} · {ffmpeg.version ?? ""}
              </p>
            ) : (
              <div>
                <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
                  {t("onboard.ffmpegMissing")}
                </p>
                {ffmpegPct !== null ? (
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                    <div className="h-full bg-indigo-500 transition-all" style={{ width: `${ffmpegPct}%` }} />
                  </div>
                ) : (
                  <button
                    onClick={downloadFfmpeg}
                    className="rounded-lg bg-indigo-600 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-indigo-500"
                  >
                    {t("onboard.ffmpegDownload")}
                  </button>
                )}
                {ffmpegErr && <p className="mt-2 text-[11px] text-rose-500">{ffmpegErr}</p>}
              </div>
            )}
          </div>
        )}

        {step === 2 && (
          <div>
            <h2 className="mb-2 text-base font-semibold">{t("onboard.engineTitle")}</h2>
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between rounded-lg bg-slate-100 px-3 py-2 dark:bg-slate-800">
                <span>Vulkan</span>
                <span className={info?.vulkanCompiled ? "text-emerald-500" : "text-slate-400"}>
                  {info?.vulkanCompiled ? t("onboard.ready") : t("onboard.notAvailable")}
                </span>
              </div>
              <div className="flex justify-between rounded-lg bg-slate-100 px-3 py-2 dark:bg-slate-800">
                <span>LibTorch (CUDA/ROCm)</span>
                <span className={info?.libtorchCompiled && info.cudaAvailable ? "text-emerald-500" : "text-slate-400"}>
                  {info?.libtorchCompiled && info.cudaAvailable
                    ? t("onboard.ready")
                    : t("onboard.notAvailable")}
                </span>
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div>
            <h2 className="mb-2 text-base font-semibold">{t("onboard.doneTitle")}</h2>
            <p className="text-xs leading-5 text-slate-500 dark:text-slate-400">{t("onboard.doneText")}</p>
          </div>
        )}

        {footer}
      </div>
    </div>
  );
}
