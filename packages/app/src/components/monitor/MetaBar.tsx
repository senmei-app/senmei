// Source→output meta overlay on the video surface. Renders nothing until
// toggled open by the Info button in the Monitor's play row. Values are
// click-to-copy. In full video mode it sits above the bottom control bar.

import { useState } from "react";
import { Check, Copy, X } from "lucide-react";
import type { VideoInfo } from "@senmei/bridge";
import { useI18n } from "../../i18n";
import type { PipelineStep } from "../../steps";

export interface OutputMeta {
  width: number | null;
  height: number | null;
  fps: number | null;
  duration: number | null;
  codec: string | null;
  container: string | null;
  pixFmt: string | null;
  colorTransfer: string | null;
  colorPrimaries: string | null;
}

function toFactor(v: string | null | undefined): number | null {
  const f = Number(v ?? "");
  return Number.isFinite(f) && f > 0 ? f : null;
}

/** Estimate the configured output's meta from the enabled pipeline steps. */
export function computeOutputMeta(info: VideoInfo | null, steps: PipelineStep[]): OutputMeta {
  const out: OutputMeta = {
    width: info?.width ?? null,
    height: info?.height ?? null,
    fps: info?.fps ?? null,
    duration: info?.duration ?? null,
    codec: null,
    container: null,
    pixFmt: null,
    colorTransfer: info?.colorTransfer ?? null,
    colorPrimaries: info?.colorPrimaries ?? null,
  };
  for (const s of steps) {
    if (!s.enabled) continue;
    // Resolution: upscale first, then the (post-scale) resize step.
    if (s.stepType === "upscale" && out.width != null && out.height != null) {
      const sc = s.params?.scale ?? 1;
      out.width = Math.round(out.width * sc);
      out.height = Math.round(out.height * sc);
    }
    if (s.stepType === "interpolation") {
      const m = s.params?.fpsMultiplier ?? 1;
      if (out.fps != null && m > 0) out.fps = out.fps * m;
    }
    if (s.stepType === "output") {
      out.codec = s.params?.videoCodec ?? null;
      out.container = s.params?.container ?? null;
      out.pixFmt = s.params?.pixFmt ?? null;
      if (s.params?.colorTransfer) out.colorTransfer = s.params.colorTransfer;
      if (s.params?.colorPrimaries) out.colorPrimaries = s.params.colorPrimaries;
    }
  }
  for (const s of steps) {
    if (!s.enabled) continue;
    if (s.stepType === "resize") {
      const f = toFactor(s.params?.factor);
      if (f != null && out.width != null && out.height != null) {
        out.width = Math.round(out.width * f);
        out.height = Math.round(out.height * f);
      }
    }
  }
  return out;
}

const dash = "–";

function fmtRes(w: number | null, h: number | null): string {
  return w != null && h != null ? `${w}×${h}` : dash;
}

function fmtFps(f: number | null): string {
  return f != null ? f.toFixed(3).replace(/\.?0+$/, "") : dash;
}

function fmtClock(ms: number): string {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

// "hevc" → "HEVC", "yuv420p10le" → "yuv420p10le".
function prettyCodec(c: string): string {
  return c.toUpperCase();
}

export default function MetaBar({
  info,
  steps,
  open,
  onClose,
  fullVideo = false,
}: {
  info: VideoInfo | null;
  steps: PipelineStep[];
  open: boolean;
  onClose: () => void;
  fullVideo?: boolean;
}) {
  const { t } = useI18n();
  const [copied, setCopied] = useState<string | null>(null);
  const [copyErr, setCopyErr] = useState(false);
  const out = computeOutputMeta(info, steps);

  const srcCodec = info?.videoCodec ?? null;
  const srcColor = info?.colorTransfer ?? info?.colorPrimaries ?? null;
  const outColor = out.colorTransfer ?? out.colorPrimaries ?? null;
  const outCodec = out.codec ? `${prettyCodec(out.codec)}${out.container ? ` · ${out.container}` : ""}` : dash;

  const rows: { key: string; src: string; out: string }[] = [
    { key: t("meta.resolution"), src: fmtRes(info?.width ?? null, info?.height ?? null), out: fmtRes(out.width, out.height) },
    { key: t("meta.fps"), src: fmtFps(info?.fps ?? null), out: fmtFps(out.fps) },
    { key: t("meta.codec"), src: srcCodec ? prettyCodec(srcCodec) : dash, out: outCodec },
    { key: t("meta.duration"), src: fmtClock((info?.duration ?? 0) * 1000), out: fmtClock((out.duration ?? 0) * 1000) },
    { key: t("meta.color"), src: srcColor ?? dash, out: outColor ?? dash },
  ];

  const block = [
    `${t("meta.source")}:`,
    ...rows.map((r) => `  ${r.key}: ${r.src}`),
    `${t("meta.output")}:`,
    ...rows.map((r) => `  ${r.key}: ${r.out}`),
  ].join("\n");

  const copy = (id: string, value: string) => {
    const done = () => {
      setCopied(id);
      window.setTimeout(() => setCopied((c) => (c === id ? null : c)), 1200);
    };
    const fail = () => {
      setCopyErr(true);
      window.setTimeout(() => setCopyErr(false), 1200);
    };
    if (!navigator.clipboard) {
      fail();
      return;
    }
    navigator.clipboard.writeText(value).then(done).catch(fail);
  };

  if (!open) return null;

  return (
    <div
      className={
        "absolute right-3 z-30 w-[22rem] max-w-[calc(100%-1.5rem)] rounded-xl border border-white/10 bg-black/75 p-2 font-mono text-[11px] text-slate-300 shadow-2xl backdrop-blur" +
        (fullVideo ? " bottom-24" : " bottom-3")
      }
    >
      <div className="absolute top-1 right-1 z-10 flex items-center gap-1">
        <button
          onClick={() => copy("all", block)}
          title={t("meta.copy")}
          aria-label={t("meta.copy")}
          className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[11px] text-slate-300 hover:border-indigo-400/50 hover:text-white"
        >
          {copyErr ? <X className="h-3.5 w-3.5 text-rose-400" /> : copied === "all" ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
        <button
          onClick={onClose}
          title={t("meta.toggle")}
          aria-label={t("meta.toggle")}
          className="rounded-md px-1.5 py-0.5 text-[11px] text-slate-400 hover:bg-white/10 hover:text-slate-200"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-x-5 pr-10">
        <div>
          <div className="mb-0.5 text-[11px] uppercase tracking-wider text-slate-500">{t("meta.source")}</div>
          {rows.map((r) => (
            <button
              key={r.key}
              onClick={() => copy(`s-${r.key}`, r.src)}
              title={t("meta.copy")}
              className="flex w-full items-center justify-between gap-2 py-0.5 text-left hover:text-white"
            >
              <span className="shrink-0 text-slate-500">{r.key}</span>
              <span
                className={
                  copied === `s-${r.key}`
                    ? "shrink-0 text-emerald-400"
                    : "truncate text-slate-200"
                }
              >
                {copied === `s-${r.key}` ? <Check className="h-3 w-3" /> : r.src}
              </span>
            </button>
          ))}
        </div>
        <div>
          <div className="mb-0.5 text-[11px] uppercase tracking-wider text-slate-500">{t("meta.output")}</div>
          {rows.map((r) => (
            <button
              key={r.key}
              onClick={() => copy(`o-${r.key}`, r.out)}
              title={t("meta.copy")}
              className="flex w-full items-center justify-between gap-2 py-0.5 text-left hover:text-white"
            >
              <span className="shrink-0 text-slate-500">{r.key}</span>
              <span
                className={
                  copied === `o-${r.key}`
                    ? "shrink-0 text-emerald-400"
                    : "truncate text-indigo-300"
                }
              >
                {copied === `o-${r.key}` ? <Check className="h-3 w-3" /> : r.out}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
