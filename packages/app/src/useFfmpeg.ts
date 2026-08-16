import { useCallback, useEffect, useState } from "react";
import { isTauri, Channel } from "@tauri-apps/api/core";
import {
  downloadFfmpeg,
  getFfmpegStatus,
  type DownloadProgress,
  type FfmpegStatus,
} from "@senmei/bridge";

export function useFfmpeg() {
  const [status, setStatus] = useState<FfmpegStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!isTauri()) return;
    getFfmpegStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const download = useCallback(() => {
    if (!isTauri()) return;
    setDownloading(true);
    setPct(0);
    setError(null);
    const ch = new Channel<DownloadProgress>();
    ch.onmessage = (p) =>
      setPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0);
    downloadFfmpeg(ch)
      .then(() => refresh())
      .catch((e) => setError(String(e)))
      .finally(() => setDownloading(false));
  }, [refresh]);

  return { status, downloading, pct, error, download };
}
