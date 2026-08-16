import { useCallback, useEffect, useState } from "react";
import { isTauri, Channel } from "@tauri-apps/api/core";
import type { DownloadProgress } from "@senmei/bridge";

interface DownloadableOptions<TStatus> {
  getStatus: () => Promise<TStatus>;
  download: (onProgress: Channel<DownloadProgress>) => Promise<unknown>;
}

/** Shared status/refresh/download state for one-off downloads (ffmpeg). */
export function useDownloadable<TStatus>({
  getStatus,
  download,
}: DownloadableOptions<TStatus>) {
  const [status, setStatus] = useState<TStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!isTauri()) return;
    getStatus().then(setStatus).catch(() => setStatus(null));
  }, [getStatus]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const run = useCallback(() => {
    if (!isTauri()) return;
    setDownloading(true);
    setPct(0);
    setError(null);
    const ch = new Channel<DownloadProgress>();
    ch.onmessage = (p) =>
      setPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0);
    download(ch)
      .then(() => refresh())
      .catch((e) => setError(String(e)))
      .finally(() => setDownloading(false));
  }, [download, refresh]);

  return { status, downloading, pct, error, download: run };
}
