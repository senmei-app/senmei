import { useCallback, useEffect, useState } from "react";
import type { DownloadProgress } from "@senmei/bridge";

interface DownloadableOptions<TStatus> {
  getStatus: () => Promise<TStatus>;
  download: (onProgress: (p: DownloadProgress) => void) => Promise<void>;
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
    getStatus().then(setStatus).catch(() => setStatus(null));
  }, [getStatus]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const run = useCallback(() => {
    setDownloading(true);
    setPct(0);
    setError(null);
    download((p) => setPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0))
      .then(() => refresh())
      .catch((e) => setError(String(e)))
      .finally(() => setDownloading(false));
  }, [download, refresh]);

  return { status, downloading, pct, error, download: run };
}
