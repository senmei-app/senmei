import { useCallback, useState } from "react";
import { isTauri, Channel } from "@tauri-apps/api/core";
import { downloadModel, type DownloadProgress } from "@senmei/bridge";

export function useModelDownload() {
  const [downloading, setDownloading] = useState(false);
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const download = useCallback((modelId: string) => {
    if (!isTauri()) return;
    setDownloading(true);
    setPct(0);
    setError(null);
    const ch = new Channel<DownloadProgress>();
    ch.onmessage = (p) =>
      setPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0);
    downloadModel(modelId, ch)
      .catch((e) => setError(String(e)))
      .finally(() => setDownloading(false));
  }, []);

  return { downloading, pct, error, download };
}
