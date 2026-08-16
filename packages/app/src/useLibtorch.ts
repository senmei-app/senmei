import { useCallback, useEffect, useState } from "react";
import { isTauri, Channel } from "@tauri-apps/api/core";
import { downloadLibtorch, getLibtorchStatus, type LibTorchInfo } from "@senmei/bridge";

export function useLibtorch() {
  const [status, setStatus] = useState<LibTorchInfo | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!isTauri()) return;
    getLibtorchStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const download = useCallback(() => {
    if (!isTauri()) return;
    setDownloading(true);
    setPct(0);
    setError(null);
    const ch = new Channel<{ downloaded: number; total: number }>();
    ch.onmessage = (p) =>
      setPct(p.total ? Math.round((p.downloaded / p.total) * 100) : 0);
    downloadLibtorch(ch)
      .then(() => refresh())
      .catch((e) => setError(String(e)))
      .finally(() => setDownloading(false));
  }, [refresh]);

  return { status, downloading, pct, error, download };
}
