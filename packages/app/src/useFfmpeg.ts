import { useCallback } from "react";
import type { DownloadProgress, FfmpegStatus } from "@senmei/bridge";
import { backend } from "./backend";
import { useDownloadable } from "./useDownloadable";

export function useFfmpeg() {
  // Stable references so useDownloadable's refresh effect runs once, not per
  // render (StatusBar re-renders every second on the hardware poll).
  const getStatus = useCallback(() => backend().then((b) => b.getFfmpegStatus()), []);
  const download = useCallback(
    (onProgress: (p: DownloadProgress) => void) => backend().then((b) => b.downloadFfmpeg(onProgress)),
    [],
  );
  return useDownloadable<FfmpegStatus>({ getStatus, download });
}
