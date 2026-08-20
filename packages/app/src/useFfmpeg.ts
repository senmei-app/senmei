import type { FfmpegStatus } from "@senmei/bridge";
import { backend } from "./backend";
import { useDownloadable } from "./useDownloadable";

export function useFfmpeg() {
  return useDownloadable<FfmpegStatus>({
    getStatus: () => backend().then((b) => b.getFfmpegStatus()),
    download: (onProgress) => backend().then((b) => b.downloadFfmpeg(onProgress)),
  });
}
