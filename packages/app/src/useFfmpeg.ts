import {
  downloadFfmpeg,
  getFfmpegStatus,
  type FfmpegStatus,
} from "@senmei/bridge";
import { useDownloadable } from "./useDownloadable";

export function useFfmpeg() {
  return useDownloadable<FfmpegStatus>({
    getStatus: getFfmpegStatus,
    download: (onProgress) => downloadFfmpeg(onProgress),
  });
}
