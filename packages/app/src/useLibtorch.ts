import { downloadLibtorch, getLibtorchStatus, type LibTorchInfo } from "@senmei/bridge";
import { useDownloadable } from "./useDownloadable";

export function useLibtorch() {
  return useDownloadable<LibTorchInfo>({
    getStatus: getLibtorchStatus,
    download: (onProgress) => downloadLibtorch(onProgress),
  });
}
