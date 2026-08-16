import { invoke, type Channel } from "@tauri-apps/api/core";

export interface RenderProgress {
  framesProcessed: number;
  totalFrames: number;
}

export async function healthCheck(): Promise<string> {
  return invoke<string>("health_check");
}

export async function render(
  input: string,
  output: string,
  onProgress: Channel<RenderProgress>,
): Promise<string> {
  return invoke<string>("render", { input, output, onProgress });
}
