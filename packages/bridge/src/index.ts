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

export async function importFolder(dir: string): Promise<string[]> {
  return invoke<string[]>("import_folder", { dir });
}

export interface ProjectEntry {
  name: string;
  path: string;
}

export async function listProjects(): Promise<ProjectEntry[]> {
  return invoke<ProjectEntry[]>("list_projects");
}

export async function createProject(name: string): Promise<string> {
  return invoke<string>("create_project", { name });
}
