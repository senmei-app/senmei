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

export interface Settings {
  language: string;
  theme: string;
}

export async function getSettings(): Promise<Settings> {
  return invoke<Settings>("get_settings");
}

export async function saveSettings(settings: Settings): Promise<void> {
  await invoke("save_settings", { settings });
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
