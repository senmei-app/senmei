import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

export interface UpdateInfo {
  version: string;
  date?: string;
  body?: string;
}

let checked = false;

/** Check for updates once per app session. Returns update info if available. */
export async function checkForUpdates(): Promise<UpdateInfo | null> {
  if (checked) return null;
  checked = true;

  try {
    const update = await check();
    if (!update) return null;

    return {
      version: update.version,
      date: update.date ?? undefined,
      body: update.body ?? undefined,
    };
  } catch {
    // Silently ignore — network errors, no endpoint configured in dev, etc.
    return null;
  }
}

/** Download, install, and relaunch. Throws if the update is no longer available. */
export async function downloadAndRelaunch(): Promise<void> {
  const update = await check();
  if (!update) throw new Error("Update is no longer available");
  await update.downloadAndInstall();
  await relaunch();
}
