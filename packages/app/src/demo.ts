// Lazy demo backend for the non-Tauri (browser) mode. The mock module is only
// loaded when `!isTauri()`, so it never ships in the production bundle's main
// chunk. `loadDemo` caches the module, so the mutable demo arrays (projects,
// models, videos) stay a single shared instance.
type DemoModule = typeof import("./mock");

let demoPromise: Promise<DemoModule> | null = null;

export function loadDemo(): Promise<DemoModule> {
  demoPromise ??= import("./mock");
  return demoPromise;
}
