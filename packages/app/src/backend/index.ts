//! Backend selection for the UI. Tauri mode → Tauri IPC; browser mode → HTTP
//! to `senmei-server --http`. Loaded lazily so each surface only ships where
//! it is used. `VITE_SENMEI_MOCK=1` forces the in-memory demo backend (dev).

import { isTauri } from "@tauri-apps/api/core";
import type { Backend } from "./types";

export type { Backend, FrameSource } from "./types";

/// Synchronous transport check: true when running outside Tauri (web/mock).
/// Use for render-path decisions that have no server equivalent (e.g. sample
/// renders can't write into a browser-only project dir).
export const isWeb = () => !isTauri();

let promise: Promise<Backend> | null = null;
let cached: Backend | null = null;

export function backend(): Promise<Backend> {
  promise ??= (async () => {
    if (import.meta.env.VITE_SENMEI_MOCK === "1") {
      cached = (await import("../mock")).mockBackend;
    } else if (isTauri()) {
      cached = (await import("./tauri")).tauriBackend;
    } else {
      cached = (await import("./http")).httpBackend;
    }
    return cached;
  })();
  return promise;
}

/// The resolved backend, or `null` before the first `backend()` completes.
/// Only for synchronous transport queries (e.g. native-video URLs); async
/// callers should `await backend()`.
export function backendSync(): Backend | null {
  return cached;
}
