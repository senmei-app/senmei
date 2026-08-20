//! Promise-based path-input dialog for the web backend (no native picker).
//! `openPathDialog` parks a resolver; the `PathDialog` component (mounted at
//! the app root) shows the modal and resolves it via `submitPath`.

export interface PathDialogOptions {
  title: string;
  placeholder?: string;
  default?: string;
  multiple?: boolean;
}

let current: PathDialogOptions | null = null;
let resolveFn: ((value: string | null) => void) | null = null;
const listeners = new Set<() => void>();

function notify() {
  for (const l of listeners) l();
}

export function openPathDialog(opts: PathDialogOptions): Promise<string | null> {
  current = opts;
  notify();
  return new Promise((resolve) => {
    resolveFn = resolve;
  });
}

export function submitPath(value: string | null) {
  const r = resolveFn;
  resolveFn = null;
  current = null;
  notify();
  r?.(value);
}

export function getPathDialog(): PathDialogOptions | null {
  return current;
}

export function subscribePathDialog(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}
