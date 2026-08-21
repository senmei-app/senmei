// Pure time/format helpers for the Monitor.

// "mm:ss.cc" clock from ms.
export function fmt(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const cs = Math.floor((ms % 1000) / 10);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(sec)}.${pad(cs)}`;
}

// "55s", "10m", "1h", "1m30s" or a bare number (seconds).
export function parseDuration(input: string): number | null {
  const s = input.trim().toLowerCase();
  if (!s) return null;
  if (/^\d+(\.\d+)?$/.test(s)) return Math.round(Number(s) * 1000);
  const re = /(\d+)([smh])/g;
  let ms = 0;
  let m: RegExpExecArray | null;
  let any = false;
  while ((m = re.exec(s)) !== null) {
    any = true;
    const v = Number(m[1]);
    ms += m[2] === "s" ? v * 1000 : m[2] === "m" ? v * 60000 : v * 3600000;
  }
  return any ? ms : null;
}

export function fmtDuration(ms: number): string {
  if (ms % 60000 === 0) return `${Math.round(ms / 60000)}m`;
  if (ms % 1000 === 0) return `${Math.round(ms / 1000)}s`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// Round a time to the nearest frame boundary (whole ms) so a rendered sample
// starts on the exact source frame that contains it (keeps compare in lockstep).
export function snapFrame(ms: number, fps: number): number {
  const frameMs = fps > 0 ? 1000 / fps : 0;
  return frameMs > 0 ? Math.round(Math.round(ms / frameMs) * frameMs) : Math.round(ms);
}
