// Cross-platform path helpers: file paths come from the OS (Windows uses `\`),
// so splitting handles both separators; joins use `/`, which Windows APIs
// also accept.
const SEP = /[\\/]/;

export function basename(p: string): string {
  const parts = p.split(SEP);
  return parts[parts.length - 1] ?? "";
}

export function dirname(p: string): string {
  const parts = p.split(SEP);
  parts.pop();
  return parts.join("/") || ".";
}

export function joinPath(...parts: string[]): string {
  return parts.filter((p) => p.length > 0).join("/");
}
