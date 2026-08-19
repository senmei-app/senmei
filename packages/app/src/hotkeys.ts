// Configurable key-bindings. Action ids are stable keys persisted in the
// settings file; only overrides are stored, the defaults fill the rest.

export interface HotkeyAction {
  id: string;
  labelKey: string;
  default: string;
}

export const HOTKEY_ACTIONS: HotkeyAction[] = [
  { id: "openFile", labelKey: "hotkeys.openFile", default: "Ctrl+O" },
  { id: "selectAll", labelKey: "hotkeys.selectAll", default: "Ctrl+A" },
  { id: "deleteSelected", labelKey: "hotkeys.deleteSelected", default: "Delete" },
  { id: "render", labelKey: "hotkeys.render", default: "Ctrl+R" },
  { id: "exportProject", labelKey: "hotkeys.exportProject", default: "Ctrl+E" },
  { id: "togglePlay", labelKey: "hotkeys.togglePlay", default: "Space" },
  { id: "toggleFullscreen", labelKey: "hotkeys.toggleFullscreen", default: "F11" },
];

export function defaultHotkey(id: string): string {
  return HOTKEY_ACTIONS.find((a) => a.id === id)?.default ?? "";
}

/// Merge persisted overrides over the defaults.
export function resolveHotkeys(overrides: Record<string, string>): Record<string, string> {
  const resolved: Record<string, string> = {};
  for (const a of HOTKEY_ACTIONS) resolved[a.id] = overrides[a.id] ?? a.default;
  return resolved;
}

/// Normalize a KeyboardEvent to a combo like "Ctrl+Shift+R" or "Space".
/// Ctrl and Cmd are interchangeable; modifier-only presses yield "".
export function comboFromEvent(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push("Ctrl");
  if (e.altKey) parts.push("Alt");
  if (e.shiftKey) parts.push("Shift");
  const key = e.key === " " ? "Space" : e.key.length === 1 ? e.key.toUpperCase() : e.key;
  if (!key || key === "Control" || key === "Meta" || key === "Alt" || key === "Shift") return "";
  parts.push(key);
  return parts.join("+");
}
