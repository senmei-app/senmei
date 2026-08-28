// Configurable key-bindings. Action ids are stable keys persisted in the
// settings file; only overrides are stored, the defaults fill the rest.

export interface HotkeyAction {
  id: string;
  labelKey: string;
  default: string;
  /** Settings grouping: "global" | "playback" | "view" | "media". */
  group: string;
}

export const HOTKEY_ACTIONS: HotkeyAction[] = [
  { id: "openFile", labelKey: "hotkeys.openFile", default: "Ctrl+O", group: "global" },
  { id: "selectAll", labelKey: "hotkeys.selectAll", default: "Ctrl+A", group: "global" },
  { id: "deleteSelected", labelKey: "hotkeys.deleteSelected", default: "Delete", group: "global" },
  { id: "undo", labelKey: "hotkeys.undo", default: "Ctrl+Z", group: "global" },
  { id: "redo", labelKey: "hotkeys.redo", default: "Ctrl+Shift+Z", group: "global" },
  { id: "render", labelKey: "hotkeys.render", default: "Ctrl+R", group: "global" },
  { id: "exportProject", labelKey: "hotkeys.exportProject", default: "Ctrl+E", group: "global" },
  { id: "toggleFullscreen", labelKey: "hotkeys.toggleFullscreen", default: "F11", group: "global" },
  { id: "togglePlay", labelKey: "hotkeys.togglePlay", default: "Space", group: "playback" },
  { id: "mute", labelKey: "hotkeys.mute", default: "M", group: "playback" },
  { id: "volumeUp", labelKey: "hotkeys.volumeUp", default: "ArrowUp", group: "playback" },
  { id: "volumeDown", labelKey: "hotkeys.volumeDown", default: "ArrowDown", group: "playback" },
  { id: "seekBack", labelKey: "hotkeys.seekBack", default: "ArrowLeft", group: "playback" },
  { id: "seekForward", labelKey: "hotkeys.seekForward", default: "ArrowRight", group: "playback" },
  { id: "toggleMeta", labelKey: "hotkeys.toggleMeta", default: "I", group: "view" },
  { id: "modeSource", labelKey: "hotkeys.modeSource", default: "1", group: "view" },
  { id: "modeResult", labelKey: "hotkeys.modeResult", default: "2", group: "view" },
  { id: "modeCompare", labelKey: "hotkeys.modeCompare", default: "3", group: "view" },
  { id: "modeAB", labelKey: "hotkeys.modeAB", default: "4", group: "view" },
  { id: "viewLibrary", labelKey: "hotkeys.viewLibrary", default: "Ctrl+1", group: "media" },
  { id: "viewQueue", labelKey: "hotkeys.viewQueue", default: "Ctrl+2", group: "media" },
  { id: "toggleMultiSelect", labelKey: "hotkeys.toggleMultiSelect", default: "Ctrl+Shift+A", group: "media" },
  { id: "renderSample", labelKey: "hotkeys.renderSample", default: "Ctrl+Shift+R", group: "media" },
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
