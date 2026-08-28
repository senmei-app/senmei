import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useI18n } from "../i18n";

interface MenuItem {
  key: string;
  label?: string;
  action?: () => void;
  separator?: boolean;
  shortcut?: string;
  disabled?: boolean;
  children?: MenuItem[];
}

interface Menu {
  key: string;
  label: string;
  items: MenuItem[];
}

export default function MenuBar({
  hotkeys,
  onImportFile,
  onImportFolder,
  onBatchFolder,
  onCloseProject,
  onExportProject,
  onSettings,
  onGithub,
  onAbout,
  onSelectAll,
  onDeleteSelected,
  onAddAllToQueue,
  onAddSelectedToQueue,
  onProcessSelected,
  onProcessAll,
  onToggleFullscreen,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  hasFiles,
  hasSelection,
}: {
  hotkeys: Record<string, string>;
  onImportFile: () => void;
  onImportFolder: () => void;
  onBatchFolder: () => void;
  onCloseProject: () => void;
  onExportProject: () => void;
  onSettings: () => void;
  onGithub: () => void;
  onAbout: () => void;
  onSelectAll: () => void;
  onDeleteSelected: () => void;
  onAddAllToQueue: () => void;
  onAddSelectedToQueue: () => void;
  onProcessSelected: () => void;
  onProcessAll: () => void;
  onToggleFullscreen: () => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  hasFiles: boolean;
  hasSelection: boolean;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState<string | null>(null);
  const [focusIdx, setFocusIdx] = useState(0);
  const menuKeys = ["file", "edit", "view", "process", "help"];
  const navRef = useRef<HTMLDivElement>(null);

  const menuKeysRef = useRef(menuKeys);
  menuKeysRef.current = menuKeys;
  const openRef = useRef(open);
  openRef.current = open;
  const focusIdxRef = useRef(focusIdx);
  focusIdxRef.current = focusIdx;

  // Alt+letter opens menus; Escape closes; arrow keys navigate.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && openRef.current) {
        e.preventDefault();
        setOpen(null);
        return;
      }
      if (!openRef.current) {
        if (!e.altKey || e.ctrlKey || e.metaKey) return;
        const map: Record<string, string> = { f: "file", e: "edit", v: "view", p: "process", h: "help" };
        const key = map[e.key.toLowerCase()];
        if (key) {
          e.preventDefault();
          setOpen(key);
          setFocusIdx(0);
        }
        return;
      }
      // Menu is open — handle arrow navigation + Enter
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setFocusIdx((i) => i + 1);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setFocusIdx((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        const idx = menuKeysRef.current.indexOf(openRef.current);
        const next = menuKeysRef.current[(idx + 1) % menuKeysRef.current.length];
        setOpen(next);
        setFocusIdx(0);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        const idx = menuKeysRef.current.indexOf(openRef.current);
        const prev = menuKeysRef.current[(idx - 1 + menuKeysRef.current.length) % menuKeysRef.current.length];
        setOpen(prev);
        setFocusIdx(0);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Scroll focused item into view when navigating with arrows.
  useEffect(() => {
    navRef.current?.scrollIntoView({ block: "nearest" });
  }, [focusIdx, open]);

  // Simulate a keyboard event so Monitor's hotkey listener picks it up.
  const dispatchHotkey = (combo: string) => {
    const parts = combo.split("+");
    const key = parts.pop()!;
    window.dispatchEvent(new KeyboardEvent("keydown", {
      key: key === "Space" ? " " : key.toLowerCase(),
      ctrlKey: parts.includes("Ctrl"),
      shiftKey: parts.includes("Shift"),
      altKey: parts.includes("Alt"),
      bubbles: true,
    }));
  };

  const menus: Menu[] = useMemo(() => [
    {
      key: "file",
      label: t("menu.file"),
      items: [
        {
          key: "import",
          label: t("menu.importVideos"),
          children: [
            { key: "import-file", label: t("menu.importFile"), shortcut: hotkeys.openFile, action: onImportFile },
            { key: "import-folder", label: t("menu.importFolder"), action: onImportFolder },
          ],
        },
        { key: "close", label: t("menu.closeProject"), action: onCloseProject },
        { key: "export", label: t("menu.exportProject"), shortcut: hotkeys.exportProject, action: onExportProject },
        { key: "sep", separator: true },
        { key: "settings", label: t("menu.settings"), action: onSettings },
      ],
    },
    {
      key: "edit",
      label: t("menu.edit"),
      items: [
        { key: "undo", label: t("menu.undo"), shortcut: hotkeys.undo, action: onUndo, disabled: !canUndo },
        { key: "redo", label: t("menu.redo"), shortcut: hotkeys.redo, action: onRedo, disabled: !canRedo },
        { key: "sep", separator: true },
        { key: "select-all", label: t("menu.selectAll"), shortcut: hotkeys.selectAll, action: onSelectAll, disabled: !hasFiles },
        { key: "delete-selected", label: t("menu.deleteSelected"), shortcut: hotkeys.deleteSelected, action: onDeleteSelected, disabled: !hasSelection },
      ],
    },
    {
      key: "view",
      label: t("menu.view"),
      items: [
        { key: "view-original", label: t("menu.viewOriginal"), shortcut: hotkeys.modeSource, action: () => dispatchHotkey(hotkeys.modeSource) },
        { key: "view-result", label: t("menu.viewResult"), shortcut: hotkeys.modeResult, action: () => dispatchHotkey(hotkeys.modeResult) },
        { key: "view-compare", label: t("menu.viewCompare"), shortcut: hotkeys.modeCompare, action: () => dispatchHotkey(hotkeys.modeCompare) },
        { key: "view-ab", label: t("menu.viewAB"), shortcut: hotkeys.modeAB, action: () => dispatchHotkey(hotkeys.modeAB) },
        { key: "sep-view", separator: true },
        { key: "toggle-meta", label: t("menu.toggleMeta"), shortcut: hotkeys.toggleMeta, action: () => dispatchHotkey(hotkeys.toggleMeta) },
        { key: "sep-view2", separator: true },
        { key: "full-video", label: t("menu.fullVideo"), shortcut: hotkeys.toggleFullscreen, action: onToggleFullscreen },
      ],
    },
    {
      key: "process",
      label: t("menu.process"),
      items: [
        { key: "add-all", label: t("menu.addAllQueue"), action: onAddAllToQueue, disabled: !hasFiles },
        { key: "add-selected", label: t("menu.addSelectedQueue"), action: onAddSelectedToQueue, disabled: !hasSelection },
        { key: "batch-folder", label: t("menu.batchFolder"), action: onBatchFolder },
        { key: "sep1", separator: true },
        { key: "process-selected", label: t("menu.processSelected"), action: onProcessSelected, disabled: !hasSelection },
        { key: "process-queue", label: t("menu.processQueue"), shortcut: hotkeys.render, action: onProcessAll, disabled: !hasFiles },
      ],
    },
    {
      key: "help",
      label: t("menu.help"),
      items: [
        { key: "github", label: t("menu.github"), action: onGithub },
        { key: "about", label: t("menu.about"), action: onAbout },
      ],
    },
  ], [t, hotkeys, onImportFile, onCloseProject, onExportProject, onUndo, onRedo, canUndo, canRedo, onSelectAll, onDeleteSelected, hasFiles, hasSelection, onToggleFullscreen, onAddAllToQueue, onAddSelectedToQueue, onBatchFolder, onProcessSelected, onProcessAll, onGithub, onAbout, onSettings]);

  const itemCls = (disabled?: boolean) =>
    "flex w-full items-center justify-between gap-4 px-3 py-1.5 text-left text-[11px] " +
    (disabled
      ? "cursor-default text-slate-400 opacity-60 dark:text-slate-600"
      : "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800");

  const renderItem = (item: MenuItem): ReactNode => {
    if (item.separator) {
      return <div key={item.key} className="my-1 border-t border-slate-200 dark:border-slate-800" />;
    }

    if (item.children) {
      return (
        <div key={item.key} className="group/item relative">
          <div className="flex items-center justify-between px-3 py-1.5 text-[11px] text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800">
            <span>{item.label}</span>
            <span className="ml-2 text-slate-400 dark:text-slate-500">›</span>
          </div>
          <div className="absolute left-full top-0 z-50 hidden w-48 rounded-lg border border-slate-200 bg-white py-1 shadow-xl group-hover/item:block dark:border-slate-800 dark:bg-slate-900">
            {item.children.map((child) => (
              <button
                key={child.key}
                disabled={child.disabled}
                onClick={() => {
                  setOpen(null);
                  child.action?.();
                }}
                className={itemCls(child.disabled)}
              >
                <span>{child.label}</span>
                {child.shortcut && (
                  <span className="text-[11px] text-slate-400 dark:text-slate-500">{child.shortcut}</span>
                )}
              </button>
            ))}
          </div>
        </div>
      );
    }

    return (
      <button
        key={item.key}
        disabled={item.disabled}
        onClick={() => {
          setOpen(null);
          item.action?.();
        }}
        className={itemCls(item.disabled)}
      >
        <span>{item.label}</span>
        {item.shortcut && (
          <span className="text-[11px] text-slate-400 dark:text-slate-500">{item.shortcut}</span>
        )}
      </button>
    );
  };

  return (
    <nav className="relative flex items-center gap-4 font-medium text-slate-600 dark:text-slate-400">
      {open && <div className="fixed inset-0 z-40" onClick={() => setOpen(null)} />}

      {menus.map((menu) => (
        <div key={menu.key} className="relative">
          <button
            onClick={() => setOpen(open === menu.key ? null : menu.key)}
            className="transition hover:text-slate-900 dark:hover:text-slate-100"
          >
            {menu.label}
          </button>

          {open === menu.key && (
            <div className="absolute left-0 top-full z-50 mt-1 w-48 rounded-lg border border-slate-200 bg-white py-1 shadow-xl dark:border-slate-800 dark:bg-slate-900">
              {menu.items.map((item, idx) => {
                const btn = renderItem(item);
                if (item.separator) return btn;
                // Clone to attach ref for auto-scroll on keyboard nav.
                if (idx === focusIdx && btn && typeof btn === "object" && "props" in btn) {
                  return (
                    <div key={item.key} ref={navRef}>
                      {btn}
                    </div>
                  );
                }
                return btn;
              })}
            </div>
          )}
        </div>
      ))}
    </nav>
  );
}
