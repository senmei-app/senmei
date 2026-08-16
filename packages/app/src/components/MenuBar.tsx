import { useState, type ReactNode } from "react";
import { useI18n } from "../i18n";

interface MenuItem {
  key: string;
  label?: string;
  action?: () => void;
  separator?: boolean;
  children?: MenuItem[];
}

interface Menu {
  key: string;
  label: string;
  items: MenuItem[];
}

export default function MenuBar({
  onImportFile,
  onImportFolder,
  onCloseProject,
  onGithub,
}: {
  onImportFile: () => void;
  onImportFolder: () => void;
  onCloseProject: () => void;
  onGithub: () => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState<string | null>(null);

  const menus: Menu[] = [
    {
      key: "file",
      label: t("menu.file"),
      items: [
        {
          key: "import",
          label: t("menu.importVideos"),
          children: [
            { key: "import-file", label: t("menu.importFile"), action: onImportFile },
            { key: "import-folder", label: t("menu.importFolder"), action: onImportFolder },
          ],
        },
        { key: "close", label: t("menu.closeProject"), action: onCloseProject },
        { key: "sep", separator: true },
        { key: "settings", label: t("menu.settings") },
      ],
    },
    {
      key: "edit",
      label: t("menu.edit"),
      items: [
        { key: "undo", label: t("menu.undo") },
        { key: "redo", label: t("menu.redo") },
      ],
    },
    {
      key: "help",
      label: t("menu.help"),
      items: [
        { key: "github", label: t("menu.github"), action: onGithub },
        { key: "about", label: t("menu.about") },
      ],
    },
  ];

  const renderItem = (item: MenuItem): ReactNode => {
    if (item.separator) {
      return <div key={item.key} className="my-1 border-t border-slate-800" />;
    }

    if (item.children) {
      return (
        <div key={item.key} className="group/item relative">
          <div className="flex items-center justify-between px-3 py-1.5 text-[11px] text-slate-300 hover:bg-slate-800">
            <span>{item.label}</span>
            <span className="ml-2 text-slate-500">›</span>
          </div>
          <div className="absolute left-full top-0 z-50 hidden w-40 rounded-lg border border-slate-800 bg-slate-900 py-1 shadow-xl group-hover/item:block">
            {item.children.map((child) => (
              <button
                key={child.key}
                onClick={() => {
                  setOpen(null);
                  child.action?.();
                }}
                className="block w-full px-3 py-1.5 text-left text-[11px] text-slate-300 hover:bg-slate-800 hover:text-slate-100"
              >
                {child.label}
              </button>
            ))}
          </div>
        </div>
      );
    }

    return (
      <button
        key={item.key}
        onClick={() => {
          setOpen(null);
          item.action?.();
        }}
        className="block w-full px-3 py-1.5 text-left text-[11px] text-slate-300 hover:bg-slate-800 hover:text-slate-100"
      >
        {item.label}
      </button>
    );
  };

  return (
    <nav className="relative flex items-center space-x-4 text-slate-400 font-medium">
      {open && <div className="fixed inset-0 z-40" onClick={() => setOpen(null)} />}

      {menus.map((menu) => (
        <div key={menu.key} className="relative">
          <button
            onClick={() => setOpen(open === menu.key ? null : menu.key)}
            className="hover:text-slate-100 transition"
          >
            {menu.label}
          </button>

          {open === menu.key && (
            <div className="absolute left-0 top-full z-50 mt-1 w-48 rounded-lg border border-slate-800 bg-slate-900 py-1 shadow-xl">
              {menu.items.map(renderItem)}
            </div>
          )}
        </div>
      ))}
    </nav>
  );
}
