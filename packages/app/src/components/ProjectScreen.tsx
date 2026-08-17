import { useState } from "react";
import { Button } from "@senmei/ui";
import type { ProjectEntry } from "@senmei/bridge";
import { useI18n } from "../i18n";
import FfmpegIndicator from "./FfmpegIndicator";
import WindowControls from "./WindowControls";

export default function ProjectScreen({
  projects,
  onCreate,
  onOpen,
  onBrowse,
  onDelete,
}: {
  projects: ProjectEntry[];
  onCreate: (name: string) => void;
  onOpen: (path: string) => void;
  onBrowse: () => void;
  onDelete: (path: string) => void;
}) {
  const { t } = useI18n();
  const [name, setName] = useState("");

  const create = () => {
    if (name.trim()) {
      onCreate(name.trim());
      setName("");
    }
  };

  const remove = (path: string) => {
    if (window.confirm(t("project.deleteConfirm"))) onDelete(path);
  };

  return (
    <div className="flex h-screen w-full flex-col bg-slate-100 font-sans text-slate-900 select-none antialiased dark:bg-slate-950 dark:text-slate-200">
      <header className="flex h-12 w-full items-center justify-between px-4">
        <div data-tauri-drag-region className="flex-1 self-stretch" />
        <WindowControls />
      </header>
      <FfmpegIndicator />
      <div className="flex flex-1 flex-col items-center justify-center">
        <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-600 text-3xl font-bold text-white shadow-lg shadow-indigo-500/30">
          鮮
        </div>
        <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">Senmei</h1>
        <p className="mt-1 text-xs text-slate-500">{t("project.subtitle")}</p>

        <div className="mt-8 flex w-72 flex-col gap-2">
          <label className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">{t("project.new")}</label>
          <div className="flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && create()}
              placeholder={t("project.namePlaceholder")}
              className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
            />
            <Button onClick={create} disabled={!name.trim()}>{t("project.create")}</Button>
          </div>
        </div>

        <div className="mt-6 w-72">
          <label className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">{t("project.existing")}</label>
          <div className="mt-2 flex max-h-48 flex-col gap-1 overflow-y-auto">
            {projects.length === 0 ? (
              <p className="text-xs text-slate-500">{t("project.none")}</p>
            ) : (
              projects.map((p) => (
                <div
                  key={p.path}
                  className="group flex items-center rounded-lg border border-slate-200 bg-white transition hover:bg-slate-200 dark:border-slate-800 dark:bg-slate-900 dark:hover:bg-slate-800"
                >
                  <button
                    onClick={() => onOpen(p.path)}
                    className="flex-1 truncate px-3 py-2 text-left text-sm text-slate-700 hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100"
                  >
                    {p.name}
                  </button>
                  <button
                    onClick={() => remove(p.path)}
                    title={t("project.delete")}
                    className="px-2.5 py-2 text-sm text-slate-400 opacity-0 transition hover:text-red-500 group-hover:opacity-100 dark:hover:text-red-400"
                  >
                    🗑
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <Button onClick={onBrowse} variant="secondary" className="mt-6 w-72 py-2.5">
          {t("project.browse")}
        </Button>
      </div>
    </div>
  );
}
