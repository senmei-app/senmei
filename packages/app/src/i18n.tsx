import { createContext, useContext, useState, type ReactNode } from "react";

export type Lang = "en" | "de";

const messages: Record<Lang, Record<string, string>> = {
  en: {
    "menu.file": "File",
    "menu.edit": "Edit",
    "menu.process": "Process",
    "menu.view": "View",
    "menu.help": "Help",
    "render.start": "Start Render",
    "media.title": "Media Library",
    "media.drop": "Drop video here",
    "media.formats": "MP4, MKV, MOV up to 4K",
    "monitor.placeholder": "[ Live Monitor / Split-View Canvas ]",
    "monitor.original": "Original: 1920x1080",
    "monitor.senmei": "Senmei: 3840x2160 (60fps)",
    "timeline.sample10": "10s",
    "timeline.sample15": "15s Sample",
    "timeline.sample30": "30s",
    "inspector.title": "Inspector Settings",
    "fi.title": "Frame Interpolation",
    "fi.model": "Model",
    "fi.fps": "FPS Multiplier",
    "up.title": "AI Upscaling",
    "up.model": "Model",
    "sample.preview": "Preview sample (15s)",
    "status.cuda": "CUDA: RTX 4080 (16GB VRAM)",
    "status.backend": "Backend: libtorch",
    "status.ready": "Ready",
  },
  de: {
    "menu.file": "Datei",
    "menu.edit": "Bearbeiten",
    "menu.process": "Prozess",
    "menu.view": "Ansicht",
    "menu.help": "Hilfe",
    "render.start": "Render Starten",
    "media.title": "Medien-Bibliothek",
    "media.drop": "Video hier ablegen",
    "media.formats": "MP4, MKV, MOV bis 4K",
    "monitor.placeholder": "[ Live Monitor / Split-View Canvas ]",
    "monitor.original": "Original: 1920x1080",
    "monitor.senmei": "Senmei: 3840x2160 (60fps)",
    "timeline.sample10": "10s",
    "timeline.sample15": "15s Sample",
    "timeline.sample30": "30s",
    "inspector.title": "Inspector Settings",
    "fi.title": "Frame Interpolation",
    "fi.model": "Modell",
    "fi.fps": "FPS Multiplikator",
    "up.title": "AI Upscaling",
    "up.model": "Modell",
    "sample.preview": "Sample vorschauen (15s)",
    "status.cuda": "CUDA: RTX 4080 (16GB VRAM)",
    "status.backend": "Backend: libtorch",
    "status.ready": "Bereit",
  },
};

interface I18nValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string) => string;
}

const I18nContext = createContext<I18nValue>({
  lang: "en",
  setLang: () => {},
  t: (key) => key,
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");

  const t = (key: string) => messages[lang][key] ?? messages.en[key] ?? key;

  return <I18nContext.Provider value={{ lang, setLang, t }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
