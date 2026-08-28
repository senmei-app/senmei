import { createContext, useContext, type ReactNode } from "react";

import { de } from "./de";
import { en } from "./en";
import { ja } from "./ja";
import { zh } from "./zh";

export type Lang = "en" | "de" | "zh" | "ja";

const messages: Record<Lang, Record<string, string>> = { en, de, zh, ja };

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

export function I18nProvider({
  lang,
  setLang,
  children,
}: {
  lang: Lang;
  setLang: (lang: Lang) => void;
  children: ReactNode;
}) {
  const t = (key: string) => messages[lang][key] ?? messages.en[key] ?? key;

  return <I18nContext.Provider value={{ lang, setLang, t }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
