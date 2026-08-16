import type { ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost";

const styles: Record<Variant, string> = {
  primary: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-md shadow-indigo-600/30",
  secondary:
    "bg-slate-200 text-slate-700 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700",
  ghost: "text-slate-600 hover:bg-slate-200/60 dark:text-slate-400 dark:hover:bg-slate-800/60",
};

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: Variant;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 text-xs font-medium transition active:scale-95 disabled:opacity-40 ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Chip({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      className={
        active
          ? "rounded-md bg-emerald-500/15 px-2 py-1 font-mono text-[11px] text-emerald-600 dark:text-emerald-400"
          : "rounded-md bg-slate-200 px-2 py-1 font-mono text-[11px] text-slate-400 dark:bg-slate-800 dark:text-slate-600"
      }
    >
      {label}
    </span>
  );
}
