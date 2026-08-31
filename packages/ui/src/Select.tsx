import { useCallback, useEffect, useRef, useState } from "react";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  className?: string;
  /** Compact sizing for dense toolbars/headers. */
  size?: "sm" | "md";
}

const Chevron = ({ open }: { open: boolean }) => (
  <svg
    viewBox="0 0 20 20"
    fill="currentColor"
    className={`h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
  >
    <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
  </svg>
);

export function Select({ value, onChange, options, className = "", size = "md" }: SelectProps) {
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(() => options.findIndex((o) => o.value === value));
  const ref = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  const selected = options.find((o) => o.value === value);
  const selectedLabel = selected?.label ?? value;

  // Sync active index when dropdown opens or value changes externally
  useEffect(() => {
    if (open) setActiveIdx(options.findIndex((o) => o.value === value));
  }, [open, value]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const pick = useCallback(
    (val: string) => {
      onChange(val);
      setOpen(false);
      btnRef.current?.focus();
    },
    [onChange],
  );

  const moveActive = (dir: 1 | -1) => {
    setActiveIdx((prev) => {
      let next = prev + dir;
      // skip disabled
      while (next >= 0 && next < options.length && options[next].disabled) next += dir;
      return next >= 0 && next < options.length ? next : prev;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        moveActive(1);
        break;
      case "ArrowUp":
        e.preventDefault();
        moveActive(-1);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (activeIdx >= 0 && !options[activeIdx].disabled) pick(options[activeIdx].value);
        break;
      case "Escape":
        e.preventDefault();
        setOpen(false);
        btnRef.current?.focus();
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  return (
    <div ref={ref} className={`relative ${className}`} onKeyDown={handleKeyDown}>
      <button
        ref={btnRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`flex w-full items-center justify-between gap-1 rounded-lg border border-slate-300 bg-white text-left text-slate-800 outline-none focus:border-indigo-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 ${
          size === "sm" ? "px-2 py-0.5 text-[11px]" : "p-1.5"
        }`}
      >
        <span className="truncate">{selectedLabel}</span>
        <Chevron open={open} />
      </button>

      {open && (
        <ul className="absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-lg border border-slate-300 bg-white py-1 shadow-lg dark:border-slate-700 dark:bg-slate-950">
          {options.map((opt, i) => (
            <li key={opt.value}>
              <button
                type="button"
                disabled={opt.disabled}
                onClick={() => pick(opt.value)}
                onMouseEnter={() => setActiveIdx(i)}
                className={`block w-full px-2.5 py-1.5 text-left text-xs ${
                  i === activeIdx ? "bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300" : "text-slate-700 dark:text-slate-200"
                } ${opt.value === value ? "font-medium" : ""} ${opt.disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800"}`}
              >
                {opt.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
