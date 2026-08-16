import type { ReactNode } from "react";

export function Button({
  children,
  onClick,
}: {
  children: ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded bg-zinc-800 px-3 py-1.5 text-sm text-zinc-100 hover:bg-zinc-700"
    >
      {children}
    </button>
  );
}
