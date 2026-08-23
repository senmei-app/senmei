import { useEffect, useRef } from "react";
import type { RawFrame } from "../../backend/types";

/// Renders a decoded preview frame (raw RGB24, base64) into a canvas via
/// `putImageData` — no `<img>`/PNG round-trip, works in any webview. The
/// canvas keeps its aspect and is contained within the parent (like
/// `object-contain`), so wrap it in a flex-centered box.
export default function FrameCanvas({
  frame,
  className,
}: {
  frame: RawFrame;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const bin = atob(frame.data);
    const w = frame.width;
    const h = frame.height;
    // RGB24 -> RGBA for ImageData.
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < bin.length; i += 3, j += 4) {
      rgba[j] = bin.charCodeAt(i);
      rgba[j + 1] = bin.charCodeAt(i + 1);
      rgba[j + 2] = bin.charCodeAt(i + 2);
      rgba[j + 3] = 255;
    }
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
  }, [frame]);

  return (
    <canvas
      ref={ref}
      className={className}
      style={{ width: "100%", height: "100%", objectFit: "contain" }}
    />
  );
}
