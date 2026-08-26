import { useEffect, useRef } from "react";
import type { RawFrame } from "../../backend/types";

/// Renders a decoded preview frame (raw RGB24, base64) into a canvas. Frames
/// are decoded into an offscreen buffer, then composited with one `drawImage`
/// so rapid updates — or a resolution change — never show a torn/blank canvas
/// on webkit2gtk. The visible canvas is only resized when the frame dimensions
/// actually change (resizing clears it and can flicker at playback rate). The
/// canvas keeps its aspect and is contained within the parent (like
/// `object-contain`).
export default function FrameCanvas({
  frame,
  className,
}: {
  frame: RawFrame;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const off = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const src = frame.data;
    const w = frame.width;
    const h = frame.height;
    // RGB24 -> RGBA for ImageData.
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < src.length; i += 3, j += 4) {
      rgba[j] = src[i];
      rgba[j + 1] = src[i + 1];
      rgba[j + 2] = src[i + 2];
      rgba[j + 3] = 255;
    }
    // Decode into an offscreen buffer, then composite atomically.
    if (!off.current) off.current = document.createElement("canvas");
    const buf = off.current;
    buf.width = w;
    buf.height = h;
    buf.getContext("2d")?.putImageData(new ImageData(rgba, w, h), 0, 0);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    canvas.getContext("2d")?.drawImage(buf, 0, 0);
  }, [frame]);

  return (
    <canvas
      ref={ref}
      className={className}
      style={{ width: "100%", height: "100%", objectFit: "contain" }}
    />
  );
}
