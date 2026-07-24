// FloatingScribe — a <Scribe> that hovers over the viewport (fixed, starts
// top-right) and can be dragged anywhere on screen. This is the reusable
// "detached, draggable" behavior; any route that wants a floating companion
// (create form, live job page, …) renders this instead of duplicating the
// pointer-drag plumbing. All drawing/mood lives in <Scribe>.
import { useRef, useState } from "react";
import { Scribe, type ScribeMood } from "./Scribe";

function clamp(v: number, min: number, max: number): number {
  return Math.min(Math.max(v, min), max);
}

export function FloatingScribe({
  mood,
  scale,
  speech,
  speechIntervalMs,
  className = "",
}: {
  mood: ScribeMood;
  scale?: number;
  speech?: string | string[];
  speechIntervalMs?: number;
  // Extra classes on the fixed root — e.g. "hidden lg:block" to gate by screen.
  className?: string;
}) {
  // `pos` is null until first dragged (default top-right CSS position applies);
  // once set, it fixes Scribe at those viewport coords.
  const rootRef = useRef<HTMLDivElement>(null);
  const grab = useRef<{ dx: number; dy: number } | null>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    grab.current = { dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!grab.current) return;
    const el = rootRef.current;
    const w = el?.offsetWidth ?? 0;
    const h = el?.offsetHeight ?? 0;
    setPos({
      x: clamp(e.clientX - grab.current.dx, 0, window.innerWidth - w),
      y: clamp(e.clientY - grab.current.dy, 0, window.innerHeight - h),
    });
  };
  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    grab.current = null;
    setDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div
      ref={rootRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      style={pos ? { left: pos.x, top: pos.y, right: "auto" } : undefined}
      className={`fixed right-6 top-24 z-40 touch-none ${
        dragging ? "cursor-grabbing" : "cursor-grab"
      } ${className}`}
      title="Drag me anywhere"
    >
      <Scribe
        mood={mood}
        scale={scale}
        speech={speech}
        speechIntervalMs={speechIntervalMs}
      />
    </div>
  );
}
