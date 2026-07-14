// Small shared UI primitives. Neutral and dependency-free — just Tailwind
// classes — so every route renders the same spinners, skeletons, pills, and
// verdict badges instead of ad-hoc glyphs and "Loading…" text.
import type { ReactNode } from "react";

// Spinning ring for in-flight state (replaces the static ◔ glyph). Inherits
// `currentColor`, so the caller sets the color via text-* on a wrapper.
export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="loading"
      className={`inline-block animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
    />
  );
}

// A single shimmering placeholder block for loading states.
export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-neutral-200 ${className}`} />;
}

// Semantic status → light fill + inset ring + leading dot. Only meaning-bearing
// colors here; no brand hue (kept neutral for the MVP).
const PILL: Record<string, { wrap: string; dot: string }> = {
  completed: { wrap: "bg-green-50 text-green-700 ring-green-600/20", dot: "bg-green-500" },
  running: { wrap: "bg-blue-50 text-blue-700 ring-blue-600/20", dot: "bg-blue-500" },
  queued: { wrap: "bg-amber-50 text-amber-700 ring-amber-600/20", dot: "bg-amber-500" },
  failed: { wrap: "bg-red-50 text-red-700 ring-red-600/20", dot: "bg-red-500" },
  canceled: { wrap: "bg-neutral-100 text-neutral-600 ring-neutral-500/20", dot: "bg-neutral-400" },
};

const PILL_FALLBACK = {
  wrap: "bg-neutral-100 text-neutral-600 ring-neutral-500/20",
  dot: "bg-neutral-400",
};

export function StatusPill({ status }: { status: string }) {
  const s = PILL[status] ?? PILL_FALLBACK;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${s.wrap}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${s.dot} ${status === "running" ? "animate-pulse" : ""}`}
      />
      {status}
    </span>
  );
}

// Pass / fail / n-a verdict badge — SVG glyphs render identically on every OS,
// unlike the ✅/❌ emoji they replace.
export function Verdict({ value }: { value: boolean | null }) {
  if (value === true) {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-green-100 text-green-700">
        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3.5 8.5l3 3 6-7" />
        </svg>
      </span>
    );
  }
  if (value === false) {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-red-100 text-red-700">
        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
          <path d="M4 4l8 8M12 4l-8 8" />
        </svg>
      </span>
    );
  }
  return (
    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-neutral-100 text-neutral-400">
      –
    </span>
  );
}

// Centered empty state: icon slot, title, optional description and action.
export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-4 py-16 text-center">
      {icon && <div className="mb-3 text-neutral-300">{icon}</div>}
      <p className="text-sm font-medium text-neutral-700">{title}</p>
      {description && (
        <p className="mt-1 max-w-sm text-sm text-neutral-500">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

// The ClipScribe mark — a clapperboard whose body doubles as a written slate.
// Same artwork as public/favicon.svg; reused in the nav header. `size` is px.
export function Logo({ size = 24 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <rect width="32" height="32" rx="7" fill="#171717" />
      <rect x="6" y="7.5" width="20" height="5.5" rx="1" fill="#fafafa" />
      <g stroke="#171717" strokeWidth="1.6">
        <line x1="10" y1="7" x2="8" y2="13.5" />
        <line x1="15" y1="7" x2="13" y2="13.5" />
        <line x1="20" y1="7" x2="18" y2="13.5" />
        <line x1="25" y1="7" x2="23" y2="13.5" />
      </g>
      <rect x="6" y="14.5" width="20" height="10.5" rx="1.5" fill="#fafafa" />
      <g stroke="#171717" strokeWidth="1.4" strokeLinecap="round">
        <line x1="9" y1="18.5" x2="20" y2="18.5" />
        <line x1="9" y1="21.5" x2="16" y2="21.5" />
      </g>
    </svg>
  );
}
