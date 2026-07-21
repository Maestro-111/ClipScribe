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

// Download menu for the ABCD report. `baseHref` is the export endpoint (e.g.
// "/api/runs/{id}/parser/export"); each item appends ?format=. The server sets
// Content-Disposition: attachment, so a plain anchor triggers the download.
// Native <details> handles open/close (and click-away) without extra state;
// the marker is hidden (incl. Safari's ::-webkit-details-marker) so we control
// the chevron, which flips when open.
export function ExportMenu({
  baseHref,
  label = "Export",
}: {
  baseHref: string;
  label?: string;
}) {
  return (
    <details className="group relative inline-block [&_summary::-webkit-details-marker]:hidden">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 whitespace-nowrap rounded-md border border-neutral-300 bg-white px-3 py-1.5 text-sm font-medium text-neutral-700 shadow-sm transition-colors hover:border-neutral-400 hover:bg-neutral-50">
        {/* download tray icon */}
        <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4 text-neutral-500">
          <path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.3 8.23a.75.75 0 1 0-1.1 1.02l4.25 4.5a.75.75 0 0 0 1.1 0l4.25-4.5a.75.75 0 1 0-1.1-1.02l-2.95 3.134V2.75Z" />
          <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
        </svg>
        {label}
        {/* chevron flips on open */}
        <svg
          viewBox="0 0 20 20"
          fill="currentColor"
          className="h-3.5 w-3.5 text-neutral-400 transition-transform group-open:rotate-180"
        >
          <path
            fillRule="evenodd"
            d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.17l3.71-3.94a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
            clipRule="evenodd"
          />
        </svg>
      </summary>
      <div className="absolute right-0 z-20 mt-1.5 w-40 overflow-hidden rounded-lg border border-neutral-200 bg-white py-1 shadow-lg ring-1 ring-black/5">
        <a
          href={`${baseHref}?format=xlsx`}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-50"
        >
          <span className="inline-flex h-5 w-5 items-center justify-center rounded bg-green-100 text-[10px] font-semibold text-green-700">
            X
          </span>
          Excel (.xlsx)
        </a>
        <a
          href={`${baseHref}?format=csv`}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-neutral-700 hover:bg-neutral-50"
        >
          <span className="inline-flex h-5 w-5 items-center justify-center rounded bg-neutral-100 text-[10px] font-semibold text-neutral-600">
            ,
          </span>
          CSV (.csv)
        </a>
      </div>
    </details>
  );
}

// Prev/Next pager shared by the jobs list and the per-run batch table. Kept
// dumb: the caller owns the page index and decides when each direction is
// available (server-side "full page" heuristic for the list, exact slice
// bounds for the client-side table). `label` shows the current range/page.
export function Pagination({
  canPrev,
  canNext,
  onPrev,
  onNext,
  label,
}: {
  canPrev: boolean;
  canNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  label?: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-neutral-500">{label}</span>
      <div className="flex items-center gap-2">
        <button
          onClick={onPrev}
          disabled={!canPrev}
          className="rounded border border-neutral-300 bg-white px-2.5 py-1 font-medium text-neutral-700 hover:border-neutral-400 hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          ← Prev
        </button>
        <button
          onClick={onNext}
          disabled={!canNext}
          className="rounded border border-neutral-300 bg-white px-2.5 py-1 font-medium text-neutral-700 hover:border-neutral-400 hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next →
        </button>
      </div>
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
