// Small display helpers. Kept dependency-free on purpose.

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// Duration between two ISO timestamps, as "1m 23s".
export function formatDuration(
  start: string | null | undefined,
  end: string | null | undefined,
): string {
  if (!start || !end) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (Number.isNaN(ms) || ms < 0) return "—";
  const s = Math.round(ms / 1000);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

// Tailwind classes per job status, for status pills.
export function statusColor(status: string): string {
  switch (status) {
    case "completed":
      return "bg-green-100 text-green-800";
    case "running":
      return "bg-blue-100 text-blue-800";
    case "queued":
      return "bg-amber-100 text-amber-800";
    case "failed":
      return "bg-red-100 text-red-800";
    case "canceled":
      return "bg-neutral-200 text-neutral-700";
    default:
      return "bg-neutral-100 text-neutral-700";
  }
}
