// Small display helpers. Kept dependency-free on purpose.

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// Seconds as "45s" / "1m 23s".
function formatSeconds(sec: number): string {
  const s = Math.round(sec);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

// Duration between two ISO timestamps, as "1m 23s". "—" if either end is
// missing or the range is invalid (used for a single run's start→finish).
export function formatDuration(
  start: string | null | undefined,
  end: string | null | undefined,
): string {
  if (!start || !end) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (Number.isNaN(ms) || ms < 0) return "—";
  return formatSeconds(ms / 1000);
}

// Total wall-clock a batch was actively working, as "1m 23s".
//
// Children run with bounded concurrency (today concurrency=1, but not
// guaranteed), so neither a naive span (max finished − min started, which
// counts idle gaps between runs) nor a plain sum (which double-counts any
// overlap) is right. Instead take the UNION of each child's [started, finished]
// interval: sort by start, greedily extend the current interval while the next
// one intersects it, and bank + restart when a gap appears. Summing the merged
// lengths gives the true time at least one run was in flight. Children without
// both timestamps (queued/running/never-dispatched) are skipped; "—" if none
// have run.
export function formatMergedDuration(
  intervals: ReadonlyArray<{
    started_at?: string | null;
    finished_at?: string | null;
  }>,
): string {
  const spans: Array<[number, number]> = [];
  for (const it of intervals) {
    if (!it.started_at || !it.finished_at) continue;
    const start = new Date(it.started_at).getTime();
    const end = new Date(it.finished_at).getTime();
    if (Number.isNaN(start) || Number.isNaN(end) || end < start) continue;
    spans.push([start, end]);
  }
  if (!spans.length) return "—";

  spans.sort((a, b) => a[0] - b[0]);
  let totalMs = 0;
  let [curStart, curEnd] = spans[0]!;
  for (let i = 1; i < spans.length; i++) {
    const [start, end] = spans[i]!;
    if (start <= curEnd) {
      // Overlaps (or touches) the current interval — extend it.
      if (end > curEnd) curEnd = end;
    } else {
      // Gap — bank the merged interval and start a fresh one.
      totalMs += curEnd - curStart;
      curStart = start;
      curEnd = end;
    }
  }
  totalMs += curEnd - curStart;
  return formatSeconds(totalMs / 1000);
}
