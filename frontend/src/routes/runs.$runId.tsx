import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Fragment, useCallback, useMemo, useRef, useState } from "react";
import {
  useRun,
  useRunAudioSegments,
  useRunFrames,
  useRunGlobalStats,
  useRunParser,
  useRunSiblings,
} from "../api/hooks";
import type {
  AudioSegment,
  DetectionSource,
  FrameDetection,
  ParserResult,
  ShotBoundary,
} from "../lib/run-types";
import { ChatPanel } from "../components/ChatPanel";
import { Verdict } from "../components/ui";

export const Route = createFileRoute("/runs/$runId")({
  component: RunInspectorRoute,
});

// Keying the inspector by runId remounts it when switching between sibling runs
// of a batch job, so per-run state (playhead, video dims) never leaks across.
function RunInspectorRoute() {
  const { runId } = Route.useParams();
  return <RunInspector key={runId} runId={runId} />;
}

// ── constants ──────────────────────────────────────────────────────────────

// Only two layers are surfaced in the inspector:
//   - sam_mask → tracked objects (shown as "Object"; DINO records the same
//     boxes upstream, so we drop DINO here to avoid duplicates);
//   - ocr → on-screen text (shown as "Text").
// mtcnn (faces) is intentionally excluded — it is unreliable and SAM already
// tracks faces as objects.
const SOURCES: DetectionSource[] = ["sam_mask", "ocr"];

const SOURCE_COLORS: Record<DetectionSource, string> = {
  dino: "#3b82f6",
  ocr: "#22c55e",
  mtcnn: "#f59e0b",
  sam_mask: "#a855f7",
};

const SOURCE_LABELS: Record<DetectionSource, string> = {
  dino: "DINO",
  ocr: "Text",
  mtcnn: "Face",
  sam_mask: "Object",
};

// ── helpers ────────────────────────────────────────────────────────────────

// Bounds for a source's hold window (see holdWindowsFor). A detection stays on
// screen for at most this long after its sample, regardless of the source rate.
const MIN_HOLD_SEC = 0.4;
const MAX_HOLD_SEC = 2.5;
const FRAME_WINDOW_STEP_SEC = 5;
const FRAME_WINDOW_AHEAD_SEC = 5;

// Per-source "hold window": how long one sample keeps showing after its
// timestamp. Derived from the source's own median inter-sample gap (×3) so it
// self-tunes to each source's sampling rate:
//   - dense sources (sam_mask, ~every frame) → tiny window → never linger;
//   - sparse clustered sources (ocr/dino, sampled every N frames) → a window
//     wide enough to stay continuous WITHIN a cluster, but far smaller than a
//     real gap (a stretch with no text/objects), so stale boxes clear instead
//     of lingering until the next distant sample.
function holdWindowsFor(
  bySource: Map<string, FrameDetection[]>,
): Map<string, number> {
  const holds = new Map<string, number>();
  for (const [src, dets] of bySource) {
    const gaps: number[] = [];
    for (let i = 1; i < dets.length; i++) {
      const g = (dets[i]!.timestamp_sec ?? 0) - (dets[i - 1]!.timestamp_sec ?? 0);
      if (g > 0) gaps.push(g);
    }
    gaps.sort((a, b) => a - b);
    const median = gaps.length ? gaps[gaps.length >> 1]! : 0;
    holds.set(src, Math.min(MAX_HOLD_SEC, Math.max(MIN_HOLD_SEC, median * 3)));
  }
  return holds;
}

// Currently-active detections. For EACH source independently, find its most
// recent sample ≤ currentTime and show it only if we are still within that
// source's hold window (otherwise we are in a real gap and show nothing).
// Per-source is essential: a single global "latest timestamp" lets a dense
// source (sam_mask) constantly evict a sparse source's (ocr) boxes.
function activeDetections(
  bySource: Map<string, FrameDetection[]>,
  holdWindows: Map<string, number>,
  currentTime: number,
): FrameDetection[] {
  const out: FrameDetection[] = [];
  for (const [src, dets] of bySource) {
    // dets is sorted ascending by timestamp. Find the greatest ts <= now.
    let latestTs: number | null = null;
    for (const d of dets) {
      const ts = d.timestamp_sec ?? 0;
      if (ts <= currentTime) latestTs = ts;
      else break;
    }
    if (latestTs === null) continue;
    if (currentTime - latestTs > (holdWindows.get(src) ?? MAX_HOLD_SEC)) continue;
    for (const d of dets) {
      if (d.timestamp_sec === latestTs) out.push(d);
    }
  }
  return out;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// The shot index covering `t`, or null if none/unknown. Used to clear boxes at
// scene cuts: a sampled detection is held only while the playhead is still in
// the same shot it was sampled in, so old boxes never linger past a hard cut.
function shotIndexAt(shots: ShotBoundary[], t: number): number | null {
  for (const s of shots) {
    const start = s.start_sec ?? 0;
    const end = s.end_sec ?? Infinity;
    if (t >= start && t < end) return s.shot_index ?? null;
  }
  return null;
}

function frameWindowFor(currentTime: number, duration: number) {
  if (duration <= 0) return null;
  const bucketStart =
    Math.floor(currentTime / FRAME_WINDOW_STEP_SEC) * FRAME_WINDOW_STEP_SEC;
  return {
    fromSec: Math.max(0, bucketStart - MAX_HOLD_SEC),
    toSec: Math.min(
      duration,
      bucketStart + FRAME_WINDOW_STEP_SEC + FRAME_WINDOW_AHEAD_SEC,
    ),
    enabled: true,
  };
}

// ── VideoOverlay ─────────────────────────────────────────────────────────
// Absolutely-positioned SVG that sits over the <video>. The viewBox matches
// the video's natural pixel dimensions and preserveAspectRatio="xMidYMid meet"
// mirrors the video element's default object-fit:contain, so box coordinates
// need no manual scaling.

interface VideoDims {
  w: number;
  h: number;
  duration: number;
}

interface OverlayProps {
  detections: FrameDetection[];
  dims: VideoDims | null;
}

function VideoOverlay({ detections, dims }: OverlayProps) {
  if (!dims || !detections.length) return null;
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${dims.w} ${dims.h}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {detections.map((d) => {
        const x = d.box_x1 ?? 0;
        const y = d.box_y1 ?? 0;
        const w = (d.box_x2 ?? 0) - x;
        const h = (d.box_y2 ?? 0) - y;
        const color = SOURCE_COLORS[d.source as DetectionSource] ?? "#ef4444";
        const pct =
          d.confidence != null ? Math.round(d.confidence * 100) : null;
        const label = (d.label ?? d.text ?? d.source ?? "").slice(0, 16);
        return (
          <g key={d.id}>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill="none"
              stroke={color}
              strokeWidth={2}
            />
            {/* label tag inside top of box, only when the box is tall enough */}
            {h > 18 && (
              <>
                <rect
                  x={x}
                  y={y}
                  width={Math.min(w, label.length * 6.5 + 8)}
                  height={15}
                  fill={color}
                  fillOpacity={0.85}
                />
                <text
                  x={x + 3}
                  y={y + 11}
                  fontSize={10}
                  fill="white"
                  fontFamily="monospace"
                >
                  {label}
                  {pct != null ? ` ${pct}%` : ""}
                </text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ── RunTimeline ───────────────────────────────────────────────────────────
// Two horizontal tracks: shots (blue) and audio (green).
// Click anywhere to seek the video.

interface TimelineProps {
  duration: number;
  shots: ShotBoundary[];
  audio: AudioSegment[];
  currentTime: number;
  onSeek: (t: number) => void;
}

function RunTimeline({
  duration,
  shots,
  audio,
  currentTime,
  onSeek,
}: TimelineProps) {
  if (duration <= 0) return null;

  const pct = (t: number) => `${((t / duration) * 100).toFixed(3)}%`;

  // The segment being spoken at the playhead — drives the live caption below
  // and highlights its block in the audio track.
  const activeAudio = audio.find(
    (a) =>
      currentTime >= (a.start_time ?? 0) && currentTime <= (a.end_time ?? 0),
  );

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - rect.left) / rect.width) * duration);
  };

  return (
    <div className="space-y-1 select-none">
      <div className="flex justify-between text-xs text-neutral-400">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      <div
        className="relative flex h-14 cursor-pointer flex-col overflow-hidden rounded border bg-neutral-50"
        onClick={handleClick}
      >
        {/* Playhead — spans all tracks */}
        <div
          className="pointer-events-none absolute inset-y-0 z-10 w-px bg-red-500"
          style={{ left: pct(currentTime) }}
        />

        {/* Shot track */}
        <div className="relative flex-1 border-b border-neutral-200">
          <span className="pointer-events-none absolute left-1 top-0 z-20 text-xs text-neutral-400">
            shots
          </span>
          {shots.map((s) => (
            <div
              key={s.shot_index}
              className="absolute inset-y-1 rounded-sm border border-blue-300 bg-blue-100"
              style={{
                left: pct(s.start_sec ?? 0),
                width: pct(s.duration_sec ?? 0),
              }}
              title={`Shot ${s.shot_index} — ${formatTime(s.start_sec ?? 0)}–${formatTime(s.end_sec ?? 0)}`}
            />
          ))}
        </div>

        {/* Audio track */}
        <div className="relative flex-1">
          <span className="pointer-events-none absolute left-1 top-0 z-20 text-xs text-neutral-400">
            audio
          </span>
          {audio.map((a) => (
            <div
              key={a.id}
              className={`absolute inset-y-1 overflow-hidden truncate rounded-sm border px-0.5 text-xs ${
                a.id === activeAudio?.id
                  ? "border-green-500 bg-green-300 text-green-900"
                  : "border-green-300 bg-green-100 text-green-800"
              }`}
              style={{
                left: pct(a.start_time ?? 0),
                width: pct((a.end_time ?? 0) - (a.start_time ?? 0)),
              }}
              title={a.text ?? ""}
            />
          ))}
        </div>
      </div>

      {/* Live caption: the transcript for the segment under the playhead. */}
      <div className="min-h-[2.5rem] rounded border border-neutral-200 bg-neutral-50 px-3 py-1.5 text-sm">
        {activeAudio?.text ? (
          <p className="text-neutral-700">
            <span className="mr-2 align-middle text-xs text-green-600">🔊</span>
            {activeAudio.text}
          </p>
        ) : (
          <p className="italic text-neutral-300">— no speech at this moment —</p>
        )}
      </div>
    </div>
  );
}

// ── ParserTable ───────────────────────────────────────────────────────────
// Criteria table grouped by feature_category. Each row expands to show the
// LLM explanation, criteria text, and (optionally) the full prompt.

function ParserTable({ results }: { results: ParserResult[] }) {
  // Allow any number of rows open at once (independent toggles).
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggleRow = (id: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  if (!results.length) {
    return (
      <p className="text-sm text-neutral-500">No parser results for this run.</p>
    );
  }

  const byCategory = results.reduce<Record<string, ParserResult[]>>((acc, r) => {
    const cat = r.feature_category ?? "Other";
    (acc[cat] ??= []).push(r);
    return acc;
  }, {});

  // Headline counts so the pass/fail split is visible before scanning the table.
  const passed = results.filter((r) => r.evaluation === true).length;
  const failed = results.filter((r) => r.evaluation === false).length;
  const passPct = results.length
    ? Math.round((passed / results.length) * 100)
    : 0;

  return (
    <div className="space-y-5">
      {/* score summary */}
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3">
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-semibold text-neutral-900">{passPct}%</span>
          <span className="text-sm text-neutral-500">pass rate</span>
        </div>
        <div className="ml-auto flex items-center gap-4 text-sm">
          <span className="flex items-center gap-1.5 text-green-700">
            <span className="h-2 w-2 rounded-full bg-green-500" />
            {passed} passed
          </span>
          <span className="flex items-center gap-1.5 text-red-700">
            <span className="h-2 w-2 rounded-full bg-red-500" />
            {failed} failed
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
          <div
            className="h-full rounded-full bg-green-500"
            style={{ width: `${passPct}%` }}
          />
        </div>
      </div>

      {Object.entries(byCategory).map(([cat, rows]) => (
        <div key={cat}>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-neutral-500">
            {cat}
          </h3>
          <table className="w-full text-sm">
            <tbody>
              {rows.map((row) => (
                <Fragment key={row.id}>
                  <tr
                    className="cursor-pointer border-t hover:bg-neutral-50"
                    onClick={() => toggleRow(row.id)}
                  >
                    <td className="w-8 py-1.5">
                      <Verdict value={row.evaluation} />
                    </td>
                    <td className="py-1.5 pr-4">{row.feature_name ?? "—"}</td>
                    <td className="py-1.5 text-xs text-neutral-400">
                      {expanded.has(row.id) ? "▲" : "▼"}
                    </td>
                  </tr>
                  {expanded.has(row.id) && (
                    <tr>
                      <td
                        colSpan={3}
                        className="bg-neutral-50 px-4 py-3 text-xs"
                      >
                        <div className="space-y-2">
                          {row.feature_criteria && (
                            <p className="text-neutral-600">
                              <span className="font-medium">Criteria: </span>
                              {row.feature_criteria}
                            </p>
                          )}
                          {row.llm_explanation && (
                            <p className="text-neutral-700">
                              <span className="font-medium">Explanation: </span>
                              {row.llm_explanation}
                            </p>
                          )}
                          {row.llm_prompt && (
                            <details>
                              <summary className="cursor-pointer text-neutral-500">
                                Show prompt
                              </summary>
                              <pre className="mt-1 overflow-auto whitespace-pre-wrap rounded bg-neutral-100 p-2 text-neutral-700">
                                {row.llm_prompt}
                              </pre>
                            </details>
                          )}
                          {row.langsmith_run_id && (
                            <p>
                              <span className="font-medium">LangSmith: </span>
                              <code className="text-blue-600">
                                {row.langsmith_run_id}
                              </code>
                            </p>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

// ── RunInspector ──────────────────────────────────────────────────────────

function RunInspector({ runId }: { runId: string }) {
  const navigate = useNavigate();

  const siblings = useRunSiblings(runId);

  const siblingRuns = siblings.data ?? [];
  const currentIdx = siblingRuns.findIndex((s) => s.run_id === runId);
  const currentSibling = currentIdx >= 0 ? siblingRuns[currentIdx] : undefined;
  const shouldFetchRun =
    siblings.isSuccess &&
    (currentSibling ? currentSibling.status === "completed" : true);
  const run = useRun(runId, shouldFetchRun);
  const detailQueriesEnabled = shouldFetchRun && !!run.data;
  const globalStats = useRunGlobalStats(runId, detailQueriesEnabled);
  const audioSegments = useRunAudioSegments(runId, detailQueriesEnabled);
  const parser = useRunParser(runId, detailQueriesEnabled);

  const videoRef = useRef<HTMLVideoElement>(null);
  // Fullscreen the wrapper (video + SVG overlay together), not the <video>:
  // native video fullscreen promotes only the video element to the fullscreen
  // layer, so the sibling overlay would vanish.
  const stageRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [dims, setDims] = useState<VideoDims | null>(null);
  const [enabledSources, setEnabledSources] = useState<Set<DetectionSource>>(
    new Set(SOURCES),
  );

  const duration = dims?.duration ?? 0;
  const frameWindow = useMemo(
    () => frameWindowFor(currentTime, duration),
    [currentTime, duration],
  );
  const frames = useRunFrames(
    runId,
    frameWindow
      ? { ...frameWindow, enabled: detailQueriesEnabled }
      : { fromSec: 0, toSec: 0, enabled: false },
  );

  const toggleSource = useCallback((src: DetectionSource) => {
    setEnabledSources((prev) => {
      const next = new Set(prev);
      if (next.has(src)) next.delete(src);
      else next.add(src);
      return next;
    });
  }, []);

  // Group detections by source once (each group sorted ascending by time).
  // activeDetections() then holds each source at its own most-recent sample.
  const bySource = useMemo(() => {
    const map = new Map<string, FrameDetection[]>();
    for (const d of frames.data ?? []) {
      const src = d.source ?? "unknown";
      (map.get(src) ?? map.set(src, []).get(src)!).push(d);
    }
    for (const dets of map.values()) {
      dets.sort((a, b) => (a.timestamp_sec ?? 0) - (b.timestamp_sec ?? 0));
    }
    return map;
  }, [frames.data]);

  // Per-source hold windows, recomputed only when the grouped data changes.
  const holdWindows = useMemo(() => holdWindowsFor(bySource), [bySource]);

  // The shot under the playhead — used to evict boxes from a previous shot the
  // instant the video cuts, instead of holding the last sample across the cut.
  const currentShotIndex = useMemo(
    () => shotIndexAt(globalStats.data?.shot_boundaries ?? [], currentTime),
    [globalStats.data, currentTime],
  );

  // Currently-active detections, filtered by the layer toggles and constrained
  // to the current shot (confidence is intentionally not filtered — show every
  // detection). The shot constraint only applies when both the playhead's shot
  // and the detection's shot are known.
  const visibleDetections = useMemo(() => {
    return activeDetections(bySource, holdWindows, currentTime)
      .filter((d) => enabledSources.has(d.source as DetectionSource))
      .filter(
        (d) =>
          currentShotIndex == null ||
          d.shot_index == null ||
          d.shot_index === currentShotIndex,
      );
  }, [bySource, holdWindows, currentTime, enabledSources, currentShotIndex]);

  const handleSeek = useCallback((t: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      setCurrentTime(t);
    }
  }, []);

  const shots = globalStats.data?.shot_boundaries ?? [];
  const audio = audioSegments.data ?? [];
  const parserResults = parser.data ?? [];

  // Sibling runs share this run's batch job; the switcher flips between them.
  // Only completed runs have inspectable data (the extractor writes the run row
  // at the end), so navigation is restricted to completed siblings and the
  // current run shows a "still processing" notice until it finishes.
  const hasSiblings = siblingRuns.length > 1;
  const runReady = detailQueriesEnabled;
  const waitingStatus =
    currentSibling && currentSibling.status !== "completed"
      ? currentSibling.status
      : siblings.isLoading || run.isLoading
        ? "loading"
        : "unavailable";
  // All siblings share one parent; use it for the "back to batch" link.
  const parentJobId = siblingRuns[0]?.parent_job_id ?? null;
  const prevCompleted =
    currentIdx > 0
      ? [...siblingRuns.slice(0, currentIdx)]
          .reverse()
          .find((s) => s.status === "completed")
      : undefined;
  const nextCompleted =
    currentIdx >= 0
      ? siblingRuns.slice(currentIdx + 1).find((s) => s.status === "completed")
      : undefined;
  const goToRun = (rid: string) =>
    void navigate({ to: "/runs/$runId", params: { runId: rid } });

  return (
    <div className="space-y-6">
      {/* heading */}
      <div>
        {parentJobId && (
          <Link
            to="/jobs/$jobId"
            params={{ jobId: parentJobId }}
            className="text-sm text-blue-600 hover:underline"
          >
            ← Batch
          </Link>
        )}
        <h1 className="mt-1 text-2xl font-semibold">Run inspector</h1>
        {run.data?.video_name && (
          <p className="text-neutral-500">{run.data.video_name}</p>
        )}
        {run.error && (
          <p className="text-sm text-red-600">
            {(run.error as Error).message}
          </p>
        )}
      </div>

      {/* ── run switcher (batch jobs only) ── */}
      {hasSiblings && (
        <div className="flex items-center gap-3 rounded-lg border border-neutral-200 bg-white px-3 py-2 text-sm shadow-sm">
          <span className="text-neutral-500">
            Run {currentIdx + 1} of {siblingRuns.length} in this job
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              disabled={!prevCompleted}
              onClick={() => prevCompleted && goToRun(prevCompleted.run_id)}
              className="rounded border px-2 py-1 text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
            >
              ← Prev
            </button>
            <select
              value={runId}
              onChange={(e) => goToRun(e.target.value)}
              className="max-w-[16rem] rounded border px-2 py-1"
            >
              {siblingRuns.map((s, i) => (
                <option
                  key={s.run_id}
                  value={s.run_id}
                  // Only completed runs can be inspected; others stay visible
                  // (so you can see the batch) but aren't selectable.
                  disabled={s.status !== "completed" && s.run_id !== runId}
                >
                  {i + 1}. {s.video_name ?? s.run_id} — {s.status}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={!nextCompleted}
              onClick={() => nextCompleted && goToRun(nextCompleted.run_id)}
              className="rounded border px-2 py-1 text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
            >
              Next →
            </button>
          </div>
        </div>
      )}

      {!runReady ? (
        <section className="rounded-lg border border-neutral-200 bg-white p-8 text-center shadow-sm">
          <p className="text-neutral-600">
            {waitingStatus === "loading"
              ? "Loading run status..."
              : waitingStatus === "unavailable"
                ? "Run data is unavailable."
                : `This run is still ${waitingStatus} - nothing to inspect yet.`}
          </p>
          {hasSiblings && (
            <p className="mt-1 text-sm text-neutral-400">
              Pick a completed run above, or check back when it finishes.
            </p>
          )}
        </section>
      ) : (
        <>
      {/* ── video + SVG overlay ── */}
      <section className="space-y-3 rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <h2 className="font-medium">Video</h2>

        <div
          ref={stageRef}
          className="video-stage relative overflow-hidden rounded bg-black"
        >
          <video
            ref={videoRef}
            src={`/api/runs/${runId}/video`}
            controls
            // Steer users to the wrapper-fullscreen button below; the native
            // video fullscreen button would drop the overlay.
            controlsList="nofullscreen"
            className="w-full"
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              setDims({ w: v.videoWidth, h: v.videoHeight, duration: v.duration });
            }}
          />
          <VideoOverlay detections={visibleDetections} dims={dims} />
          <button
            type="button"
            onClick={() => {
              if (document.fullscreenElement) void document.exitFullscreen();
              else void stageRef.current?.requestFullscreen?.();
            }}
            title="Toggle fullscreen"
            className="absolute right-2 top-2 z-10 rounded bg-black/60 px-2 py-1 text-xs text-white hover:bg-black/80"
          >
            ⛶ Fullscreen
          </button>
        </div>

        {/* layer controls */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
          <div className="flex gap-3">
            {SOURCES.map((src) => (
              <label
                key={src}
                className="flex cursor-pointer select-none items-center gap-1"
              >
                <input
                  type="checkbox"
                  checked={enabledSources.has(src)}
                  onChange={() => toggleSource(src)}
                />
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: SOURCE_COLORS[src] }}
                />
                {SOURCE_LABELS[src]}
              </label>
            ))}
          </div>

          <span className="ml-auto text-xs text-neutral-400">
            {frames.isLoading
              ? "Loading detections…"
              : frames.data
                ? `${visibleDetections.length} shown`
                : ""}
          </span>
        </div>

        <a
          href={`/api/runs/${runId}/tracked-video`}
          className="inline-block text-sm text-blue-600 hover:underline"
        >
          Download tracked_output.mp4
        </a>
      </section>

      {/* ── timeline ── */}
      <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 font-medium">Timeline</h2>
        {globalStats.isLoading ? (
          <p className="text-sm text-neutral-500">Loading…</p>
        ) : (
          <RunTimeline
            duration={duration}
            shots={shots}
            audio={audio}
            currentTime={currentTime}
            onSeek={handleSeek}
          />
        )}
      </section>

      {/* ── ABCD criteria ── */}
      <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 font-medium">ABCD criteria</h2>
        {parser.isLoading ? (
          <p className="text-sm text-neutral-500">Loading…</p>
        ) : (
          <ParserTable results={parserResults} />
        )}
      </section>

      {/* ── Advisory chat (web-app-plan §13) ── */}
      <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 font-medium">Ask about this video</h2>
        <ChatPanel runId={runId} />
      </section>
        </>
      )}
    </div>
  );
}
