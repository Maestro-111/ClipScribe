import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useReducer, useRef } from "react";
import {
  useCancelJob,
  useJob,
  useJobProgress,
  useRetryJob,
  type JobChild,
} from "../api/hooks";
import { Spinner, StatusPill } from "../components/ui";

// "/jobs/{id}" — the job page (web-app-plan §7 page 3, step 9).
//
// A job is either a batch parent (has children) or a leaf run. The route
// dispatches: parents render a per-run batch panel; leaf runs render the live
// SSE view below.
export const Route = createFileRoute("/jobs/$jobId")({
  component: JobPage,
});

function JobPage() {
  const { jobId } = Route.useParams();
  const job = useJob(jobId);
  // Wait for the first fetch before choosing a view: rendering the leaf SSE
  // view for what turns out to be a parent would open a needless stream to the
  // parent's (empty) progress channel.
  if (!job.data) {
    return <p className="text-neutral-500">Loading…</p>;
  }
  // A parent (batch) job carries children; a leaf run does not.
  return (job.data.children?.length ?? 0) > 0 ? <BatchJob /> : <LeafJob />;
}

// ── Live progress state ────────────────────────────────────────────────────
type PhaseName =
  | "scene_detection"
  | "audio"
  | "shot_processing"
  | "finalize"
  | "parse";

type PhaseStatus = "pending" | "running" | "completed";

const PHASE_LABELS: Record<PhaseName, string> = {
  scene_detection: "Scene detection",
  audio: "Audio",
  shot_processing: "Shots",
  finalize: "Finalize",
  parse: "Parse",
};

// Weights approximate observed wall-clock share (web-app-plan §7); normalized
// over whatever phases the current mode actually runs.
const PHASE_WEIGHT: Record<PhaseName, number> = {
  scene_detection: 0.05,
  audio: 0.15,
  shot_processing: 0.7,
  finalize: 0.1,
  parse: 0.3,
};

const LOG_TAIL_CAP = 500;

interface CurrentShot {
  idx: number;
  description?: string;
  dinoPrompt?: string;
  targets?: string[];
  framesDone: number;
}

interface LiveState {
  streamStatus: "connecting" | "live" | "closed";
  videoName?: string;
  phaseOrder: PhaseName[];
  phases: Partial<Record<PhaseName, PhaseStatus>>;
  totalShots?: number;
  shotsCompleted: number;
  currentShot?: CurrentShot;
  audioSegments: { start: number; end: number; text: string }[];
  logs: { level: string; message: string }[];
  runId?: string;
  error?: string;
}

const initialState: LiveState = {
  streamStatus: "connecting",
  phaseOrder: [],
  phases: {},
  shotsCompleted: 0,
  audioSegments: [],
  logs: [],
};

// A stream event as delivered by the SSE endpoint: {type, data}.
type Event = { type: string; data: Record<string, unknown> };
type Action = Event | { type: "@stream/open" } | { type: "@stream/close" };

function reducer(state: LiveState, action: Action): LiveState {
  switch (action.type) {
    case "@stream/open":
      return { ...state, streamStatus: "live" };
    case "@stream/close":
      return { ...state, streamStatus: "closed" };

    case "job.started": {
      const phases = (action.data.phases as PhaseName[]) ?? [];
      return {
        ...state,
        videoName: (action.data.video_name as string) ?? state.videoName,
        phaseOrder: phases,
        phases: Object.fromEntries(phases.map((p) => [p, "pending"])),
      };
    }
    case "phase.started": {
      const phase = action.data.phase as PhaseName;
      const totalShots =
        phase === "shot_processing"
          ? (action.data.total_shots as number) ?? state.totalShots
          : state.totalShots;
      return {
        ...state,
        totalShots,
        phases: { ...state.phases, [phase]: "running" },
      };
    }
    case "phase.completed": {
      const phase = action.data.phase as PhaseName;
      const totalShots =
        (action.data.total_shots as number | undefined) ?? state.totalShots;
      return {
        ...state,
        totalShots,
        phases: { ...state.phases, [phase]: "completed" },
      };
    }
    case "shot.started":
      return {
        ...state,
        currentShot: { idx: action.data.shot_idx as number, framesDone: 0 },
      };
    case "shot.scene_described":
      return {
        ...state,
        currentShot: state.currentShot && {
          ...state.currentShot,
          description: action.data.description as string,
          dinoPrompt: action.data.dino_prompt as string,
        },
      };
    case "shot.taxonomy_resolved":
      return {
        ...state,
        currentShot: state.currentShot && {
          ...state.currentShot,
          targets: action.data.targets as string[],
        },
      };
    case "shot.frame_processed":
      return {
        ...state,
        currentShot: state.currentShot && {
          ...state.currentShot,
          framesDone: state.currentShot.framesDone + 1,
        },
      };
    case "shot.completed":
      return { ...state, shotsCompleted: state.shotsCompleted + 1 };
    case "audio.segment":
      return {
        ...state,
        audioSegments: [
          ...state.audioSegments,
          {
            start: action.data.start as number,
            end: action.data.end as number,
            text: action.data.text as string,
          },
        ],
      };
    case "log":
      return {
        ...state,
        logs: [
          ...state.logs.slice(-(LOG_TAIL_CAP - 1)),
          {
            level: (action.data.level as string) ?? "INFO",
            message: (action.data.message as string) ?? "",
          },
        ],
      };
    case "job.completed":
      return {
        ...state,
        runId: (action.data.run_id as string) ?? state.runId,
        streamStatus: "closed",
      };
    case "job.failed":
      return {
        ...state,
        error: (action.data.error as string) ?? "Job failed",
        streamStatus: "closed",
      };
    default:
      return state;
  }
}

function overallProgress(state: LiveState): number {
  if (!state.phaseOrder.length) return 0;
  let total = 0;
  let done = 0;
  for (const phase of state.phaseOrder) {
    const w = PHASE_WEIGHT[phase] ?? 0.1;
    total += w;
    const status = state.phases[phase];
    if (status === "completed") {
      done += w;
    } else if (status === "running") {
      if (phase === "shot_processing" && state.totalShots) {
        done += w * Math.min(state.shotsCompleted / state.totalShots, 0.99);
      } else {
        done += w * 0.1;
      }
    }
  }
  return total ? done / total : 0;
}

// ── Leaf run (live SSE view) ────────────────────────────────────────────────
function LeafJob() {
  const { jobId } = Route.useParams();
  const job = useJob(jobId);
  const parentJobId = job.data?.parent_job_id ?? null;
  const cancel = useCancelJob();
  const [state, dispatch] = useReducer(reducer, initialState);
  const logRef = useRef<HTMLDivElement>(null);

  // Open the SSE stream once per job id; the endpoint replays history first, so
  // this fills in even if the page loads mid-run or just after completion.
  useEffect(() => {
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.onopen = () => dispatch({ type: "@stream/open" });
    es.onmessage = (e) => {
      try {
        dispatch(JSON.parse(e.data) as Event);
      } catch {
        /* ignore malformed frame */
      }
    };
    es.onerror = () => {
      // The server closes the stream on a terminal event; EventSource surfaces
      // that as an error. Close so the browser doesn't auto-reconnect forever.
      es.close();
      dispatch({ type: "@stream/close" });
    };
    return () => es.close();
  }, [jobId]);

  // Keep the log tail pinned to the newest line.
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [state.logs]);

  const status = job.data?.status ?? "queued";
  const runId = job.data?.run_id ?? state.runId;
  const isActive = status === "queued" || status === "running";
  const pct = Math.round(overallProgress(state) * 100);

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          {parentJobId ? (
            <Link
              to="/jobs/$jobId"
              params={{ jobId: parentJobId }}
              className="text-sm text-blue-600 hover:underline"
            >
              ← Batch
            </Link>
          ) : (
            <Link to="/" className="text-sm text-blue-600 hover:underline">
              ← Jobs
            </Link>
          )}
          <h1 className="mt-1 text-2xl font-semibold">
            {state.videoName ?? job.data?.video_name ?? "Job"}
          </h1>
        </div>
        <StatusPill status={status} />
      </div>

      {/* progress bar + actions */}
      <section className="space-y-3 rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <div className="h-3 w-full overflow-hidden rounded-full bg-neutral-100">
          <div
            className={`h-full rounded-full transition-all ${
              status === "failed" ? "bg-red-500" : "bg-blue-500"
            } ${status === "running" ? "progress-active" : ""}`}
            style={{ width: `${status === "completed" ? 100 : pct}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-neutral-500">
            {status === "completed"
              ? "Done"
              : status === "failed"
                ? "Failed"
                : `${pct}%`}
          </span>
          <div className="flex items-center gap-3">
            {status === "completed" && runId && (
              <Link
                to="/runs/$runId"
                params={{ runId }}
                className="rounded bg-blue-600 px-3 py-1 font-medium text-white hover:bg-blue-700"
              >
                View run →
              </Link>
            )}
            {isActive && (
              <button
                onClick={() => cancel.mutate(jobId)}
                disabled={cancel.isPending}
                className="rounded border border-red-200 bg-white px-3 py-1 font-medium text-red-600 hover:border-red-400 hover:bg-red-50 disabled:opacity-50"
              >
                ■ Cancel
              </button>
            )}
          </div>
        </div>
        {state.error && <p className="text-sm text-red-600">{state.error}</p>}
      </section>

      <div className="grid gap-6 md:grid-cols-2">
        {/* phase tree */}
        <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
          <h2 className="mb-3 font-medium">Phases</h2>
          {state.phaseOrder.length === 0 ? (
            <p className="text-sm text-neutral-500">Waiting for the job to start…</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {state.phaseOrder.map((phase) => {
                const ps = state.phases[phase] ?? "pending";
                const shots =
                  phase === "shot_processing" && state.totalShots
                    ? ` ${state.shotsCompleted}/${state.totalShots}`
                    : "";
                return (
                  <li key={phase} className="flex items-center gap-2.5">
                    <span className="flex h-4 w-4 items-center justify-center">
                      {ps === "completed" ? (
                        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-green-100 text-green-700">
                          <svg viewBox="0 0 16 16" className="h-2.5 w-2.5" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M3.5 8.5l3 3 6-7" />
                          </svg>
                        </span>
                      ) : ps === "running" ? (
                        <Spinner className="h-3.5 w-3.5 text-blue-500" />
                      ) : (
                        <span className="h-2 w-2 rounded-full bg-neutral-300" />
                      )}
                    </span>
                    <span
                      className={
                        ps === "pending"
                          ? "text-neutral-400"
                          : ps === "running"
                            ? "font-medium text-neutral-900"
                            : "text-neutral-700"
                      }
                    >
                      {PHASE_LABELS[phase]}
                      {shots}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        {/* current shot */}
        <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
          <h2 className="mb-3 font-medium">Current shot</h2>
          {state.currentShot ? (
            <div className="space-y-3 text-sm">
              <p className="font-medium">
                Shot {state.currentShot.idx}
                <span className="ml-2 font-normal text-neutral-500">
                  {state.currentShot.framesDone} frames
                </span>
              </p>
              {state.currentShot.description && (
                <div>
                  <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Scene description
                  </h3>
                  <p className="text-neutral-700">
                    {state.currentShot.description}
                  </p>
                </div>
              )}
              {state.currentShot.dinoPrompt && (
                <div>
                  <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Scene visual targets
                  </h3>
                  <p className="text-xs text-neutral-600">
                    {state.currentShot.dinoPrompt}
                  </p>
                </div>
              )}
              {state.currentShot.targets && state.currentShot.targets.length > 0 && (
                <div>
                  <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Scene vocabulary
                  </h3>
                  <div className="flex flex-wrap gap-1">
                    {state.currentShot.targets.map((t) => (
                      <span
                        key={t}
                        className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs text-neutral-600"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-neutral-500">No shot in progress.</p>
          )}
        </section>
      </div>

      {/* log tail */}
      <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-medium">Logs</h2>
          <span className="flex items-center gap-1.5 text-xs text-neutral-400">
            {state.streamStatus === "live" ? (
              <>
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-500" />
                live
              </>
            ) : state.streamStatus === "connecting" ? (
              "connecting…"
            ) : (
              "closed"
            )}
          </span>
        </div>
        <div
          ref={logRef}
          className="h-56 overflow-y-auto rounded bg-neutral-900 p-3 font-mono text-xs text-neutral-100"
        >
          {state.logs.length === 0 ? (
            <p className="text-neutral-500">No log output yet.</p>
          ) : (
            state.logs.map((l, i) => (
              <div key={i} className="whitespace-pre-wrap">
                <span
                  className={
                    l.level === "ERROR" || l.level === "WARNING"
                      ? "text-amber-400"
                      : "text-neutral-400"
                  }
                >
                  {l.level}
                </span>{" "}
                {l.message}
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}

// ── Batch job (per-run panel) ───────────────────────────────────────────────
// A parent job fans out to one child run per video. This lists the children
// with per-run status/progress and links into each run's live view + inspector.
function BatchJob() {
  const { jobId } = Route.useParams();
  const job = useJob(jobId);
  const cancel = useCancelJob();

  const children = job.data?.children ?? [];
  const status = job.data?.status ?? "queued";
  const total = children.length;
  const done = children.filter((c) => c.status === "completed").length;
  const failed = children.filter((c) => c.status === "failed").length;
  const canceled = children.filter((c) => c.status === "canceled").length;
  const isActive = status === "queued" || status === "running";
  const pct = total ? Math.round((done / total) * 100) : 0;

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link to="/" className="text-sm text-blue-600 hover:underline">
            ← Jobs
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">
            {job.data?.video_name ?? "Batch job"}
          </h1>
          <p className="text-sm text-neutral-500">
            {total} run{total === 1 ? "" : "s"}
          </p>
        </div>
        <StatusPill status={status} />
      </div>

      {/* batch progress + actions */}
      <section className="space-y-3 rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
        <div className="h-3 w-full overflow-hidden rounded-full bg-neutral-100">
          <div
            className={`h-full rounded-full transition-all ${
              failed ? "bg-amber-500" : "bg-blue-500"
            } ${isActive ? "progress-active" : ""}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-neutral-500">
            {done}/{total} complete
            {failed ? ` · ${failed} failed` : ""}
            {canceled ? ` · ${canceled} canceled` : ""}
          </span>
          {isActive && (
            <button
              onClick={() => cancel.mutate(jobId)}
              disabled={cancel.isPending}
              className="rounded border border-red-200 bg-white px-3 py-1 font-medium text-red-600 hover:border-red-400 hover:bg-red-50 disabled:opacity-50"
            >
              ■ Cancel all
            </button>
          )}
        </div>
      </section>

      {/* per-run table */}
      <section className="overflow-hidden rounded-lg border border-neutral-200 bg-white shadow-sm">
        <table className="w-full text-sm">
          <thead className="border-b border-neutral-200 bg-neutral-50 text-left text-xs uppercase tracking-wide text-neutral-500">
            <tr>
              <th className="px-3 py-2 font-medium">Video</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {children.map((child) => (
              <ChildRow key={child.job_id} child={child} />
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

// One child-run row: status, a live progress bar while running, and links into
// the run's live view and (once complete) its inspector.
const CHILD_TERMINAL = new Set(["completed", "failed", "canceled"]);

function ChildRow({ child }: { child: JobChild }) {
  const running = child.status === "running";
  const progress = useJobProgress(child.job_id, running);
  const pct = Math.round(progress.data?.percent ?? 0);
  const retry = useRetryJob();
  const cancel = useCancelJob();
  const isTerminal = CHILD_TERMINAL.has(child.status);
  const isCancellable = child.status === "queued" || running;

  return (
    <tr className="border-t border-neutral-100 hover:bg-neutral-50">
      <td className="px-3 py-2 font-medium">
        <Link
          to="/jobs/$jobId"
          params={{ jobId: child.job_id }}
          className="hover:underline"
        >
          {child.video_name ?? "—"}
        </Link>
        {child.error_text && (
          <p className="text-xs text-red-500">{child.error_text}</p>
        )}
      </td>
      <td className="px-3 py-2">
        <StatusPill status={child.status} />
        {running && (
          <div className="mt-1 w-28">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200">
              <div
                className="progress-active h-full rounded-full bg-blue-500 transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )}
      </td>
      <td className="px-3 py-2 text-right">
        <div className="flex items-center justify-end gap-3">
          {child.status === "completed" && child.run_id && (
            <Link
              to="/runs/$runId"
              params={{ runId: child.run_id }}
              className="text-blue-600 hover:underline"
            >
              inspect →
            </Link>
          )}
          {isCancellable && (
            <button
              onClick={() => cancel.mutate(child.job_id)}
              disabled={cancel.isPending}
              className="rounded border border-red-200 bg-white px-2 py-0.5 text-xs font-medium text-red-600 hover:border-red-400 hover:bg-red-50 disabled:opacity-50"
              title="Cancel this run"
            >
              ■ Cancel
            </button>
          )}
          {isTerminal && (
            <button
              onClick={() => retry.mutate(child.job_id)}
              disabled={retry.isPending}
              className="rounded border border-neutral-300 bg-white px-2 py-0.5 text-xs font-medium text-neutral-700 hover:border-neutral-400 hover:bg-neutral-50 disabled:opacity-50"
              title={
                child.status === "completed"
                  ? "Re-run this video as a fresh run"
                  : "Retry this run"
              }
            >
              ↺ {child.status === "completed" ? "Re-run" : "Retry"}
            </button>
          )}
        </div>
      </td>
    </tr>
  );
}
