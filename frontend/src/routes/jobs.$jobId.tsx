import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useReducer, useRef } from "react";
import { useCancelJob, useJob } from "../api/hooks";
import { statusColor } from "../lib/format";

// "/jobs/{id}" — the live job page (web-app-plan §7 page 3, step 9).
//
// Two data sources feed this page:
//   - useJob(id): the canonical `jobs` row (status, run_id), polled until
//     terminal. It's the source of truth for navigation + final status.
//   - EventSource → GET /api/jobs/{id}/events: the live progress stream. The
//     API replays the whole stream on connect, so a late-loaded page still
//     fills in, then tails. Events drive the reducer below.
export const Route = createFileRoute("/jobs/$jobId")({
  component: LiveJob,
});

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

// ── Component ───────────────────────────────────────────────────────────────
function LiveJob() {
  const { jobId } = Route.useParams();
  const job = useJob(jobId);
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
          <Link to="/" className="text-sm text-blue-600 hover:underline">
            ← Jobs
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">
            {state.videoName ?? job.data?.video_name ?? "Job"}
          </h1>
        </div>
        <span
          className={`rounded px-2 py-0.5 text-sm ${statusColor(status)}`}
        >
          {status}
        </span>
      </div>

      {/* progress bar + actions */}
      <section className="space-y-3 rounded border bg-white p-4">
        <div className="h-3 w-full overflow-hidden rounded-full bg-neutral-100">
          <div
            className={`h-full rounded-full transition-all ${
              status === "failed" ? "bg-red-500" : "bg-blue-500"
            }`}
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
        <section className="rounded border bg-white p-4">
          <h2 className="mb-3 font-medium">Phases</h2>
          {state.phaseOrder.length === 0 ? (
            <p className="text-sm text-neutral-500">Waiting for the job to start…</p>
          ) : (
            <ul className="space-y-1.5 text-sm">
              {state.phaseOrder.map((phase) => {
                const ps = state.phases[phase] ?? "pending";
                const shots =
                  phase === "shot_processing" && state.totalShots
                    ? ` ${state.shotsCompleted}/${state.totalShots}`
                    : "";
                return (
                  <li key={phase} className="flex items-center gap-2">
                    <span>
                      {ps === "completed" ? "✓" : ps === "running" ? "◔" : "○"}
                    </span>
                    <span
                      className={
                        ps === "pending" ? "text-neutral-400" : "text-neutral-800"
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
        <section className="rounded border bg-white p-4">
          <h2 className="mb-3 font-medium">Current shot</h2>
          {state.currentShot ? (
            <div className="space-y-2 text-sm">
              <p className="font-medium">
                Shot {state.currentShot.idx}
                <span className="ml-2 font-normal text-neutral-500">
                  {state.currentShot.framesDone} frames
                </span>
              </p>
              {state.currentShot.description && (
                <p className="text-neutral-700">{state.currentShot.description}</p>
              )}
              {state.currentShot.dinoPrompt && (
                <p className="text-xs text-neutral-500">
                  DINO: {state.currentShot.dinoPrompt}
                </p>
              )}
              {state.currentShot.targets && (
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
              )}
            </div>
          ) : (
            <p className="text-sm text-neutral-500">No shot in progress.</p>
          )}
        </section>
      </div>

      {/* log tail */}
      <section className="rounded border bg-white p-4">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-medium">Logs</h2>
          <span className="text-xs text-neutral-400">
            {state.streamStatus === "live"
              ? "● live"
              : state.streamStatus === "connecting"
                ? "connecting…"
                : "closed"}
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
