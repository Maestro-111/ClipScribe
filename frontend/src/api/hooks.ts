// React hooks that wrap the typed API client with TanStack Query.
//
// Pattern to internalize:
//   - useQuery  = READ. Cached, auto-refetched, deduped. Keyed by `queryKey`.
//   - useMutation = WRITE. Runs on demand (e.g. form submit); on success we
//     invalidate the relevant queryKey so the read re-fetches fresh data.
//
// The component never calls fetch() or touches loading flags manually — it reads
// `data`, `isLoading`, `error` off the hook. Because `api` is typed from the
// OpenAPI schema, `data` below is fully typed with zero manual annotations.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, unwrap } from "./client";
import type {
  AudioSegment,
  FrameDetection,
  GlobalStatsResponse,
  ParserResult,
  Run,
  TextEvent,
} from "../lib/run-types";
import type { components } from "./types";

// The request body for POST /jobs, taken straight from the generated schema
// (FastAPI named it after the Pydantic model). This is the type the form must
// satisfy — change the Python model, re-run `pnpm gen:api`, and the form stops
// compiling until it matches. That's the anti-drift guarantee, made concrete.
export type JobCreateRequest = components["schemas"]["JobCreateRequest"];
export type JobResponse = components["schemas"]["JobResponse"];
export type JobChild = components["schemas"]["JobChild"];

export interface RunFramesWindow {
  fromSec: number;
  toSec: number;
  enabled?: boolean;
}

// Centralized query keys. Keeping them in one factory means invalidation and
// fetching always agree on the exact key array (a common source of "why won't
// my list refresh?" bugs).
export const keys = {
  jobs: (status?: string, limit?: number, offset?: number) =>
    ["jobs", { status, limit, offset }] as const,
  job: (id: string) => ["jobs", id] as const,
  run: (id: string) => ["runs", id] as const,
  runFrames: (id: string, window?: RunFramesWindow) =>
    [
      "runs",
      id,
      "frames",
      { fromSec: window?.fromSec, toSec: window?.toSec },
    ] as const,
  runGlobalStats: (id: string) => ["runs", id, "global-stats"] as const,
  runParser: (id: string) => ["runs", id, "parser"] as const,
  runAudioSegments: (id: string) => ["runs", id, "audio-segments"] as const,
  runTextEvents: (id: string) => ["runs", id, "text-events"] as const,
  platforms: () => ["platforms"] as const,
  inputs: () => ["inputs"] as const,
};

const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "canceled"]);

// --- Jobs list (paginated) ---
// Server-side offset pagination via the existing /jobs `limit`/`offset` query
// params. `placeholderData` keeps the previous page on screen while the next
// one loads, so paging doesn't flash the skeleton or drop the 2s poll.
export function useJobs(status?: string, limit = 20, offset = 0) {
  return useQuery({
    queryKey: keys.jobs(status, limit, offset),
    // openapi-fetch returns a Promise<{ data, error }>; we await it, then
    // `unwrap` throws on error (so Query flips to its error state) or returns
    // the typed data.
    queryFn: async () =>
      unwrap(
        await api.GET("/jobs", {
          params: { query: { ...(status ? { status } : {}), limit, offset } },
        }),
      ),
    placeholderData: (previousData) => previousData,
    refetchInterval: (query) =>
      query.state.data?.jobs.some((job) => !TERMINAL_JOB_STATUSES.has(job.status))
        ? 2000
        : false,
  });
}

// --- Coarse live progress for the jobs-list bar. ---
// Polls GET /jobs/{id}/progress every 2s while enabled (i.e. the row is
// running). Cheap server-side reduction of the Redis event stream — no SSE
// connection per row. `enabled` gates the query so terminal rows never poll.
export function useJobProgress(jobId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["jobs", jobId, "progress"] as const,
    queryFn: async () =>
      unwrap(
        await api.GET("/jobs/{job_id}/progress", {
          params: { path: { job_id: jobId } },
        }),
      ),
    enabled,
    refetchInterval: enabled ? 2000 : false,
  });
}

// --- Single job. Polls while the job is not in a terminal state. ---
export function useJob(jobId: string) {
  return useQuery({
    queryKey: keys.job(jobId),
    queryFn: async () =>
      unwrap(
        await api.GET("/jobs/{job_id}", {
          params: { path: { job_id: jobId } },
        }),
      ),
    // refetchInterval can be a function of the latest data: keep polling every
    // 2s until the job finishes, then stop. The live page uses SSE for detailed
    // progress; this query remains the canonical jobs-row state.
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "completed" || s === "failed" || s === "canceled"
        ? false
        : 2000;
    },
  });
}

// --- Create job (mutation) ---
// On success we invalidate the jobs list so it shows the new row without a
// manual refresh.
export function useCreateJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: JobCreateRequest) =>
      unwrap(await api.POST("/jobs", { body })),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// --- Run inspector reads ---
export function useRun(runId: string, enabled = true) {
  return useQuery({
    queryKey: keys.run(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}", {
          params: { path: { run_id: runId } },
        }),
      ) as unknown as Run,
    enabled,
    staleTime: Infinity,
  });
}

export function useRunFrames(runId: string, window?: RunFramesWindow) {
  return useQuery({
    queryKey: keys.runFrames(runId, window),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/frames", {
          params: {
            path: { run_id: runId },
            query: window
              ? { from: window.fromSec, to: window.toSec }
              : undefined,
          },
        }),
      ) as unknown as FrameDetection[],
    enabled: window?.enabled ?? true,
    placeholderData: (previousData) => previousData,
    staleTime: Infinity,
  });
}

export function useRunGlobalStats(runId: string, enabled = true) {
  return useQuery({
    queryKey: keys.runGlobalStats(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/global-stats", {
          params: { path: { run_id: runId } },
        }),
      ) as unknown as GlobalStatsResponse,
    enabled,
    staleTime: Infinity,
  });
}

export function useRunParser(runId: string, enabled = true) {
  return useQuery({
    queryKey: keys.runParser(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/parser", {
          params: { path: { run_id: runId } },
        }),
      ) as unknown as ParserResult[],
    enabled,
    staleTime: Infinity,
  });
}

export function useRunAudioSegments(runId: string, enabled = true) {
  return useQuery({
    queryKey: keys.runAudioSegments(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/audio-segments", {
          params: { path: { run_id: runId } },
        }),
      ) as unknown as AudioSegment[],
    enabled,
    staleTime: Infinity,
  });
}

export function useRunTextEvents(runId: string) {
  return useQuery({
    queryKey: keys.runTextEvents(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/text-events", {
          params: { path: { run_id: runId } },
        }),
      ) as unknown as TextEvent[],
    staleTime: Infinity,
  });
}

// --- New-job form metadata ---
export function usePlatforms() {
  return useQuery({
    queryKey: keys.platforms(),
    queryFn: async () => unwrap(await api.GET("/platforms", {})),
    staleTime: Infinity, // platform specs don't change at runtime
  });
}

export function useInputs() {
  return useQuery({
    queryKey: keys.inputs(),
    queryFn: async () => unwrap(await api.GET("/inputs", {})),
  });
}

// Delete a terminal job record (DELETE /jobs/{id} → 204).
export function useDeleteJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const resp = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
      if (!resp.ok) {
        const p = (await resp.json().catch(() => ({}))) as {
          title?: string;
          detail?: string;
        };
        throw new ApiError(
          resp.status,
          p.title ?? "Delete failed",
          p.detail ?? "Server returned an error",
        );
      }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// Cancel a queued job (POST /jobs/{id}/cancel → 204).
export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const resp = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      if (!resp.ok) {
        const p = (await resp.json().catch(() => ({}))) as {
          title?: string;
          detail?: string;
        };
        throw new ApiError(
          resp.status,
          p.title ?? "Cancel failed",
          p.detail ?? "Server returned an error",
        );
      }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// Retry a failed or canceled job. This uses plain fetch because the lifecycle
// helpers predate the typed wrapper call sites; the response shape is still
// generated from OpenAPI and cast below.
export function useRetryJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const resp = await fetch(`/api/jobs/${jobId}/retry`, { method: "POST" });
      if (!resp.ok) {
        const p = (await resp.json().catch(() => ({}))) as {
          title?: string;
          detail?: string;
        };
        throw new ApiError(
          resp.status,
          p.title ?? "Retry failed",
          p.detail ?? "Server returned an error",
        );
      }
      return (await resp.json()) as components["schemas"]["JobCreatedResponse"];
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export type UploadedVideo = { name: string; path: string; size_bytes: number };

// Upload one or more video files through the server's video-storage backend in
// a single request (POST /uploads accepts a list). openapi-fetch can't represent
// File in the generated schema (it emits string[]), so we use plain fetch +
// FormData.
export function useUploadVideos() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (files: File[]) => {
      const form = new FormData();
      for (const file of files) form.append("files", file);
      const resp = await fetch("/api/uploads", { method: "POST", body: form });
      if (!resp.ok) {
        const p = (await resp.json().catch(() => ({}))) as {
          title?: string;
          detail?: string;
        };
        throw new ApiError(
          resp.status,
          p.title ?? "Upload failed",
          p.detail ?? "Server returned an error",
        );
      }
      const body = (await resp.json()) as { uploaded: UploadedVideo[] };
      return body.uploaded;
    },
    // Refresh the input picker after a successful upload.
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.inputs() });
    },
  });
}

// Sibling runs of the run belonging to the same batch job (for the run
// inspector's run switcher). Empty for a standalone run.
export type RunSibling = components["schemas"]["RunSibling"];

export function useRunSiblings(runId: string) {
  return useQuery({
    queryKey: ["runs", runId, "siblings"] as const,
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/siblings", {
          params: { path: { run_id: runId } },
        }),
      ) as unknown as RunSibling[],
    // Poll while any sibling is still working, so the switcher enables runs as
    // they finish; stop once every run is terminal.
    refetchInterval: (query) => {
      const data = query.state.data as RunSibling[] | undefined;
      const pending = data?.some(
        (s) => !TERMINAL_JOB_STATUSES.has(s.status),
      );
      return pending ? 3000 : false;
    },
  });
}
