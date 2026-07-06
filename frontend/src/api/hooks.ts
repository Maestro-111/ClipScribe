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
import { api, unwrap } from "./client";
import type { components } from "./types";

// The request body for POST /jobs, taken straight from the generated schema
// (FastAPI named it after the Pydantic model). This is the type the form must
// satisfy — change the Python model, re-run `pnpm gen:api`, and the form stops
// compiling until it matches. That's the anti-drift guarantee, made concrete.
export type JobCreateRequest = components["schemas"]["JobCreateRequest"];

// Centralized query keys. Keeping them in one factory means invalidation and
// fetching always agree on the exact key array (a common source of "why won't
// my list refresh?" bugs).
export const keys = {
  jobs: (status?: string) => ["jobs", { status }] as const,
  job: (id: string) => ["jobs", id] as const,
  run: (id: string) => ["runs", id] as const,
  runParser: (id: string) => ["runs", id, "parser"] as const,
  platforms: () => ["platforms"] as const,
  inputs: () => ["inputs"] as const,
};

// --- Jobs list ---
export function useJobs(status?: string) {
  return useQuery({
    queryKey: keys.jobs(status),
    // openapi-fetch returns a Promise<{ data, error }>; we await it, then
    // `unwrap` throws on error (so Query flips to its error state) or returns
    // the typed data.
    queryFn: async () =>
      unwrap(
        await api.GET("/jobs", {
          params: { query: status ? { status } : {} },
        }),
      ),
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
    // 2s until the job finishes, then stop. This is our stand-in for live
    // progress until the SSE stream (plan step 9) lands.
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
export function useRun(runId: string) {
  return useQuery({
    queryKey: keys.run(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}", {
          params: { path: { run_id: runId } },
        }),
      ),
  });
}

export function useRunParser(runId: string) {
  return useQuery({
    queryKey: keys.runParser(runId),
    queryFn: async () =>
      unwrap(
        await api.GET("/runs/{run_id}/parser", {
          params: { path: { run_id: runId } },
        }),
      ),
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
