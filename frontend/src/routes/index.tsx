import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import {
  useCancelJob,
  useDeleteJob,
  useJobs,
  useRetryJob,
  type JobResponse,
} from "../api/hooks";
import { formatDateTime, formatDuration } from "../lib/format";
import { EmptyState, Pagination, Skeleton, StatusPill } from "../components/ui";

// "/" — the jobs list (web-app-plan §7, page 1).
export const Route = createFileRoute("/")({
  component: JobsList,
});

const STATUSES = ["", "queued", "running", "completed", "failed", "canceled"];
const PAGE_SIZE = 20;

// A job is a batch parent with one run per video. The list row shows how many
// of those runs have finished; a completed job links to the per-run batch view
// regardless of run count.
function runCounts(job: JobResponse) {
  const children = job.children ?? [];
  const total = children.length;
  const done = children.filter((c) => c.status === "completed").length;
  return { total, done };
}

function JobsList() {
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(0);
  const { data, isLoading, error } = useJobs(
    status || undefined,
    PAGE_SIZE,
    page * PAGE_SIZE,
  );
  // No total count from the API, so infer "there's a next page" from a full
  // page of results. `isLoading` is only true on the very first fetch (later
  // pages reuse placeholderData), so it's a safe gate for the skeleton.
  const count = data?.jobs.length ?? 0;
  const canNext = count === PAGE_SIZE;
  const canPrev = page > 0;
  const paginationLabel =
    count > 0
      ? `Showing ${page * PAGE_SIZE + 1}–${page * PAGE_SIZE + count}`
      : `No jobs on page ${page + 1}`;
  const retry = useRetryJob();
  const cancel = useCancelJob();
  const del = useDeleteJob();

  return (
    // Full-bleed: the dashboard spans the whole viewport (see styles.css) while
    // every other route stays in the shared max-w-6xl column. px matches the
    // page gutter so content isn't glued to the screen edges.
    <div className="full-bleed px-6 lg:px-10">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Jobs</h1>
        <Link
          to="/jobs/new"
          className="rounded-md bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-neutral-800"
        >
          New job
        </Link>
      </div>

      <div className="mb-4 flex gap-2">
        {STATUSES.map((s) => (
          <button
            key={s || "all"}
            onClick={() => {
              setStatus(s);
              setPage(0);
            }}
            className={`rounded-md px-2.5 py-1 text-sm capitalize transition-colors ${
              status === s
                ? "bg-neutral-900 text-white"
                : "border border-neutral-200 bg-white text-neutral-600 hover:bg-neutral-50"
            }`}
          >
            {s || "all"}
          </button>
        ))}
      </div>

      {error && (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600">
          {(error as Error).message}
        </p>
      )}

      {isLoading && (
        <div className="overflow-hidden rounded-lg border border-neutral-200 bg-white shadow-sm">
          <div className="space-y-3 p-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="ml-auto h-4 w-24" />
              </div>
            ))}
          </div>
        </div>
      )}

      {data && data.jobs.length === 0 && (
        <div className="rounded-lg border border-neutral-200 bg-white shadow-sm">
          <EmptyState
            icon={
              <svg width="48" height="48" viewBox="0 0 32 32" fill="none">
                <rect x="4" y="9" width="24" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" />
                <path d="M4 13h24" stroke="currentColor" strokeWidth="1.5" />
                <path d="M9 9l1.5-3M15 9l1.5-3M21 9l1.5-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            }
            title={canPrev ? "No jobs on this page" : "No jobs yet"}
            description={
              canPrev
                ? "Go back to see earlier jobs."
                : "Create a job to process a video through the ClipScribe pipeline and watch it run live."
            }
            action={
              canPrev ? undefined : (
                <Link
                  to="/jobs/new"
                  className="inline-block rounded-md bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-neutral-800"
                >
                  Create your first job →
                </Link>
              )
            }
          />
        </div>
      )}

      {data && data.jobs.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-neutral-200 bg-white shadow-sm">
          <table className="w-full text-base">
            <thead className="border-b border-neutral-200 bg-neutral-50 text-left text-sm uppercase tracking-wide text-neutral-500">
              <tr>
                <th className="px-4 py-3 font-medium">Video</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Platform</th>
                <th className="px-4 py-3 font-medium">Mode</th>
                <th className="px-4 py-3 font-medium">Created</th>
                <th className="px-4 py-3 font-medium">Duration</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {data.jobs.map((job) => (
                <tr key={job.job_id} className="border-t border-neutral-100 hover:bg-neutral-50">
                  <td className="px-4 py-3 font-medium">
                    <Link
                      to="/jobs/$jobId"
                      params={{ jobId: job.job_id }}
                      className="hover:underline"
                    >
                      {job.video_name ?? "—"}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <StatusPill status={job.status} />
                    {(() => {
                      const { total, done } = runCounts(job);
                      // Show run progress once there's more than one run, or
                      // while a batch is still working through them.
                      if (total > 1 || (total >= 1 && job.status === "running")) {
                        return (
                          <span className="ml-2 text-[10px] text-neutral-400">
                            {done}/{total} runs
                          </span>
                        );
                      }
                      return null;
                    })()}
                  </td>
                  <td className="px-4 py-3 text-neutral-600">
                    {job.platform ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-neutral-600">{job.mode ?? "—"}</td>
                  <td className="px-4 py-3 text-neutral-600">
                    {formatDateTime(job.created_at)}
                  </td>
                  <td className="px-4 py-3 text-neutral-600">
                    {formatDuration(job.started_at, job.finished_at)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-3">
                      {job.status === "completed" && (
                        <Link
                          to="/jobs/$jobId"
                          params={{ jobId: job.job_id }}
                          className="text-blue-600 hover:underline"
                        >
                          runs →
                        </Link>
                      )}
                      {job.status === "queued" && (
                        <button
                          onClick={() => cancel.mutate(job.job_id)}
                          disabled={cancel.isPending}
                          className="rounded border border-red-200 bg-white px-2 py-0.5 text-xs font-medium text-red-600 hover:border-red-400 hover:bg-red-50 disabled:opacity-50"
                        >
                          ■ Stop
                        </button>
                      )}
                      {(job.status === "failed" ||
                        job.status === "canceled" ||
                        job.status === "completed") && (
                        <button
                          onClick={() => retry.mutate(job.job_id)}
                          disabled={retry.isPending}
                          className="rounded border border-neutral-300 bg-white px-2 py-0.5 text-xs font-medium text-neutral-700 hover:border-neutral-400 hover:bg-neutral-50 disabled:opacity-50"
                          title={
                            job.status === "completed"
                              ? "Re-run this job as a new run"
                              : "Retry this job"
                          }
                        >
                          ↺ {job.status === "completed" ? "Re-run" : "Retry"}
                        </button>
                      )}
                      {(job.status === "completed" ||
                        job.status === "failed" ||
                        job.status === "canceled") && (
                        <button
                          onClick={() => {
                            if (confirm(`Delete job "${job.video_name ?? job.job_id}" and all its run results?`)) {
                              del.mutate(job.job_id);
                            }
                          }}
                          disabled={del.isPending}
                          className="rounded border border-neutral-200 bg-white px-2 py-0.5 text-xs font-medium text-neutral-400 hover:border-red-300 hover:text-red-500 disabled:opacity-50"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && (canPrev || canNext) && (
        <div className="mt-4">
          <Pagination
            canPrev={canPrev}
            canNext={canNext}
            onPrev={() => setPage((p) => Math.max(0, p - 1))}
            onNext={() => setPage((p) => p + 1)}
            label={paginationLabel}
          />
        </div>
      )}
    </div>
  );
}
