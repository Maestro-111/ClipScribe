import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import {
  useCancelJob,
  useDeleteJob,
  useJobs,
  useRetryJob,
  type JobResponse,
} from "../api/hooks";
import { formatDateTime, formatDuration, statusColor } from "../lib/format";

// "/" — the jobs list (web-app-plan §7, page 1).
export const Route = createFileRoute("/")({
  component: JobsList,
});

const STATUSES = ["", "queued", "running", "completed", "failed", "canceled"];

// A job is a batch parent with one run per video. The list row shows how many
// of those runs have finished; a single completed run links straight to its
// inspector, otherwise the job name opens the per-run batch view.
function runCounts(job: JobResponse) {
  const children = job.children ?? [];
  const total = children.length;
  const done = children.filter((c) => c.status === "completed").length;
  return { total, done };
}

// The lone completed run of a single-video job, for the direct "inspect" link.
function soleCompletedRun(job: JobResponse): string | null {
  const completed = (job.children ?? []).filter(
    (c) => c.status === "completed" && c.run_id,
  );
  return completed.length === 1 ? completed[0]!.run_id! : null;
}

function JobsList() {
  const [status, setStatus] = useState("");
  const { data, isLoading, error } = useJobs(status || undefined);
  const retry = useRetryJob();
  const cancel = useCancelJob();
  const del = useDeleteJob();

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Jobs</h1>
        <Link
          to="/jobs/new"
          className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        >
          New job
        </Link>
      </div>

      <div className="mb-4 flex gap-2">
        {STATUSES.map((s) => (
          <button
            key={s || "all"}
            onClick={() => setStatus(s)}
            className={`rounded px-2 py-1 text-sm ${
              status === s ? "bg-neutral-900 text-white" : "bg-white border"
            }`}
          >
            {s || "all"}
          </button>
        ))}
      </div>

      {isLoading && <p className="text-neutral-500">Loading…</p>}
      {error && <p className="text-red-600">{(error as Error).message}</p>}

      {data && (
        <div className="overflow-hidden rounded border bg-white">
          <table className="w-full text-sm">
            <thead className="bg-neutral-100 text-left text-neutral-600">
              <tr>
                <th className="px-3 py-2">Video</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Platform</th>
                <th className="px-3 py-2">Mode</th>
                <th className="px-3 py-2">Created</th>
                <th className="px-3 py-2">Duration</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {data.jobs.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-3 py-6 text-center text-neutral-500">
                    No jobs yet.
                  </td>
                </tr>
              )}
              {data.jobs.map((job) => (
                <tr key={job.job_id} className="border-t">
                  <td className="px-3 py-2 font-medium">
                    <Link
                      to="/jobs/$jobId"
                      params={{ jobId: job.job_id }}
                      className="hover:underline"
                    >
                      {job.video_name ?? "—"}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded px-2 py-0.5 text-xs ${statusColor(job.status)}`}
                    >
                      {job.status}
                    </span>
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
                  <td className="px-3 py-2 text-neutral-600">
                    {job.platform ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-neutral-600">{job.mode ?? "—"}</td>
                  <td className="px-3 py-2 text-neutral-600">
                    {formatDateTime(job.created_at)}
                  </td>
                  <td className="px-3 py-2 text-neutral-600">
                    {formatDuration(job.started_at, job.finished_at)}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex items-center justify-end gap-3">
                      {job.status === "completed" &&
                        (soleCompletedRun(job) ? (
                          <Link
                            to="/runs/$runId"
                            params={{ runId: soleCompletedRun(job)! }}
                            className="text-blue-600 hover:underline"
                          >
                            inspect →
                          </Link>
                        ) : (
                          <Link
                            to="/jobs/$jobId"
                            params={{ jobId: job.job_id }}
                            className="text-blue-600 hover:underline"
                          >
                            runs →
                          </Link>
                        ))}
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
    </div>
  );
}
