import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useJobs } from "../api/hooks";
import { formatDateTime, formatDuration, statusColor } from "../lib/format";

// "/" — the jobs list (web-app-plan §7, page 1).
export const Route = createFileRoute("/")({
  component: JobsList,
});

const STATUSES = ["", "queued", "running", "completed", "failed", "canceled"];

function JobsList() {
  // Local UI state (the status filter) lives in React state; the server data
  // lives in TanStack Query. Changing the filter changes the queryKey, so Query
  // fetches (and caches) each filter independently.
  const [status, setStatus] = useState("");
  const { data, isLoading, error } = useJobs(status || undefined);

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
                <th className="px-3 py-2">Mode</th>
                <th className="px-3 py-2">Created</th>
                <th className="px-3 py-2">Duration</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {data.jobs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-6 text-center text-neutral-500">
                    No jobs yet.
                  </td>
                </tr>
              )}
              {data.jobs.map((job) => (
                <tr key={job.job_id} className="border-t">
                  <td className="px-3 py-2 font-medium">
                    {job.video_name ?? "—"}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded px-2 py-0.5 text-xs ${statusColor(job.status)}`}
                    >
                      {job.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-neutral-600">{job.mode ?? "—"}</td>
                  <td className="px-3 py-2 text-neutral-600">
                    {formatDateTime(job.created_at)}
                  </td>
                  <td className="px-3 py-2 text-neutral-600">
                    {formatDuration(job.started_at, job.finished_at)}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {job.run_id && job.status === "completed" && (
                      <Link
                        to="/runs/$runId"
                        params={{ runId: job.run_id }}
                        className="text-blue-600 hover:underline"
                      >
                        inspect →
                      </Link>
                    )}
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
