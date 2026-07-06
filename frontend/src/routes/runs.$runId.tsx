import { createFileRoute } from "@tanstack/react-router";
import { useRun, useRunParser } from "../api/hooks";

// "/runs/:runId" — the run inspector (web-app-plan §7, page 4).
// This is the step-6 version: run metadata, the video player, and the parser
// (ABCD) results table. The SVG detection overlay + timeline tracks are step 7.
export const Route = createFileRoute("/runs/$runId")({
  component: RunInspector,
});

function RunInspector() {
  // useParams reads the dynamic segment. Because of the file name ($runId) and
  // the router type augmentation, `runId` is typed as a string here.
  const { runId } = Route.useParams();
  const run = useRun(runId);
  const parser = useRunParser(runId);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Run {runId}</h1>

      {run.isLoading && <p className="text-neutral-500">Loading…</p>}
      {run.error && <p className="text-red-600">{(run.error as Error).message}</p>}

      {run.data && (
        <section className="rounded border bg-white p-4">
          <h2 className="mb-2 font-medium">Metadata</h2>
          <pre className="overflow-auto rounded bg-neutral-50 p-3 text-xs">
            {JSON.stringify(run.data, null, 2)}
          </pre>
        </section>
      )}

      <section className="rounded border bg-white p-4">
        <h2 className="mb-2 font-medium">Video</h2>
        {/*
          The <video> source is the ORIGINAL input, served Range-aware by the
          artifacts route (plan §9.17: raw input + our own overlay beats the
          baked tracked mp4). We hit it through the /api proxy.

          STEP 7 (inspector overlay) will wrap this <video> in a relative-
          positioned container and lay an absolutely-positioned <svg> on top.
          On the video's `timeupdate` event we look up frame_detections for the
          current time (GET /runs/{id}/frames) and draw <rect> boxes scaled to
          the video's displayed size. That's the piece that makes this feel real.
        */}
        <video
          src={`/api/runs/${runId}/video`}
          controls
          className="w-full rounded bg-black"
        />
        <a
          href={`/api/runs/${runId}/tracked-video`}
          className="mt-2 inline-block text-sm text-blue-600 hover:underline"
        >
          Download tracked_output.mp4
        </a>
      </section>

      <section className="rounded border bg-white p-4">
        <h2 className="mb-2 font-medium">ABCD criteria</h2>
        {parser.isLoading && <p className="text-neutral-500">Loading…</p>}
        {parser.data && parser.data.length === 0 && (
          <p className="text-neutral-500">No parser results for this run.</p>
        )}
        {parser.data && parser.data.length > 0 && (
          <table className="w-full text-sm">
            <thead className="text-left text-neutral-600">
              <tr>
                <th className="py-1">Feature</th>
                <th className="py-1">Result</th>
              </tr>
            </thead>
            <tbody>
              {/* The /runs/{id}/parser endpoint currently returns untyped
                  dict rows (dict[str, Any] on the backend), so fields arrive as
                  `unknown` — hence the String()/Boolean() coercions. Step 7 can
                  give these a real Pydantic model and delete the casts. */}
              {parser.data.map((row, i) => (
                <tr key={i} className="border-t align-top">
                  <td className="py-1 pr-4">
                    {String(row.feature_name ?? "—")}
                  </td>
                  <td className="py-1">{row.evaluation ? "✅" : "❌"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
