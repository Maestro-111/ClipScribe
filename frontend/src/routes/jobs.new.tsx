import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useCreateJob, useInputs } from "../api/hooks";

// "/jobs/new" — the create-job form (web-app-plan §7, page 2).
export const Route = createFileRoute("/jobs/new")({
  component: NewJob,
});

function NewJob() {
  const navigate = useNavigate();
  const inputs = useInputs(); // server-side video picker (plan §9.1, option c)
  const createJob = useCreateJob();

  // Form state. For an initial sketch we keep it as plain useState fields; a
  // form library (react-hook-form) can come later if this grows.
  const [mode, setMode] = useState<"full" | "extract" | "parse">("full");
  const [videoPath, setVideoPath] = useState("");
  const [videoType, setVideoType] = useState("");
  const [brandName, setBrandName] = useState("");
  const [products, setProducts] = useState(""); // comma-separated in the UI

  const canSubmit = mode === "parse" ? false : Boolean(videoPath);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const videoName = videoPath; // picker returns a bare filename == the name
    createJob.mutate(
      {
        mode,
        platform: "youtube",
        video_path: videoPath,
        video_name: videoName,
        video_type: videoType || null,
        platform_params: {
          brand_name: brandName,
          // Split the comma field into the string[] the API expects.
          branded_products: products
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
        },
        // Required by the generated schema (the Pydantic field has a default,
        // but FastAPI still lists it). Send the default explicitly.
        generate_hint_from_name: false,
      },
      {
        // On success the API returns the new job id; send the user back to the
        // list where they can watch status. (A dedicated live page is plan
        // step 9.)
        onSuccess: () => {
          void navigate({ to: "/" });
        },
      },
    );
  }

  return (
    <div className="max-w-xl">
      <h1 className="mb-4 text-2xl font-semibold">New job</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <Field label="Mode">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as typeof mode)}
            className="w-full rounded border px-2 py-1.5"
          >
            <option value="full">full (extract + parse)</option>
            <option value="extract">extract only</option>
            <option value="parse" disabled>
              parse (needs an existing run — later)
            </option>
          </select>
        </Field>

        <Field label="Video">
          {inputs.isLoading && <p className="text-sm text-neutral-500">Loading…</p>}
          {inputs.data && (
            <select
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              className="w-full rounded border px-2 py-1.5"
            >
              <option value="">— pick a video from input/ —</option>
              {inputs.data.videos.map((v) => (
                <option key={v.path} value={v.path}>
                  {v.name}
                </option>
              ))}
            </select>
          )}
          <p className="mt-1 text-xs text-neutral-500">
            Files under the server's <code>input/</code> directory. Upload support
            comes later (plan §9.1a).
          </p>
        </Field>

        <Field label="Video type (optional)">
          <input
            value={videoType}
            onChange={(e) => setVideoType(e.target.value)}
            placeholder="e.g. car ad"
            className="w-full rounded border px-2 py-1.5"
          />
        </Field>

        <Field label="Brand name">
          <input
            value={brandName}
            onChange={(e) => setBrandName(e.target.value)}
            placeholder="e.g. RAM"
            className="w-full rounded border px-2 py-1.5"
          />
        </Field>

        <Field label="Branded products (comma-separated)">
          <input
            value={products}
            onChange={(e) => setProducts(e.target.value)}
            placeholder="e.g. RAM 1500, RAM 2500"
            className="w-full rounded border px-2 py-1.5"
          />
        </Field>

        {createJob.error && (
          <p className="text-sm text-red-600">
            {(createJob.error as Error).message}
          </p>
        )}

        <button
          type="submit"
          disabled={!canSubmit || createJob.isPending}
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {createJob.isPending ? "Submitting…" : "Create job"}
        </button>
      </form>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      {children}
    </label>
  );
}
