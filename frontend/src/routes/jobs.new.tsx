import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  useCreateJob,
  useInputs,
  usePlatforms,
  useUploadVideos,
  type JobCreateRequest,
} from "../api/hooks";

// "/jobs/new" — the create-job form (web-app-plan §7, page 2).
export const Route = createFileRoute("/jobs/new")({
  component: NewJob,
});

// Split a comma-separated string into a trimmed, non-empty string array.
function splitComma(s: string): string[] {
  return s.split(",").map((x) => x.trim()).filter(Boolean);
}

const ALLOWED_SUFFIXES = [".mp4", ".mov", ".mkv", ".webm", ".m4v"];

function hasAllowedSuffix(name: string): boolean {
  const lower = name.toLowerCase();
  return ALLOWED_SUFFIXES.some((s) => lower.endsWith(s));
}

// One video queued for the job. `path` is the server-side INPUT_DIR path; `name`
// is what the user sees. A job fans out to one run per entry here.
interface SelectedVideo {
  path: string;
  name: string;
}

function NewJob() {
  const navigate = useNavigate();
  const inputs = useInputs();
  const platforms = usePlatforms();
  const createJob = useCreateJob();

  // Mode is always "full" from the UI: "extract"/"parse" are developer-only
  // paths run against the API / main.py directly, not user-facing.
  const mode = "full" as const;

  // The batch of videos this job will process. All share the platform, params,
  // and hints below (same brand/product context).
  const [selected, setSelected] = useState<SelectedVideo[]>([]);
  const [videoType, setVideoType] = useState("");
  const [videoTab, setVideoTab] = useState<"pick" | "upload">("pick");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadVideos();

  // `webkitdirectory` isn't in React's input typings; set it on the DOM node.
  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  const addVideos = (videos: SelectedVideo[]) =>
    setSelected((prev) => {
      const byPath = new Map(prev.map((v) => [v.path, v]));
      for (const v of videos) byPath.set(v.path, v);
      return [...byPath.values()];
    });

  const removeVideo = (path: string) =>
    setSelected((prev) => prev.filter((v) => v.path !== path));

  const isSelected = (path: string) => selected.some((v) => v.path === path);

  const handleUpload = (fileList: FileList | null) => {
    if (!fileList) return;
    const files = Array.from(fileList).filter((f) => hasAllowedSuffix(f.name));
    if (!files.length) return;
    upload.mutate(files, {
      onSuccess: (uploaded) => {
        addVideos(uploaded.map((u) => ({ path: u.path, name: u.name })));
        setVideoTab("pick");
      },
    });
  };

  // Platform selection. Options come from GET /platforms; today the backend
  // only advertises "youtube", so this renders as a single-option select.
  const [platform, setPlatform] = useState("youtube");
  const platformOptions = platforms.data?.platforms.map((p) => p.name) ?? [
    "youtube",
  ];

  // YouTube platform params (matches YouTubePlatformParams in backend/app/models.py)
  const [brandName, setBrandName] = useState("");
  const [brandedProducts, setBrandedProducts] = useState("");
  const [brandedProductsCategories, setBrandedProductsCategories] = useState("");
  const [callToActions, setCallToActions] = useState("");

  // Hint fields
  const [userHints, setUserHints] = useState("");
  const [generateHintFromName, setGenerateHintFromName] = useState(false);

  const canSubmit = selected.length > 0;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    createJob.mutate(
      {
        mode,
        platform: platform as JobCreateRequest["platform"],
        videos: selected.map((v) => ({
          video_path: v.path,
          video_name: v.name,
          video_type: videoType || null,
        })),
        platform_params: (platform === "youtube"
          ? {
              brand_name: brandName,
              branded_products: splitComma(brandedProducts),
              branded_products_categories: splitComma(brandedProductsCategories),
              call_to_actions: splitComma(callToActions),
            }
          : {}) as JobCreateRequest["platform_params"],
        user_hints: splitComma(userHints).length ? splitComma(userHints) : null,
        generate_hint_from_name: generateHintFromName,
      },
      {
        onSuccess: (created) => {
          // Land on the live job page so the user watches progress stream in.
          void navigate({
            to: "/jobs/$jobId",
            params: { jobId: created.job_id },
          });
        },
      },
    );
  }

  return (
    <div className="max-w-xl">
      <h1 className="mb-6 text-2xl font-semibold">New job</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* ── Job ─────────────────────────────────────────────────── */}
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-500">
            Job
          </h2>
          <div className="space-y-4">
            <div>
              <span className="mb-1 block text-sm font-medium">
                Videos
                <span className="ml-2 font-normal text-neutral-500">
                  — one run per video; all share the settings below
                </span>
              </span>

              {/* Selected videos */}
              {selected.length > 0 && (
                <ul className="mb-2 space-y-1">
                  {selected.map((v) => (
                    <li
                      key={v.path}
                      className="flex items-center justify-between rounded border bg-neutral-50 px-2 py-1 text-sm"
                    >
                      <span className="truncate">{v.name}</span>
                      <button
                        type="button"
                        onClick={() => removeVideo(v.path)}
                        className="ml-2 text-neutral-400 hover:text-red-500"
                        title="Remove"
                      >
                        ✕
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              {/* Tab toggle */}
              <div className="mb-2 flex gap-1">
                {(["pick", "upload"] as const).map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setVideoTab(tab)}
                    className={`rounded px-2 py-0.5 text-xs ${
                      videoTab === tab
                        ? "bg-neutral-900 text-white"
                        : "border bg-white text-neutral-600"
                    }`}
                  >
                    {tab === "pick" ? "Pick existing" : "Upload new"}
                  </button>
                ))}
              </div>

              {videoTab === "pick" && (
                <>
                  {inputs.isLoading && (
                    <p className="text-sm text-neutral-500">Loading…</p>
                  )}
                  {inputs.data && inputs.data.videos.length === 0 && (
                    <p className="text-sm text-neutral-500">
                      No videos in input/ yet — use "Upload new" to add some.
                    </p>
                  )}
                  {inputs.data && inputs.data.videos.length > 0 && (
                    <div className="max-h-48 space-y-1 overflow-y-auto rounded border p-2">
                      {inputs.data.videos.map((v) => (
                        <label
                          key={v.path}
                          className="flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 text-sm hover:bg-neutral-50"
                        >
                          <input
                            type="checkbox"
                            checked={isSelected(v.path)}
                            onChange={(e) =>
                              e.target.checked
                                ? addVideos([{ path: v.path, name: v.name }])
                                : removeVideo(v.path)
                            }
                          />
                          <span className="truncate">{v.name}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </>
              )}

              {videoTab === "upload" && (
                <div className="space-y-2">
                  {/* Hidden native inputs; the styled buttons below trigger them. */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".mp4,.mov,.mkv,.webm,.m4v"
                    className="sr-only"
                    onChange={(e) => handleUpload(e.target.files)}
                  />
                  <input
                    ref={folderInputRef}
                    type="file"
                    multiple
                    className="sr-only"
                    onChange={(e) => handleUpload(e.target.files)}
                  />
                  <div className="flex gap-2">
                    <button
                      type="button"
                      disabled={upload.isPending}
                      onClick={() => fileInputRef.current?.click()}
                      className="flex flex-1 flex-col items-center gap-1 rounded-lg border-2 border-dashed border-neutral-300 bg-neutral-50 px-4 py-6 text-center hover:border-blue-400 hover:bg-blue-50 disabled:cursor-wait disabled:opacity-60"
                    >
                      <span className="text-2xl">🎬</span>
                      <span className="text-sm font-medium text-neutral-700">
                        {upload.isPending ? "Uploading…" : "Select video files"}
                      </span>
                      <span className="text-xs text-neutral-400">
                        one or more · .mp4 .mov .mkv .webm .m4v
                      </span>
                    </button>
                    <button
                      type="button"
                      disabled={upload.isPending}
                      onClick={() => folderInputRef.current?.click()}
                      className="flex flex-1 flex-col items-center gap-1 rounded-lg border-2 border-dashed border-neutral-300 bg-neutral-50 px-4 py-6 text-center hover:border-blue-400 hover:bg-blue-50 disabled:cursor-wait disabled:opacity-60"
                    >
                      <span className="text-2xl">📁</span>
                      <span className="text-sm font-medium text-neutral-700">
                        {upload.isPending ? "Uploading…" : "Select a folder"}
                      </span>
                      <span className="text-xs text-neutral-400">
                        uploads every video in the folder
                      </span>
                    </button>
                  </div>
                  {upload.error && (
                    <p className="text-sm text-red-600">
                      {(upload.error as Error).message}
                    </p>
                  )}
                </div>
              )}
            </div>

            <Field
              label="Video type"
              hint="Short description of the creative format, applied to every video as context for scene descriptions."
            >
              <input
                value={videoType}
                onChange={(e) => setVideoType(e.target.value)}
                placeholder="e.g. car ad, product launch"
                className="w-full rounded border px-2 py-1.5"
              />
            </Field>
          </div>
        </section>

        {/* ── Platform ────────────────────────────────────────────── */}
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-500">
            Platform
          </h2>
          <div className="space-y-4">
            <Field
              label="Evaluation platform"
              hint="The platform whose ABCD criteria the parser evaluates against."
            >
              <select
                value={platform}
                onChange={(e) => setPlatform(e.target.value)}
                className="w-full rounded border px-2 py-1.5"
              >
                {platformOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>

            {/* YouTube-specific params. Rendered only when YouTube is selected;
                each platform owns its own field block. */}
            {platform === "youtube" && (
              <>
                <Field
                  label="Brand name"
                  hint="The brand being advertised. Used by parser agents to identify brand presence."
                  required
                >
                  <input
                    value={brandName}
                    onChange={(e) => setBrandName(e.target.value)}
                    placeholder="e.g. RAM"
                    className="w-full rounded border px-2 py-1.5"
                  />
                </Field>

                <Field
                  label="Branded products"
                  hint="Specific products shown or mentioned. Comma-separated."
                >
                  <input
                    value={brandedProducts}
                    onChange={(e) => setBrandedProducts(e.target.value)}
                    placeholder="e.g. RAM 1500, RAM 2500"
                    className="w-full rounded border px-2 py-1.5"
                  />
                </Field>

                <Field
                  label="Branded product categories"
                  hint="Category phrasings used to match products when exact names aren't spoken. Comma-separated."
                >
                  <input
                    value={brandedProductsCategories}
                    onChange={(e) => setBrandedProductsCategories(e.target.value)}
                    placeholder="e.g. pickup truck, heavy-duty truck"
                    className="w-full rounded border px-2 py-1.5"
                  />
                </Field>

                <Field
                  label="Call-to-actions"
                  hint="CTA phrases the parser should detect in speech or on-screen text. Comma-separated."
                >
                  <input
                    value={callToActions}
                    onChange={(e) => setCallToActions(e.target.value)}
                    placeholder="e.g. learn more, visit us today"
                    className="w-full rounded border px-2 py-1.5"
                  />
                </Field>
              </>
            )}
          </div>
        </section>

        {/* ── Hints ───────────────────────────────────────────────── */}
        <section>
          <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-neutral-500">
            Taxonomy hints
          </h2>
          <p className="mb-3 text-xs text-neutral-500">
            Optional guidance for object detection. Hints seed the GroundingDINO
            taxonomy so the extractor looks for things it might otherwise miss.
          </p>
          <div className="space-y-4">
            <Field
              label="User hints"
              hint="Objects or concepts to prioritize. Comma-separated."
            >
              <input
                value={userHints}
                onChange={(e) => setUserHints(e.target.value)}
                placeholder="e.g. truck bed, tow hitch, off-road terrain"
                className="w-full rounded border px-2 py-1.5"
              />
            </Field>

            <label className="flex items-start gap-2">
              <input
                type="checkbox"
                checked={generateHintFromName}
                onChange={(e) => setGenerateHintFromName(e.target.checked)}
                className="mt-0.5"
              />
              <span className="text-sm">
                <span className="font-medium">Generate hints from video name</span>
                <span className="ml-1 text-neutral-500">
                  — asks the LLM to infer taxonomy hints from each video filename
                  (costs one extra OpenAI call per video).
                </span>
              </span>
            </label>
          </div>
        </section>

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
          {createJob.isPending
            ? "Submitting…"
            : selected.length > 1
              ? `Create job (${selected.length} videos)`
              : "Create job"}
        </button>
      </form>
    </div>
  );
}

function Field({
  label,
  hint,
  required,
  children,
}: {
  label: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block">
        <span className="mb-1 block text-sm font-medium">
          {label}
          {required && <span className="ml-1 text-red-500">*</span>}
        </span>
        {children}
      </label>
      {hint && <p className="mt-1 text-xs text-neutral-500">{hint}</p>}
    </div>
  );
}
