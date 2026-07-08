import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState } from "react";
import {
  useCreateJob,
  useInputs,
  usePlatforms,
  useUploadVideo,
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

function NewJob() {
  const navigate = useNavigate();
  const inputs = useInputs();
  const platforms = usePlatforms();
  const createJob = useCreateJob();

  // Job-level fields
  const [mode, setMode] = useState<"full" | "extract" | "parse">("full");
  const [videoPath, setVideoPath] = useState("");
  const [videoType, setVideoType] = useState("");
  const [videoTab, setVideoTab] = useState<"pick" | "upload">("pick");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadVideo();

  // Platform selection. Options come from GET /platforms; today the backend
  // only advertises "youtube", so this renders as a single-option select.
  // When more platforms are added, list them here without touching this form.
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

  const canSubmit = mode !== "parse" && Boolean(videoPath);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    createJob.mutate(
      {
        mode,
        platform: platform as JobCreateRequest["platform"],
        video_path: videoPath,
        video_name: videoPath, // picker returns a bare filename, which is the name
        video_type: videoType || null,
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
        onSuccess: () => {
          void navigate({ to: "/" });
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
            <Field label="Mode">
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as typeof mode)}
                className="w-full rounded border px-2 py-1.5"
              >
                <option value="full">full — extract + evaluate</option>
                <option value="extract">extract only — no parser evaluation</option>
                <option value="parse" disabled>
                  parse — re-evaluate an existing run (not yet supported)
                </option>
              </select>
            </Field>

            <div>
              <span className="mb-1 block text-sm font-medium">Video</span>

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
                      No videos in input/ yet — use "Upload new" to add one.
                    </p>
                  )}
                  {inputs.data && inputs.data.videos.length > 0 && (
                    <select
                      value={videoPath}
                      onChange={(e) => setVideoPath(e.target.value)}
                      className="w-full rounded border px-2 py-1.5"
                    >
                      <option value="">— pick a video —</option>
                      {inputs.data.videos.map((v) => (
                        <option key={v.path} value={v.path}>
                          {v.name}
                        </option>
                      ))}
                    </select>
                  )}
                </>
              )}

              {videoTab === "upload" && (
                <div className="space-y-2">
                  {/* Hidden native input; the styled zone below triggers it. */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp4,.mov,.mkv,.webm,.m4v"
                    className="sr-only"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      upload.mutate(file, {
                        onSuccess: (uploaded) => {
                          if (uploaded) {
                            setVideoPath(uploaded.path);
                            setVideoTab("pick");
                          }
                        },
                      });
                    }}
                  />
                  <button
                    type="button"
                    disabled={upload.isPending}
                    onClick={() => fileInputRef.current?.click()}
                    className="flex w-full flex-col items-center gap-2 rounded-lg border-2 border-dashed border-neutral-300 bg-neutral-50 px-4 py-8 text-center hover:border-blue-400 hover:bg-blue-50 disabled:cursor-wait disabled:opacity-60"
                  >
                    {upload.isPending ? (
                      <>
                        <span className="text-2xl">⏳</span>
                        <span className="text-sm font-medium text-neutral-700">
                          Uploading…
                        </span>
                      </>
                    ) : (
                      <>
                        <span className="text-3xl">🎬</span>
                        <span className="text-sm font-medium text-neutral-700">
                          Click to select a video file
                        </span>
                        <span className="text-xs text-neutral-400">
                          .mp4 · .mov · .mkv · .webm · .m4v
                        </span>
                      </>
                    )}
                  </button>
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
              hint="Short description of the creative format, used as context for scene descriptions."
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
                  — asks the LLM to infer taxonomy hints from the video filename
                  (costs one extra OpenAI call).
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
          {createJob.isPending ? "Submitting…" : "Create job"}
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
