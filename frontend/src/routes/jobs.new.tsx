import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState } from "react";
import {
  useCreateJob,
  useInputs,
  usePlatforms,
  useUploadVideos,
  type JobCreateRequest,
} from "../api/hooks";
import { PipelineAnimation } from "../components/PipelineAnimation";
import { JobSummary, RunOutputs } from "../components/JobSidebar";

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

// One video queued for the job; a job fans out to one run per entry here.
// `name` is what the user sees. A video is either already stored (`path` set —
// picked from the existing list) or a local file pending upload (`file` set —
// nothing is sent to the server until "Create job"). On submit, pending files
// are uploaded and their returned storage keys fill in `path`.
interface SelectedVideo {
  name: string;
  path?: string;
  file?: File;
}

// Stable identity for dedup/removal: the storage key for stored videos, or a
// name+size signature for files not yet uploaded.
function videoKey(v: SelectedVideo): string {
  return v.path ?? `pending:${v.file?.name}:${v.file?.size}`;
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
  const upload = useUploadVideos();
  // Aggregate upload completion in [0, 1], driven by XHR progress events.
  const [uploadPct, setUploadPct] = useState(0);

  const addVideos = (videos: SelectedVideo[]) =>
    setSelected((prev) => {
      const byKey = new Map(prev.map((v) => [videoKey(v), v]));
      for (const v of videos) byKey.set(videoKey(v), v);
      return [...byKey.values()];
    });

  const removeVideo = (key: string) =>
    setSelected((prev) => prev.filter((v) => videoKey(v) !== key));

  const isSelected = (path: string) => selected.some((v) => v.path === path);

  // Queue chosen files locally — nothing uploads until "Create job", so
  // deselecting a video never leaves an orphaned upload behind.
  const handleSelectFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const files = Array.from(fileList).filter((f) => hasAllowedSuffix(f.name));
    if (!files.length) return;
    addVideos(files.map((f) => ({ name: f.name, file: f })));
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

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    // Upload any files not yet on the server, in one request, and map their
    // returned storage keys back onto the queued entries (order preserved). On
    // failure, bail — the error surfaces via `upload.error` and nothing is
    // created.
    const pending = selected.filter((v) => v.file);
    let resolved = selected;
    if (pending.length) {
      let uploaded;
      try {
        setUploadPct(0);
        uploaded = await upload.mutateAsync({
          files: pending.map((v) => v.file!),
          onProgress: setUploadPct,
        });
      } catch {
        return;
      }
      const keyByPending = new Map(
        pending.map((v, i) => [videoKey(v), uploaded[i]]),
      );
      resolved = selected.map((v) => {
        const u = v.file ? keyByPending.get(videoKey(v)) : undefined;
        return u ? { name: u.name, path: u.path } : v;
      });
    }

    createJob.mutate(
      {
        mode,
        platform: platform as JobCreateRequest["platform"],
        videos: resolved.map((v) => ({
          video_path: v.path!,
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
    <div className="grid gap-10 lg:grid-cols-[minmax(0,34rem)_1fr] lg:gap-20">
      <div className="max-w-xl">
        <h1 className="mb-6 text-2xl font-semibold">New job</h1>

        <form onSubmit={(e) => void handleSubmit(e)} className="space-y-6">
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
                      key={videoKey(v)}
                      className="flex items-center justify-between rounded border bg-neutral-50 px-2 py-1 text-sm"
                    >
                      <span className="truncate">{v.name}</span>
                      <button
                        type="button"
                        onClick={() => removeVideo(videoKey(v))}
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
                  {/* Hidden native inputs; the styled buttons below trigger them.
                      Files are only queued here — they upload on "Create job". */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".mp4,.mov,.mkv,.webm,.m4v"
                    className="sr-only"
                    onChange={(e) => handleSelectFiles(e.target.files)}
                  />
                  {/* `webkitdirectory` isn't in React's input typings, so set it
                      via a callback ref — which runs whenever this node mounts,
                      unlike a mount-only effect that fires before the tab exists. */}
                  <input
                    ref={(el) => el?.setAttribute("webkitdirectory", "")}
                    type="file"
                    multiple
                    className="sr-only"
                    onChange={(e) => handleSelectFiles(e.target.files)}
                    id="folder-input"
                  />
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex flex-1 flex-col items-center gap-1 rounded-lg border-2 border-dashed border-neutral-300 bg-neutral-50 px-4 py-6 text-center hover:border-blue-400 hover:bg-blue-50"
                    >
                      <span className="text-2xl">🎬</span>
                      <span className="text-sm font-medium text-neutral-700">
                        Select video files
                      </span>
                      <span className="text-xs text-neutral-400">
                        one or more · .mp4 .mov .mkv .webm .m4v
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        document.getElementById("folder-input")?.click()
                      }
                      className="flex flex-1 flex-col items-center gap-1 rounded-lg border-2 border-dashed border-neutral-300 bg-neutral-50 px-4 py-6 text-center hover:border-blue-400 hover:bg-blue-50"
                    >
                      <span className="text-2xl">📁</span>
                      <span className="text-sm font-medium text-neutral-700">
                        Select a folder
                      </span>
                      <span className="text-xs text-neutral-400">
                        queues every video in the folder
                      </span>
                    </button>
                  </div>
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
                >
                  <input
                    value={brandName}
                    onChange={(e) => setBrandName(e.target.value)}
                    placeholder="e.g. RAM"
                    className="w-full rounded border px-2 py-1.5"
                  />
                  {!brandName.trim() && <FieldWarning>The brand-presence criteria (brand in speech, brand logo on screen) can't identify anything without a brand name — those checks will effectively pass on nothing.</FieldWarning>}
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
                  {!callToActions.trim() && <FieldWarning>The call-to-action criteria (CTA in speech, CTA on screen) have no phrases to match against and will effectively pass on nothing.</FieldWarning>}
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

        {(createJob.error || upload.error) && (
          <p className="text-sm text-red-600">
            {((createJob.error || upload.error) as Error).message}
          </p>
        )}

        {(upload.isPending || createJob.isPending || createJob.isSuccess) && (
          <SubmitProgress
            uploadCount={selected.filter((v) => v.file).length}
            uploadPct={uploadPct}
            uploading={upload.isPending}
            creating={createJob.isPending}
            created={createJob.isSuccess}
          />
        )}

        <button
          type="submit"
          disabled={!canSubmit || upload.isPending || createJob.isPending}
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {upload.isPending
            ? "Uploading…"
            : createJob.isPending
              ? "Submitting…"
              : selected.length > 1
                ? `Create job (${selected.length} videos)`
                : "Create job"}
          </button>
        </form>
      </div>

      {/* Right canvas: fill the otherwise-empty space with context for the job
          being created — the pipeline each video runs through, a live summary
          of the current form state, and a preview of the run's outputs. Dropped
          entirely below lg where the form goes full width. */}
      <aside className="hidden space-y-8 lg:block">
        <PipelineAnimation />
        <JobSummary
          videoCount={selected.length}
          platform={platform}
          brandName={brandName}
          videoType={videoType}
          hintTermCount={splitComma(userHints).length}
          generateHintFromName={generateHintFromName}
        />
        <RunOutputs />
      </aside>
    </div>
  );
}

// Submit-time progress: a small stepper that makes the upload → create → open
// sequence legible. The upload step shows a real byte-progress bar (the actual
// wait); create + open are near-instant but shown so the transition isn't a
// mysterious freeze. The upload step is omitted entirely when every video was
// picked from existing storage (nothing to upload).
type StepState = "pending" | "active" | "done";

function SubmitProgress({
  uploadCount,
  uploadPct,
  uploading,
  creating,
  created,
}: {
  uploadCount: number;
  uploadPct: number;
  uploading: boolean;
  creating: boolean;
  created: boolean;
}) {
  const steps: { key: string; label: string; state: StepState; pct?: number }[] =
    [];

  if (uploadCount > 0) {
    steps.push({
      key: "upload",
      label: `Uploading ${uploadCount} video${uploadCount > 1 ? "s" : ""}`,
      // Once we've moved past uploading, this step is done (create/open follow).
      state: uploading ? "active" : "done",
      pct: uploadPct,
    });
  }
  steps.push({
    key: "create",
    label: "Creating job",
    state: creating ? "active" : created ? "done" : "pending",
  });
  steps.push({
    key: "open",
    label: "Opening job",
    state: created ? "active" : "pending",
  });

  return (
    <div className="space-y-2 rounded-lg border border-neutral-200 bg-neutral-50 p-3">
      {steps.map((s) => (
        <div key={s.key} className="space-y-1">
          <div className="flex items-center gap-2 text-sm">
            <StepIcon state={s.state} />
            <span
              className={
                s.state === "pending" ? "text-neutral-400" : "text-neutral-700"
              }
            >
              {s.label}
              {s.key === "upload" && s.state === "active" && s.pct != null
                ? ` — ${Math.round(s.pct * 100)}%`
                : ""}
            </span>
          </div>
          {s.key === "upload" && s.state === "active" && (
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200">
              <div
                className="h-full rounded-full bg-blue-500 transition-[width] duration-150"
                style={{ width: `${Math.round((s.pct ?? 0) * 100)}%` }}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "done") return <span className="text-green-600">✓</span>;
  if (state === "active")
    return (
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
    );
  return (
    <span className="inline-block h-3 w-3 rounded-full border border-neutral-300" />
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

// Non-blocking advisory shown under an optional field whose emptiness weakens
// the parser (e.g. brand-presence / CTA criteria have nothing to match). Amber,
// not red — the job is still valid to submit.
function FieldWarning({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-1 flex items-start gap-1 text-xs text-amber-600">
      <span aria-hidden>⚠</span>
      <span>{children}</span>
    </p>
  );
}
