// Right-column companions to the create-job form (jobs.new.tsx). Two pieces:
//   • JobSummary  — a live readout of the current form state plus a grounded
//     estimate of the OpenAI work the batch will trigger.
//   • RunOutputs  — a static "what you'll get" preview of the artifacts a
//     finished run produces, tying this page to the run inspector.
// Both are presentational; JobSummary derives everything from props (the form
// state), so it never touches the network.
import type { ReactNode } from "react";

// Per-video LLM-call constants, measured from the backend pipeline:
//   - The YouTube parser evaluates 17 agentic criteria, each one LLM call,
//     independent of video length (src/parser/youtube/criteria.py +
//     evaluator_base.py). The other criteria are baseline/no-LLM.
//   - "Generate hints from name" adds one call per video (taxonomy_core.py).
// Scene description + taxonomy add ~2 GPT calls per *detected shot*, which we
// can't know until the video is processed — surfaced as a separate note.
const PARSER_CALLS_PER_VIDEO = 17;
const SHOT_CALLS_EACH = 2;

interface JobSummaryProps {
  videoCount: number;
  platform: string;
  brandName: string;
  videoType: string;
  hintTermCount: number;
  generateHintFromName: boolean;
}

export function JobSummary({
  videoCount,
  platform,
  brandName,
  videoType,
  hintTermCount,
  generateHintFromName,
}: JobSummaryProps) {
  const hintCalls = generateHintFromName ? videoCount : 0;
  const fixedCalls = videoCount * PARSER_CALLS_PER_VIDEO + hintCalls;

  const hintLabel =
    [
      hintTermCount > 0
        ? `${hintTermCount} term${hintTermCount === 1 ? "" : "s"}`
        : null,
      generateHintFromName ? "from filename" : null,
    ]
      .filter(Boolean)
      .join(" · ") || "—";

  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        Job summary
      </h3>

      {videoCount === 0 ? (
        <p className="text-sm text-neutral-400">
          Add a video to see what this job will run.
        </p>
      ) : (
        <>
          <dl className="space-y-2.5 text-sm">
            <Row label="Videos">
              {videoCount} → {videoCount} run{videoCount === 1 ? "" : "s"}
            </Row>
            <Row label="Platform">{platform}</Row>
            <Row label="Brand">{brandName.trim() || "—"}</Row>
            <Row label="Video type">{videoType.trim() || "—"}</Row>
            <Row label="Hints">{hintLabel}</Row>
          </dl>

          {/* Grounded estimate. The fixed part is exact; the shot-scaled part
              depends on how many shots the videos break into, so it's a note
              rather than a number we'd be making up. */}
          <div className="mt-4 border-t border-neutral-100 pt-4">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-neutral-600">
                OpenAI calls, up front
              </span>
              <span className="text-lg font-semibold text-neutral-900">
                ≈ {fixedCalls}
              </span>
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-neutral-400">
              {videoCount} × {PARSER_CALLS_PER_VIDEO} parser criteria
              {hintCalls > 0 && ` + ${hintCalls} filename hint`}, plus about{" "}
              {SHOT_CALLS_EACH} scene calls per detected shot as each video is
              processed.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="shrink-0 text-neutral-500">{label}</dt>
      <dd className="truncate text-right font-medium text-neutral-900">
        {children}
      </dd>
    </div>
  );
}

// One artifact a finished run produces. Icons are inline SVG to stay
// dependency-free, matching ui.tsx / PipelineAnimation.tsx.
interface Output {
  title: string;
  detail: string;
  icon: ReactNode;
}

function OutIcon({ children }: { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-4 w-4"
    >
      {children}
    </svg>
  );
}

const OUTPUTS: Output[] = [
  {
    title: "ABCD scorecard",
    detail: "Pass/fail per criterion — export to Excel or CSV",
    icon: (
      <OutIcon>
        <rect x="3.5" y="2.5" width="13" height="15" rx="1.5" />
        <path d="M6.5 7h7M6.5 10h7M6.5 13h4" />
      </OutIcon>
    ),
  },
  {
    title: "Annotated video",
    detail: "Detection boxes tracked across every shot",
    icon: (
      <OutIcon>
        <rect x="2.5" y="4.5" width="15" height="11" rx="2" />
        <rect x="6" y="7" width="5" height="5" rx="0.5" />
      </OutIcon>
    ),
  },
  {
    title: "Transcript",
    detail: "Timestamped speech from Whisper",
    icon: (
      <OutIcon>
        <path d="M4 5.5h9M4 9h12M4 12.5h9M4 16h6" />
      </OutIcon>
    ),
  },
  {
    title: "On-screen text & faces",
    detail: "OCR reads and detected people",
    icon: (
      <OutIcon>
        <circle cx="10" cy="7" r="2.5" />
        <path d="M4.5 16a5.5 5.5 0 0 1 11 0" />
      </OutIcon>
    ),
  },
  {
    title: "Advisory chat",
    detail: "Ask questions about the results",
    icon: (
      <OutIcon>
        <path d="M4 4.5h12a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H8l-4 3v-3a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1Z" />
      </OutIcon>
    ),
  },
];

export function RunOutputs() {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        What you'll get
      </h3>
      <ul className="space-y-3">
        {OUTPUTS.map((o) => (
          <li key={o.title} className="flex gap-3">
            <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-neutral-100 text-neutral-500">
              {o.icon}
            </span>
            <div className="min-w-0">
              <p className="text-sm font-medium text-neutral-800">{o.title}</p>
              <p className="text-xs leading-relaxed text-neutral-500">
                {o.detail}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
