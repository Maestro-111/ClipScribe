// Decorative animation of the ClipScribe core pipeline, shown beside the
// create-job form to fill the otherwise-empty canvas. It mirrors the real
// engine flow (backend/src/clip_scribe/engine.py + extractor_core.py): digest
// the video, transcribe audio, split into scenes, then per-scene build targets
// → detect (GroundingDINO) → track (SAM2), finalize, and hand off to the
// parser layer. Purely presentational — no data, no props — so it never blocks
// or depends on the form. Hovering a step reveals a fuller explanation.
import { useEffect, useState, type ReactNode } from "react";

// One node in the flow. `icon` is a compact inline SVG so the component stays
// dependency-free (no icon library), matching ui.tsx's hand-rolled glyphs.
// `detail` is the one-line subtitle; `long` is the on-hover explanation.
interface Stage {
  key: string;
  title: string;
  detail: string;
  long: string;
  icon: ReactNode;
}

// Small rounded-stroke glyphs sized to 20×20. Kept inline and minimal.
function Glyph({ children }: { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
    >
      {children}
    </svg>
  );
}

// The ordered core stages, in the sequence the pipeline actually runs them.
const STAGES: Stage[] = [
  {
    key: "digest",
    title: "Digest video",
    detail: "Load & decode frames",
    long: "The uploaded video is ingested and decoded frame-by-frame, and its metadata (duration, resolution, fps) is read to drive the rest of the pipeline.",
    icon: (
      <Glyph>
        <rect x="2.5" y="4.5" width="15" height="11" rx="2" />
        <path d="M8.5 7.5l4 2.5-4 2.5z" fill="currentColor" stroke="none" />
      </Glyph>
    ),
  },
  {
    key: "audio",
    title: "Transcribe audio",
    detail: "Whisper speech-to-text",
    long: "The audio track is transcribed with OpenAI Whisper into timestamped speech segments, later used for call-to-action and messaging checks.",
    icon: (
      <Glyph>
        <path d="M10 3v9m0 0a2 2 0 1 1-2 2M14 7v5a2 2 0 1 1-2-2M6 7v5a2 2 0 1 1-2-2" />
      </Glyph>
    ),
  },
  {
    key: "scenes",
    title: "Scene detection",
    detail: "Break into shots",
    long: "The video is segmented into distinct shots at cut boundaries, so each visually-coherent scene is analyzed independently.",
    icon: (
      <Glyph>
        <rect x="2.5" y="4.5" width="15" height="11" rx="1.5" />
        <path d="M6.5 4.5v11M13.5 4.5v11" />
      </Glyph>
    ),
  },
  {
    key: "targets",
    title: "Generate targets",
    detail: "GPT vision + LLM taxonomy",
    long: "For each scene, GPT vision describes what's on screen and an LLM turns that into a canonical set of detection targets (the taxonomy) for the detector to search for.",
    icon: (
      <Glyph>
        <circle cx="10" cy="4" r="2" />
        <circle cx="4.5" cy="15" r="2" />
        <circle cx="15.5" cy="15" r="2" />
        <path d="M10 6v3m0 0-4.5 4m4.5-4 4.5 4" />
      </Glyph>
    ),
  },
  {
    key: "detect",
    title: "Detect objects",
    detail: "GroundingDINO",
    long: "GroundingDINO runs open-vocabulary detection using the scene's targets, drawing a bounding box around each matched object.",
    icon: (
      <Glyph>
        <path d="M3 6.5V4h2.5M14.5 4H17v2.5M17 13.5V16h-2.5M5.5 16H3v-2.5" />
        <circle cx="10" cy="10" r="2.5" />
      </Glyph>
    ),
  },
  {
    key: "track",
    title: "Track objects",
    detail: "SAM2",
    long: "SAM2 propagates the detected objects across the frames of the scene, giving each a stable mask and identity over time.",
    icon: (
      <Glyph>
        <rect x="2.5" y="6" width="6" height="6" rx="1" />
        <path d="M9 9h4.5" strokeDasharray="1.5 2" />
        <path d="M13 6l4 3-4 3z" fill="currentColor" stroke="none" />
      </Glyph>
    ),
  },
  {
    key: "finalize",
    title: "Finalize",
    detail: "Merge across shots",
    long: "Per-shot results are consolidated: object identities are merged across shots, OCR text and detected faces are attached, and the structured run is persisted.",
    icon: (
      <Glyph>
        <path d="M4 10.5l3.5 3.5L16 6" />
      </Glyph>
    ),
  },
  {
    key: "parse",
    title: "Parser layer",
    detail: "Score against platform",
    long: "A LangGraph parser queries the persisted run and evaluates it against the selected platform's criteria (e.g. YouTube ABCD), producing the final report.",
    icon: (
      <Glyph>
        <rect x="3.5" y="2.5" width="13" height="15" rx="1.5" />
        <path d="M6.5 6.5h7M6.5 10h7M6.5 13.5h4" />
      </Glyph>
    ),
  },
];

// ms each stage stays "active" before the pulse advances. Tuned so a full loop
// stays lively but not distracting next to a form the user is filling in.
const STEP_MS = 1150;

export function PipelineAnimation() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const id = setInterval(
      () => setActive((i) => (i + 1) % STAGES.length),
      STEP_MS,
    );
    return () => clearInterval(id);
  }, []);

  return (
    <div className="select-none">
      <div className="mb-6 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-blue-500" />
        </span>
        ClipScribe pipeline
      </div>

      <ol className="relative space-y-1">
        {STAGES.map((stage, i) => {
          const isActive = i === active;
          // Stages the pulse has already passed this loop stay lightly filled.
          const isDone = i < active;
          return (
            <li key={stage.key} className="group relative flex gap-3">
              {/* Connector rail: a vertical line linking the nodes, with the
                  segment above the active node tinted to suggest flow. */}
              {i < STAGES.length - 1 && (
                <span
                  aria-hidden
                  className={`absolute left-[19px] top-10 h-[calc(100%-1rem)] w-0.5 transition-colors duration-500 ${
                    active > i ? "bg-blue-300" : "bg-neutral-200"
                  }`}
                />
              )}

              {/* Node marker */}
              <span
                className={`relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border transition-all duration-500 ${
                  isActive
                    ? "border-blue-500 bg-blue-500 text-white shadow-lg shadow-blue-500/30 scale-110"
                    : isDone
                      ? "border-blue-200 bg-blue-50 text-blue-500"
                      : "border-neutral-200 bg-white text-neutral-400"
                }`}
              >
                {stage.icon}
                {isActive && (
                  <span className="absolute inset-0 animate-ping rounded-xl border-2 border-blue-400 opacity-60" />
                )}
              </span>

              {/* Label */}
              <div className="min-w-0 flex-1 pb-4 pt-1">
                <p
                  className={`text-sm font-medium transition-colors duration-500 ${
                    isActive ? "text-neutral-900" : "text-neutral-500"
                  }`}
                >
                  {stage.title}
                </p>
                <p
                  className={`truncate text-xs transition-colors duration-500 ${
                    isActive ? "text-neutral-500" : "text-neutral-400"
                  }`}
                >
                  {stage.detail}
                </p>
              </div>

              {/* Hover explanation. Anchored from just past the icon to the
                  right edge of the column, so it never clips horizontally; it
                  overlays the label on hover and fades/rises in. pointer-events
                  are off so it can't trap the cursor. */}
              <div
                role="tooltip"
                className="pointer-events-none absolute left-14 right-1 top-0 z-30 translate-y-1 rounded-lg border border-neutral-200 bg-white p-3 opacity-0 shadow-xl ring-1 ring-black/5 transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100"
              >
                <p className="text-sm font-semibold text-neutral-900">
                  {stage.title}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-neutral-600">
                  {stage.long}
                </p>
              </div>
            </li>
          );
        })}
      </ol>

      <p className="mt-4 max-w-xs text-xs leading-relaxed text-neutral-400">
        Every video runs through this pipeline once created. Hover a step for
        details.
      </p>
    </div>
  );
}
