// Scribe — the shared, presentational ClipScribe mascot. This is the reusable
// core: it only draws the mark, floats it, shows an optional speech bubble, and
// expresses a `mood`. It has NO positioning, dragging, or data of its own, so
// any route can drop it in for a consistent character. Behavior-specific
// wrappers (e.g. ScribeMascot, which adds fixed positioning + drag on the
// create-job form) compose around this.
//
// Moods only change the accent glyph and whether the accent halo shows; the
// float/halo/shadow animations run continuously and never swap animation-name,
// so nothing restarts/flickers when the mood changes. Animations live in
// styles.css and respect prefers-reduced-motion.
import { useEffect, useState } from "react";
import { Logo } from "./ui";

export type ScribeMood =
  | "bored"
  | "excited"
  | "working"
  | "done"
  | "thinking"
  | "lost";

// Moods that read as "engaged" — they light up the accent halo + glyph.
const LIVELY = new Set<ScribeMood>(["excited", "working", "done", "thinking"]);

// Small corner glyph per mood (none for the quiet moods).
const ACCENT: Partial<Record<ScribeMood, { glyph: string; className: string }>> =
  {
    excited: { glyph: "✦", className: "text-blue-500" },
    working: { glyph: "✦", className: "text-blue-500" },
    done: { glyph: "✓", className: "text-green-600" },
    lost: { glyph: "?", className: "font-bold text-amber-500" },
  };

// Pick a random phrase that isn't `avoid`, so lines don't repeat back-to-back
// (a single-entry pool just returns its one line).
function pickPhrase(pool: string[], avoid?: string): string {
  if (pool.length === 1) return pool[0]!;
  let next = avoid;
  while (next === avoid) next = pool[Math.floor(Math.random() * pool.length)];
  return next!;
}

// Drives the speech line. A single string shows statically (and re-fades when
// it changes); an array rotates on `intervalMs`. Undefined → no bubble.
function useSpeech(
  speech: string | string[] | undefined,
  intervalMs: number,
): string | null {
  const pool = speech == null ? null : Array.isArray(speech) ? speech : [speech];
  // Content signature so the effect re-runs when the lines change, but not on
  // every render (a fresh array literal each render would otherwise re-fire
  // it). Newline is a safe separator — phrases are single-line.
  const poolKey = pool ? pool.join("\n") : "";
  const [phrase, setPhrase] = useState(() => (pool?.length ? pool[0]! : ""));

  useEffect(() => {
    const current = poolKey ? poolKey.split("\n") : [];
    if (!current.length) return;
    setPhrase((prev) => pickPhrase(current, prev));
    if (current.length === 1) return; // static line — no rotation timer
    const id = setInterval(
      () => setPhrase((prev) => pickPhrase(current, prev)),
      intervalMs,
    );
    return () => clearInterval(id);
  }, [poolKey, intervalMs]);

  return pool?.length ? phrase : null;
}

export function Scribe({
  mood,
  scale = 1,
  speech,
  speechIntervalMs = 10_000,
  float = true,
  className = "",
}: {
  mood: ScribeMood;
  scale?: number;
  speech?: string | string[];
  speechIntervalMs?: number;
  float?: boolean;
  className?: string;
}) {
  const phrase = useSpeech(speech, speechIntervalMs);
  const lively = LIVELY.has(mood);
  const accent = ACCENT[mood];

  return (
    <div
      className={`flex select-none flex-col items-center gap-5 ${className}`}
    >
      {/* Speech bubble with a tail pointing down at Scribe. min-h keeps the
          layout steady as lines of different length swap in; `key` on the text
          replays the fade on each change. */}
      {phrase != null && (
        <div className="relative flex min-h-[3rem] max-w-[15rem] items-center rounded-2xl border border-neutral-200 bg-white px-3.5 py-2 text-center text-sm font-medium text-neutral-700 shadow-sm">
          <span key={phrase} className="scribe-say block w-full">
            {phrase}
          </span>
          <span className="absolute -bottom-1.5 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 border-b border-r border-neutral-200 bg-white" />
        </div>
      )}

      {/* Figure. Only the figure scales (bubble text stays readable); origin is
          top so growth goes downward, away from the bubble. */}
      <div
        style={{ transform: `scale(${scale})`, transformOrigin: "top center" }}
        className="relative flex h-28 w-28 items-end justify-center transition-transform duration-500"
      >
        <span
          aria-hidden
          className={`scribe-halo pointer-events-none absolute bottom-6 h-20 w-20 rounded-full bg-blue-400/40 blur-2xl transition-opacity duration-500 ${
            lively ? "opacity-100" : "opacity-0"
          }`}
        />

        <div className={`pointer-events-none ${float ? "scribe-bob" : ""}`}>
          <div className="relative drop-shadow-lg">
            <Logo size={76} />
            {accent && (
              <span
                className={`absolute -right-1 -top-1 text-lg transition-opacity duration-300 ${accent.className}`}
              >
                {accent.glyph}
              </span>
            )}
          </div>
        </div>

        {/* Ground shadow — only shown while floating (a pulsing shadow under a
            static figure reads oddly). */}
        {float && (
          <span
            aria-hidden
            className="scribe-shadow pointer-events-none absolute bottom-0 h-2 w-16 rounded-[50%] bg-neutral-900/30 blur-[3px]"
          />
        )}
      </div>
    </div>
  );
}
