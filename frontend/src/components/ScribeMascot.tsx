// ScribeMascot — the create-job form's floating Scribe. It supplies this
// screen's state to the reusable <FloatingScribe> (drag/fixed behavior): a mood
// + phrase pool, and a size that grows with the number of queued videos.
import { FloatingScribe } from "./FloatingScribe";
import { type ScribeMood } from "./Scribe";

// The form only ever reaches these three moods.
export type ScribeState = Extract<ScribeMood, "bored" | "excited" | "working">;

// A characterful phrase pool per mood; <Scribe> rotates them.
const PHRASES: Record<ScribeState, string[]> = {
  bored: [
    "Waiting for a video to sink my teeth into…",
    "It's awfully quiet in here.",
    "I could be parsing frames right now, you know.",
    "*taps foot* Any videos yet?",
    "Drop a clip on me anytime.",
    "Zzz… oh! Ready whenever you are.",
  ],
  excited: [
    "Ooh, a video! Let's gooo.",
    "Can't wait to dig into this one.",
    "Frames, transcripts, objects — gimme!",
    "This one looks juicy.",
    "Load me up, I can take more!",
    "Sharpening my pencils…",
  ],
  working: [
    "On it — digging in!",
    "Rolling up my sleeves.",
    "Decoding frames…",
    "Let's find those ABCDs.",
    "Here we gooo!",
  ],
};

// Video count at which Scribe reaches full size; more don't grow it.
const MAX_VIDEOS = 50;

export function ScribeMascot({
  state,
  videoCount,
}: {
  state: ScribeState;
  videoCount: number;
}) {
  // Grows from ~0.85× (idle) to 1.7× at MAX_VIDEOS, then flat.
  const n = Math.min(Math.max(videoCount, 0), MAX_VIDEOS);
  const figScale = 0.85 + (n / MAX_VIDEOS) * 0.85;

  return (
    <FloatingScribe mood={state} scale={figScale} speech={PHRASES[state]} />
  );
}
