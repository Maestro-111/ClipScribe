# Core Extractor Algorithm

> Scope: this document describes the end-to-end control flow of the core
> extraction algorithm in `backend/src/extractor/extractor_core.py`
> (`VideoInformationExtractor.extract`). It covers how a video is turned into a
> structured `ExtractionSummary` — scene detection, audio, the per-shot
> detection/tracking/OCR loop, and finalization.
>
> The SAM2 propagation loop, DINOv2 re-ID embedding accumulation, and
> cross-shot identity merging are covered in depth in
> [`sam2-tracking-and-identity.md`](./sam2-tracking-and-identity.md). This
> document treats those as a subsystem and references them rather than
> repeating their internals. GroundingDINO, SAM2, MTCNN, Whisper, and
> PaddleOCR are treated as black boxes accessed through their wrappers.

---

## 1. Inputs and output

`extract(video_type, video_path, video_name, run_id)` is the single entry
point. It returns an `ExtractionSummary` (a `TypedDict`) and also writes
`extraction_summary.json` plus per-frame/visualization artifacts under
`artifacts/<run_id>/` (keyed by `run_id`, not `video_name`, so repeated runs
over the same video never collide).

The `ExtractionSummary` bundles seven result streams:

| Key                 | Produced by                        | Meaning |
|---------------------|------------------------------------|---------|
| `global_stats`      | `_digest_video`                    | Shot counts, duration, pacing flags. |
| `shot_boundaries`   | `_digest_video`                    | Per-shot start/end/duration (timeline view). |
| `audio_segments`    | `_analyze_audio`                   | Whisper transcript segments above confidence. |
| `scene_descriptions`| per-shot GPT describe              | Narrative description per shot. |
| `visual_objects`    | `_finalize_data`                   | Cross-shot-resolved objects + motion metrics. |
| `text_events`       | per-frame OCR → `_save_metadata`   | Distinct on-screen text per second. |
| `frame_detections`  | `_record_detection` (all sources)  | Raw per-(frame, box) rows for the UI overlay. |

Progress is emitted throughout via a `ProgressReporter` (null reporter in
CLI/tests), so the web layer can observe phases and per-shot events without the
extractor importing any web/Redis code.

---

## 2. High-level phases

`extract` runs four sequential phases, each bracketed by
`progress.phase_started` / `phase_completed`:

```
_state_init            open capture, video writer, SAM inference state
   │
   ├─ Phase.SCENE_DETECTION   → _digest_video     (shots + pacing stats)
   ├─ Phase.AUDIO             → _analyze_audio     (Whisper transcript)
   ├─ Phase.SHOT_PROCESSING   → per-shot loop      (the core of the algorithm)
   └─ Phase.FINALIZE          → _finalize_data     (identity + metrics + JSON)
```

### 2.1 `_state_init`

Opens the video with OpenCV, reads FPS and dimensions (defaulting FPS to 30 if
the container reports `<= 0`), creates the `tracked_output.mp4` writer, and
initializes the SAM2 inference state. `total_frames` comes from SAM2's
inference state, not the OpenCV capture.

### 2.2 Scene detection — `_digest_video`

Runs PySceneDetect's `ContentDetector(threshold=27.0)` to split the video into
**shots** (scene cuts on adjacent-frame HSV content difference > 27). If no
cuts are found, the whole video becomes a single shot `(0, total_frames)`.

From the shot list it builds `shot_data` (index/start/end/duration in seconds)
and derives pacing statistics into `global_stats`:

- **Dynamic start** — first shot shorter than 3.0s.
- **Quick-pacing intro** — ≥ 5 shots start within `t = 0..5s`.
- **Quick-pacing general** — any 5-second window (anchored at each shot start)
  containing ≥ 5 shots; qualifying windows are recorded as `rapid_fire_intervals`.

These flags exist so the downstream parser can evaluate platform pacing rules
(e.g. YouTube) against persisted numbers rather than re-deriving them.

### 2.3 Audio — `_analyze_audio`

Transcribes the whole video with Whisper (`no_speech_threshold=0.6`,
`condition_on_previous_text=False`). Each segment's `avg_logprob` is converted
to a 0–1 confidence via `exp(avg_logprob)`; segments below `audio_confidence`
are dropped. Kept segments are appended to `audio_registry` and emitted as
`AUDIO_SEGMENT` progress events.

---

## 3. The per-shot loop (the core)

`Phase.SHOT_PROCESSING` iterates every shot. This is where detection, semantic
resolution, tracking, OCR, and face detection are coordinated. For each shot:

### 3.1 Reset and scene understanding

1. `sam_model.reset_state(...)` and clear `active_trackers`; set
   `current_frame = start_f`.
2. **Adaptive frame sampling:** the number of frames sent to GPT scales with
   shot length — `num_samples = clamp(min_samples, max_samples,
   ceil(sampling_rate * sqrt(shot_duration)))`. Longer shots get more (but
   sub-linearly more) sampled frames; the samples are spread evenly across the
   shot.
3. **Scene description:** a single `scene_describer.describe_scene(frames)` call
   over all sampled frames returns both a narrative `raw_context` (stored in
   `scene_description_registry`) and a `final_context` GroundingDINO prompt.
   Emitted as `SHOT_SCENE_DESCRIBED`.
4. **Per-shot taxonomy:** `taxonomy_generator.generate_taxonomy_prompt(...)`
   builds a prompt from the video type, scene context, DINO prompt, and user
   hints (also written to `taxonomy_prompt_<shot>.txt`);
   `generate_taxonomy_targets(...)` produces the canonical target list, which is
   loaded into `taxonomy_resolver.set_active_targets(...)`. Emitted as
   `SHOT_TAXONOMY_RESOLVED`.

The key design point: **detection prompt and taxonomy are regenerated per
shot**, so each shot is detected and labeled in its own semantic context.

### 3.2 Frame walk: detect → resolve → track

Within the shot, the algorithm advances in chunks of `detection_interval`
frames. At each `current_frame` it runs *detection on one frame*, then *SAM2
tracking* over the following chunk:

```
while current_frame < end_f:
    read frame
    ── DETECTION on this frame ─────────────────────────────
    DINO.detect(frame, prompt=final_context)   → raw object boxes
    OCR.detect(frame)                          → text boxes  (recorded raw)
    MTCNN.detect(frame)                        → face boxes

    for each DINO box:
        label = taxonomy_resolver.resolve(raw_label, threshold)   # SBERT
        if label is None: skip                    # not in taxonomy
        record_detection(source="dino")
        if _is_new_object(box, label): _add_new_tracker(...)      # register w/ SAM2

    for each MTCNN face (prob ≥ torch_face_cong):
        label = "human face"
        record_detection(source="mtcnn")
        if _is_new_object(box, label): _add_new_tracker(...)

    ── TRACKING over the next chunk ────────────────────────
    frames_to_track = min(detection_interval, frames_left_in_shot)
    if no active trackers: current_frame += frames_to_track; continue
    for (frame_idx, obj_ids, masks) in SAM2.propagate_in_video(...):
        _save_metadata(...)          # OCR filter, mask→box, re-ID embeddings
        visualize_sam_tracking(...)  # write overlay frame to mp4
    current_frame = last_propagated_frame + 1
```

Three things worth calling out:

- **Semantic gating:** a raw DINO label only becomes a tracked object if
  `TaxonomyResolver.resolve` (SBERT cosine similarity vs. the active targets)
  snaps it to a canonical label above `word_similarity_threshold`. Labels that
  don't map are logged and skipped — they never enter tracking.
- **New-object test:** `_is_new_object` compares the candidate box against
  every active tracker of a *semantically matching* class (IoU > 0.5 +
  `_labels_match`, which uses string inclusion and WordNet synonym/hypernym
  relations). Only genuinely new objects are registered with SAM2 via
  `_add_new_tracker`, preventing duplicate tracks for the same object.
- **Detection vs. tracking cadence:** detection (the expensive DINO/OCR/MTCNN
  calls) runs once per `detection_interval` frames; SAM2 propagation fills in
  the frames between. If a chunk has no active trackers, it is skipped entirely.

OCR text and SAM masks are turned into records inside `_save_metadata`
(text filtering, mask→box conversion, periodic DINOv2 embedding accumulation) —
see the SAM2 doc for that machinery. All detection sources (`dino`, `ocr`,
`mtcnn`, `sam_mask`) funnel through `_record_detection` into `frame_detections`
with a shared row shape.

### 3.3 Detection sources and their roles

| Source     | Detector      | Produces                        | Tracked by SAM2? |
|------------|---------------|---------------------------------|------------------|
| `dino`     | GroundingDINO | Taxonomy-resolved object boxes  | Yes (if new)     |
| `mtcnn`    | MTCNN         | "human face" boxes              | Yes (if new)     |
| `ocr`      | PaddleOCR     | On-screen text boxes            | No               |
| `sam_mask` | SAM2          | Per-frame tracked object masks  | (is the tracker) |

OCR text is aggregated per second into `text_registry` (after
`_is_valid_text` filtering for size, confidence, and boilerplate terms like
"msrp", "copyright", URLs); faces and DINO objects seed trackers.

---

## 4. Finalization — `_finalize_data`

After all shots:

1. **Cross-shot identity resolution** (`_resolve_identities`) merges local
   per-shot object IDs into global IDs using DINOv2 embedding cosine similarity
   plus label matching, with a visual-override path for high-similarity /
   differing-label cases. Full detail in the SAM2 doc.
2. **Per-object motion metrics** (`_calculate_metrics`) derive velocity
   (px/sec), growth factor, screen coverage, dominant direction, centrality,
   screen-time ratio, and a 3×3 quadrant from each object's box/timestamp
   history.
3. Objects are grouped by global ID into `VisualObjectSummary` records (one
   object, potentially many per-shot `occurrences`).
4. `frame_detections` are rewritten so each `object_id` points at the resolved
   **global** ID via `_frame_detections_with_global_object_ids`.
5. Text events are flattened from `text_registry`, and everything is assembled
   into the `ExtractionSummary` and written to `extraction_summary.json`.

The returned summary is what the persistence layer (`backend/src/db/`) writes
to the database and what the parser later evaluates.

---

## 5. Key tunables

These constructor parameters shape the algorithm's cost/accuracy trade-offs:

| Parameter                                   | Controls |
|---------------------------------------------|----------|
| `detection_interval`                        | Frames between expensive detection passes (tracking fills the gap). |
| `min_samples` / `max_samples` / `sampling_rate` | Adaptive per-shot frame sampling for GPT. |
| `word_similarity_threshold`                 | SBERT cutoff for snapping DINO labels to the taxonomy. |
| `dino_box_conf` / `dino_text_conf`          | GroundingDINO detection thresholds. |
| `torch_face_cong`                           | MTCNN face-probability cutoff. |
| `audio_confidence`                          | Whisper segment confidence cutoff. |
| `label_match_merge_threshold` / `label_no_match_merge_threshold` | Cross-shot merge thresholds (see SAM2 doc). |
| `reid_model_frame_check_freq` / `reid_similarity_difference` | Re-ID embedding sampling cadence and novelty gate. |
| `max_artifact_files`                        | Caps per-frame visualization PNGs (mp4 + JSON always kept). |

---

## 6. Related documents

- [`sam2-tracking-and-identity.md`](./sam2-tracking-and-identity.md) — SAM2
  propagation, `_save_metadata`, DINOv2 re-ID, and `_resolve_identities` in
  depth.
- `backend/src/extractor/CLAUDE.md` — module ownership and where to make
  specific kinds of changes.
