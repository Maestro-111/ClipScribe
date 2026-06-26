# ClipScribe Web App — Migration Plan

A planning doc, not a spec. It maps the current CLI-driven pipeline onto a
two-tier web app: a TypeScript single-page dashboard and a Python backend split
into a thin FastAPI process and one or more Celery workers.

This file is a living checklist. Sections marked **Open question** are decisions
to make before that piece is built.

---

## 1. Target architecture

```
                       ┌──────────────────────────┐
                       │   Browser (TS SPA)        │
                       │   Vite + React + TS       │
                       └────────────┬──────────────┘
                                    │
                  REST  ────────────┼──────────────  SSE
                                    │
                       ┌────────────▼──────────────┐
                       │     FastAPI (slim)        │
                       │  - request validation     │
                       │  - DB reads               │
                       │  - artifact serving       │
                       │  - SSE pub/sub bridge     │
                       │  - enqueue Celery tasks   │
                       └─────┬────────────┬────────┘
                             │            │
                             ▼            ▼
                       ┌─────────┐  ┌──────────────┐
                       │  Redis  │  │  Postgres    │
                       │ broker  │  │  schema/runs │
                       │ + pubsub│  │  + jobs +    │
                       │         │  │  detections  │
                       └────┬────┘  └──────────────┘
                            │
                ┌───────────▼──────────────┐
                │  Celery worker(s) (heavy)│
                │  loads SAM2 / DINO /     │
                │  Whisper / DINOv2 /      │
                │  MTCNN / PaddleOCR once  │
                │  per worker process      │
                └──────────────────────────┘
```

- **API container**: small, fast restart, scales horizontally.
- **Worker container**: big (8–15 GB), GPU/MPS-bound, one job at a time per
  GPU. Multiple machines = multiple workers, same Redis queue.
- **Redis**: both the Celery broker and the live-progress pub/sub channel.
- **Postgres**: existing schema (`src/db/schema.py`) plus three new tables
  (`jobs`, `frame_detections`, `parser_results`) and one widened table
  (`shot_boundaries` extracted from `global_stats`).

---

## 2. Repo restructure (monorepo)

```
clipscribe/
  backend/
    src/                          # existing Python, mostly untouched
      clip_scribe/  extractor/  ocr/  parser/  db/  dino/  sam2/  utils/
    app/                          # NEW — web layer
      main.py                     # FastAPI app + lifespan
      deps.py                     # injection helpers (DB, redis, settings)
      routes/
        jobs.py                   # POST /jobs, GET /jobs, GET /jobs/{id}/events
        runs.py                   # GET /runs, GET /runs/{id}/...
        artifacts.py              # Range video, PNGs, /frames
        health.py                 # /healthz, /readyz
      models.py                   # Pydantic request/response schemas
      events.py                   # ProgressReporter + Redis pub/sub helpers
      tasks.py                    # Celery task definitions
      celery_app.py               # Celery app + worker_process_init
      progress.py                 # Event vocabulary + reducer reference
    pyproject.toml
    Dockerfile.api                # slim image
    Dockerfile.worker             # heavy image
  frontend/
    src/
      api/                        # generated TS client from OpenAPI
      lib/                        # state, hooks, utils
      pages/                      # JobsList / NewJob / JobLive / RunInspector
      components/                 # VideoOverlay, Timeline, PhaseTree, LogTail
    package.json
    Dockerfile
    vite.config.ts
  docker-compose.yml              # api + worker + redis + postgres + nginx
  Makefile
  docs/web-app-plan.md            # this file
```

`backend/src/` is the current `src/`. The CLI entry (`main.py`) becomes a thin
shim that calls into the same job-execution function the Celery task uses, so
the CLI path keeps working.

**Open question**: keep `src/` at the repo root and use a Python `src-layout` so
the package import path doesn't change, or move under `backend/`? Moving means
updating every `from src.x import y` site — there are many. Recommended: keep
`src/` at root, put new web code at `app/` at root, and treat "backend" as
logical not filesystem-level. Drop the `backend/` directory from the layout
above if that's the call.

---

## 3. Builder refactor (load-once vs per-job)

This is the single biggest non-trivial Python change. Without it, every job
re-loads SAM2 + DINOv2 + Whisper + DINO + MTCNN + PaddleOCR (~30–60s).

### Process model recap

The API process and the worker process are **separate**. The API never imports
torch and never loads a model — it only validates requests, reads the DB, and
sends Celery tasks into Redis. The worker imports everything heavy. Workers
can run on the same machine as the API or on remote machines (e.g. a GPU box);
they only need network reach to Redis (broker + pubsub) and Postgres. Routing
is automatic via the Celery queue.

For Mac dev: run the worker natively (so MPS works) and the rest in Docker.
For Linux+GPU prod: everything in Docker, worker container gets `--gpus all`.

### Current tangle (`src/clip_scribe/build_clip_scribe.py`)

`ClipScribeBuilder.build_clip_scribe(...)` mixes process-level params (device,
model weights, thresholds) with per-job params (`video_name`, `video_path`,
`user_hints`, `clib_scribe_platform_conf`, `generate_hint_from_name`,
`clib_scribe_mode`) in a single call. Inside `build_extractor`, the only lines
that are actually per-job are:
- 190–194: `combined_hints` derivation from `user_hints` + `video_name`.
- 276: `taxonomy_user_hints=combined_hints` baked into the extractor
  constructor.

Everything else (lines 137–267) loads models or reads yaml defaults — all
process-level. The "tangle" is therefore narrow: two locations in one method.

### Boot-vs-per-job classification

Line numbers refer to `src/clip_scribe/build_clip_scribe.py` as of this writing.

#### Boot once (worker startup)

| What | Source line(s) | Cost |
|---|---|---|
| Read `clip_scribe.yaml` | 66–88 | trivial |
| DB engine + reader + writer | 313–331 | low |
| `sam2_size`, `taxonomy_objects_num`, `dino_size` | 137, 139, 162 | trivial |
| Resolved model names (hint/target/scene/parser) | 141–156, 104–108 | trivial |
| Default thresholds (`audio_confidence`, `dino_text_conf`, `dino_box_conf`, `torch_face_cong`, `label_*_merge_threshold`, `word_similarity_threshold`, `detection_interval`, `reid_model_frame_check_freq`) | 158–181 | trivial |
| Scene analysis numeric params (`min_samples`, `max_samples`, `sampling_rate`, `max_frame_dim`, `image_detail`) | 205–209 | trivial |
| `ProfilesPile()` | 188 | trivial |
| `TaxonomyResolver(logger)` (owns SBERT) | 196 | **HEAVY** |
| `TaxonomyGenerator(...)` (OpenAI client + profiles) | 197–202 | low |
| `DinoDetector(...)` | 211 | **HEAVY** (GroundingDINO checkpoint) |
| `GPTSceneDescriber(...)` (OpenAI client) | 212–217 | low |
| Device resolution (`sam2_device`, `dino_reid_device`, `whisper_device`) | 219–235 | trivial |
| `OCRSystem(logger)` | 237 | **HEAVY** (PaddleOCR) |
| `build_sam2_video_predictor(...)` | 238–240 | **HEAVY** (SAM2 checkpoint; biggest) |
| `torch.hub.load("facebookresearch/dinov2", ...)` | 246–250 | **HEAVY** (DINOv2) |
| `whisper.load_model("base", ...)` | 254 | **HEAVY** |
| `embedding_transform` | 256–265 | trivial |
| `MTCNN(keep_all=True, device="cpu")` | 267 | moderate |
| `VideoInformationExtractor(...)` constructor (sans hints) | 269–296 | trivial — just stores refs |
| Parser agent defaults (`parser_max_parallel`, `recursion_limit`, `parser_detection_model`) | 101–108 | trivial |

#### Per job (Celery task payload)

| What | Source line(s) | Notes |
|---|---|---|
| `video_name`, `video_path`, `video_type` | 124, plus engine args | from request |
| `mode` (`full`/`extract`/`parse`) | 305, 336, 352 | from request |
| `user_hints` | 125, 190 | from request |
| `generate_hint_from_name` + the `generate_hints_from_video_name` OpenAI call | 126, 191–194 | per-job because it depends on `video_name` |
| `platform_name` + `platform_params` → `BasePlatformConf` | 307, 308 (built in `build_clip_scribe_plalform.py`) | per-job |
| Optional threshold overrides | n/a today | new in the API |
| `ProgressReporter` instance bound to `job_id` | n/a today | new in the API |
| Per-run state inside `VideoInformationExtractor` (`active_trackers`, `text_registry`, `object_registry`, `audio_registry`, `scene_description_registry`, `obj_id_counter`, `current_frame`, `cap`, `video_writer`, `inference_state`, `shot_boundaries`, `global_stats`) | inside `extract()` | must be reset between jobs |

### Refactor target

Split into three layers:

1. **`ModelRegistry`** — owns everything in the "boot once" table above. Built
   once at worker startup via Celery's `worker_process_init` signal. Holds a
   single long-lived `VideoInformationExtractor` instance whose constructor
   **no longer takes `taxonomy_user_hints`**.

2. **`JobAssembler`** — takes a `ModelRegistry` + per-job `JobParams` and
   returns a `ClipScribeEngine` ready to call `.run()`. Cheap (target <100 ms).
   It:
   - resolves `combined_hints` (calling OpenAI only if
     `generate_hint_from_name=True`),
   - calls `registry.extractor.setup_for_job(...)` to rebind hints/reporter
     and clear per-run state,
   - builds the per-job `VideoInformationParser` via
     `registry.build_parser_for_job(platform_conf, reporter)`,
   - constructs the per-job `ClipScribeEngine` wrapper.

3. **`ClipScribeEngine`** — unchanged conceptually. Accepts an already-built
   extractor/parser. Gains a `reporter` arg and emits `job.started` /
   `job.completed` / `job.failed` around the existing flow.

### Required edits

#### `src/extractor/extractor_core.py`
- Remove `taxonomy_user_hints` from `__init__` (currently line 266).
- Add a `setup_for_job(...)` method that does both rebind and reset:
  ```python
  def setup_for_job(
      self,
      user_hints: list[str] | None,
      reporter: "ProgressReporter",
      thresholds_override: dict | None = None,
  ) -> None:
      self.taxonomy_user_hints = user_hints
      self.reporter = reporter

      # reset per-run state
      self.current_frame = 0
      self.obj_id_counter = 1
      self.active_trackers = {}
      self.id_to_label = {}
      self.text_registry = defaultdict(set)
      self.object_registry = {}
      self.audio_registry = []
      self.scene_description_registry = []

      # optional per-job threshold overrides (kept narrow)
      if thresholds_override:
          for k, v in thresholds_override.items():
              if not hasattr(self, k):
                  raise ValueError(f"unknown threshold override: {k}")
              setattr(self, k, v)
  ```
- `extract()` signature stays as it is.
- Sprinkle `self.reporter.emit(...)` calls at the hook sites listed in §5.

#### `src/clip_scribe/build_clip_scribe.py`
- Replace `build_extractor(...)` with **`build_extractor_models(self) -> ExtractorBundle`** that takes no per-job args. Returns the populated extractor + the resolver/generator references the assembler may need to introspect.
- Replace `build_parser(...)` with two methods:
  - **`build_parser_defaults(self) -> ParserDefaults`** — agent LLM, max_parallel, recursion_limit, output_dir.
  - **`build_parser_for_job(self, defaults, platform_conf, reporter) -> VideoInformationParser`** — cheap, called per job.
- Replace `build_clip_scribe(...)` with two distinct entry points:
  - **`build_registry(self) -> ModelRegistry`** — boot path. Used by both the
    worker (via `worker_process_init`) and the CLI.
  - **`assemble_engine(self, registry, job_params) -> ClipScribeEngine`** —
    per-job path. Used by both the Celery task and the CLI.

#### `src/clip_scribe/engine.py`
- Accept `reporter: ProgressReporter` as a constructor arg.
- Wrap the existing `run()` body so it emits `job.started` /
  `job.completed` / `job.failed` and propagates `reporter` into both the
  extractor and parser.

### Worker integration

```python
# app/celery_app.py
from celery import Celery
from celery.signals import worker_process_init
from src.clip_scribe.build_clip_scribe import ClipScribeBuilder

celery_app = Celery("clipscribe", broker=settings.redis_url)
REGISTRY = None  # set in worker_process_init

@worker_process_init.connect
def boot_registry(**_):
    global REGISTRY
    REGISTRY = ClipScribeBuilder().build_registry()
```

```python
# app/tasks.py
@celery_app.task(bind=True)
def run_clip_scribe(self, job_params_json: dict):
    params = JobParams.model_validate(job_params_json)
    reporter = RedisProgressReporter(params.job_id, REGISTRY.redis)
    engine = ClipScribeBuilder().assemble_engine(REGISTRY, params, reporter)
    engine.run(run_id=params.run_id)
```

The worker boots once (slow: ~30–60s), then services jobs in a tight loop
where each job costs only the `assemble_engine` call (<100 ms) plus the actual
extraction work.

### CLI compatibility

`main.py` (the existing entry point at line 65) collapses to:

```python
builder = ClipScribeBuilder()
registry = builder.build_registry()
job_params = JobParams(...)   # populated from the hardcoded values currently in main.py
engine = builder.assemble_engine(registry, job_params, NullProgressReporter())
engine.run(run_id="")
```

Same wall-clock as today (models were going to load anyway), but the code path
is now identical to the worker — guaranteeing parity between CLI runs and
API-driven runs.

---

## 4. Database changes

### Existing (`src/db/schema.py`) — keep
- `runs`, `global_stats`, `visual_object_occurrences`, `text_events`,
  `audio_segments`, `scene_descriptions`, `field_descriptions`.

### New tables

**`jobs`** — orchestration state, separate from `runs` (which is extractor
output).
```
job_id          TEXT PK         # ULID, also exposed in API URLs
run_id          TEXT FK runs    # populated when extractor writes the run
status          TEXT            # queued | running | completed | failed | canceled
celery_task_id  TEXT
mode            TEXT            # full | extract | parse
video_name      TEXT
video_path      TEXT
video_type      TEXT
device          TEXT
platform        TEXT            # youtube | ...
params_json     JSONB           # full request payload for reproducibility
error_text      TEXT
created_at      TIMESTAMP
started_at      TIMESTAMP
finished_at     TIMESTAMP
created_by      TEXT            # nullable until auth lands
```

**`frame_detections`** — raw boxes for the UI overlay.
```
id                INTEGER PK
run_id            TEXT
shot_index        INTEGER
frame_idx         INTEGER
timestamp_sec     FLOAT
source            TEXT          # dino | ocr | mtcnn | sam_mask
label             TEXT          # resolved taxonomy label or None
text              TEXT          # OCR text or None
box_x1,y1,x2,y2   FLOAT
confidence        FLOAT
object_id         INTEGER       # local SAM2 id, joinable to global_id
INDEX (run_id, frame_idx)
INDEX (run_id, object_id)
```

Rough volume estimate: a 15-second car ad with detections every 10 frames at
30 fps = ~45 frames × ~10 boxes each ≈ 450 rows. Negligible.

**`parser_results`** — what's currently in `abcd_report.csv`. Lets the UI render
the report without parsing CSV.
```
id              INTEGER PK
run_id          TEXT
platform        TEXT
feature_category TEXT
feature_name    TEXT
feature_criteria TEXT
evaluation      BOOLEAN
llm_prompt      TEXT
llm_explanation TEXT
langsmith_run_id TEXT          # link to trace
created_at      TIMESTAMP
```

### Widened: shot boundaries

`global_stats` currently encodes pacing/dynamic-start info but does **not**
persist per-shot boundaries (only `qp_intro_shots`). The timeline view needs
them. Add:

**`shot_boundaries`**
```
id            INTEGER PK
run_id        TEXT
shot_index    INTEGER
start_sec     FLOAT
end_sec       FLOAT
duration_sec  FLOAT
```

Hook: `extractor_core.py:850` already builds `shot_data` — write it.

### Migrations

- Adopt Alembic. SQLite + Postgres both supported via SQLAlchemy already.
- First migration is the new tables only; existing tables are untouched.

### Default backend

Default to Postgres in the web-app deployment; keep SQLite working for local
single-machine dev. `docker-compose.yml` already has Postgres scaffolding.

---

## 5. Event vocabulary (worker → UI live updates)

Worker publishes to two Redis pub/sub channels per job:
- `job:{id}:events` — structured JSON events (the vocabulary below)
- `job:{id}:logs` — raw `logging.Handler` passthrough (level, msg, ts)

FastAPI's SSE endpoint subscribes to both and multiplexes into one stream.

### Event types

```jsonc
{"type":"job.started",       "ts":..., "job_id":"...", "video_name":"...", "phases":["scene_detection","audio","shot_processing","finalize"]}

{"type":"phase.started",     "phase":"scene_detection"}
{"type":"phase.completed",   "phase":"scene_detection", "data":{"total_shots":12, "video_duration":15.3}}

{"type":"phase.started",     "phase":"audio"}
{"type":"audio.segment",     "data":{"start":0.0,"end":3.84,"text":"..."}}
{"type":"phase.completed",   "phase":"audio", "data":{"segments_kept":4}}

{"type":"phase.started",     "phase":"shot_processing", "data":{"total_shots":12}}
{"type":"shot.started",      "data":{"shot_idx":0,"start":0.0,"end":1.58}}
{"type":"shot.scene_described","data":{"shot_idx":0,"description":"...","dino_prompt":"..."}}
{"type":"shot.taxonomy_resolved","data":{"shot_idx":0,"targets":["car","logo",...]}}
{"type":"shot.frame_processed","data":{"shot_idx":0,"frame_idx":9,"detections":7,"ocr_lines":2,"faces":0}}
{"type":"shot.completed",    "data":{"shot_idx":0,"objects_tracked":3}}
{"type":"phase.completed",   "phase":"shot_processing"}

{"type":"phase.started",     "phase":"finalize"}
{"type":"identity.merged",   "data":{"from_ids":[2,7],"to_global_id":1,"similarity":0.91}}
{"type":"phase.completed",   "phase":"finalize"}

{"type":"parser.criterion_started","data":{"feature_name":"Brand Mention (Speech)"}}
{"type":"parser.criterion_completed","data":{"feature_name":"...","evaluation":true}}

{"type":"job.completed",     "ts":..., "run_id":"..."}
{"type":"job.failed",        "ts":..., "error":"...","phase":"shot_processing"}
```

### Publish sites in the engine

| Event | Hook |
|---|---|
| `job.started` / `job.failed` / `job.completed` | `ClipScribeEngine.run` outer try/except |
| `phase.started/completed (scene_detection)` | around `_digest_video` (`extractor_core.py:999`) |
| `phase.started (audio)` + `audio.segment` + `phase.completed` | `_analyze_audio` segment loop (`extractor_core.py:966`) |
| `phase.started (shot_processing)` | top of shot loop (`extractor_core.py:1004`) |
| `shot.started` | start of each iteration |
| `shot.scene_described` | after `describe_scene` (`extractor_core.py:1054`) |
| `shot.taxonomy_resolved` | after `set_active_targets` (`extractor_core.py:1071`) |
| `shot.frame_processed` | bottom of inner frame loop (after `extractor_core.py:1207`) |
| `shot.completed` | end of shot iteration |
| `identity.merged` | inside `_resolve_identities` when `should_merge=True` (`extractor_core.py:708`) |
| `parser.criterion_*` | `src/parser/parser_core.py` per-criterion run |

### Implementation

A `ProgressReporter` interface injected into the extractor and parser. Two
implementations:
- `RedisProgressReporter(job_id, redis_client)` — used in worker.
- `NullProgressReporter()` — used in CLI and tests.

`extractor_core.py` only depends on the interface, not Redis. Same applies to
the parser.

A custom `logging.Handler` is attached at task start that mirrors `INFO+` log
records to `job:{id}:logs`. The handler reads job id from a contextvar so no
function signatures change.

---

## 6. API surface

OpenAPI-generated; TS client codegen via `openapi-typescript`. All routes
return Pydantic-validated JSON. Errors: RFC7807 (`type`/`title`/`detail`).

### Jobs
- `POST   /jobs`                       — create + enqueue. Request body mirrors `main.py` params (see §7).
- `GET    /jobs`                       — paginated, filterable by status.
- `GET    /jobs/{id}`                  — full state.
- `GET    /jobs/{id}/events`           — SSE; multiplexes `events` + `logs`.
- `POST   /jobs/{id}/cancel`           — cooperative cancel (see §10).

### Runs (read-only views of extractor + parser output)
- `GET /runs/{id}`                     — `runs` row + summary.
- `GET /runs/{id}/global-stats`        — `global_stats` + `shot_boundaries`.
- `GET /runs/{id}/objects`             — `visual_object_occurrences` grouped by `global_id`.
- `GET /runs/{id}/text-events`         — `text_events`.
- `GET /runs/{id}/audio-segments`      — `audio_segments`.
- `GET /runs/{id}/scenes`              — `scene_descriptions`.
- `GET /runs/{id}/frames?from=X&to=Y`  — `frame_detections` in time window.
- `GET /runs/{id}/parser`              — `parser_results`.

### Artifacts (filesystem-backed)
- `GET /runs/{id}/video`               — original input, `Range`-aware.
- `GET /runs/{id}/tracked-video`       — `tracked_output.mp4`, `Range`-aware.
- `GET /runs/{id}/png/{filename}`      — DINO/OCR/face viz PNGs (fallback only).

### Health
- `GET /healthz`                       — liveness.
- `GET /readyz`                        — DB + Redis reachability.

### Metadata
- `GET /platforms`                     — list, with required params.
- `GET /defaults`                      — current yaml config exposed (read-only).

---

## 7. Frontend (SPA)

### Stack
- Vite + React + TypeScript (strict mode).
- TanStack Router (file-based; cleaner than React Router for app-shell apps).
- TanStack Query for REST data + `EventSource` for SSE.
- Zustand (or just `useReducer`) for the per-job live state.
- Tailwind + shadcn/ui for components.
- visx or d3-scale for the timeline (don't bring full d3 unless needed).
- Type-safe API client from OpenAPI (`openapi-typescript` + `openapi-fetch`).

### Pages

1. **Jobs list** (`/`)
   - Table of jobs (status, video_name, created_at, duration, ABCD pass rate
     when available).
   - Filters: status, platform, date range, brand.
   - "New job" button.

2. **New job** (`/jobs/new`)
   - Form mirroring `main.py:26–57` (`platform_params`, `user_hints`,
     `video_type`, `device`, `mode`, `platform`).
   - Video field: upload OR pick from server-side `input/` directory (see
     Open question §8).
   - Defaults pre-populated from `GET /defaults` so the form shows the yaml
     values and the user only overrides what they care about.
   - Submit → `POST /jobs` → redirect to `/jobs/{id}`.

3. **Live job** (`/jobs/{id}`)
   - Layout from prior chat sketch:
     - Top: progress bar + estimated time + "Cancel" button.
     - Left: phase tree (scene detection ✓, audio ✓, shots N/M, finalize, parse).
     - Right: current-shot panel (description, dino prompt, taxonomy
       targets, frames processed).
     - Bottom: live log tail (ring buffer ~500 lines, level filter).
   - State driven by SSE; reducer keyed off `event.type`.
   - On `job.completed`, auto-redirect (or show CTA) to `/runs/{id}`.

4. **Run inspector** (`/runs/{id}`)
   - Top: video player with SVG overlay (see §8).
   - Right rail: layer toggles (DINO / OCR / faces / SAM bbox), confidence
     slider, "active detections at t=..." list.
   - Center-bottom: stacked timeline tracks (shots, audio, per-object lifespans,
     OCR seconds).
   - Bottom: ABCD criteria table from `parser_results`, each row expandable to
     show `llm_prompt` + `llm_explanation` + LangSmith trace link.
   - Download menu: tracked_output.mp4, abcd_report.csv,
     extraction_summary.json.

### Live progress state shape

```ts
type JobProgress = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'canceled';
  startedAt?: number;
  finishedAt?: number;
  phases: Record<PhaseName, PhaseState & Partial<PhaseExtra>>;
  audioSegments: AudioSegment[];
  identityMerges: IdentityMerge[];
  currentShot?: { idx: number; description?: string; dinoPrompt?: string; targets?: string[]; framesDone: number };
  logTail: LogLine[];
};
```

Derived progress: `0.05 * scenePct + 0.15 * audioPct + 0.70 * shotPct + 0.10 *
finalizePct` — weights based on observed wall-clock distribution; tune later.

### Inspector overlay (SVG on `<video>`)

`useFramesForRun(runId)` pulls all `frame_detections` once on mount (small) and
caches. On `timeupdate`, find the most recent frame ≤ current playback time and
render its boxes as SVG over the video. See chat for the reference component
sketch.

---

## 8. Things we have not nailed down yet (open questions)

These are decisions to make before the corresponding implementation step.

1. **Video ingest.**
   How does a video get from the user to the worker?
   - (a) Multipart upload to API → API writes to a shared volume → enqueues
     job with the path.
   - (b) Pre-signed S3-style upload → worker pulls from object storage.
   - (c) Server-side picker over an existing `input/` directory (closest to
     today; simplest for internal use).
   - Probably (c) first, (a) when multi-user. (b) only if deploying to cloud.

2. **Artifact storage.**
   `extractor_artifacts/<video_name>/` is currently keyed by video filename. If
   two jobs share a video name they collide. Switch to
   `artifacts/<run_id>/` once we have ULIDs. Decide: filesystem (shared volume)
   vs object storage. Object storage is more cloud-friendly but adds
   `boto3`/`minio-py` dependency.

3. **Authentication / multi-user.**
   None today. Likely deferred to post-MVP. If we add it, keep `created_by` on
   `jobs` and gate everything on a session.

4. **Job cancellation semantics.**
   Celery `revoke(terminate=True)` is hard-kill (SIGTERM). The engine holds
   open files and a CUDA/MPS context — abrupt termination leaks both. Need
   cooperative cancellation: a "should_cancel" flag the shot loop checks each
   iteration. Decide if we want partial results saved on cancel.

5. **Resumability after worker crash.**
   Probably out of scope. Worth deciding because today the extractor writes the
   `tracked_output.mp4` incrementally — a crash leaves a half-written file.
   Workaround: write to `.partial` and rename atomically at the end.

6. **Postgres or SQLite for the web deployment.**
   Recommend Postgres for the API since SQLite has poor concurrent-write
   behavior and we now have a writer worker + reader API hitting it. Local CLI
   keeps SQLite for convenience.

7. **MPS in Docker on Mac.**
   Doesn't work — MPS is not available inside Linux containers. Dev on Mac:
   run worker natively, dockerize only API + Redis + Postgres + frontend.
   Prod on Linux+NVIDIA: full docker-compose with `--gpus all`. Document both
   in the README.

8. **Cost & telemetry.**
   Each job calls OpenAI for hint generation, target generation, scene
   description (per shot), and parser agents (~30 criteria). No tracking
   today. Worth capturing per-job token usage from the OpenAI client and
   surfacing on the run page. LangSmith already traces parser agents — just
   need to persist the run id per criterion (already on the table above).

9. **Concurrency story for multiple jobs.**
   Worker `--concurrency=1` per machine; queue handles backlog. Decide: does
   the UI let users submit a 2nd job while one is running? It can — Redis
   queues it. Show queue position on the live page.

10. **Logging refactor.**
    `src/utils/clip_scribe_logging.py` is a singleton. With a worker pool +
    per-job event streams we need a contextvar with `job_id` propagated into
    every log record so the SSE handler can route correctly. No call-site
    changes if done via a `logging.Filter` that reads the contextvar.

11. **Tests.**
    Test suite is "minimal" per CLAUDE.md. New code should land with tests:
    Pydantic model round-trips, `ProgressReporter` event ordering, `frame_detections`
    population, SSE multiplexer. Don't try to test the engine end-to-end —
    that needs models.

12. **OpenAPI → TS codegen as CI step.**
    Generate `frontend/src/api/types.ts` from the FastAPI schema on every API
    change. Either pre-commit hook or a `make codegen` step. Avoid drift
    between Python and TS types.

13. **CORS / dev proxy.**
    Vite dev server on 5173, FastAPI on 8000. Use Vite proxy in dev so the
    frontend can call `/api/*` without CORS gymnastics. In prod, nginx in
    front does the same thing.

14. **Disk retention.**
    `extractor_artifacts/` grows unboundedly. Decide a retention policy
    (delete after N days, keep last K runs, manual purge) before the disk
    fills. Add a `POST /runs/{id}` DELETE endpoint that cleans both DB rows
    and artifact directory.

15. **Per-job artifact directory keying.**
    Use `artifacts/<run_id>/` not `artifacts/<video_name>/` once `run_id` is a
    ULID. Migrate existing data by renaming or just leave old runs alone.

16. **What does the SSE channel do when there are zero subscribers?**
    Redis pub/sub drops messages with no subscribers. If a user opens the live
    page after a job is already running, they miss the early events. Two
    options: (a) also write events to a Redis stream (XADD) keyed by job_id
    so the SSE handler can replay history on connect; (b) keep the last N
    events in a per-job list in Postgres `jobs.events_json`. (a) is cleaner.

17. **`tracked_output.mp4` vs raw input video for the player.**
    Use the raw input as the `<video>` source and overlay our own SVG boxes —
    full control, supports layer toggles. The baked tracked mp4 stays as a
    download.

---

## 9. Sequencing — what to build first

Strictly ordered; each step is shippable on its own.

1. **DB migrations: `jobs`, `frame_detections`, `parser_results`,
   `shot_boundaries`.** Alembic baseline + first migration. No behavior
   change yet; just schema.

2. **Builder refactor: `ModelRegistry` + `JobAssembler`.** CLI still works.
   Models load once. No FastAPI yet. This is the load-bearing refactor.

3. **`ProgressReporter` interface + Null impl.** Wire publish calls into
   `extractor_core.py` and `parser_core.py`. CLI passes Null; nothing changes
   for the user. Unit test the event ordering.

4. **Persist raw detections.** Hook `frame_detections` writes inside
   `_save_metadata` and OCR/MTCNN branches. Persist `shot_boundaries` from
   `_digest_video`. Persist parser results from the parser. CLI is the proving
   ground.

5. **FastAPI app, sync path only.** `POST /jobs` runs the job inline (no
   Celery yet) so we can shake out request shape, error handling, OpenAPI
   schema, and CORS. Implement read-only `/runs/*` and artifact endpoints.

6. **Frontend bootstrap.** Vite + React + TS + Tailwind + TanStack Router /
   Query. Pages: Jobs list, New job, Run inspector (against existing DB data).
   No live progress yet — just submit + poll completed status.

7. **Inspector overlay.** Use `frame_detections` to draw SVG boxes on
   `<video>`. Layer toggles, confidence filter, timeline tracks. This is
   where the project starts to feel real.

8. **Celery + Redis.** Move `POST /jobs` to enqueue. `worker_process_init`
   builds `ModelRegistry`. One worker, concurrency 1. `jobs` table tracks
   status transitions.

9. **Redis pub/sub bridge + SSE.** Swap `NullProgressReporter` for
   `RedisProgressReporter` in the worker. FastAPI multiplexes
   `job:{id}:events` + `job:{id}:logs` into the SSE response. Frontend live
   page renders from the reducer.

10. **Cooperative cancel.** "should_cancel" flag honored by the shot loop +
    parser. `POST /jobs/{id}/cancel`. Partial result handling.

11. **Docker split.** `Dockerfile.api` (slim) + `Dockerfile.worker` (heavy).
    `docker-compose.yml` for the full stack. Document Mac-MPS caveat.

12. **Polish:** retention policy, auth (if/when needed), cost tracking,
    OpenAPI codegen in CI, expand tests.

---

## 10. Risks / things that could derail this

- **Builder refactor is bigger than it looks.** Hints get passed deep into
  the extractor and into the GPT taxonomy generator. Untangling these so the
  registry is truly per-process will touch `taxonomy_core.py`,
  `extractor_core.py`, and the builder.

- **MPS in Docker on Mac.** Already flagged. Lots of dev pain if we forget.

- **Redis pub/sub is lossy without subscribers.** Must use streams or
  Postgres-backed replay for late subscribers (open question §16).

- **OpenAI cost.** Adding a "Run job" button in a web UI makes it easy to
  burn through credits. Cost cap or per-user budget should exist before
  this is anything but internal.

- **Disk usage.** Unbounded `extractor_artifacts/`. Retention story needs to
  exist before turning the app on for more than one person.

- **Cancellation correctness.** Hard-killing a worker mid-job leaves OpenCV
  file handles, partial mp4s, and possibly a corrupted SAM2 inference state
  on the GPU. Cooperative cancel is the only safe path; it requires touching
  the shot loop.

- **Type drift Python ↔ TS.** Without OpenAPI codegen wired into CI, the
  shapes will drift the moment a Pydantic field is added.
