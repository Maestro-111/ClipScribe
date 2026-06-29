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

**Status: DONE.** Layout below reflects the current state on disk.

```
clipscribe/
  backend/                        # Python project
    pyproject.toml                # root of the uv project; readme line removed
    uv.lock
    main.py                       # CLI entry — instantiates ClipScribeBuilder and runs one job
    src/                          # existing core, paths unchanged
      clip_scribe/  extractor/  ocr/  parser/  db/  dino/  sam2/  utils/
    app/                          # NEW — web layer (currently just __init__.py)
      celery_app.py               # SHARED — Celery app + worker_process_init signal
      tasks.py                    # WORKER-ONLY — imports ClipScribeBuilder (heavy)
      main.py                     # API-ONLY — FastAPI app + lifespan
      routes/                     # API-ONLY — jobs.py, runs.py, artifacts.py, health.py
      models.py                   # SHARED — Pydantic request/response schemas
      events.py                   # ProgressReporter + Redis pub/sub helpers
    docker/
      api/
        Dockerfile                # slim API image
        deploy.sh
      core/
        Dockerfile                # heavy worker image (a.k.a. "scribe core")
        deploy.sh
    checkpoints/                  # all model weights live here (see §8)
    data/                         # SQLite db lives here
    input/                        # video inputs for CLI / picker
    test/
  frontend/                       # TS SPA
    src/
      api/                        # generated TS client from OpenAPI
      lib/                        # state, hooks, utils
      pages/                      # JobsList / NewJob / JobLive / RunInspector
      components/                 # VideoOverlay, Timeline, PhaseTree, LogTail
    package.json
    vite.config.ts
  docker-compose.yml              # dev stack: api + worker + redis + postgres
  Makefile
  docs/web-app-plan.md            # this file
```

Notes on the layout as it stands:

- `from src.x import y` imports still work because `pyproject.toml` lives at
  `backend/` and declares `packages = ["src"]`. After `uv sync` from `backend/`,
  `src` is installed as a top-level package.
- `PROJECT_ROOT = Path(__file__).resolve().parents[2]` inside
  `build_clip_scribe.py` resolves to `backend/`, which is where `data/`,
  `input/`, `checkpoints/` now live — so relative-path resolution works
  without code changes.
- Pre-commit must be invoked from `backend/` because the config lives at
  `backend/.pre-commit-config.yaml`. See root `CLAUDE.md` § Commands.
- The root `Makefile` is still partially broken (it references paths that
  moved). Either move it under `backend/` or update each target to
  `cd backend && ...`. Tracked as a cleanup, not a blocker.

---

## 3. Builder refactor (load-once vs per-job)

**Status: DONE.** Implemented in
`backend/src/clip_scribe/build_clip_scribe.py`. The shape that landed is
simpler than the original `ModelRegistry` / `JobAssembler` proposal — a
single `ClipScribeBuilder` class whose `__init__` loads everything heavy,
and whose existing `build_clip_scribe(...)` method becomes the cheap
per-job entry point.

### Why the refactor matters

Without it, every job re-loads SAM2 + DINOv2 + Whisper + DINO + MTCNN +
PaddleOCR + SBERT (~30–60s). A Celery worker is a long-lived process that
should pay that cost once at boot and amortize it across every job it ever
handles. The refactor is what makes that possible.

### Process model recap

The API process and the worker process are **separate**. The API never
imports torch and never loads a model — it only validates requests, reads
the DB, and sends Celery tasks into Redis by name (see §8 on the import
boundary trick). The worker imports everything heavy. Workers can run on
the same machine as the API or on remote machines (e.g. a GPU box); they
only need network reach to Redis (broker + pubsub) and Postgres. Routing
is automatic via the Celery queue.

### What actually changed

`ClipScribeBuilder.__init__` now calls two private setup methods:

- `_assemble_db()` — builds the SQLAlchemy engine + reader + writer; stored
  on `self.writer_db` / `self.reader_db`.
- `_assemble_heavy_extractor_utils()` — loads every heavy model and stores
  it on `self` (`self.dino`, `self.sam2`, `self.ocr`, `self.reid_model`,
  `self.audio_model`, `self.embedding_transform`, `self.face_detection`,
  `self.taxonomy_resolver`, `self.taxonomy_generator`).

`build_clip_scribe(...)` is now cheap: it consults `self.*` for the heavy
deps, calls the OpenAI hint generator only if
`generate_hint_from_name=True`, constructs a fresh
`VideoInformationExtractor` (cheap — it's just storing pointers to the
shared models) plus a fresh `VideoInformationParser`, and wraps both in a
`ClipScribeEngine`.

Key consequences:

- **No `setup_for_job` method on the extractor was needed.** Because
  `VideoInformationExtractor` is instantiated fresh per job, all per-run
  state (`active_trackers`, `text_registry`, `object_registry`,
  `audio_registry`, `scene_description_registry`, `obj_id_counter`,
  `current_frame`) starts empty automatically. The heavy GPU-resident
  models are passed by reference into the new instance, so no copies.
- **No new `ModelRegistry` class.** The builder itself plays that role.
- **`taxonomy_user_hints` is still a constructor arg of
  `VideoInformationExtractor`** — that's fine because the extractor is
  recreated per job.
- **CLI is unchanged.** `main.py` still does
  `builder = ClipScribeBuilder()` then
  `builder.build_clip_scribe(...).run(run_id=...)` — same two lines as
  before. Wall-clock for a single video is identical to pre-refactor; the
  win shows up the moment a second job runs in the same process.

### Boot vs per-job, current state

#### Boot once (`ClipScribeBuilder()` → `__init__`)

- Read `clip_scribe.yaml`.
- Resolve `models_weights_dir` and all yaml param dicts.
- `_assemble_db()` → DB engine + reader + writer.
- `_assemble_heavy_extractor_utils()` →
  GroundingDINO, SAM2, PaddleOCR, DINOv2 (reid), Whisper,
  MTCNN, `embedding_transform`, `ProfilesPile`, `TaxonomyResolver` (SBERT),
  `TaxonomyGenerator`.

#### Per job (`build_clip_scribe(...)`)

- `video_name`, `video_path`, `video_type`, `mode`, `device`,
  `platform_name`, `platform_conf`, `user_hints`, `generate_hint_from_name`
  arrive as call args.
- `combined_hints` derivation (only OpenAI-roundtrip if
  `generate_hint_from_name=True`).
- Fresh `GPTSceneDescriber` (cheap — OpenAI client wrapper, not a model).
- Fresh `VideoInformationExtractor` wrapping `self.dino`, `self.sam2`, …
- Fresh `VideoInformationParser` bound to the per-job `platform_conf`.
- `ClipScribeEngine(...)` wrapper around extractor + parser + `self.*_db`.

Target cost: a few hundred ms (dominated by the optional hint-generation
OpenAI call when enabled), versus 30–60s pre-refactor.

### Worker integration (the shape that will land in §10)

The actual Celery wiring is unchanged from the planned approach — it's
just thinner now because there's no `ModelRegistry` / `assemble_engine`
plumbing in the middle:

```python
# backend/app/celery_app.py
from celery import Celery
import os
celery_app = Celery("clipscribe", broker=os.environ["REDIS_URL"])
```

```python
# backend/app/tasks.py
from celery.signals import worker_process_init
from app.celery_app import celery_app
from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
from src.clip_scribe.build_clip_scribe_plalform import build_platform

BUILDER = None

@worker_process_init.connect
def boot(**_):
    global BUILDER
    BUILDER = ClipScribeBuilder()   # 30–60s, ONCE per worker process

@celery_app.task(name="app.tasks.run_job")
def run_job(job_params: dict):
    platform_conf = build_platform(
        job_params["platform_name"], **job_params["platform_params"]
    )
    engine = BUILDER.build_clip_scribe(
        video_name=job_params["video_name"],
        video_path=job_params["video_path"],
        video_type=job_params["video_type"],
        clib_scribe_mode=job_params["mode"],
        clib_scribe_device=job_params["device"],
        clib_scribe_platform_name=job_params["platform_name"],
        clib_scribe_platform_conf=platform_conf,
        user_hints=job_params.get("user_hints"),
        generate_hint_from_name=job_params.get("generate_hint_from_name", False),
    )
    engine.run(run_id=job_params.get("run_id", ""))
```

### What still needs to happen in this file later (tracked elsewhere)

- Inject a `ProgressReporter` into the engine + extractor + parser
  (§5; needed for live UI updates).
- Pass `download_root` to `whisper.load_model` and the equivalent dirs to
  `OCRSystem` so all weight downloads land under
  `backend/checkpoints/` instead of `~/.cache/...` (§8).
- Optional cleanup: move `hint_generation_model` and `scene_detection_model`
  string resolution into `__init__` for consistency with
  `target_generation_model`. Cost-neutral.

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
     Open question §9).
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

## 8. Docker & checkpoint strategy

Two images, two roles, one shared volume for weights. Both Dockerfiles
live under `backend/docker/`:

```
backend/docker/
  api/
    Dockerfile          # slim — fastapi/pydantic/redis/sqlalchemy only
    deploy.sh
  core/
    Dockerfile          # heavy — full torch + CV stack + weights story
    deploy.sh
```

### Image responsibilities

| | API image (`docker/api/`) | Core image (`docker/core/`) |
|---|---|---|
| Role | FastAPI; never executes ML | Celery worker; runs the pipeline |
| Heavy deps | none (no `torch`, no `whisper`, no `paddleocr`) | full stack |
| Imports `src/clip_scribe/build_clip_scribe.py`? | **no** | yes |
| Imports `app/tasks.py`? | **no** — sends tasks by name | yes |
| Talks to Redis | yes (broker client + pubsub subscriber) | yes (broker consumer + pubsub publisher) |
| Talks to Postgres | yes (read-mostly + jobs writes) | yes (writer + reader) |
| Mounts `backend/checkpoints/`? | no | yes in dev, baked in prod |
| Approx size | ~300 MB | 8–15 GB |

### The import-boundary trick (avoids dragging torch into the API)

To enqueue without importing the task:

```python
# backend/app/routes/jobs.py — API only
from app.celery_app import celery_app          # lightweight (just Celery)
celery_app.send_task("app.tasks.run_job", args=[params.model_dump()])
```

`backend/app/celery_app.py` is small and lives in both images.
`backend/app/tasks.py` contains the heavy imports (`ClipScribeBuilder`,
torch transitively) and is **only ever imported by the worker**, never by
the API. The contract between them is the string task name.

### Checkpoint / weight strategy

All model weights live under `backend/checkpoints/`, organized by source:

```
backend/checkpoints/
  dino/         GroundingDINO .pth   (explicit path; loaded today)
  sam2/         SAM2 .pt             (explicit path; loaded today)
  torch_hub/    ← TORCH_HOME         (DINOv2 + MTCNN auto-download)
  huggingface/  ← HF_HOME            (SBERT inside TaxonomyResolver)
  whisper/      ← download_root arg  (Whisper auto-download)
  paddleocr/    ← model_dir arg      (PaddleOCR auto-download)
  nltk/         ← NLTK_DATA          (WordNet)
```

Env vars set once at process start (`.env` in dev, Dockerfile `ENV` block
in prod) redirect every auto-downloader into a subdir of
`backend/checkpoints/`:

```bash
TORCH_HOME=$REPO/backend/checkpoints/torch_hub
HF_HOME=$REPO/backend/checkpoints/huggingface
NLTK_DATA=$REPO/backend/checkpoints/nltk
```

The builder reads them implicitly — the underlying libraries
(`torch.hub.load`, `sentence-transformers`, `facenet_pytorch`, `nltk`)
honor those env vars and write into the right place.

Two libraries don't honor env vars and need explicit args from the
builder:

- **Whisper**:
  `whisper.load_model("base", device=..., download_root=str(self.models_weights_dir / "whisper"))`
- **PaddleOCR**: pass `det_model_dir` / `rec_model_dir` / `cls_model_dir`
  through `OCRSystem` to the PaddleOCR constructor.

A `backend/scripts/prewarm.py` one-liner triggers every download by
constructing the builder:

```python
# backend/scripts/prewarm.py
from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
ClipScribeBuilder()
print("prewarm complete")
```

### Dev image — mount strategy

`backend/docker/core/Dockerfile` (dev variant) is slim: install deps, copy
code, no weight downloads. The `docker-compose.yml` does the work:

```yaml
worker:
  build:
    context: .
    dockerfile: backend/docker/core/Dockerfile
  env_file: backend/.env                       # TORCH_HOME, HF_HOME, NLTK_DATA
  volumes:
    - ./backend/checkpoints:/app/backend/checkpoints   # persist weights
    - ./backend/src:/app/backend/src                   # hot-reload code
    - ./backend/app:/app/backend/app
```

- First worker boot ever to use the volume: downloads ~5–10 GB (~5 min).
- Every subsequent boot: instant, cache is reused.
- Optional manual prewarm to avoid lazy first-job downloads:
  `docker compose run --rm worker uv run python scripts/prewarm.py`.

### Prod image — bake strategy

A separate Dockerfile path (could be a multi-stage with build args, or a
distinct `Dockerfile.prod`; choice deferred) sets the env vars at build
time and runs `prewarm.py` as a `RUN` step so weights are baked into an
image layer:

```dockerfile
ENV TORCH_HOME=/app/backend/checkpoints/torch_hub \
    HF_HOME=/app/backend/checkpoints/huggingface \
    NLTK_DATA=/app/backend/checkpoints/nltk
RUN cd /app/backend && uv run python scripts/prewarm.py
```

No volume mount needed at runtime. Image is 8–15 GB and starts instantly.
Good for prod where images are pushed rarely.

### MPS in Docker on Mac — important caveat

MPS is not available inside Linux containers. Two consequences:

- **Mac dev**: run the worker natively (so `device=mps` works), and put
  the rest (API + Redis + Postgres + frontend) in `docker-compose`.
  Document this in the README.
- **Prod on Linux + NVIDIA**: everything in Docker, worker container gets
  `--gpus all`. No special handling beyond that.

The same `core` image *can* run on Mac if you accept CPU fallback — useful
for verifying the worker integration end-to-end, but ~10× slower than
native MPS. Treat it as smoke-test only.

### API image — always slim

The `docker/api/Dockerfile` only installs the lean dependency set,
copies `backend/app/`, `backend/src/db/`, and
`backend/src/clip_scribe/platform_configs/`. It never copies the rest of
`src/` and never installs torch. Rebuilds in seconds.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock /app/backend/
COPY backend/app /app/backend/app
COPY backend/src/db /app/backend/src/db
COPY backend/src/clip_scribe/platform_configs /app/backend/src/clip_scribe/platform_configs
RUN cd /app/backend && uv sync --no-dev   # consider an `api` extra to skip torch entirely
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

To keep the API image truly slim, the eventual `pyproject.toml`
should split `[project.optional-dependencies]` into `api` and `worker`
extras so `uv sync --no-dev --extra api` skips torch / whisper /
paddleocr entirely.

---

## 9. Things we have not nailed down yet (open questions)

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

7. **MPS in Docker on Mac.** Resolved — see §8 "Docker & checkpoint
   strategy". Dev on Mac runs the worker natively; everything else can be
   dockerized.

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

## 10. Sequencing — what to build first

Strictly ordered; each step is shippable on its own.

1. **DB migrations: `jobs`, `frame_detections`, `parser_results`,
   `shot_boundaries`.** Alembic baseline + first migration. No behavior
   change yet; just schema.

2. **Builder refactor: load-once via `ClipScribeBuilder.__init__`.**
   **DONE.** Heavy assembly moved into `__init__` (`_assemble_db`,
   `_assemble_heavy_extractor_utils`). `build_clip_scribe(...)` is now
   the cheap per-job entry point. CLI works unchanged. See §3.

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

## 11. Risks / things that could derail this

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
