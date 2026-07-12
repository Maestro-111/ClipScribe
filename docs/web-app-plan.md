# ClipScribe Web App — Migration Plan

A planning doc, not a spec. It maps the current CLI-driven pipeline onto a
two-tier web app: a TypeScript single-page dashboard and a Python backend split
into a thin FastAPI process and one or more Celery workers.

ClipScribe is intended to run in **two operating modes** that share the same
core (`ClipScribeBuilder` + `ClipScribeEngine`):

1. **Web app** (this doc's main subject) — interactive dashboard with live
   progress. Run locally via `docker-compose` for development; a GCP shape is
   sketched in §12 but not a near-term build target.
2. **CLI / batch** — no UI, no realtime. Runs the core over one or many videos
   and persists results. Locally: process a list sequentially. Remotely: fan
   out over a Kubernetes **Indexed Job**, one video per GPU pod. See §12.

Both modes reuse the same "core" worker image with different entrypoints, so
nothing built for one blocks the other.

This file is a living checklist. Sections marked **Open question** are decisions
to make before that piece is built.

Current checked-in state: the backend relocation, Alembic migrations,
load-once builder, progress seam, raw detection persistence, the FastAPI app,
**Celery/Redis job dispatch**, and the Docker split have landed. `POST /jobs` runs either
in-process (single-slot executor) or via a Redis-backed Celery worker, selected
by `CLIPSCRIBE_JOB_BACKEND`; the API exposes uploads, input listing, job
polling/progress, read-only run views, advisory chat, artifact serving, health,
and metadata endpoints.
**SSE live progress has landed** — a per-job Redis stream feeds `GET
/jobs/{id}/events`, the frontend live page, and jobs-list progress bars.
Cooperative mid-run interruption is still planned.

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
                       │  - SSE stream bridge      │
                       │  - enqueue Celery tasks   │
                       └─────┬────────────┬────────┘
                             │            │
                             ▼            ▼
                       ┌─────────┐  ┌──────────────┐
                       │  Redis  │  │  Postgres    │
                       │ broker  │  │  schema/runs │
                       │ +streams │  │  + jobs +    │
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
- **Redis**: both the Celery broker/result backend and the live-progress Streams store.
- **Postgres**: existing schema (`backend/src/db/schema.py`) plus web-app
  tables (`jobs`, `frame_detections`, `parser_results`, `shot_boundaries`,
  `chat_messages`).

---

## 2. Repo restructure (monorepo)

**Status: DONE for the current local/container split.** The Python project,
FastAPI app, frontend bootstrap, Celery/Redis dispatch, SSE progress pieces,
and Docker image split now live in the monorepo.

```
clipscribe/
  backend/                        # Python project
    pyproject.toml                # root of the uv project
    uv.lock
    main.py                       # CLI entry — instantiates ClipScribeBuilder and runs one job
    src/                          # existing core, paths unchanged
      clip_scribe/  extractor/  ocr/  parser/  db/  dino/  sam2/  utils/
    app/                          # FastAPI layer (inline + celery dispatch)
      celery_app.py               # thin shared Celery handle (broker=REDIS_URL, no torch)
      tasks.py                    # WORKER-ONLY — imports ClipScribeBuilder; run_job task
      job_execution.py            # run_job_core — lifecycle shared by inline + celery
      main.py                     # FastAPI app + lifespan; builder inline / DB-only for celery
      job_runner.py               # JobService: validate, persist, dispatch (inline|celery)
      settings.py                 # CLIPSCRIBE_* API env settings (job_backend, redis_url)
      errors.py                   # RFC7807 problem+json handlers
      routes/                     # jobs, runs, artifacts, chat, health, meta, uploads
      models.py                   # Pydantic request/response schemas
      events.py                   # Redis Streams progress helpers + log bridge
    docker/
      api/
        Dockerfile                # slim API image; also runs migrate
        deploy.sh
      core/
        cpu/
          Dockerfile              # heavy CPU worker/prewarm image
          deploy.sh
        gpu/
          Dockerfile              # heavy CUDA worker image; weights baked
          deploy.sh
    checkpoints/                  # all model weights live here (see §8)
    data/                         # SQLite db lives here
    input/                        # video inputs for CLI / picker
    test/
  frontend/                       # TS SPA
    src/
      api/                        # generated TS client + query hooks
      lib/                        # state, formatting, run types
      routes/                     # JobsList / NewJob / JobLive / RunInspector
      components/                 # ChatPanel and reusable UI pieces
    package.json
    vite.config.ts
  docker-compose.yml              # full local stack: postgres, redis, migrate, prewarm, api, worker, frontend
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
- The root `Makefile` delegates into `backend/`: `make migrate` applies
  Alembic migrations, while `make setup`, `make prewarm`, and
  `make checkpoints` fetch or verify model assets under `backend/checkpoints/`.
  The model setup targets are intentionally heavyweight and can download several
  GB.

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

The API process and the worker process are **separate** in celery mode. The API
loads no pipeline models — it validates requests, reads the DB, serves SSE from
Redis Streams, and sends Celery tasks into Redis by name (see §8 on the import
boundary trick). The worker imports everything heavy. Workers can run on the
same machine as the API or on remote machines (e.g. a GPU box); they only need
network reach to Redis (broker + streams) and Postgres. Routing is automatic via
the Celery queue.

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
- **CLI entry remains hardcoded.** `backend/main.py` still instantiates one
  `ClipScribeBuilder`, builds one `ClipScribeEngine`, and runs it; the video,
  mode, platform params, run id, and optional device override are edited in the
  file. Wall-clock for a single video is identical to pre-refactor; the win
  shows up the moment a second job runs in the same process.

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

- `video_name`, `video_path`, `video_type`, `mode`,
  `platform_name`, `platform_conf`, `user_hints`, `generate_hint_from_name`
  arrive as call args.
- Device is no longer a per-job argument. `ClipScribeBuilder(device=...)`
  resolves one process-wide pipeline device at boot: the web API and Celery
  worker pass `CLIPSCRIBE_DEVICE`, while `backend/main.py` passes its hardcoded
  local override or can omit it to fall back to `clip_scribe.device` from yaml.
- `combined_hints` derivation (only OpenAI-roundtrip if
  `generate_hint_from_name=True`).
- Fresh `GPTSceneDescriber` (cheap — OpenAI client wrapper, not a model).
- Fresh `VideoInformationExtractor` wrapping `self.dino`, `self.sam2`, …
- Fresh `VideoInformationParser` bound to the per-job `platform_conf`.
- `ClipScribeEngine(...)` wrapper around extractor + parser + `self.*_db`.

Target cost: a few hundred ms (dominated by the optional hint-generation
OpenAI call when enabled), versus 30–60s pre-refactor.

### Worker integration

The Celery wiring is implemented in `backend/app/celery_app.py`,
`backend/app/tasks.py`, and `backend/app/job_execution.py`. The API dispatches
by task name, the worker loads one long-lived `ClipScribeBuilder` per process
(`worker_process_init` or lazy first task for `--pool=solo`), and both inline
and celery paths call the same `run_job_core(...)` lifecycle.

### Remaining cleanup in this area

- Optional cleanup: move `hint_generation_model` and `scene_detection_model`
  string resolution into `__init__` for consistency with
  `target_generation_model`. Cost-neutral.

---

## 4. Database changes

### Existing (`backend/src/db/schema.py`) — keep
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
device          TEXT            # resolved process device, not a request field
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

**`chat_messages`** — user-facing transcript for the advisory chat (§13). The
agent's working memory lives in the LangGraph checkpointer; this table is what
the UI lists and replays.
```
id               INTEGER PK
run_id           TEXT
session_id       TEXT          # = LangGraph thread_id
role             TEXT          # user | assistant
content          TEXT
tool_calls_json  JSONB         # optional: tools the agent invoked, for UI transparency
created_at       TIMESTAMP
INDEX (run_id, session_id)
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

- Alembic is adopted under `backend/alembic/`. SQLite + Postgres both use the
  same SQLAlchemy metadata from `backend/src/db/schema.py`. `env.py` resolves
  the DB URL via `resolve_database_url()` (config + env), never the static
  `alembic.ini` placeholder — so migrations always target the app's DB.
- Current revisions are a baseline migration for the existing schema, a
  migration adding `jobs`, `frame_detections`, `parser_results`, and
  `shot_boundaries`, and a chat migration adding `chat_messages`.
- Runtime DB setup no longer calls `metadata.create_all`; run
  `uv run alembic upgrade head` from `backend/` (or `make migrate` from the
  repository root). Authoring a new migration: `make revision m="..."` then
  review the generated script (delete it if the diff was empty).
- In deployment, `upgrade head` runs **once per release** as a discrete step
  (a compose one-shot service / K8s Job / deploy command), **not** per worker
  or API replica — all replicas share one DB. It can run from the slim API
  image (it has `src/db` + the scripts, no torch needed).

### run_id is now minted up front

`run_id` is a **ULID** minted by `ClipScribeEngine.run()` at the start of an
extract/full run (or the provided id for parse), stored on `self.run_id`, and
threaded into `extractor.extract(run_id=...)` and `writer.save_run(run_id=...)`.
This lets the extractor key its artifact directory and raw `frame_detections`
by the same id **before** the `runs` row exists. ULIDs sort lexicographically
by creation time, which the jobs-list / run-history ordering relies on.
Generation lives in `backend/src/utils/ids.py` (`new_ulid`).

### Default backend

Default to Postgres in the web-app deployment; keep SQLite working for local
single-machine dev. `docker-compose.yml` already has Postgres scaffolding.

---

## 5. Event vocabulary (worker → UI live updates)

**Status: DONE.** The core emits event type + payload through
`backend/src/utils/progress.py`, and the web layer now publishes and serves them
(step 9). Per-criterion parser events remain reserved (no per-criterion emit is
wired yet).

The transport is a **single Redis stream per job**, `job:{id}:stream` — not the
two pub/sub channels originally sketched. Both structured events *and* mirrored
log lines are `XADD`ed to it, each entry tagged with a `type` (`"log"` for log
records). One stream, so ordering is preserved and the SSE handler does one
read. Streams (not pub/sub) so a late subscriber can replay history (§16);
`MAXLEN` bounds growth and a TTL is set once a terminal event is written.

`GET /jobs/{id}/events` is an async `XREAD BLOCK` generator that replays from id
`0` then tails, closing on a terminal event.

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

// Reserved for a future parser-evaluator hook; current parser emits parse phase events only.
{"type":"parser.criterion_started","data":{"feature_name":"Brand Mention (Speech)"}}
{"type":"parser.criterion_completed","data":{"feature_name":"...","evaluation":true}}

{"type":"job.completed",     "ts":..., "run_id":"..."}
{"type":"job.failed",        "ts":..., "error":"...","phase":"shot_processing"}
```

### Publish sites in the engine

| Event | Hook |
|---|---|
| `job.started` / `job.failed` / `job.completed` | `ClipScribeEngine.run` outer try/except |
| `phase.started/completed (scene_detection)` | around `_digest_video` in `extractor_core.py` |
| `phase.started (audio)` + `audio.segment` + `phase.completed` | `_analyze_audio` segment loop in `extractor_core.py` |
| `phase.started (shot_processing)` | top of shot loop in `extractor_core.py` |
| `shot.started` | start of each iteration |
| `shot.scene_described` | after `describe_scene` |
| `shot.taxonomy_resolved` | after `set_active_targets` |
| `shot.frame_processed` | bottom of inner frame loop |
| `shot.completed` | end of shot iteration |
| `identity.merged` | inside `_resolve_identities` when `should_merge=True` |
| `phase.started/completed (parse)` | around `VideoInformationParser.parse` evaluation |
| `parser.criterion_*` | reserved; constants exist but no per-criterion emit is wired yet |

### Implementation

A `ProgressReporter` interface is injected into the engine, extractor, and
parser. Implementations:
- `NullProgressReporter()` — used by CLI, tests, and as the fallback when Redis
  is unreachable.
- `RedisProgressReporter(job_id, client)` (`app/events.py`) — `XADD`s events to
  `job:{id}:stream`. Built by `make_reporter(...)` and wired into the engine by
  `run_job_core` for both the inline and celery paths.

`extractor_core.py` only depends on the interface, not Redis. Same applies to
the parser.

`JobLogStreamHandler` (`app/events.py`) mirrors `INFO+` `clip_scribe` log
records into the same stream tagged `type: "log"`. It reads the job id from the
`current_job_id` contextvar (set in `run_job_core`), so no function signatures
change; records emitted outside a job context are dropped.

---

## 6. API surface

OpenAPI-generated; TS client codegen via `openapi-typescript`. All routes
return Pydantic-validated JSON. Errors: RFC7807 (`type`/`title`/`detail`).
Implemented routes cover the inline and Celery dispatch paths, Redis
Stream-backed SSE progress, read-only run views, artifacts, metadata, uploads,
and advisory chat.

### Jobs
- `POST   /jobs`                       — create a queued job and submit it to the configured backend (`inline` single-slot executor or `celery`). Request body mirrors `main.py` params except device is config-owned. `parse` requires an existing `run_id`; `extract` still writes artifacts only and does not create a `runs` row.
- `GET    /jobs`                       — paginated, filterable by status.
- `GET    /jobs/{id}`                  — full state.
- `GET    /jobs/{id}/events`           — SSE live progress. Replays the job's Redis stream from the start (events + logs interleaved), then tails; closes on a terminal event.
- `GET    /jobs/{id}/progress`         — coarse percent summary derived from the Redis stream for jobs-list bars.
- `POST   /jobs/{id}/cancel`           — cancel a queued job or mark a running job canceled. Cooperative mid-run interruption is still planned (see §10.10).
- `POST   /jobs/{id}/retry`            — create a fresh job from a failed/canceled job's stored request payload.
- `DELETE /jobs/{id}`                  — delete a completed, failed, or canceled job row.

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
- `GET /readyz`                        — DB + Redis reachable, plus (inline mode only) the heavy builder loaded; celery mode is ready without models.

### Metadata
- `GET /platforms`                     — list, with required params.
- `GET /defaults`                      — current yaml config exposed (read-only).
- `GET /inputs`                        — list videos under `CLIPSCRIBE_INPUT_DIR`.

### Uploads
- `POST /uploads`                      — stream uploaded video(s) to `CLIPSCRIBE_INPUT_DIR`; returned `path` values are valid `JobCreateRequest.video_path` values.

### Chat (advisory agent — post-run Q&A, see §13)
- `POST   /runs/{id}/chat`             — ask a question; streamed (SSE) answer. Body: `{session_id?, message}`.
- `GET    /runs/{id}/chat/sessions`    — list chat sessions for the run.
- `GET    /runs/{id}/chat/{session_id}`— message history for one session.
- `DELETE /runs/{id}/chat/{session_id}`— delete a session.

---

## 7. Frontend (SPA)

### Stack
- Vite + React + TypeScript (strict mode).
- TanStack Router (file-based; cleaner than React Router for app-shell apps).
- TanStack Query for REST data + `EventSource` for SSE.
- Zustand (or just `useReducer`) for the per-job live state.
- Tailwind v4 utilities for components; shadcn/ui remains deferred.
- Lightweight custom timeline rendering; no charting library is currently needed.
- Type-safe API client from OpenAPI (`openapi-typescript` + `openapi-fetch`).

### Pages

1. **Jobs list** (`/`)
   - Table of jobs (video, status/progress, platform, mode, created time,
     duration, and lifecycle actions).
   - Status filter.
   - "New job" button.

2. **New job** (`/jobs/new`)
   - Form for user-facing `full` jobs: `platform_params`, `user_hints`,
     `video_type`, and `platform`. `extract` and parser-only `parse` remain
     API/developer paths. Device is process configuration, not submitted in the
     job request.
   - Video field: upload via `POST /uploads` OR pick from server-side
     `input/` directory via `GET /inputs`.
   - `GET /defaults` is available for config-driven form expansion; the current
     first-pass form only renders the fields it submits.
  - Submit → `POST /jobs` → redirect to `/jobs/{job_id}` and watch the SSE
    progress stream until the response contains a completed `run_id`.

3. **Live job** (`/jobs/{id}`)
   - Top progress bar + cancel action.
   - Phase tree (scene detection, audio, shots N/M, finalize, parse).
   - Current-shot panel (description, dino prompt, taxonomy targets, frames
     processed).
   - Live log tail.
   - Implemented live state is driven by SSE with a reducer keyed off
     `event.type`, while `GET /jobs/{job_id}` remains the canonical status row.
   - On completed status, auto-redirect (or show CTA) to `/runs/{run_id}`.

4. **Run inspector** (`/runs/{id}`)
   - Top: video player with SVG overlay.
   - Layer toggles for tracked objects and OCR text, plus active detection count.
   - Timeline tracks for shots and audio.
   - Bottom: ABCD criteria table from `parser_results`, each row expandable to
     show `llm_prompt` + `llm_explanation` + LangSmith trace link.
   - Tracked video download and advisory chat panel.

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

The current local stack uses three runtime images plus two one-shot compose
services. All Docker builds use the repository root as context so the images can
copy both `backend/` and `frontend/` without moving files around.

```
backend/docker/
  api/
    Dockerfile          # slim — FastAPI, DB, Redis/Celery client, advisory chat
    deploy.sh
  core/
    cpu/
      Dockerfile        # heavy CPU Celery worker + prewarm image
      deploy.sh
    gpu/
      Dockerfile        # CUDA worker image; prewarms weights at build time
      deploy.sh
frontend/
  Dockerfile            # Vite build stage + nginx static/proxy stage
```

### Image responsibilities

| | API image (`backend/docker/api/`) | Worker image (`backend/docker/core/{cpu,gpu}/`) | Frontend image (`frontend/Dockerfile`) |
|---|---|---|---|
| Role | FastAPI, REST/SSE, artifact serving, Celery dispatch, Alembic migrate | Celery worker; runs the full pipeline | Static SPA + `/api/*` nginx reverse proxy |
| Heavy deps | none (no `torch`, `whisper`, `paddleocr`) | full CV/ML stack | Node build deps only in build stage |
| Imports `src/clip_scribe/build_clip_scribe.py`? | no | yes | no |
| Imports `app/tasks.py`? | no — sends tasks by name | yes | no |
| Talks to Redis | yes (broker client + stream reader) | yes (broker consumer + stream publisher) | no |
| Talks to Postgres | yes (reader/writer + jobs) | yes (writer + reader) | no |
| Uses `backend/checkpoints/`? | no | CPU: compose volume; GPU: baked layer | no |

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

### Environment variables (.env → compose → prod)

Runtime config lives in a single **repo-root `.env`** (gitignored; holds secrets
too). It is the source of vars for local runs. Two consumers read it:

- **The app processes** load it at import via `find_dotenv` — both
  `app/settings.py` (so the celery-mode API and the worker's `celery_app` see
  `REDIS_URL` / `CLIPSCRIBE_JOB_BACKEND` before reading `os.environ`) and
  `build_clip_scribe.py` (for the core). `load_dotenv(..., override=False)`, so a
  var already set in the real environment wins over the file.
- **Compose** passes it into containers with `env_file: .env` (path is relative
  to the compose file at the repo root), then overrides network-sensitive vars
  and device settings in each service's `environment:` block.

Backend vars (current):

| Var | Purpose | Local (native) | Container |
|---|---|---|---|
| `CLIPSCRIBE_JOB_BACKEND` | `inline` \| `celery` dispatch | `celery` | `celery` |
| `CLIPSCRIBE_DB_BACKEND` | overrides `database.backend` | unset (yaml `sqlite`) or `postgresql` | `postgresql` |
| `REDIS_URL` | broker + Redis Streams | `redis://localhost:6379/0` | `redis://redis:6379/0` |
| `POSTGRESQL_URL` | DB (when backend=postgresql) | `…@localhost:5433/…` | `…@postgres:5432/…` |
| `SQLITE_URL` | DB (when backend=sqlite) | `sqlite:///data/…` | (needs shared volume; prefer PG) |
| `CLIPSCRIBE_DEVICE` | builder device (web app) | `cpu` default; set `mps` for native GPU | `cpu` (Mac) / `cuda` (GPU box) |
| `OPENAI_API_KEY`, `LANGCHAIN_*` | LLM + tracing | secret | secret |

**Device precedence.** `settings.clip_scribe_device` reads `CLIPSCRIBE_DEVICE`
(default `cpu`) and is passed into `ClipScribeBuilder(device=...)` by **both** the
inline API (`main.lifespan`) and the worker (`tasks.get_builder`). The builder's
own signature is `device: str | None = None`, and the arg overrides the yaml
value — so the web app is env-driven while the **CLI** (`backend/main.py`) can
use a hardcoded local override or omit the arg to fall back to the yaml
`device`. The builder then guards the choice: `mps`/`cuda` requested but
unavailable → CPU. So to run the native Mac worker on MPS, set
`CLIPSCRIBE_DEVICE=mps` in the host env; the default `cpu` keeps a Linux
container safe with no config.

**The one gotcha — host vs. service names (§12).** `.env` holds the *native-host*
values (localhost + compose-mapped ports), which is what the native worker and a
host-run API need. A *containerized* service must instead reach `postgres:5432` /
`redis:6379`. So compose **overrides** the network-sensitive vars
(`POSTGRESQL_URL`, `REDIS_URL`) per service in its `environment:` block while
still pulling the rest from `env_file`. `CLIPSCRIBE_DEVICE` is set per service
too (`cpu` on a Mac worker container, `cuda` on a GPU box) — this is what lets
full-container compose run on Mac at all.

**Prod.** Replace `.env` with GCP **Secret Manager** for secrets
(`OPENAI_API_KEY`, DB creds) injected as env at deploy, and bake the non-secret
knobs (`CLIPSCRIBE_DEVICE=cuda`, weight-dir vars) into the image `ENV` block.

### Checkpoint / weight strategy

All model weights live under `backend/checkpoints/`, organized by source:

```
backend/checkpoints/
  groundingdino_swinb_cogcoor.pth    GroundingDINO base checkpoint
  groundingdino_swint_ogc.pth        GroundingDINO tiny checkpoint
  sam2.1_hiera_tiny.pt               SAM2 tiny checkpoint
  sam2.1_hiera_small.pt              SAM2 small checkpoint
  torch_hub/    ← TORCH_HOME         (DINOv2 auto-download)
  huggingface/  ← HF_HOME            (SBERT inside TaxonomyResolver)
  whisper/      ← download_root arg  (Whisper auto-download)
  paddleocr/    ← PADDLE_OCR_BASE_DIR (PaddleOCR auto-download)
  nltk/         ← NLTK_DATA          (WordNet)
```

Env vars set once at process start (`.env` in dev, Dockerfile `ENV` block
in prod) redirect every auto-downloader into a subdir of
`backend/checkpoints/`:

```bash
TORCH_HOME=$REPO/backend/checkpoints/torch_hub
HF_HOME=$REPO/backend/checkpoints/huggingface
NLTK_DATA=$REPO/backend/checkpoints/nltk
PADDLE_OCR_BASE_DIR=$REPO/backend/checkpoints/paddleocr
```

The builder and prewarm script read those settings before importing the
libraries that use them. Whisper does not honor the cache env vars, so
`build_clip_scribe.py` and `prewarm.py` pass
`download_root=str(checkpoints / "whisper")` explicitly. MTCNN weights ship in
the `facenet-pytorch` wheel; spaCy's `en_core_web_sm` is installed as a wheel by
`make spacy` or baked into the worker images.

`backend/scripts/prewarm.py` is the single prefetch entry point. It downloads
the GroundingDINO and SAM2 checkpoint files, loads DINOv2 through `torch.hub`,
loads SBERT, downloads NLTK WordNet into `checkpoints/nltk`, loads Whisper into
`checkpoints/whisper`, constructs PaddleOCR so it populates
`PADDLE_OCR_BASE_DIR`, verifies spaCy, and writes `.prewarm_complete` after a
successful run.

### Local compose stack

`docker-compose.yml` has Postgres, Redis, one-shot `migrate`, slim `api`,
one-shot `prewarm`, heavy CPU `worker`, and the nginx-served `frontend`. The CPU
worker and prewarm services are pinned to `linux/amd64` because PaddlePaddle has
no Linux arm64 wheel and amd64 is the deploy target.

```yaml
prewarm:
  build:
    dockerfile: backend/docker/core/cpu/Dockerfile
  platform: linux/amd64
  command: ["python", "scripts/prewarm.py"]
  volumes:
    - ./backend/checkpoints:/app/backend/checkpoints

worker:
  build:
    context: .
    dockerfile: backend/docker/core/cpu/Dockerfile
  platform: linux/amd64
  env_file: .env
  environment:
    CLIPSCRIBE_JOB_BACKEND: celery
    CLIPSCRIBE_DB_BACKEND: postgresql
    CLIPSCRIBE_DEVICE: cpu
    POSTGRESQL_URL: postgresql://clipscribe:clipscribe@postgres:5432/clipscribe
    REDIS_URL: redis://redis:6379/0
  volumes:
    - ./backend/checkpoints:/app/backend/checkpoints   # persist weights
    - ./backend/artifacts:/app/backend/artifacts
    - ./backend/input:/app/backend/input
```

- First `docker compose up`: the `prewarm` service downloads several GB, then
  the worker starts.
- Every subsequent `up`: `prewarm.py` short-circuits on
  `backend/checkpoints/.prewarm_complete`.
- Force a refetch with
  `docker compose run --rm prewarm python scripts/prewarm.py --force`.

### GPU image — bake strategy

`backend/docker/core/gpu/Dockerfile` is the CUDA worker image for Linux +
NVIDIA hosts. It uses a CUDA runtime base, installs torch/torchvision from the
CUDA wheel index, exports the locked `worker` group from `uv.lock`, strips the
torch and `nvidia-*` lines from that export, installs the rest pinned, bakes
spaCy, and runs `python scripts/prewarm.py` before copying app/source code so
the large weights layer stays cacheable:

```dockerfile
ENV TORCH_HOME=/app/backend/checkpoints/torch_hub \
    HF_HOME=/app/backend/checkpoints/huggingface \
    NLTK_DATA=/app/backend/checkpoints/nltk \
    PADDLE_OCR_BASE_DIR=/app/backend/checkpoints/paddleocr
RUN python scripts/prewarm.py
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

The CPU worker image can run on Mac under amd64 emulation if you accept CPU
fallback — useful for verifying the worker integration end-to-end, but ~10×
slower than native MPS. Treat it as smoke-test only.

### API image — always slim

The API Dockerfile installs only the locked `api` dependency group using
`uv export --frozen --only-group api`, then copies the API package and only the
source trees it imports at module load: `src/db`, `src/parser`, `src/utils`, and
`src/clip_scribe`. It deliberately omits extractor/OCR/DINO/SAM2 and never
installs torch.

```dockerfile
FROM python:3.12-slim
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --only-group api \
  -o /tmp/api-reqs.txt && uv pip install --no-cache -r /tmp/api-reqs.txt
COPY backend/app ./app
COPY backend/src/db ./src/db
COPY backend/src/parser ./src/parser
COPY backend/src/utils ./src/utils
COPY backend/src/clip_scribe ./src/clip_scribe
```

`backend/pyproject.toml` already has a slim `[dependency-groups].api` group for
local API-container work. Use `uv sync --only-group api`; unlike an optional
extra, it excludes the heavy main dependencies (torch / whisper / paddleocr).

---

## 9. Things we have not nailed down yet (open questions)

These are decisions to make before the corresponding implementation step.

1. **Video ingest.** **Resolved for the sync path.**
   The API now supports both near-term local flows: `POST /uploads` streams
   browser uploads into `CLIPSCRIBE_INPUT_DIR`, and `GET /inputs` lists videos
   already present in that directory. `POST /jobs` accepts the returned
   server-side relative path and rejects traversal or missing files before
   enqueuing. Still open for cloud/multi-user: pre-signed object-storage upload.

2. **Artifact storage.** **Mostly resolved.**
   The extractor now writes to `artifacts/<run_id>/` (keyed by the ULID, not the
   video name — no more collisions). A `remote_artifact_write` config flag
   (default `false`) selects an `ArtifactUploader` (`backend/src/utils/artifacts.py`):
   `NullArtifactUploader` (local only) or `SimulatedGCSArtifactUploader`, which
   currently just **logs** the single bundle it would push
   (`gs://…/<run_id>/artifacts.tar.gz`) at the end of the run. Swapping in a
   real GCS uploader later is a drop-in replacement — flip the flag, implement
   the body, no call-site changes. Still open: whether the real backend is GCS
   vs a shared filesystem volume for the local/compose case.

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

10. **Logging refactor.** **Resolved for job streams.**
    `JobLogStreamHandler` reads the `current_job_id` contextvar set by
    `run_job_core` and mirrors `INFO+` `clip_scribe` records into the same Redis
    stream as structured progress events. No call-site changes were needed.

11. **Tests.**
    Test suite is "minimal" per CLAUDE.md. New code should land with tests:
    Pydantic model round-trips, `ProgressReporter` event ordering, `frame_detections`
    population, API job validation/routes, and SSE stream replay/tailing.
    Don't try to test the engine end-to-end — that needs models.

12. **OpenAPI → TS codegen as CI step.**
    Generate `frontend/src/api/types.ts` from the FastAPI schema on every API
    change. Either pre-commit hook or a `make codegen` step. Avoid drift
    between Python and TS types.

13. **CORS / dev proxy.**
    Vite dev server on 5173, FastAPI on 8000. Use Vite proxy in dev so the
    frontend can call `/api/*` without CORS gymnastics. In prod, nginx in
    front does the same thing.

14. **Disk retention.** **Partially mitigated.**
    A `max_artifact_files` config cap (default 350) now bounds the per-frame
    visualization PNGs written per run (the unbounded growth source); the
    tracked mp4 and `extraction_summary.json` are always kept. Still open: a
    run-level retention policy (delete after N days / keep last K) and a
    `DELETE /runs/{id}` endpoint that cleans both DB rows and the artifact dir.

15. **Per-job artifact directory keying.** **Resolved.**
    Artifacts are written to `artifacts/<run_id>/` (ULID). The path convention
    lives in one place: `run_artifact_dir(run_id)` in
    `backend/src/utils/artifacts.py`, used by both the extractor and the engine
    (for the upload call). Old `extractor_artifacts/<video_name>/` runs are left
    as-is.

16. **What does the SSE channel do when there are zero subscribers?**
    **Resolved.** Live progress uses one Redis stream per job (`XADD` to
    `job:{id}:stream`) instead of pub/sub. The SSE handler replays from id `0`
    on connect and then tails with `XREAD BLOCK`, so late subscribers receive
    prior events while the stream is retained.

17. **`tracked_output.mp4` vs raw input video for the player.**
    Use the raw input as the `<video>` source and overlay our own SVG boxes —
    full control, supports layer toggles. The baked tracked mp4 stays as a
    download.

18. **Advisory chat scope & memory (§13).**
    Conversation memory via LangGraph checkpointer (`MemorySaver` in dev, the
    Postgres checkpointer when deployed so sessions survive API restarts and
    span replicas). Decide: one implicit session per run vs. multiple named
    sessions; whether the agent may ever compare across runs (default **no** —
    tools stay strictly bound to a single `run_id`); and whether transcripts are
    retained/purged alongside run retention (§9.14).

---

## 10. Sequencing — what to build first

Strictly ordered; each step is shippable on its own.

1. **DB migrations: `jobs`, `frame_detections`, `parser_results`,
   `shot_boundaries`.** **DONE.** Alembic baseline + second migration are in
   `backend/alembic/versions/`; schema creation is migration-owned.

2. **Builder refactor: load-once via `ClipScribeBuilder.__init__`.**
   **DONE.** Heavy assembly moved into `__init__` (`_assemble_db`,
   `_assemble_heavy_extractor_utils`). `build_clip_scribe(...)` is now
   the cheap per-job entry point. CLI works unchanged. See §3.

3. **`ProgressReporter` interface + Null impl.** Wire publish calls into
   `engine.py`, `extractor_core.py`, and `parser_core.py`. **DONE.** CLI/tests
   pass Null by default; event ordering has focused unit coverage.

4. **Persist raw detections.** **DONE.** The extractor collects
   `frame_detections` (sources `dino`/`ocr`/`mtcnn`/`sam_mask`) and
   `shot_boundaries` into its returned `ExtractionSummary` (staying DB-free);
   the writer persists them in `save_run`, keyed by the up-front `run_id`. The
   parser persists per-criterion `parser_results` via `writer.save_parser_results`
   (feature fields read by `getattr`, so non-YouTube platforms still persist the
   common columns). Also landed here: run_id-up-front ULID (§4), artifact dir
   keyed by run_id (§9.15), the `max_artifact_files` PNG cap (§9.14), and the
   `remote_artifact_write` `ArtifactUploader` seam (§9.2). Unit-tested with an
   in-memory DB; full end-to-end proof is the first real `main.py` run.

5. **FastAPI app, inline path.** **DONE.** `POST /jobs` writes a queued
   job and submits it to a single-slot in-process executor in `inline` mode, so
   the HTTP contract is asynchronous from the client's perspective.
   Implemented routes include uploads, input listing, job list/get, read-only
   `/runs/*`, filesystem artifacts, health, and metadata; errors use RFC7807.
   Request shape intentionally omits device, using process configuration
   (`CLIPSCRIBE_DEVICE` in web mode) instead.

6. **Frontend bootstrap.** **DONE.** Vite + React + TS + Tailwind + TanStack Router /
   Query. Pages: Jobs list, New job, live Job page, and Run inspector against
   existing DB data.

7. **Inspector overlay.** **DONE (first pass).** Uses `frame_detections` to draw
   SVG boxes on `<video>`, with layer toggles, active detections, timeline
   tracks, parser results, and tracked-video download. Remaining polish lives in
   step 12.

8. **Celery + Redis.** **DONE (backend wiring).** `POST /jobs` dispatches on
   `settings.job_backend`: `inline` (the step-5 executor) or `celery`
   (`celery_app.send_task("app.tasks.run_job", …)`). New files: `app/celery_app.py`
   (thin shared broker handle, no torch — the §8 import boundary),
   `app/tasks.py` (worker-only; `worker_process_init` / lazy `get_builder()`
   loads one long-lived `ClipScribeBuilder` per process), and
   `app/job_execution.py` (`run_job_core`, the single lifecycle both paths
   share). In celery mode the API loads **no models** — lifespan builds only a
   standalone reader/writer, and `get_reader`/`get_writer` read from
   `app.state`. Cancel `revoke`s the task (no `terminate`; cooperative cancel is
   step 10). `docker-compose.yml` gains a `redis` service; `celery` + `redis`
   added to deps + the `api` group. Config: `CLIPSCRIBE_JOB_BACKEND=celery`,
   `REDIS_URL`. Run the worker natively on macOS/MPS with
   `uv run celery -A app.celery_app worker --pool=solo --concurrency=1`.
   Remaining: an end-to-end run against live Redis + a real worker (needs models).

9. **Redis Streams bridge + SSE.** **DONE.** Live progress uses a per-job Redis
   **stream** (`job:{id}:stream`, `XADD` with `MAXLEN`), not pub/sub — pub/sub
   drops events with no subscriber, so a late-loading page would miss history
   (§16). `app/events.py` holds `RedisProgressReporter` (torch-free), a
   `make_reporter` factory that falls back to `NullProgressReporter` when Redis
   is down, and `JobLogStreamHandler` — a `logging.Handler` that reads a
   `current_job_id` contextvar and mirrors `INFO+` `clip_scribe` records into the
   same stream tagged `type: "log"` (§9.10, no call-site changes). `run_job_core`
   builds the reporter, installs the log bridge, and sets the contextvar, so
   **both** the inline and celery paths publish. `GET /jobs/{id}/events` is an
   async SSE generator over `redis.asyncio` `XREAD BLOCK` from id `0`: it replays
   the whole stream to a late subscriber, then tails; it closes on a terminal
   event, with the job row's terminal status as a backstop for jobs that never
   emit one (queued-then-canceled, or Redis down at run time). `/readyz` gained a
   Redis ping and no longer requires a loaded builder in celery mode. Frontend:
   `jobs.$jobId.tsx` live page renders from a reducer keyed on `event.type`
   (progress bar, phase tree, current-shot panel, log tail) fed by an
   `EventSource`; `POST /jobs` now redirects there and the jobs list links to it.
   Tested with `fakeredis` in `test/test_api_events.py`.

10. **Cooperative cancel.** Queue/running cancel endpoint exists and marks the
    job canceled, but the engine cannot yet interrupt mid-run. Remaining:
    "should_cancel" flag honored by the shot loop + parser, plus partial result
    handling.

11. **Docker split.** **DONE.** `backend/docker/api/Dockerfile` builds the
    slim torch-free API/migrate image; `backend/docker/core/cpu/Dockerfile`
    builds the heavy CPU worker/prewarm image; `backend/docker/core/gpu/Dockerfile`
    builds the CUDA worker image with weights baked during build; `frontend/Dockerfile`
    builds the SPA with Vite and serves it through nginx. `docker-compose.yml`
    wires the full local CPU stack plus the hybrid native-MPS workflow.

12. **Polish:** retention policy, auth (if/when needed), cost tracking,
    OpenAPI codegen in CI, expand tests.

13. **Advisory chat agent — backend (§13).** **DONE.** `query_parser_results`
    tool + `"advisory"` tool group in `backend/src/parser/tools.py`;
    `build_advisory_agent(model, reader_db, run_id)` in `backend/src/parser/advisory.py`;
    `app/chat.py` `ChatService` (SSE streaming, DB-as-memory) + `chat_messages`
    table (migration `c1a2d3e4f5a6`) + reader/writer methods; routes in
    `app/routes/chat.py` (`POST /runs/{id}/chat` streamed, session
    list/history/delete). **No models, no worker, no GPU.** One caveat vs the
    original "no torch" claim: the LLM client (`langchain_openai`/`langgraph`)
    may transitively import torch in a full local environment, so the route
    lazy-imports the service to keep `import app.main` torch-free. The slim API
    image includes `langgraph`, `langchain-openai`, and `backend/src/parser`
    through the `api` dependency group/source copy while still resolving without torch.
    Tested in `test/test_api_chat.py`.

14. **Advisory chat agent — frontend (§13).** **DONE.** `ChatPanel`
    (`frontend/src/components/ChatPanel.tsx`) mounted in the run inspector:
    streams the answer token-by-token (POST + manual SSE parse over `fetch`),
    shows tool-call chips, keeps a session id for multi-turn continuity, and
    offers starter-prompt buttons. Follow-up: an "ask about this" shortcut on
    failed criterion rows (needs lifting chat state above `ParserTable`).

---

## 11. Risks / things that could derail this

- **Builder refactor is bigger than it looks.** Hints get passed deep into
  the extractor and into the GPT taxonomy generator. Untangling these so the
  registry is truly per-process will touch `taxonomy_core.py`,
  `extractor_core.py`, and the builder.

- **MPS in Docker on Mac.** Already flagged. Lots of dev pain if we forget.

- **Live-progress replay depends on Redis Stream retention.** Streams solve the
  subscriber-loss problem, but `MAXLEN` and terminal TTL still bound how much
  history a late user can replay.

- **OpenAI cost.** Adding a "Run job" button in a web UI makes it easy to
  burn through credits. Cost cap or per-user budget should exist before
  this is anything but internal.

- **Disk usage.** `backend/artifacts/` is now run-id keyed and per-frame PNGs
  are capped, but run-level retention still needs to
  exist before turning the app on for more than one person.

- **Cancellation correctness.** Hard-killing a worker mid-job leaves OpenCV
  file handles, partial mp4s, and possibly a corrupted SAM2 inference state
  on the GPU. Cooperative cancel is the only safe path; it requires touching
  the shot loop.

- **Type drift Python ↔ TS.** Without OpenAPI codegen wired into CI, the
  shapes will drift the moment a Pydantic field is added.

---

## 12. Operating modes & deployment

Both modes share the same **core image** (torch + CV stack + `ClipScribeBuilder`)
with **two entrypoints**: `celery worker` (web app) and a "process one video"
batch entrypoint (CLI / K8s Job). Same builder, same engine, different launcher.

### Celery worker model (web app)

- A worker is a Python process. With the default **prefork pool** it forks
  `--concurrency=N` child processes **once at startup**; each child loads the
  models a single time (`worker_process_init`). Tasks are dispatched to a free
  child; when all children are busy, extra tasks **wait in the Redis queue**,
  not inside the worker.
- Run **`--concurrency=1`** (or `--pool=solo`, no fork at all): models load
  once, one job at a time. `N>1` would multiply the 8–15 GB model load per
  child and contend for one GPU.
- **One GPU → one effective worker slot.** A machine hosts more workers only if
  it has more GPUs (one worker per GPU, pinned via `CUDA_VISIBLE_DEVICES`).
  Throughput scales by adding machines/GPUs to the same Redis queue; backlog
  just queues.

### Docker networking (compose)

Container-to-container addressing uses the **service name + internal port**,
not the host port mapping. Consequences for `POSTGRESQL_URL` / `REDIS_URL`:

| Caller | Postgres | Redis |
|---|---|---|
| Process on the host (e.g. the **native Mac worker**) | `…@localhost:<mapped>` (e.g. `5433`) | `redis://localhost:6379/0` |
| A **container** on the compose network | `…@postgres:5432` | `redis://redis:6379/0` |

Because the Mac dev worker runs **natively** (for MPS) while Postgres/Redis are
dockerized, the native worker uses `localhost:<mapped-port>` while the API
container uses `postgres:5432`. So services get **different env values** by
where they run.

### Mode A — web app deployment

- **Local dev (near term):** the API can run natively from `backend/` in
  `inline` mode so `device=mps` works while `docker-compose` supplies Postgres
  and Redis. In `celery` mode, `docker-compose` supplies Redis + Postgres and
  the worker can run natively on the Mac for MPS. No cloud, no K8s. This mode is
  primarily a learning/dev surface.
- **GCP shape (sketch, not a build target):**

  | Component | GCP service | Note |
  |---|---|---|
  | API (slim, no pipeline models) | **Cloud Run** | Enqueues to Redis, serves SSE. Mind Cloud Run request timeouts for long SSE streams. |
  | Redis (broker + streams) | **Memorystore for Redis** | Managed. |
  | Postgres | **Cloud SQL for PostgreSQL** | Managed. |
  | Celery GPU worker | **GKE GPU node pool** | Deployment with `nvidia.com/gpu: 1`, concurrency 1; autoscale pods on **queue depth** via KEDA (Redis scaler), scale to zero when idle. |

  The GPU worker is the awkward piece: Cloud Run's scale-to-zero, request-driven
  model fights a long-lived broker consumer holding models in GPU memory, so GKE
  (or a plain GPU VM) is the better host.

### Mode B — CLI / batch over K8s

No UI, no realtime, and **no Redis/Celery** — the job runner *is* the
orchestrator.

- **Local mode** (local paths): loop the video list, run the engine
  sequentially on one machine (generalized `main.py`, reusing the load-once
  builder).
- **Remote mode** (GCS URIs): fan out with a Kubernetes **Indexed Job**
  (`completionMode: Indexed`, `completions: N`, `parallelism: P`). K8s creates
  N indexed pods, each mapping its index → one video; each pod requests
  `nvidia.com/gpu: 1` (one whole GPU per pod), runs the batch entrypoint, reads
  the video from GCS, writes results to Cloud SQL + artifacts to GCS, and exits.
  `parallelism` is bounded by available GPUs; the cluster autoscaler adds GPU
  nodes up to a quota and scales back to zero when the batch finishes, so GPUs
  are paid for only during the run. `backoffLimit` gives per-video retries.
- The CLI (on the operator's machine) renders and `kubectl apply`s the Job (or
  uses the Python k8s client) against a GKE cluster with a GPU node pool.

---

## 13. Advisory chat agent (post-run Q&A)

An interactive follow-up to evaluation. After a run completes, the user opens
the run inspector, sees the ABCD verdicts, and can **ask questions** of an agent
that already knows the whole video and every verdict the evaluator agents
produced — e.g. *"criterion X failed — how would we fix it?"* or *"overall,
what should change in this creative?"*. Backend and frontend support have landed
as steps 13–14 in §10.

### Why it's a clean fit (not a new subsystem)

The evaluation agents already do the hard part. `backend/src/parser/agent.py` builds a
LangGraph ReAct agent via `create_react_agent(model, tools)`, and
`backend/src/parser/tools.py` exposes read-only, run-scoped query tools
(`query_audio_segments`, `query_text_events`, `query_visual_objects`,
`query_scene_descriptions`, `query_global_stats`, `query_field_descriptions`),
grouped by feature type in `tool_map`. The advisory chat agent is the same
pattern with three deltas:

1. **All tools, not one group.** It gets a new `"advisory"` tool group that
   includes every existing query tool.
2. **One new tool — `query_parser_results`.** Reads the run's `parser_results`
   rows so the agent can cite each criterion's verdict, `llm_explanation`, and
   `llm_prompt`. This is what lets it reason about *why* something failed and
   what the evaluators saw.
3. **Conversational + advisory.** Multi-turn, free-form guidance (a strategist
   proposing concrete, testable changes) instead of the evaluators' one-shot
   structured pass/fail (`_parse_agent_response` in `agent.py`).

### The architectural win: API-only, no pipeline models

The chat agent does **only LLM calls + DB reads**. It never loads pipeline
models and never touches the Celery worker. It runs in the API process; routes
lazy-import the service because LangChain/LangGraph may transitively import
torch in this environment. Consequences:

- It ships independently of the Celery worker path.
- It reuses the API's existing `reader_db` and `OPENAI_API_KEY`; no worker
  round-trip. The slim API image includes the LLM dependencies needed for this
  route while avoiding the pipeline model stack.

### Security model — read-only and run-scoped

Every tool is bound to a single `run_id` **server-side**, exactly as the
evaluators do today (`build_tools(reader_db, run_id, tool_group)`). The client
sends a message and an optional `session_id`; it never passes a `run_id` into a
tool. The agent physically cannot read another run's data because no tool
accepts a cross-run argument. That closure is the entire isolation story.

### Components

- **`query_parser_results(feature_category=None, only_failed=False)`** — new
  read tool in `backend/src/parser/tools.py`; requires a matching
  `reader_db.get_parser_results(run_id, ...)` reader method.
- **`"advisory"` tool group** — registered in `tool_map` with all query tools
  plus `query_parser_results`.
- **`build_advisory_agent(model, reader_db, run_id)`** —
  `create_react_agent(model, advisory_tools)` with an advisory system prompt:
  persona is a senior creative strategist; must cite specific field values /
  verdicts; must fetch data via tools rather than invent it; must give concrete,
  testable recommendations.
- **Conversation memory = the DB (as built).** Each turn reloads the session's
  prior `chat_messages` and replays them into the agent as Human/AI messages, so
  history survives API restarts and spans replicas without a checkpointer — the
  `chat_messages` table is the single source of truth. (A LangGraph
  `MemorySaver`/Postgres checkpointer keyed by `thread_id = session_id` remains a
  possible optimization if replaying full history ever gets expensive.)
- **Streaming** — `agent.stream(..., stream_mode="messages")` piped over an SSE
  response. This reuses the §9 SSE *pattern*, but the event source is the LLM
  token stream directly — no Redis stream, no worker involved.

### Frontend (extends §7 page 4, Run inspector)

A chat panel below the ABCD criteria table:
- Streams the assistant answer token-by-token.
- Renders tool-call chips ("queried visual objects…", "read parser verdicts…")
  so the reasoning is transparent.
- Planned follow-up: each failed criterion row gets an **"ask about this"**
  shortcut that seeds a question like *"Criterion '{feature_name}' failed — what
  would fix it?"*.

### Open questions (tracked in §9.18 and §9.8)

- **Cost.** A turn can fan out into many tool calls over a large dataset. Cap
  reasoning depth with `recursion_limit` (as the evaluators already do) and
  consider a per-session token budget; ties into cost tracking (§9.8).
- **Context size.** Don't prefill the whole run into the system prompt — rely on
  the on-demand tool-call pattern the evaluators use, so only fetched slices
  enter the context window.
- **Model.** Advisory reasoning wants a strong model; make it configurable in
  `clip_scribe.yaml` next to the existing agent-model settings.
