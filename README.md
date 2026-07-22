# ClipScribe

ClipScribe is a multimodal video processing pipeline that extracts and structures visual, textual, and audio information from videos. It combines scene understanding, object detection and tracking, OCR, speech transcription, face detection, persistence, and platform-specific evaluation.

## Overview

ClipScribe splits a video into scenes, detects and tracks objects across shots, transcribes speech, extracts on-screen text, and assembles the result into structured metadata. A parser layer can then evaluate the extracted data against platform-specific criteria.

## Pipeline

1. **Scene Detection** - split the video into shots using content-based scene detection.
2. **Scene Comprehension** - GPT vision analyzes sampled frames per shot and produces a narrative scene description plus a GroundingDINO detection prompt.
3. **Dynamic Taxonomy** - an LLM generates canonical detection targets from the scene description and video type; SBERT maps raw labels to the taxonomy.
4. **Object Detection and Tracking** - GroundingDINO detects objects per frame; SAM2 tracks them across frames; DINOv2 embeddings support cross-shot identity resolution.
5. **Face Detection** - MTCNN detects faces in parallel.
6. **Audio Transcription** - Whisper transcribes speech with confidence filtering.
7. **OCR** - PaddleOCR extracts on-screen text frame by frame.
8. **Persistence** - structured results and scene descriptions are stored in a database.
9. **Evaluation** - parser agents query persisted extraction data and produce platform-specific reports.

## Features

- GPT-powered scene descriptions and detection prompts
- GroundingDINO object detection with configurable thresholds
- SAM2 segmentation and temporal object tracking
- Cross-shot object identity resolution using DINOv2 embeddings
- LLM-generated detection taxonomy with SBERT label resolution
- Whisper speech-to-text transcription
- PaddleOCR text extraction and text-box consolidation
- MTCNN face detection
- Pacing and dynamic-start analysis
- LangGraph/LangChain parser agents for feature evaluation
- YouTube platform evaluation support
- SQLite or PostgreSQL persistence through SQLAlchemy
- Per-run extraction artifacts, parser report generation, and CSV/XLSX ABCD exports for runs and jobs
- FastAPI API for deduplicated video uploads, jobs, Redis Stream-backed live progress, run inspection, run/job advisory chat, ABCD exports, and artifact serving
- Inline or Redis-backed Celery job dispatch
- Vite/React dashboard for job submission, live job progress, run inspection, ABCD export downloads, and run/job advisory chat

## Setup

ClipScribe is a monorepo: the Python backend lives in `backend/`, the React frontend in `frontend/`, and the git root is the repository root. **Run `uv`, Alembic, and pre-commit from the `backend/` directory** so project config and relative paths resolve correctly.

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 22+ with pnpm (`corepack enable`), and Docker (for Postgres + Redis, and the full container stack).

Install the **backend** dependencies (from `backend/`):

```bash
cd backend
uv sync --extra dev
```

Install the **frontend** dependencies (from `frontend/`):

```bash
cd frontend
pnpm install
```

Download the **model weights** — only needed to run extraction. From the repository root, `make setup` fetches the GroundingDINO/SAM2 checkpoints and the spaCy model, and pre-caches the auto-downloaded models (DINOv2, SBERT, Whisper, PaddleOCR, WordNet) under `backend/checkpoints/`:

```bash
make setup
```

`backend/scripts/prewarm.py` is the container's equivalent (run automatically by the `prewarm` compose service); it fetches the same downloaded weights but relies on the spaCy wheel being installed separately, as the worker Dockerfile does.

## Environment

Common environment variables:

- `OPENAI_API_KEY` - required for GPT scene analysis, taxonomy generation, and parser agents.
- `POSTGRESQL_URL` - required when the resolved database backend is `postgresql`.
- `SQLITE_URL` - optional when the resolved database backend is `sqlite`; defaults to `sqlite:///data/clip_scribe.db`.
- `CLIPSCRIBE_DB_BACKEND` - selects the database backend (`sqlite` | `postgresql`); defaults to `sqlite` when unset. `clip_scribe.yaml` does not carry a backend key, only pool settings.
- `CLIPSCRIBE_INPUT_DIR` - local source-video storage root relative to `backend/`; defaults to `input`. Backs the `local` backend, and is the staging/scratch dir the `gcs` backend downloads into.
- `CLIPSCRIBE_STORAGE_BACKEND` - the single selector for BOTH source-video and run-artifact storage (`local` | `gcs`); defaults to `local`. `gcs` uploads videos and artifacts to one bucket and serves them to the browser via signed-URL redirects.
- `CLIPSCRIBE_GCS_BUCKET` - required when `CLIPSCRIBE_STORAGE_BACKEND=gcs`; one bucket holds videos under the `videos/` prefix and artifacts under `artifacts/`.
- `GOOGLE_APPLICATION_CREDENTIALS` - service-account JSON for the `gcs` backend in dev. A repo-root-relative path (e.g. `service_account.json`) is resolved against the repo root, so it works even though the API/worker run from `backend/`. In prod, omit it and rely on the workload identity attached to the API/worker — which needs `roles/iam.serviceAccountTokenCreator` to sign video/artifact URLs (Bucket Admin alone lacks `iam.serviceAccounts.signBlob`).
- `CLIPSCRIBE_JOB_BACKEND` - `inline` (default single-slot in-process executor) or `celery` (Redis-backed worker dispatch).
- `REDIS_URL` - Redis connection used for Celery broker/result backend and per-job progress streams; defaults to `redis://localhost:6379/0`.
- `CLIPSCRIBE_DEVICE` - device used by the web API/worker builder (`cpu`, `mps`, or `cuda`); defaults to `cpu`.
- `CLIPSCRIBE_API_LOAD_MODELS` - set to `false` for limited schema/health/test startup without loading ML models; ignored in `celery` mode because the API loads only DB handles.
- `CLIPSCRIBE_CORS_ORIGINS` - comma-separated browser origins for the API; defaults to `http://localhost:5173`.

The main configuration file is:

```text
backend/src/clip_scribe/configs/clip_scribe.yaml
```

To use PostgreSQL, set `CLIPSCRIBE_DB_BACKEND=postgresql` and `POSTGRESQL_URL`.

Database schema is managed by Alembic. Apply migrations after creating a fresh database or pulling new migrations:

```bash
uv run alembic upgrade head
```

## Running

ClipScribe has two run surfaces: the **web app** (dashboard + API + worker) and a **local batch entry point** for one-off pipeline runs. To bring up the full web app, see [Running with Docker](#running-with-docker).

### Web app

The stack is a Vite/React dashboard, a FastAPI API, one or more Celery workers, Postgres, and Redis. `POST /jobs` returns immediately and streams live progress from a per-job Redis Stream over SSE. `CLIPSCRIBE_JOB_BACKEND` selects dispatch:

- `inline` — one long-lived `ClipScribeBuilder` and a single-slot in-process executor.
- `celery` — a model-free API that dispatches jobs to a Redis-backed Celery worker.

`POST /jobs` creates a parent job and fans its `videos` array out to one child run per video. Each `video_path` is an opaque storage key returned by `POST /uploads` or `GET /inputs`; the worker materializes that key to a local file only when extraction starts. Running child jobs stop cooperatively at extractor/parser checkpoints when canceled; parent cancellation cancels every queued or running child. The API request has no device field — the builder device comes from `CLIPSCRIBE_DEVICE`. The dashboard screens are the jobs list (`/`), the new-job form (`/jobs/new`), the batch/live job page (`/jobs/{job_id}`) with job-level chat and "Export all", and the run inspector (`/runs/{run_id}`) with sibling navigation, per-run export, and per-run advisory chat.

Useful API routes (reached from the browser under the `/api` proxy prefix):

- `POST /uploads` — upload one or more video files through the configured video storage backend; files are streamed to staging, hashed, deduplicated per user by SHA-256, and registered with their original filenames.
- `GET /inputs` — list the user's registered videos after reconciling missing storage objects; returned `path` values are valid `videos[].video_path` keys.
- `POST /jobs` — create a batch job from one or more `videos` sharing platform params and hints; parser-only `parse` is local/dev-only and rejected by the job API.
- `GET /jobs?status=&limit=&offset=` and `GET /jobs/{job_id}` — poll parent jobs with child summaries and read-time aggregated status.
- `GET /jobs/{job_id}/events` — SSE stream that replays and tails a child job's Redis Stream progress/log events.
- `GET /jobs/{job_id}/progress` — coarse percent summary for jobs-list progress bars.
- `GET /jobs/{job_id}/export?format=xlsx|csv` — export all completed runs' ABCD results; XLSX has a summary sheet plus one sheet per run, CSV is one flat table with a `Video` column.
- `POST /jobs/{job_id}/chat` and related job chat routes — stream read-only advisory Q&A across every completed run in a job.
- `POST /jobs/{job_id}/cancel` — cancel a queued/running child job, or cancel every cancellable child in a batch.
- `POST /jobs/{job_id}/retry` — retry a terminal parent as a fresh batch, or retry one child run in place with a fresh `run_id`.
- `DELETE /jobs/{job_id}` — cancel if needed, then delete job rows plus associated run data, chat transcripts, and artifacts.
- `GET /runs/{run_id}/...` — inspect persisted run data, sibling runs, frame detections, parser results, and artifacts.
- `GET /runs/{run_id}/parser/export?format=xlsx|csv` — export one run's ABCD results; XLSX includes `Detail` and `Scores` sheets, CSV is the flat detail table.
- `POST /runs/{run_id}/chat` and related chat routes — stream read-only advisory Q&A over a completed run.

### Local batch entry point (`main.py`)

`backend/main.py` is a temporary, hardcoded entry point (not a stable CLI) for local experiments — video name, mode, run id, platform params, and device are edited in the file. Modes:

- `extract` — run extraction and write local artifacts only (no DB run row); local dev only.
- `parse` — evaluate an existing persisted `run_id`.
- `full` — run extraction, then parse the saved run.

```bash
cd backend && uv run python main.py
```

> **Storage backend must be `local`.** `main.py` sets `video_path = "input/<name>"` — a direct path to a file already on the host, not an opaque storage key. It bypasses the upload → materialize seam the web app uses (there is no upload step and no bucket download), so it only works with `CLIPSCRIBE_STORAGE_BACKEND=local` (the default). Under `gcs` the path would be treated as a bucket object key and the run would fail to find the video. Unset `CLIPSCRIBE_STORAGE_BACKEND` (or set it to `local`) before running — even if the web app is otherwise configured for GCS.

Prefer changing existing builder/engine code over adding one-off run scripts.

## Running with Docker

Three images back the web app, all built from the repository root as context:

| Image | Dockerfile                           | Role                                                                                                                                                      |
| --- |--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `api` | `backend/docker/api/Dockerfile`      | Slim, torch-free FastAPI (REST + SSE, DB reads, artifact serving, ABCD exports, Celery dispatch, advisory chat). Also runs the one-shot Alembic migration.              |
| `worker` | `backend/docker/core/cpu/Dockerfile` | Heavy Celery worker running the full pipeline. CPU build here; `backend/docker/core/gpu/Dockerfile` is the CUDA variant for a Linux + NVIDIA host (see `docs/deployment.md`). |
| `frontend` | `frontend/Dockerfile`                | Vite/React SPA built with pnpm and served by nginx, which reverse-proxies `/api/*` to the `api` service (SSE-safe).                                       |

`docker-compose.yml` wires these plus `postgres`, `redis`, and two one-shot services: `migrate` (`alembic upgrade head`) and `prewarm` (model-weight download). There are two ways to run.

### Mode 1 — Everything in Compose (CPU only worker)

```bash
docker compose build
docker compose up      # api :8000, frontend :5173, worker, postgres, redis
```

The app is at `http://localhost:5173`. Model weights are fetched automatically: the one-shot `prewarm` service populates the shared `./backend/checkpoints` volume before the worker starts. The **first** `up` downloads several GB (the GroundingDINO/SAM2 `.pth` already on your host appear immediately); later `up`s short-circuit on a `.prewarm_complete` marker. Force a refetch with `docker compose run --rm prewarm python scripts/prewarm.py --force`. Compose also bind-mounts `backend/app` and `backend/src` into the API and worker containers so local source edits are visible after process reload/restart; rebuild when dependencies or Dockerfiles change.

The `worker`/`prewarm` images are built for **`linux/amd64`** (paddlepaddle has no arm64 wheel, and amd64 is the deploy target). On an **Apple Silicon** Mac they run under QEMU emulation — correct but slow, and `prewarm` loading models emulated can take a while. Mode 1 is best on a Linux/amd64 host or CI; on a Mac, use **Mode 2** for real work (native MPS worker) and treat Mode 1 as an end-to-end smoke test.

### Mode 2 — Hybrid local dev (MPS/CPU/GPU worker)

Run only Postgres + Redis (or just Redis with the default `CLIPSCRIBE_DB_BACKEND=sqlite`) in Compose and run the API, worker, and frontend natively so the worker gets MPS. Use separate shells:

```bash
# shell 0 — infra only
docker compose up postgres redis migrate

# shell 1 — migrations (once, you may skip since we run migrate container as well)
make migrate                                          # or: cd backend && uv run alembic upgrade head

# shell 2 — API (native)
cd backend && uv run uvicorn app.main:app --reload

# shell 3 — Celery worker (native, MPS)
cd backend && uv run celery -A app.celery_app worker --pool=solo --concurrency=1

# shell 4 — frontend (native, Vite dev proxy)
cd frontend && pnpm install && pnpm dev
```

Mode 2 uses the Vite dev-server `/api` proxy; Mode 1 serves the built SPA behind nginx doing the same proxy — same mental model in both.

### `.env` values per mode

Keep one repo-root `.env` with the **native-host** values. Compose feeds it to every service with `env_file` and then overrides only the network-sensitive vars per service (with in-container service names), so you never edit `.env` when switching modes:

| Var | Hybrid / native host | Full Compose (container) | Prod (GCP) |
| --- | --- | --- | --- |
| `POSTGRESQL_URL` | `…@localhost:5433/clipscribe` | `…@postgres:5432/clipscribe` | Secret Manager |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://redis:6379/0` | Memorystore URL |
| `CLIPSCRIBE_DB_BACKEND` | unset (`sqlite`) or `postgresql` | `postgresql` | `postgresql` |
| `CLIPSCRIBE_DEVICE` | `mps` | `cpu` | `cuda` |
| `CLIPSCRIBE_JOB_BACKEND` | `celery` | `celery` | `celery` |
| `CLIPSCRIBE_STORAGE_BACKEND` | `local` | `local` (or `gcs` via the overlay) | `gcs` |
| `CLIPSCRIBE_GCS_BUCKET` | unset for `local` | set when using `gcs` | bucket name |
| `GOOGLE_APPLICATION_CREDENTIALS` | `service_account.json` for `gcs` | mounted by `docker-compose.gcs.yml` | unset (workload identity) |
| `OPENAI_API_KEY`, `LANGCHAIN_*` | secrets (from `.env`) | same (from `.env`) | Secret Manager |

The container-column values live in each service's `environment:` block in `docker-compose.yml` and win over `env_file`.

### GCS storage (Compose overlay)

Both modes above default to `local` storage (videos in `backend/input`, artifacts in `backend/artifacts`, served from disk). To run the **full Compose stack against GCS** — videos and artifacts in a bucket, served to the browser via signed-URL redirects — layer the `docker-compose.gcs.yml` overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.gcs.yml up
```

Set these in the repo-root `.env` first:

```bash
CLIPSCRIBE_STORAGE_BACKEND=gcs
CLIPSCRIBE_GCS_BUCKET=your-bucket
GOOGLE_APPLICATION_CREDENTIALS=service_account.json   # repo-root file, gitignored
```

Passing multiple `-f` files makes Compose deep-merge them: the overlay only **adds** a read-only bind mount of the repo-root `service_account.json` into the `api` and `worker` containers (at `/app/service_account.json`) and pins `GOOGLE_APPLICATION_CREDENTIALS` to that absolute path — every other service field from the base file is left intact. It is kept out of the base `docker-compose.yml` on purpose: the default `local` stack must not require a credentials file, and an unconditional bind mount to a missing host file would be silently created by Docker as an empty directory.

In **prod** (GKE/Cloud Run) you drop the overlay entirely — the api/worker run under an attached workload identity with no key file, which needs `roles/iam.serviceAccountTokenCreator` to sign URLs.

### How imports resolve in the containers

The backend images set `WORKDIR /app/backend` and `PYTHONPATH=/app/backend`, then copy the code there — mirroring how you run locally from `backend/`. That is what makes both `from app.X import …` (the `app` package is not installed as a wheel) and `from src.X import …` resolve, without `pip install .`. The images install only third-party dependencies (the `api` or `worker` dependency group).

## Development Commands

Run tests:

```bash
uv run pytest -q
```

Run type checks:

```bash
uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser
```

Run all pre-commit hooks (from `backend/`):

```bash
uv run pre-commit run --all-files
```

> **Note:** pre-commit must be run from `backend/`. It discovers the config (`backend/.pre-commit-config.yaml`) from the current directory, but then executes every hook from the git root with file paths relative to that root. That is why the `exclude` patterns are `backend/`-prefixed and the mypy hook `cd backend` before running. Running pre-commit from the repository root fails with `.pre-commit-config.yaml is not a file`.

Install only the slim API dependency group for API-container work. The `api` group includes the advisory-chat LLM libraries (`langgraph`, `langchain-openai`) plus `openpyxl` for XLSX exports and resolves torch-free, so the API container stays slim while still serving chat and export routes:

```bash
uv sync --only-group api
```

Apply database migrations from the repository root via the Makefile:

```bash
make migrate
```

Working Makefile targets include `make setup` (model prefetch), `make checkpoints`, `make migrate`, and `make revision`.

## Project Structure

```text
clipscribe/
├── backend/
│   ├── app/                    FastAPI: routes, settings, inline/Celery dispatch, Redis events, chat, exports
│   ├── src/
│   │   ├── clip_scribe/        Engine, builder, platform configs, clip_scribe.yaml
│   │   ├── extractor/          Scene extraction, taxonomy, tracking, scene description
│   │   ├── parser/             LangGraph parser agents, query tools, evaluators, reports
│   │   ├── ocr/                PaddleOCR wrapper and box consolidation
│   │   ├── db/                 SQLAlchemy schema, engine, reader, writer
│   │   ├── dino/               GroundingDINO wrapper (+ vendored groundingdino, third-party)
│   │   ├── sam2/               Vendored SAM2 (third-party)
│   │   └── utils/              Progress, artifacts, video storage, ids, logging
│   ├── scripts/                prewarm.py + model-download helpers
│   ├── docker/
│   │   ├── api/                Slim, torch-free API image
│   │   └── core/{cpu,gpu}/     Heavy Celery worker images
│   ├── alembic/                Migration environment and versions
│   ├── checkpoints/            Model weights (gitignored; populated by prewarm)
│   ├── input/ artifacts/ parser_artifacts/ data/ logs/    Local storage and generated I/O (gitignored)
│   ├── main.py                 Local batch entry point
│   └── pyproject.toml, uv.lock
├── frontend/
│   ├── src/
│   │   ├── routes/             Jobs list, new job, live job, run inspector
│   │   ├── api/                Generated OpenAPI types + client/hooks
│   │   ├── components/         ChatPanel, Markdown, JobSidebar, PipelineAnimation, shared UI
│   │   └── lib/                State, formatting, run types
│   ├── Dockerfile, nginx.conf  Build + nginx-serve the SPA
│   └── package.json
├── docs/                       web-app-plan.md, deployment.md, deep-dive docs
├── docker-compose.yml          Full local stack (postgres, redis, migrate, prewarm, api, worker, frontend)
└── Makefile                    setup, migrate, revision helpers
```

## Artifacts And Data

Generated artifacts are intentionally kept out of the core source tree:

- `backend/artifacts/<run_id>/` - tracked videos, extraction summaries, and capped per-frame visualization PNGs.
- `backend/input/` - local source-video storage for `CLIPSCRIBE_STORAGE_BACKEND=local` (and the download scratch dir under `gcs`); uploaded objects are named by opaque storage keys and tracked in the `videos` registry.
- `backend/parser_artifacts/` - generated parser reports and scores.
- `backend/data/` - local database files.
- `backend/logs/` - runtime logs.

Do not hardcode absolute paths to these directories. Use project-relative paths or configuration values. The `artifacts.max_artifact_files` setting caps per-frame PNGs only; `tracked_output.mp4` and `extraction_summary.json` are always kept. Remote upload is not a config flag — it follows `CLIPSCRIBE_STORAGE_BACKEND`: under `gcs`, a finished run's `tracked_output.mp4` is uploaded loose (served via signed URL) and the remaining debug artifacts are bundled into `artifacts.tar.gz`, both under `artifacts/<run_id>/` in the bucket; under `local` everything stays on disk.

## Important
- `main.py` is hardcoded. it's not a real cli, this script is intend to be an entry point for local runs.


## AI Agent Notes

For AI coding-agent instructions, see:

- `CLAUDE.md`
- `AGENTS.md`

Those files include stricter rules about third-party code boundaries, typing expectations, and avoiding unnecessary new code.

## Third-Party Code

This project includes third-party components:

- SAM2 by Meta Platforms, Inc. (Apache License 2.0): <https://github.com/facebookresearch/segment-anything-2>
- GroundingDINO (Apache License 2.0 / MIT): <https://github.com/IDEA-Research/GroundingDINO>

Their respective licenses are included in the source tree.

### Deep-Dive Docs (`docs/`)
Diagram-rich explanations of the core mechanics:
- [SAM2 tracking mechanism](docs/sam2-tracking-and-identity.md) - Explains how SAM2 tracks the objects and how ClipScribe merges similar objects across the scenes
- [Extractor core algorithm](docs/extractor-core-algorithm.md) - ClipScribe extractor core algorithm
