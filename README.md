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
- Per-run extraction artifacts and parser report generation
- FastAPI API for uploads, jobs, Redis Stream-backed live progress, run inspection, advisory chat, and artifact serving
- Inline or Redis-backed Celery job dispatch
- Vite/React dashboard for job submission, live job progress, run inspection, and post-run advisory chat

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
- `CLIPSCRIBE_DB_BACKEND` - optional override for `database.backend` from `clip_scribe.yaml` (`sqlite` | `postgresql`). The env var wins over yaml, so the Docker/compose stack and prod force `postgresql` without editing the config; local CLI runs keep the yaml default.
- `CLIPSCRIBE_INPUT_DIR` - optional API input directory relative to `backend/`; defaults to `input`.
- `CLIPSCRIBE_JOB_BACKEND` - `inline` (default single-slot in-process executor) or `celery` (Redis-backed worker dispatch).
- `REDIS_URL` - Redis connection used for Celery broker/result backend and per-job progress streams; defaults to `redis://localhost:6379/0`.
- `CLIPSCRIBE_DEVICE` - device used by the web API/worker builder (`cpu`, `mps`, or `cuda`); defaults to `cpu`.
- `CLIPSCRIBE_API_LOAD_MODELS` - set to `false` for limited schema/health/test startup without loading ML models; ignored in `celery` mode because the API loads only DB handles.
- `CLIPSCRIBE_CORS_ORIGINS` - comma-separated browser origins for the API; defaults to `http://localhost:5173`.

The main configuration file is:

```text
backend/src/clip_scribe/configs/clip_scribe.yaml
```

The current checked-in config uses SQLite by default. Switch `database.backend` to `postgresql` (and set `POSTGRESQL_URL`) to use PostgreSQL.

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

Running jobs can be marked canceled, but cooperative mid-run interruption is still planned. The API request has no device field — the builder device comes from `CLIPSCRIBE_DEVICE`. The dashboard screens are the jobs list (`/`), the new-job form (`/jobs/new`), the live job page (`/jobs/{job_id}`), and the run inspector (`/runs/{run_id}`) with advisory chat.

Useful API routes (reached from the browser under the `/api` proxy prefix):

- `POST /uploads` — upload one or more video files into `CLIPSCRIBE_INPUT_DIR`.
- `GET /inputs` — list server-side input videos accepted by the job form.
- `POST /jobs` — create a `full`, `extract`, or `parse` job; `parse` requires an existing `run_id`, and `extract` writes artifacts only (no `runs` row).
- `GET /jobs` and `GET /jobs/{job_id}` — poll job state.
- `GET /jobs/{job_id}/events` — SSE stream that replays and tails Redis Stream progress/log events.
- `GET /jobs/{job_id}/progress` — coarse percent summary for jobs-list progress bars.
- `POST /jobs/{job_id}/cancel` — cancel a queued job or mark a running job canceled.
- `POST /jobs/{job_id}/retry` — create a fresh job from a failed/canceled job's stored request.
- `DELETE /jobs/{job_id}` — remove a completed, failed, or canceled job row.
- `GET /runs/{run_id}/...` — inspect persisted run data, frame detections, parser results, and artifacts.
- `POST /runs/{run_id}/chat` and related chat routes — stream read-only advisory Q&A over a completed run.

### Local batch entry point (`main.py`)

`backend/main.py` is a temporary, hardcoded entry point (not a stable CLI) for local experiments — video name, mode, run id, platform params, and device are edited in the file. Modes:

- `extract` — run extraction and write local artifacts only (no DB run row); local dev only.
- `parse` — evaluate an existing persisted `run_id`.
- `full` — run extraction, then parse the saved run.

```bash
cd backend && uv run python main.py
```

Prefer changing existing builder/engine code over adding one-off run scripts.

## Running with Docker

Three images back the web app, all built from the repository root as context:

| Image | Dockerfile                           | Role                                                                                                                                                      |
| --- |--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `api` | `backend/docker/api/Dockerfile`      | Slim, torch-free FastAPI (REST + SSE, DB reads, artifact serving, Celery dispatch, advisory chat). Also runs the one-shot Alembic migration.              |
| `worker` | `backend/docker/core/cpu/Dockerfile` | Heavy Celery worker running the full pipeline. CPU build here; `gpu/Dockerfile` is the CUDA variant for a Linux + NVIDIA host (see `docs/deployment.md`). |
| `frontend` | `frontend/Dockerfile`                | Vite/React SPA built with pnpm and served by nginx, which reverse-proxies `/api/*` to the `api` service (SSE-safe).                                       |

`docker-compose.yml` wires these plus `postgres`, `redis`, and two one-shot services: `migrate` (`alembic upgrade head`) and `prewarm` (model-weight download). There are two ways to run.

### Mode 1 — Everything in Compose (CPU worker)

```bash
docker compose build
docker compose up      # api :8000, frontend :5173, worker, postgres, redis
```

The app is at `http://localhost:5173`. Model weights are fetched automatically: the one-shot `prewarm` service populates the shared `./backend/checkpoints` volume before the worker starts. The **first** `up` downloads several GB (the GroundingDINO/SAM2 `.pth` already on your host appear immediately); later `up`s short-circuit on a `.prewarm_complete` marker. Force a refetch with `docker compose run --rm prewarm python scripts/prewarm.py --force`.

The `worker`/`prewarm` images are built for **`linux/amd64`** (paddlepaddle has no arm64 wheel, and amd64 is the deploy target). On an **Apple Silicon** Mac they run under QEMU emulation — correct but slow, and `prewarm` loading models emulated can take a while. Mode 1 is best on a Linux/amd64 host or CI; on a Mac, use **Mode 2** for real work (native MPS worker) and treat Mode 1 as an end-to-end smoke test.

### Mode 2 — Hybrid local dev (native MPS worker)

Run only Postgres + Redis in Compose and run the API, worker, and frontend natively so the worker gets MPS. Use separate shells:

```bash
# shell 0 — infra only
docker compose up postgres redis

# shell 1 — migrations (once)
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
| `CLIPSCRIBE_DB_BACKEND` | unset (yaml `sqlite`) or `postgresql` | `postgresql` | `postgresql` |
| `CLIPSCRIBE_DEVICE` | `mps` | `cpu` | `cuda` |
| `CLIPSCRIBE_JOB_BACKEND` | `celery` | `celery` | `celery` |
| `OPENAI_API_KEY`, `LANGCHAIN_*` | secrets (from `.env`) | same (from `.env`) | Secret Manager |

The container-column values live in each service's `environment:` block in `docker-compose.yml` and win over `env_file`.

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

Install only the slim API dependency group for API-container work. The `api` group includes the advisory-chat LLM libraries (`langgraph`, `langchain-openai`) and resolves torch-free, so the API container stays slim while still serving `/runs/{id}/chat`:

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
│   ├── app/                    FastAPI: routes, settings, inline/Celery dispatch, Redis events, chat
│   ├── src/
│   │   ├── clip_scribe/        Engine, builder, platform configs, clip_scribe.yaml
│   │   ├── extractor/          Scene extraction, taxonomy, tracking, scene description
│   │   ├── parser/             LangGraph parser agents, query tools, evaluators, reports
│   │   ├── ocr/                PaddleOCR wrapper and box consolidation
│   │   ├── db/                 SQLAlchemy schema, engine, reader, writer
│   │   ├── dino/               GroundingDINO wrapper (+ vendored groundingdino, third-party)
│   │   ├── sam2/               Vendored SAM2 (third-party)
│   │   └── utils/              Progress, artifacts, ids, logging
│   ├── scripts/                prewarm.py + model-download helpers
│   ├── docker/
│   │   ├── api/                Slim, torch-free API image
│   │   └── core/{cpu,gpu}/     Heavy Celery worker images
│   ├── alembic/                Migration environment and versions
│   ├── checkpoints/            Model weights (gitignored; populated by prewarm)
│   ├── input/ artifacts/ parser_artifacts/ data/ logs/    Local I/O (gitignored)
│   ├── main.py                 Local batch entry point
│   └── pyproject.toml, uv.lock
├── frontend/
│   ├── src/
│   │   ├── routes/             Jobs list, new job, live job, run inspector
│   │   ├── api/                Generated OpenAPI types + client/hooks
│   │   ├── components/         ChatPanel and shared UI
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
- `backend/parser_artifacts/` - generated parser reports and scores.
- `backend/data/` - local database files.
- `backend/logs/` - runtime logs.

Do not hardcode absolute paths to these directories. Use project-relative paths or configuration values. The `artifacts.max_artifact_files` setting caps per-frame PNGs only; `tracked_output.mp4` and `extraction_summary.json` are always kept. `artifacts.remote_artifact_write` defaults to `false` and currently only logs a simulated GCS bundle upload.

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
