# ClipScribe

ClipScribe is a multimodal video processing pipeline that extracts and structures visual, textual, and audio information from videos. It combines scene understanding, object detection and tracking, OCR, speech transcription, face detection, persistence, and platform-specific evaluation.

## Overview

ClipScribe splits a video into scenes, detects and tracks objects across shots, transcribes speech, extracts on-screen text, and assembles the result into structured metadata. A parser layer can then evaluate the extracted data against platform-specific criteria, such as YouTube ad requirements.

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

ClipScribe is a monorepo: the Python backend lives in `backend/`, while the git root is the repository root. **Run `uv`, Alembic, and pre-commit commands below from the `backend/` directory** so project config and relative paths resolve correctly.

```bash
cd backend
```

ClipScribe requires Python 3.12 or newer.

Install project dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --extra dev
```

Download checkpoints and pre-cache large models only when you need to run extraction. The helper scripts live in `backend/checkpoints/`, but the root `Makefile` setup/checkpoint targets still reference pre-monorepo paths and need cleanup before use.

## Environment

Common environment variables:

- `OPENAI_API_KEY` - required for GPT scene analysis, taxonomy generation, and parser agents.
- `POSTGRESQL_URL` - required when `database.backend` is set to `postgresql`.
- `SQLITE_URL` - optional when `database.backend` is set to `sqlite`; defaults to `sqlite:///data/clip_scribe.db`.
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

### Local Scratch Entry Point

The current `main.py` is a temporary hardcoded entry point, not a stable CLI. It is useful for local experiments, but video names, mode, run id, platform parameters, and device settings are currently edited in the file.

Current modes:

- `extract` - run extraction and write local artifacts only; it does not persist run metadata to the database.
- `parse` - evaluate an existing persisted run id.
- `full` - run extraction and then parse the saved run.

Example:

```bash
uv run python main.py
```

A proper CLI is still a TODO. Until then, prefer changing existing builder/engine code rather than adding one-off scripts for each run.

### Web API

The checked-in FastAPI app supports two job backends selected by `CLIPSCRIBE_JOB_BACKEND`. `inline` uses one long-lived `ClipScribeBuilder` and a single-slot in-process executor. `celery` keeps the API model-free and dispatches jobs to a Redis-backed Celery worker. Both backends return from `POST /jobs` immediately and publish best-effort live progress to a per-job Redis Stream served over SSE. Running jobs can be marked canceled, but the engine still finishes its current run before the terminal write is suppressed; cooperative mid-run interruption is still planned.

Start the API from `backend/`:

```bash
uv run uvicorn app.main:app --reload
```

For `CLIPSCRIBE_JOB_BACKEND=celery`, run a worker from `backend/` as well:

```bash
uv run celery -A app.celery_app worker --pool=solo --concurrency=1
```

Useful API routes:

- `POST /uploads` - upload one or more video files into `CLIPSCRIBE_INPUT_DIR`.
- `GET /inputs` - list server-side input videos accepted by the job form.
- `POST /jobs` - create a `full`, `extract`, or `parse` job; `parse` requires an existing `run_id`, and `extract` still writes artifacts only without creating a run row for `/runs`.
- `GET /jobs` and `GET /jobs/{job_id}` - poll job state.
- `GET /jobs/{job_id}/events` - SSE stream that replays and tails Redis Stream progress/log events.
- `GET /jobs/{job_id}/progress` - coarse percent summary for jobs-list progress bars.
- `POST /jobs/{job_id}/cancel` - cancel a queued job or mark a running job canceled.
- `POST /jobs/{job_id}/retry` - create a fresh job from a failed or canceled job's stored request payload.
- `DELETE /jobs/{job_id}` - remove a completed, failed, or canceled job row.
- `GET /runs/{run_id}/...` - inspect persisted run data, frame detections, parser results, and artifacts.
- `POST /runs/{run_id}/chat` and related chat session routes - stream read-only advisory Q&A over a completed run.

The API request does not accept a device field. The web app uses `CLIPSCRIBE_DEVICE` when constructing the inline API builder or Celery worker builder; `backend/main.py` still falls back to `clip_scribe.device` from `clip_scribe.yaml` when running locally.

### Frontend

The initial dashboard lives in `frontend/`. It is a Vite + React + TypeScript app using pnpm, TanStack Router, TanStack Query, Tailwind v4, and OpenAPI-generated types from the FastAPI schema.

Run it from the repository root with the API listening on port 8000:

```bash
cd frontend
pnpm install
pnpm gen:api
pnpm dev
```

The dev server runs at `http://localhost:5173` and proxies `/api/*` to the backend. Implemented screens are the jobs list (`/`) with running progress bars, the new-job form (`/jobs/new`), the live job page (`/jobs/{job_id}`), and the run inspector (`/runs/{run_id}`) with advisory chat.

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

Install only the slim API dependency group for API-container work. This is for the planned slim API image; advisory chat still needs `langgraph` and `langchain-openai` added before that image can serve chat requests:

```bash
uv sync --only-group api
```

Apply database migrations from the repository root via the Makefile:

```bash
make migrate
```

The root Makefile currently has stale setup/checkpoint/clean targets after the backend move; `make migrate` is the reliable target in the checked-in Makefile.

## Project Structure

```text
backend/src/clip_scribe/        Engine, builders, platform config, main app config
backend/app/                    FastAPI sync API, routes, settings, job runner
backend/src/extractor/          Scene extraction, taxonomy, tracking, scene description
backend/src/parser/             Parser agents, tools, evaluators, reports
backend/src/ocr/                PaddleOCR wrapper and OCR post-processing
backend/src/db/                 SQLAlchemy schema, engine, reader, writer
backend/src/utils/progress.py   Progress event interface and null reporter
backend/src/dino/dino_wrapper.py Safe wrapper around GroundingDINO
backend/alembic/                Alembic migration environment and versions
backend/checkpoints/            Model checkpoint download helpers
backend/input/                  Local input videos
backend/artifacts/              Per-run extraction outputs keyed by run id
backend/parser_artifacts/       Generated parser reports
backend/data/                   Local database files
backend/logs/                   Runtime logs
frontend/                       Vite/React dashboard and generated API types
```

## Artifacts And Data

Generated artifacts are intentionally kept out of the core source tree:

- `backend/artifacts/<run_id>/` - tracked videos, extraction summaries, and capped per-frame visualization PNGs.
- `backend/parser_artifacts/` - generated parser reports and scores.
- `backend/data/` - local database files.
- `backend/logs/` - runtime logs.

Do not hardcode absolute paths to these directories. Use project-relative paths or configuration values. The `artifacts.max_artifact_files` setting caps per-frame PNGs only; `tracked_output.mp4` and `extraction_summary.json` are always kept. `artifacts.remote_artifact_write` defaults to `false` and currently only logs a simulated GCS bundle upload.

## Current Caveats

- `main.py` is hardcoded. it's not a real cli, this script is intend to be an entry point for local runs.
- The FastAPI app has both inline and Celery dispatch paths with Redis Stream-backed SSE progress; cooperative mid-run cancellation is not implemented yet.
- Root Makefile setup/checkpoint/clean targets are stale after the backend move; `make migrate` is the current working target.
- Test coverage is minimal.
- Full extraction is resource-intensive and can trigger model downloads and API calls.

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
