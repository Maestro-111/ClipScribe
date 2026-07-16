# ClipScribe - AI Assistant Guidelines

## Project Overview
ClipScribe is a multimodal video processing pipeline that extracts and structures information from videos. It uses GPT vision for scene comprehension, an LLM for dynamic taxonomy generation, GroundingDINO for object detection, SAM2 for object tracking, MTCNN for face detection, Whisper for audio transcription, PaddleOCR for OCR, and a parser layer for platform-specific evaluation.

## Critical Rules For AI Agents
1. **Third-Party Code:** The directories `backend/src/dino/groundingdino/` and `backend/src/sam2/` contain third-party code.
   * Do not read, analyze, or modify files in these directories unless the user explicitly asks.
   * Treat them as black-box dependencies. Work through wrappers such as `backend/src/dino/dino_wrapper.py` and the SAM2 builder imports.
2. **Primary Editable Areas:** Focus architectural suggestions and refactoring on `backend/app/`, `backend/src/clip_scribe/`, `backend/src/extractor/`, `backend/src/ocr/`, `backend/src/parser/`, and `backend/src/db/`.
3. **Artifacts:** Generated outputs live under `backend/artifacts/<run_id>/`, `backend/parser_artifacts/`, `backend/logs/`, and `backend/data/`. Do not hardcode absolute paths. Use project-relative paths or configuration from `backend/src/clip_scribe/configs/clip_scribe.yaml`.
4. **Expensive Operations:** Do not run full extraction, checkpoint downloads, root Makefile setup/checkpoint targets, model prefetches, or other network/API-heavy workflows unless the user asks for them.

## Engineering Principles
- Prefer modifying and extending existing code over adding new modules, classes, or parallel implementations. Add new code only when it fits an existing ownership boundary or clearly reduces complexity.
- Avoid tech debt from one-off helper files, duplicate abstractions, hardcoded paths, and task-specific scripts that should be configuration or CLI options.
- Keep changes narrowly scoped to the requested behavior and nearby code.
- Preserve existing public interfaces unless changing them is necessary and callers are updated.
- Follow Python typing best practices:
  - Add type annotations for public functions, constructors, and non-obvious internal helpers.
  - Prefer precise domain types, `TypedDict`, dataclasses, Pydantic models, or protocols over broad `dict`, `Any`, or untyped tuples.
  - Use `Any` only at external boundaries or where a library is genuinely untyped.
  - Avoid blanket `# type: ignore`; if needed, keep it narrow and explain why.
  - Keep mypy-relevant changes compatible with the project config in `pyproject.toml`.

## High-Level Pipeline Flow
1. **Builder and Engine:** `backend/src/clip_scribe/build_clip_scribe.py` reads config, initializes long-lived dependencies, and creates `ClipScribeEngine` from `backend/src/clip_scribe/engine.py`.
2. **Extraction:** `backend/src/extractor/extractor_core.py` chunks video into scenes and coordinates scene analysis, detection, tracking, OCR, audio, face detection, and progress events.
3. **Scene Description and Taxonomy:** `backend/src/extractor/scene_describer.py` samples frames and produces scene descriptions plus GroundingDINO prompts. `backend/src/extractor/taxonomy_core.py` generates canonical targets and maps raw labels to the taxonomy.
4. **Detection and Tracking:** GroundingDINO detects raw objects, SBERT resolves labels, and SAM2 tracks objects across frames.
5. **Parallel Tasks:** Whisper extracts audio, PaddleOCR extracts text, and MTCNN extracts faces.
6. **Persistence:** `backend/src/db/` writes and reads structured run data, including uploaded video registry rows, raw frame detections, shot boundaries, and parser results. Alembic migrations in `backend/alembic/` own schema creation.
7. **Parser:** `backend/src/parser/` evaluates persisted data against platform-specific criteria such as YouTube rules.
8. **Web API:** `backend/app/` exposes the FastAPI layer for uploads, parent/child batch job creation/polling, inline or Celery dispatch, Redis Stream-backed SSE progress, cooperative cancel/retry/delete, run reads, advisory chat, artifacts, health, and metadata.
9. **Frontend:** `frontend/` contains the Vite/React dashboard for job listing, job creation, live job progress, run inspection, and advisory chat. It uses pnpm, TanStack Router/Query, Tailwind v4, and OpenAPI-generated types.

## Setup And Environment
- **Repository layout:** This is a monorepo. The Python project lives in `backend/`, but the git root is the repository root. Run `uv ...`, Alembic, and pre-commit commands from `backend/` so relative paths like `pyproject.toml` and `src/clip_scribe` resolve. The root `Makefile` delegates project commands into `backend/` for migrations and model setup.
- Python requirement: `>=3.12`.
- Install project dependencies with `uv sync`.
- Install development dependencies with `uv sync --extra dev`.
- Required environment variables depend on the mode and database backend:
  - `OPENAI_API_KEY` for scene analysis, taxonomy generation, and parser agents.
  - `CLIPSCRIBE_DB_BACKEND` selects the database backend (`sqlite` or `postgresql`); default is `sqlite`. This is the only selector — `clip_scribe.yaml` carries no backend key, just the pool knobs (`pool_size`, `max_overflow`).
  - `POSTGRESQL_URL` when `CLIPSCRIBE_DB_BACKEND=postgresql`.
  - `SQLITE_URL` is optional when the backend is `sqlite`; default is `sqlite:///data/clip_scribe.db`.
  - `CLIPSCRIBE_JOB_BACKEND`, `REDIS_URL`, `CLIPSCRIBE_DEVICE`, `CLIPSCRIBE_INPUT_DIR`, `CLIPSCRIBE_VIDEO_STORAGE`, `CLIPSCRIBE_API_LOAD_MODELS`, and `CLIPSCRIBE_CORS_ORIGINS` for the FastAPI process and Celery worker.
- Main configuration is `backend/src/clip_scribe/configs/clip_scribe.yaml`.

## Commands
- `uv run pytest -q` - run tests.
- `uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser` - typecheck the editable core.
- `uv run alembic upgrade head` - apply schema migrations. Schema is owned by Alembic, not `metadata.create_all`.
- `uv run uvicorn app.main:app --reload` - start the FastAPI app. Inline mode loads heavy models unless `CLIPSCRIBE_API_LOAD_MODELS=false`; celery mode loads DB handles only.
- `uv run celery -A app.celery_app worker --pool=solo --concurrency=1` - start a local Celery worker for `CLIPSCRIBE_JOB_BACKEND=celery`.
- `uv run pre-commit run --all-files` - run formatting, lint, and type hooks. Must be invoked from `backend/`: the config is `backend/.pre-commit-config.yaml` (pre-commit discovers the config from the current directory), but hooks always execute from the git root with paths relative to it. This is why the `exclude` patterns are prefixed with `backend/` and the mypy hook is a local hook that `cd backend` before running `uv run mypy`. Running from the repo root fails with `.pre-commit-config.yaml is not a file`.
  - Corollary: the third-party `backend/src/sam2/` and `backend/src/dino/groundingdino/` trees are protected only by the `backend/`-prefixed `exclude`. If the layout changes, update that regex or `ruff --fix` will strip side-effect imports from their `__init__.py` files (notably the GroundingDINO model-registry imports) and break model loading.
- `make setup`, `make prewarm`, and `make checkpoints` - from the repository root, fetch or verify model assets under `backend/checkpoints/`; these are expensive and can download several GB.
- `make migrate` - from the repository root, applies SQLite migrations always and then attempts PostgreSQL using repo-root `.env` values, skipping PostgreSQL if unavailable or unset.
- Frontend commands run from `frontend/`: `pnpm install`, `pnpm gen:api`, `pnpm dev`, `pnpm build`, and `pnpm typecheck`.

## Current Caveats
- `backend/main.py` is an entry point to run the pipeline for local dev, not a stable CLI.
- `backend/app/` supports inline and Celery job dispatch plus Redis Stream-backed SSE progress. Running jobs stop cooperatively at engine/extractor/parser checkpoints after cancellation.
- Root Makefile model setup/checkpoint targets are working but expensive; avoid running or recommending them unless model downloads are explicitly in scope.
- The test suite is minimal.
- Generated media, databases, logs, and parser/extractor artifacts should usually be ignored during code review unless the task is about outputs.

## Module Notes
- `backend/src/clip_scribe/`: Orchestration, dependency construction, platform configs, and config loading.
- `backend/app/`: FastAPI routes, request/response models, runtime settings, inline/Celery job dispatch, Redis Stream events, and advisory chat.
- `backend/src/extractor/`: Video extraction, scene description, taxonomy generation/resolution, tracking metrics, and cross-shot identity logic.
- `backend/src/parser/`: LangGraph/LangChain parser agents, query tools, evaluator base classes, and YouTube evaluation.
- `backend/src/ocr/`: PaddleOCR wrapper and OCR box consolidation.
- `backend/src/db/`: SQLAlchemy schema, engine creation, reader, and writer, including the `videos` registry used by the input picker.
- `backend/src/utils/progress.py`: Progress event interface and null reporter used by CLI/tests and Redis fallback.
- `backend/src/utils/cancel.py`: Cooperative cancellation interface and null token used by CLI/tests and Redis fallback.
- `backend/src/utils/video_storage.py`: Source-video storage seam. Local storage is implemented; GCS is a fail-fast reserved backend.
- `backend/src/dino/dino_wrapper.py`: Safe wrapper around GroundingDINO.
- `backend/src/utils/`: Shared utility code. Treat SAM2-derived utility files cautiously and avoid refactors unless directly needed.
- `frontend/`: Vite/React dashboard, route files, API client/hooks, and generated OpenAPI TypeScript types.
