# Module: Utilities

## Purpose
Shared helper functions, global logging configurations, and standalone scripts.

## Key Files
* `clip_scribe_logging.py`: Provides `configure_logging()` for ClipScribe's application logging. Pipeline modules retrieve `logging.getLogger("clip_scribe")` locally and the web layer (`backend/app/`) uses `logging.getLogger("api")`; the config handles formatting and file output mapping to the `backend/logs/` directory. `configure_logging(levels)` also takes per-logger levels (e.g. `{"api": 10}`, applied on every call); the API's comes from `CLIPSCRIBE_LOG_API_LEVEL` (number or name, default INFO).
* `progress.py`: Defines the structured progress event vocabulary, the `ProgressReporter` interface, phase helpers, and `NullProgressReporter` for CLI/tests.
* `clip_scribe_artifacts.py`: Defines the `artifacts/<run_id>/` directory convention and the artifact storage seam. `NullArtifactUploader` (local) is a no-op; `GCSArtifactUploader` uploads `tracked_output.mp4` loose plus an `artifacts.tar.gz` bundle and signs the tracked video for serving. Backend follows `CLIPSCRIBE_STORAGE_BACKEND`.
* `clip_scribe_cancel.py`: Defines the cooperative cancellation seam and null token used by CLI/tests.
* `ids.py`: Generates ULID run ids for extraction jobs.

## Guidelines
* Place pure functions or cross-module helpers here to avoid circular imports between `extractor`, `dino`, and `ocr`.
* The database layer lives in `backend/clip_scribe/db/`. See that package for schema, engine, reader, and writer classes.
* Keep web or worker integrations behind utility interfaces so core pipeline modules do not import FastAPI, Celery, Redis, or SSE code directly.
