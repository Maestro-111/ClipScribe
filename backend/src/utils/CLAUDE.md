# Module: Utilities

## Purpose
Shared helper functions, global logging configurations, and standalone scripts.

## Key Files
* `clip_scribe_logging.py`: Provides `configure_logging()` for ClipScribe's application logging. Editable modules retrieve `logging.getLogger("clip_scribe")` locally; the config handles formatting and file output mapping to the `backend/logs/` directory.
* `progress.py`: Defines the structured progress event vocabulary, the `ProgressReporter` interface, phase helpers, and `NullProgressReporter` for CLI/tests.
* `clip_scribe_artifacts.py`: Defines the `artifacts/<run_id>/` directory convention and the no-op/simulated remote artifact upload seam.
* `clip_scribe_cancel.py`: Defines the cooperative cancellation seam and null token used by CLI/tests.
* `clip_scribe_video_storage.py`: Defines the source-video storage seam. `LocalVideoStorage` stores opaque keys under `CLIPSCRIBE_INPUT_DIR`; `GCSVideoStorage` is a fail-fast reserved backend documenting the cloud contract.
* `ids.py`: Generates ULID run ids for extraction jobs.

## Guidelines
* Place pure functions or cross-module helpers here to avoid circular imports between `extractor`, `dino`, and `ocr`.
* The database layer lives in `backend/src/db/`. See that package for schema, engine, reader, and writer classes.
* Keep web or worker integrations behind utility interfaces so core pipeline modules do not import FastAPI, Celery, Redis, or SSE code directly.
