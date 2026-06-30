# Module: Utilities

## Purpose
Shared helper functions, global logging configurations, and standalone scripts.

## Key Files
* `clip_scribe_logging.py`: Configures the global `logger` instance used across all modules. Handles formatting and file output mapping to the `backend/logs/` directory.
* `clib_scribe_video_manager.py`: Video file management utilities.
* `progress.py`: Defines the structured progress event vocabulary, the `ProgressReporter` interface, phase helpers, and `NullProgressReporter` for CLI/tests.

## Guidelines
* Place pure functions or cross-module helpers here to avoid circular imports between `extractor`, `dino`, and `ocr`.
* The database layer lives in `backend/src/db/`. See that package for schema, engine, reader, and writer classes.
* Keep web or worker integrations behind utility interfaces so core pipeline modules do not import FastAPI, Celery, Redis, or SSE code directly.
