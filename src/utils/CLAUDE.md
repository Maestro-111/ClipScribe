# Module: Utilities

## Purpose
Shared helper functions, global logging configurations, and standalone scripts.

## Key Files
* `clip_scribe_logging.py`: Configures the global `logger` instance used across all modules. Handles formatting and file output mapping to the `logs/` directory.
* `clib_scribe_video_manager.py`: Video file management utilities.

## Guidelines
* Place pure functions or cross-module helpers here to avoid circular imports between `extractor`, `dino`, and `ocr`.
* The database layer has been moved to `src/db/`. See that package for schema, engine, reader, and writer classes.
