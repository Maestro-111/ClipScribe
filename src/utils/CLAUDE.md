# Module: Utilities

## Purpose
Shared helper functions, global logging configurations, database layer, and standalone scripts.

## Key Files
* `clip_scribe_logging.py`: Configures the global `logger` instance used across all modules. Handles formatting and file output mapping to the `logs/` directory.
* `clib_scribe_db.py`: SQLite persistence layer. `ClipScribeWriterDB` creates tables and writes extraction results (runs, global stats, visual objects, text events, audio segments, scene descriptions, field descriptions). `ClipScribeReaderDB` provides filtered read access used by parser tools.
* `clib_scribe_video_manager.py`: Video file management utilities.

## Guidelines
* Place pure functions or cross-module helpers here to avoid circular imports between `extractor`, `dino`, and `ocr`.
* When adding new data to the extraction output, add the corresponding table in `_create_tables()`, a writer method, and a reader method in `clib_scribe_db.py`.
