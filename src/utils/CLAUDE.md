# Module: Utilities

## Purpose
Shared helper functions, global logging configurations, and standalone scripts.

## Key Files
* `clip_scribe_logging.py`: Configures the global `logger` instance used across all modules. Handles formatting and file output mapping to the `logs/` directory.

## Guidelines
* Place pure functions or cross-module helpers here to avoid circular imports between `extractor`, `dino`, and `ocr`.
