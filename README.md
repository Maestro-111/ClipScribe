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
- Extraction and parser artifact generation

## Setup

ClipScribe requires Python 3.12 or newer.

Install project dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --extra dev
```

Download checkpoints and pre-cache large models only when you need to run extraction:

```bash
make setup
```

`make setup` can download large model files and may take a while.

## Environment

Common environment variables:

- `OPENAI_API_KEY` - required for GPT scene analysis, taxonomy generation, and parser agents.
- `POSTGRESQL_URL` - required when `database.backend` is set to `postgresql`.
- `SQLITE_URL` - optional when `database.backend` is set to `sqlite`; defaults to `sqlite:///data/clip_scribe.db`.

The main configuration file is:

```text
src/clip_scribe/configs/clip_scribe.yaml
```

The current checked-in config uses PostgreSQL by default. Switch `database.backend` to `sqlite` for local SQLite runs.

## Running

The current `main.py` is a temporary hardcoded entry point, not a stable CLI. It is useful for local experiments, but video names, mode, run id, platform parameters, and device settings are currently edited in the file.

Current modes:

- `extract` - run extraction and persist metadata.
- `parse` - evaluate an existing persisted run id.
- `full` - run extraction and then parse the saved run.

Example:

```bash
uv run python main.py
```

A proper CLI is still a TODO. Until then, prefer changing existing builder/engine code rather than adding one-off scripts for each run.

## Development Commands

Run tests:

```bash
uv run pytest -q
```

Run type checks:

```bash
uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser
```

Run all pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

Clean downloaded checkpoints:

```bash
make clean
```

## Project Structure

```text
src/clip_scribe/        Engine, builders, platform config, main app config
src/extractor/          Scene extraction, taxonomy, tracking, scene description
src/parser/             Parser agents, tools, evaluators, reports
src/ocr/                PaddleOCR wrapper and OCR post-processing
src/db/                 SQLAlchemy schema, engine, reader, writer
src/dino/dino_wrapper.py Safe wrapper around GroundingDINO
checkpoints/            Model checkpoint download helpers
input/                  Local input videos
extractor_artifacts/    Generated extraction outputs
parser_artifacts/       Generated parser reports
data/                   Local database files
logs/                   Runtime logs
```

## Artifacts And Data

Generated artifacts are intentionally kept out of the core source tree:

- `extractor_artifacts/` - detection visualizations, OCR outputs, tracked videos, extraction summaries.
- `parser_artifacts/` - generated parser reports and scores.
- `data/` - local database files.
- `logs/` - runtime logs.

Do not hardcode absolute paths to these directories. Use project-relative paths or configuration values.

## Current Caveats

- `main.py` is hardcoded and should become a real CLI.
- `make run_extractor` is stale and points at a module that does not currently exist.
- `make help` currently prints only a header.
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

## TODO

- Replace the hardcoded `main.py` flow with a proper CLI.
- Improve Docker and long-running job execution.
- Add API-layer support for calling the CLI or container image.
- Add GCP support utilities for BigQuery and video download workflows.
- Expand tests around taxonomy, parser tools, database persistence, and platform evaluation.
