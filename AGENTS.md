# ClipScribe - AI Assistant Guidelines

## Project Overview
ClipScribe is a multimodal video processing pipeline that extracts and structures information from videos. It uses GPT vision for scene comprehension, an LLM for dynamic taxonomy generation, GroundingDINO for object detection, SAM2 for object tracking, MTCNN for face detection, Whisper for audio transcription, PaddleOCR for OCR, and a parser layer for platform-specific evaluation.

## Critical Rules For AI Agents
1. **Third-Party Code:** The directories `src/dino/groundingdino/` and `src/sam2/` contain third-party code.
   * Do not read, analyze, or modify files in these directories unless the user explicitly asks.
   * Treat them as black-box dependencies. Work through wrappers such as `src/dino/dino_wrapper.py` and the SAM2 builder imports.
2. **Primary Editable Areas:** Focus architectural suggestions and refactoring on `src/clip_scribe/`, `src/extractor/`, `src/ocr/`, `src/parser/`, and `src/db/`.
3. **Artifacts:** Generated outputs live in `extractor_artifacts/`, `parser_artifacts/`, `logs/`, and `data/`. Do not hardcode absolute paths. Use project-relative paths or configuration from `src/clip_scribe/configs/clip_scribe.yaml`.
4. **Expensive Operations:** Do not run full extraction, checkpoint downloads, `make setup`, model prefetches, or other network/API-heavy workflows unless the user asks for them.

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
1. **Builder and Engine:** `src/clip_scribe/build_clip_scribe.py` reads config, initializes dependencies, and creates `ClipScribeEngine` from `src/clip_scribe/engine.py`.
2. **Extraction:** `src/extractor/extractor_core.py` chunks video into scenes and coordinates scene analysis, detection, tracking, OCR, audio, and face detection.
3. **Scene Description and Taxonomy:** `src/extractor/scene_describer.py` samples frames and produces scene descriptions plus GroundingDINO prompts. `src/extractor/taxonomy_core.py` generates canonical targets and maps raw labels to the taxonomy.
4. **Detection and Tracking:** GroundingDINO detects raw objects, SBERT resolves labels, and SAM2 tracks objects across frames.
5. **Parallel Tasks:** Whisper extracts audio, PaddleOCR extracts text, and MTCNN extracts faces.
6. **Persistence:** `src/db/` writes and reads structured run data.
7. **Parser:** `src/parser/` evaluates persisted data against platform-specific criteria such as YouTube rules.

## Setup And Environment
- Python requirement: `>=3.12`.
- Install project dependencies with `uv sync`.
- Install development dependencies with `uv sync --extra dev`.
- Required environment variables depend on the mode and database backend:
  - `OPENAI_API_KEY` for scene analysis, taxonomy generation, and parser agents.
  - `POSTGRESQL_URL` when `database.backend` is `postgresql`.
  - `SQLITE_URL` is optional when `database.backend` is `sqlite`; default is `sqlite:///data/clip_scribe.db`.
- Main configuration is `src/clip_scribe/configs/clip_scribe.yaml`.

## Commands
- `uv run pytest -q` - run tests. Current repo state may collect no tests until coverage is added.
- `uv run mypy --config-file=pyproject.toml --explicit-package-bases src/clip_scribe src/extractor src/ocr src/parser` - typecheck the editable core.
- `uv run pre-commit run --all-files` - run formatting, lint, and type hooks.
- `make setup` - download checkpoints and pre-cache large models. Ask before running.
- `make checkpoints` - download DINO and SAM2 checkpoints. Ask before running.
- `make clean` - remove checkpoint files.

## Current Caveats
- `main.py` is a temporary hardcoded entry point, not a stable CLI.
- `make run_extractor` currently references `src.extractor.extractor`, but the implementation lives in `src/extractor/extractor_core.py`.
- `make help` currently prints only a header.
- The test suite is minimal.
- Generated media, databases, logs, and parser/extractor artifacts should usually be ignored during code review unless the task is about outputs.

## Module Notes
- `src/clip_scribe/`: Orchestration, dependency construction, platform configs, and config loading.
- `src/extractor/`: Video extraction, scene description, taxonomy generation/resolution, tracking metrics, and cross-shot identity logic.
- `src/parser/`: LangGraph/LangChain parser agents, query tools, evaluator base classes, and YouTube evaluation.
- `src/ocr/`: PaddleOCR wrapper and OCR box consolidation.
- `src/db/`: SQLAlchemy schema, engine creation, reader, and writer.
- `src/dino/dino_wrapper.py`: Safe wrapper around GroundingDINO.
- `src/utils/`: Shared utility code. Treat SAM2-derived utility files cautiously and avoid refactors unless directly needed.
