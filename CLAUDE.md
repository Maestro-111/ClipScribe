# ClipScribe - AI Assistant Guidelines

## Project Overview
ClipScribe is a multimodal video processing pipeline that extracts and structures information from videos. It utilizes BLIP for scene comprehension, an LLM for dynamic taxonomy generation, GroundingDINO for object detection, SAM2 for object tracking, MTCNN for face detection, Whisper for audio transcription, and PaddleOCR for text extraction.

## ⚠️ CRITICAL RULES FOR AI AGENTS ⚠️
1. **Third-Party Code:** The directories `src/dino/groundingdino/` and `src/sam2/` contain third-party code (GroundingDINO and Meta's SAM 2, respectively).
   * **DO NOT** read, analyze, or modify the code in these directories unless explicitly instructed to do so by the user.
   * Assume these are black-box dependencies. Focus only on the wrapper classes (`src/dino/dino_wrapper.py` and the SAM2 builder).
2. **Read-Only vs Editable:** Focus your architectural suggestions and refactoring on `src/clip_scribe/`, `src/extractor/`, `src/ocr/`, and `src/parser/`.
3. **Artifacts:** Output artifacts (like json/csv/mp4s) are stored in `extractor_artifacts/`. Do not write code that assumes hardcoded absolute paths; always use relative paths or configurable paths from `configs/clip_scribe.yaml`.

## High-Level Pipeline Flow
1. **Engine (`src/clip_scribe/engine.py`)** initializes the pipeline.
2. **Extractor (`src/extractor/extractor.py`)** chunks the video into scenes.
3. **Prompt & Taxonomy:** BLIP generates a scene prompt -> LLM creates canonical targets (`src/extractor/taxonomy_core.py`).
4. **Detection & Tracking:** GroundingDINO detects raw objects -> SBERT maps them to the taxonomy -> SAM2 tracks them across frames.
5. **Parallel Tasks:** Whisper extracts audio, PaddleOCR extracts text, MTCNN extracts faces.
6. **Parser (`src/parser/`)** takes the final JSON output and processes it for end-use.
