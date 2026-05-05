# ClipScribe

A multimodal video processing pipeline that extracts and structures visual, textual, and audio information from videos using state-of-the-art AI models.

## Overview

ClipScribe automatically splits a video into scenes, detects and tracks objects across shots, transcribes speech, extracts on-screen text, and assembles everything into a structured JSON summary. An LLM dynamically generates a detection taxonomy tailored to each video, and results are persisted to a SQLite database for querying across runs.

## Pipeline

1. **Scene Detection** -- the video is split into shots using content-based scene detection.
2. **Scene Comprehension** -- BLIP captions each shot to provide context.
3. **Dynamic Taxonomy** -- an LLM generates canonical detection targets from the caption and video type; SBERT maps raw labels to the taxonomy.
4. **Object Detection & Tracking** -- GroundingDINO detects objects per frame; SAM2 tracks them across frames with cross-shot identity resolution via DINOv2 embeddings.
5. **Face Detection** -- MTCNN detects faces in parallel.
6. **Audio Transcription** -- Whisper transcribes speech with per-segment confidence scores.
7. **OCR** -- PaddleOCR extracts on-screen text frame by frame.
8. **Persistence** -- results are saved as JSON and written to a SQLite database (`data/clip_scribe.db`).

## Features

- **Object Detection**: GroundingDINO with configurable text/box confidence thresholds
- **Object Tracking**: SAM2 segmentation and temporal tracking with per-object metrics (velocity, growth, centrality, screen coverage, quadrant)
- **Dynamic Taxonomy**: LLM-generated detection targets adapted to each video's content and type
- **Audio Transcription**: Whisper-based speech-to-text with confidence filtering
- **OCR**: PaddleOCR text recognition across video frames
- **Face Detection**: MTCNN-based face detection
- **Pacing Analysis**: Automatic detection of dynamic starts and quick-pacing segments
- **SQLite Persistence**: Queryable database of extraction results across multiple runs
- **Artifact Generation**: Saves detection results, OCR outputs, and tracked videos to `extractor_artifacts/`

## Configuration

Pipeline parameters are set in `src/clip_scribe/configs/clip_scribe.yaml`, including detection thresholds, model sizes, and the database path.

## Third-Party Code

This project includes third-party components:

- SAM2 by Meta Platforms, Inc. (Apache License 2.0) [SAM2](https://github.com/facebookresearch/segment-anything-2)
- GroundingDINO (Apache License 2.0 / MIT) [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

Their respective licenses are included in the source tree.
