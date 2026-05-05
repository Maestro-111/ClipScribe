# Module: ClipScribe Engine & Builder

## Purpose
This directory contains the initialization and orchestration logic for the entire ClipScribe application. It is the entry point that ties all the individual AI models and processing steps together.

## Key Files
* `engine.py`: The main runtime loop. Handles the high-level `run()` method, triggering extraction and parsing, and catching global exceptions.
* `build_clip_scribe.py`: The dependency injection/factory script. It reads `configs/clip_scribe.yaml`, allocates devices (CPU/MPS/CUDA), initializes all heavy models (SAM2, DINO, Whisper, etc.), and constructs the `InformationExtractor`.

## Guidelines
* When adding new models or dependencies to the pipeline, instantiate them in `build_clip_scribe.py` and pass them into the `InformationExtractor`.
* Ensure proper device mapping (MPS vs CPU vs CUDA) is maintained when adding new PyTorch models here.
