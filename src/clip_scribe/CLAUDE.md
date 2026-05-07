# Module: ClipScribe Engine & Builder

## Purpose
This directory contains the initialization and orchestration logic for the entire ClipScribe application. It is the entry point that ties all the individual AI models and processing steps together.

## Key Files
* `engine.py`: The main runtime loop. Handles the high-level `run()` method, triggering extraction, DB persistence, and parsing, and catching global exceptions.
* `build_clip_scribe.py`: The dependency injection/factory script. It reads `configs/clip_scribe.yaml`, allocates devices (CPU/MPS/CUDA), initializes all heavy models (SAM2, DINO, Whisper, etc.), and constructs the `VideoInformationExtractor`.
* `build_clip_scribe_plalform.py`: Builds the parser-side dependencies (reader DB, platform config, evaluator) for a given platform.
* `platform_configs/`: Pluggable platform configuration classes. `base.py` defines `BasePlatformConf`; `youtube.py` holds YouTube-specific settings.

## Guidelines
* When adding new models or dependencies to the pipeline, instantiate them in `build_clip_scribe.py` and pass them into the `VideoInformationExtractor`.
* When adding a new platform, create a config in `platform_configs/` extending `BasePlatformConf` and a corresponding evaluator in `src/parser/`.
* Ensure proper device mapping (MPS vs CPU vs CUDA) is maintained when adding new PyTorch models here.
