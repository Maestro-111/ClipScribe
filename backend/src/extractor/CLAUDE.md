# Module: Information Extractor

## Purpose
This is the core execution module of ClipScribe. It handles video iteration, scene detection, tracking logic, semantic resolution, and scene description generation.

## Key Files
* `extractor_core.py`: The `VideoInformationExtractor` class. It splits the video into shots, extracts frames, triggers DINO/SAM2/OCR, calculates bounding box tracking metrics (velocity, growth, centrality), resolves object identities across different shots using DINOv2 embeddings, emits structured progress events through `ProgressReporter`, and collects GPT scene descriptions per shot into `scene_description_registry`.
* `scene_describer.py`: The `GPTSceneDescriber` class. Uses GPT vision to analyze sampled frames from a shot and produce a narrative scene description plus a GroundingDINO detection prompt.
* `taxonomy_core.py`: Contains `TaxonomyGenerator` (LLM-based object target generation) and `TaxonomyResolver` (SBERT cosine-similarity matching to snap raw DINO labels to the canonical taxonomy).
* `taxonomy_config.py`: Configuration profiles mapping video types (e.g., "car ad") to specific LLM generation ratios and categories.

## Guidelines
* Modifications to object tracking metrics, IoU overlap rules, or identity merging thresholds happen in `extractor_core.py`.
* Modifications to how raw labels are filtered or categorized happen in `taxonomy_core.py`.
* Modifications to how scenes are described or how DINO prompts are generated happen in `scene_describer.py`.
* Progress reporting changes should use `backend/src/utils/progress.py` and keep the extractor independent of Redis or web-layer imports.
