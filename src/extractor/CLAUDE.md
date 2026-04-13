# Module: Information Extractor

## Purpose
This is the core execution module of ClipScribe. It handles video iteration, scene detection, tracking logic, and semantic resolution.

## Key Files
* `extractor.py`: The `InformationExtractor` class. It splits the video into shots, extracts frames, triggers DINO/SAM2/OCR, calculates bounding box tracking metrics (velocity, growth, centrality), and resolves object identities across different shots using DINOv2 embeddings.
* `taxonomy_core.py`: Contains `TaxonomyGenerator` (LLM-based object target generation) and `TaxonomyResolver` (SBERT cosine-similarity matching to snap raw DINO labels to the canonical taxonomy).
* `taxonomy_config.py`: Configuration profiles mapping video types (e.g., "car ad") to specific LLM generation ratios and categories.

## Guidelines
* Modifications to object tracking metrics, IoU overlap rules, or identity merging thresholds happen in `extractor.py`.
* Modifications to how raw labels are filtered or categorized happen in `taxonomy_core.py`.
