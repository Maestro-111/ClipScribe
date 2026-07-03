# Module: SAM 2 (Segment Anything Model 2)

## Purpose
This directory contains the official, third-party implementation of Meta's Segment Anything Model 2 (SAM 2). In the ClipScribe pipeline, it is responsible for zero-shot video object segmentation and continuous frame-by-frame tracking of the bounding boxes initially detected by GroundingDINO.

## Directory Structure
* `sam/build_sam.py`: The primary factory script used by ClipScribe to instantiate the `sam2_video_predictor`.
* `configs/`: YAML configuration files (e.g., `sam2_hiera_t.yaml`) defining the model architectures.
* `image/`, `mask/`, `memory/`, `prompt/`: The core neural network components of the SAM 2 architecture (image encoders, memory attention mechanisms, etc.).

## ⚠️ CRITICAL RULES FOR AI AGENTS ⚠️
1. **Third-Party Black Box:** This entire directory is external, third-party code. **DO NOT** attempt to refactor, format, or optimize any files within `src/sam2/`.
2. **Debugging Protocol:** If a tracking issue arises, assume the bug is in the way `backend/src/extractor/extractor_core.py` formats the inputs (points, boxes, masks) or handles the `inference_state`, rather than a bug in the SAM 2 source code itself.
3. **Read-Only:** Only read these files if you specifically need to understand the exact expected shape, type, or format of arguments passed to functions like `propagate_in_video` or `add_new_points_or_box`.
