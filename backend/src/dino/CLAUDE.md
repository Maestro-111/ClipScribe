# Module: DINO Object Detection

## Purpose
This directory contains the GroundingDINO object detection wrapper.

## Key Files
* `dino_wrapper.py`: Wraps GroundingDINO. Takes a raw image and a text prompt, returning bounding boxes, raw string labels, and confidence scores. Includes logic to map and visualize these boxes.

## ⚠️ Important Note
* **DO NOT** edit or analyze files inside the `groundingdino/` subdirectory unless explicitly requested. It is a third-party repository clone.

## Notes
* Scene comprehension and prompt generation have moved to `src/extractor/scene_describer.py` (GPT vision).
