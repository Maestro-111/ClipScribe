# Module: DINO & Prompt Generation

## Purpose
This directory contains the visual understanding and object detection wrappers. 

## Key Files
* `dino_wrapper.py`: Wraps GroundingDINO. Takes a raw image and a text prompt, returning bounding boxes, raw string labels, and confidence scores. Includes logic to map and visualize these boxes.
* `dino_prompt.py`: Wraps BLIP. Generates a descriptive sentence of a frame, parses it with Spacy to extract physical nouns, and returns a clean DINO-compatible prompt (e.g., "car . tree . road .").

## ⚠️ Important Note
* **DO NOT** edit or analyze files inside the `groundingdino/` subdirectory unless explicitly requested. It is a third-party repository clone.