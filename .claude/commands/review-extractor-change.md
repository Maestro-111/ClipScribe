---
description: Review extractor-related changes for correctness, cost, and integration risk.
argument-hint: "[optional files or branch]"
---

# Review Extractor Change

Use this workflow for changes touching extraction, scene description, taxonomy, OCR, tracking, audio, or face detection.

Review priorities:

1. Do not inspect or modify `src/dino/groundingdino/**` or `src/sam2/**`.
2. Confirm changes reuse existing extractor, taxonomy, OCR, and wrapper APIs.
3. Check for hardcoded absolute paths, model names, thresholds, run ids, video names, and artifact paths.
4. Check Python typing quality and avoid broad `dict` or `Any` where a local model or typed structure already exists.
5. Check that expensive model/API calls are not added to import time or simple tests.
6. Verify generated artifacts remain under configured output directories.
7. Recommend focused tests with mocks instead of tests that load full ML models.

Lead with bugs and risks, then mention test gaps.
