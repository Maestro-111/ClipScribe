---
description: Inspect whether a ClipScribe database run exists and summarize it.
argument-hint: "<run_id>"
---

# Inspect DB Run

Use this workflow before parser work or when debugging persisted extraction data.

1. Get the `run_id` from `$ARGUMENTS`.
2. Check `database.backend` in `src/clip_scribe/configs/clip_scribe.yaml`.
3. Use existing `src/db/` reader APIs when practical instead of writing direct SQL.
4. If direct SQL is necessary, keep it read-only.
5. Do not modify database schema or data in this workflow.
6. Summarize whether the run exists, the video name/path/type, and which data categories appear populated.

Avoid dumping large JSON payloads unless the user asks for raw data.
