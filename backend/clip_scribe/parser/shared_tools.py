"""Run-bound data tools shared by the ABCD and exploratory chat agents.

These read the extracted data of ONE run (speech, on-screen text, objects,
scenes, pacing stats) and know nothing about either analysis type. Each closes
over ``reader_db`` and ``run_id`` so the model cannot read another run, which is
why they are built here (decorated with the bound reader) rather than defined as
plain functions. ABCD and exploration each pick the ones they need from the
returned mapping and add their own result-reading tool.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from clip_scribe.db import ClipScribeAsyncReaderDB

logger = logging.getLogger("clip_scribe")


def build_field_descriptions_tool(reader_db: ClipScribeAsyncReaderDB) -> BaseTool:
    """The field-docs tool; it needs no run, so job-level agents share it too."""

    @tool
    async def query_field_descriptions(table_name: str | None = None) -> str:
        """
        Query field descriptions to understand the meaning of database columns.
        Call this before analyzing query results to understand what each field represents.

        Args:
            table_name: Optional table name to filter (e.g., "visual_object_occurrences",
                        "audio_segments", "text_events", "scene_descriptions", "global_stats").
                        If None, returns descriptions for all tables.

        Returns:
            JSON string containing list of field descriptions with fields:
            - table_name: the database table
            - column_name: the column in that table
            - description: human-readable explanation of what this field means and how to interpret its values
        """
        try:
            descriptions = await reader_db.get_field_descriptions(table_name)
            return json.dumps(descriptions, indent=2)
        except Exception as e:
            logger.error("ClipScribe Parser: query_field_descriptions failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    return query_field_descriptions


def build_run_data_tools(
    reader_db: ClipScribeAsyncReaderDB, run_id: str
) -> dict[str, BaseTool]:
    """Build the run-scoped extraction query tools, keyed by tool name."""

    @tool
    async def query_audio_segments(max_start_time: float | None = None) -> str:
        """
        Query audio transcript segments from the database.

        Args:
            max_start_time: Optional maximum start time in seconds (e.g., 5.0 for first 5 seconds)

        Returns:
            JSON string containing list of audio segments with fields:
            - id: segment ID
            - start_time: segment start time in seconds
            - end_time: segment end time in seconds
            - text: transcribed text
            - confidence: transcription confidence score
        """
        try:
            segments = await reader_db.get_audio_segments(run_id, max_start_time)
            return json.dumps(segments, indent=2)
        except Exception as e:
            logger.error("ClipScribe Parser: query_audio_segments failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    async def query_text_events(max_second: int | None = None) -> str:
        """
        Query OCR text events (on-screen text overlays) from the database.

        Args:
            max_second: Optional maximum second (e.g., 5 for first 5 seconds)

        Returns:
            JSON string containing list of text events with fields:
            - id: event ID
            - second: the second when text appears
            - line_index: line number within that second
            - text: extracted text content
        """
        try:
            events = await reader_db.get_text_events(run_id, max_second)
            return json.dumps(events, indent=2)
        except Exception as e:
            logger.error("ClipScribe Parser: query_text_events failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    async def query_visual_objects(
        label_contains: str | None = None, max_lifespan_start: float | None = None
    ) -> str:
        """
        Query visual object occurrences from the database.

        Args:
            label_contains: Optional substring to filter object labels (e.g., "person", "vehicle")
            max_lifespan_start: Optional maximum lifespan start time in seconds (e.g., 5.0 for first 5 seconds)

        Returns:
            JSON string containing list of visual objects with fields:
            - id: occurrence ID
            - global_id: object tracking ID across shots
            - label: object label/category
            - shot_index: shot number where object appears
            - lifespan_start: start time in seconds
            - lifespan_end: end time in seconds
            - screen_coverage: percentage of screen covered
            - velocity_px_sec: movement speed in pixels per second
            - growth_factor: size change factor
            - direction: movement direction
            - centrality_score: how centered the object is
            - screen_time_ratio: proportion of shot duration
            - quadrant: screen quadrant location
        """
        try:
            objects = await reader_db.get_visual_objects(
                run_id, label_contains, max_lifespan_start
            )
            return json.dumps(objects, indent=2)
        except Exception as e:
            logger.error("ClipScribe Parser: query_visual_objects failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    async def query_global_stats() -> str:
        """
        Query global video statistics from the database.

        Returns:
            JSON string containing global stats with fields:
            - total_shots: number of shots in video
            - video_duration: total duration in seconds
            - avg_shot_duration: average shot duration in seconds
            - dynamic_start_detected: whether dynamic start was detected (0/1)
            - dynamic_start_first_shot_dur: first shot duration
            - qp_intro_detected: whether quick pacing in intro was detected (0/1)
            - qp_intro_shot_count: number of shots in intro
            - qp_general_detected: whether general quick pacing was detected (0/1)
        """
        try:
            stats = await reader_db.get_global_stats(run_id)
            if not stats:
                return json.dumps({})
            return json.dumps(stats, indent=2)
        except Exception as e:
            logger.error("ClipScribe Parser: query_global_stats failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": {}})

    @tool
    async def query_scene_descriptions(
        max_start_time: float | None = None, max_end_time: float | None = None
    ) -> str:
        """
        Query GPT-generated scene descriptions for each shot from the database.

        Args:
            max_start_time: Optional maximum start time in seconds to filter scenes
            max_end_time: Optional maximum end time in seconds to filter scenes

        Returns:
            JSON string containing list of scene descriptions with fields:
            - shot_index: zero-based shot index
            - start_time: shot start time in seconds
            - end_time: shot end time in seconds
            - description: rich narrative scene description from GPT vision analysis
        """
        try:
            descriptions = await reader_db.get_scene_descriptions(
                run_id, max_start_time, max_end_time
            )
            return json.dumps(descriptions, indent=2)
        except Exception as e:
            logger.error("ClipScribe Parser: query_scene_descriptions failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    return {
        t.name: t
        for t in (
            query_audio_segments,
            query_text_events,
            query_visual_objects,
            query_global_stats,
            query_scene_descriptions,
            build_field_descriptions_tool(reader_db),
        )
    }


def unknown_run_error(runs: list[dict], run_id: str) -> str | None:
    """An error message if ``run_id`` is not one of the job's ``runs``, else None.

    ``runs`` is ``[{"run_id": ..., "video_name": ...}]``. Job-level tools call
    this on every ``run_id`` the model passes, so it cannot read outside the job.
    """
    names = {r["run_id"]: r.get("video_name") or r["run_id"] for r in runs}
    if run_id not in names:
        known = ", ".join(f"{rid} ({name})" for rid, name in names.items())
        return f"Unknown run_id '{run_id}'. Runs in this job: {known}"
    return None


def build_job_data_tools(
    reader_db: ClipScribeAsyncReaderDB, runs: list[dict]
) -> dict[str, BaseTool]:
    """Job-scoped extraction tools, keyed by tool name.

    ``list_job_runs`` plus per-run detail tools that take an explicit ``run_id``,
    validated against ``runs`` (``[{"run_id": ..., "video_name": ...}]``). Each
    wrapper delegates to the run-bound tools of :func:`build_run_data_tools`, so
    the query code exists once.
    """
    name_of = {r["run_id"]: r.get("video_name") or r["run_id"] for r in runs}

    @tool
    def list_job_runs() -> str:
        """
        List the videos (runs) in this job. Call this first to learn which runs
        exist and their run_ids before using any per-run tool.

        Returns:
            JSON list of objects with fields:
            - run_id: the run identifier to pass to per-run tools
            - video_name: the video's file name
        """
        return json.dumps(
            [{"run_id": r["run_id"], "video_name": name_of[r["run_id"]]} for r in runs],
            indent=2,
        )

    # One set of run-bound tools per job run; the wrappers below only add the
    # run_id argument and its membership check.
    per_run = {rid: build_run_data_tools(reader_db, rid) for rid in name_of}

    async def _per_run(run_id: str, tool_name: str, **kwargs: object) -> str:
        """Run a shared per-run tool for a run that belongs to this job."""
        if err := unknown_run_error(runs, run_id):
            return json.dumps({"error": err})
        return await per_run[run_id][tool_name].ainvoke(kwargs)

    @tool
    async def query_scene_descriptions(run_id: str) -> str:
        """
        Query GPT-generated scene descriptions for one run's shots.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON list of scene descriptions (shot_index, start_time, end_time,
            description).
        """
        return await _per_run(run_id, "query_scene_descriptions")

    @tool
    async def query_visual_objects(
        run_id: str, label_contains: str | None = None
    ) -> str:
        """
        Query visual object occurrences for one run.

        Args:
            run_id: Which run (one of the job's runs).
            label_contains: Optional substring to filter object labels.

        Returns:
            JSON list of visual objects (label, shot_index, lifespan, coverage,
            motion, quadrant, ...).
        """
        return await _per_run(
            run_id, "query_visual_objects", label_contains=label_contains
        )

    @tool
    async def query_audio_segments(run_id: str) -> str:
        """
        Query audio transcript segments for one run.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON list of audio segments (start_time, end_time, text, confidence).
        """
        return await _per_run(run_id, "query_audio_segments")

    @tool
    async def query_text_events(run_id: str) -> str:
        """
        Query OCR on-screen text events for one run.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON list of text events (second, line_index, text).
        """
        return await _per_run(run_id, "query_text_events")

    @tool
    async def query_global_stats(run_id: str) -> str:
        """
        Query global video statistics (pacing, shot count, duration) for one run.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON object of global stats, or {} if none.
        """
        return await _per_run(run_id, "query_global_stats")

    return {
        t.name: t
        for t in (
            list_job_runs,
            query_scene_descriptions,
            query_visual_objects,
            query_audio_segments,
            query_text_events,
            query_global_stats,
        )
    }
