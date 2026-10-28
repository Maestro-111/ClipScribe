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
