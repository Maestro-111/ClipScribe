"""LangGraph tool functions for querying ClipScribe database."""

import json
from langchain_core.tools import tool
from src.db import ClipScribeReaderDB

TOOL_GROUP_TABLES: dict[str, list[str]] = {
    "audio": ["audio_segments", "scene_descriptions", "global_stats"],
    "text": ["text_events", "scene_descriptions", "global_stats"],
    "visual": ["visual_object_occurrences", "scene_descriptions", "global_stats"],
    "audio_text": [
        "audio_segments",
        "text_events",
        "scene_descriptions",
        "global_stats",
    ],
    "visual_text": [
        "visual_object_occurrences",
        "text_events",
        "scene_descriptions",
        "global_stats",
    ],
}


def build_tools(reader_db: ClipScribeReaderDB, run_id: str, tool_group: str) -> list:
    """
    Build LangGraph tools for a specific feature evaluation.

    Args:
        reader_db: Database reader instance
        run_id: Run identifier for the video
        tool_group: Tool group identifier (audio, text, visual, audio_text, visual_text)

    Returns:
        List of LangGraph tool functions relevant to the tool_group
    """

    # Define tool closures that capture reader_db and run_id
    @tool
    def query_audio_segments(max_start_time: float | None = None) -> str:
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
            segments = reader_db.get_audio_segments(run_id, max_start_time)
            return json.dumps(segments, indent=2)
        except Exception:
            return json.dumps([])

    @tool
    def query_text_events(max_second: int | None = None) -> str:
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
            events = reader_db.get_text_events(run_id, max_second)
            return json.dumps(events, indent=2)
        except Exception:
            return json.dumps([])

    @tool
    def query_visual_objects(
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
            objects = reader_db.get_visual_objects(
                run_id, label_contains, max_lifespan_start
            )
            return json.dumps(objects, indent=2)
        except Exception:
            return json.dumps([])

    @tool
    def query_global_stats() -> str:
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
            stats = reader_db.get_global_stats(run_id)
            if not stats:
                return json.dumps({})
            return json.dumps(stats, indent=2)
        except Exception:
            return json.dumps({})

    @tool
    def query_scene_descriptions(
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
            descriptions = reader_db.get_scene_descriptions(
                run_id, max_start_time, max_end_time
            )
            return json.dumps(descriptions, indent=2)
        except Exception:
            return json.dumps([])

    @tool
    def query_field_descriptions(table_name: str | None = None) -> str:
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
            descriptions = reader_db.get_field_descriptions(table_name)
            return json.dumps(descriptions, indent=2)
        except Exception:
            return json.dumps([])

    # Map tool groups to relevant tools
    tool_map = {
        "audio": [
            query_audio_segments,
            query_scene_descriptions,
            query_global_stats,
            query_field_descriptions,
        ],
        "text": [
            query_text_events,
            query_scene_descriptions,
            query_global_stats,
            query_field_descriptions,
        ],
        "visual": [
            query_visual_objects,
            query_scene_descriptions,
            query_global_stats,
            query_field_descriptions,
        ],
        "audio_text": [
            query_audio_segments,
            query_text_events,
            query_scene_descriptions,
            query_global_stats,
            query_field_descriptions,
        ],
        "visual_text": [
            query_visual_objects,
            query_text_events,
            query_scene_descriptions,
            query_global_stats,
            query_field_descriptions,
        ],
    }

    if tool_group not in tool_map:
        raise ValueError(f"Unknown tool_group: {tool_group}")

    return tool_map[tool_group]
