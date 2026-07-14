"""LangChain tool functions for querying ClipScribe database."""

import json
import logging
from langchain_core.tools import tool
from src.db import ClipScribeReaderDB

logger = logging.getLogger("clip_scribe")

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
        except Exception as e:
            logger.error("query_audio_segments failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

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
        except Exception as e:
            logger.error("query_text_events failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

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
        except Exception as e:
            logger.error("query_visual_objects failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

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
        except Exception as e:
            logger.error("query_global_stats failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": {}})

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
        except Exception as e:
            logger.error("query_scene_descriptions failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

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
        except Exception as e:
            logger.error("query_field_descriptions failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    def query_parser_results(
        feature_category: str | None = None, only_failed: bool = False
    ) -> str:
        """
        Query the platform ABCD evaluation verdicts already produced for this
        video. Use this to ground advice in what the evaluators concluded and why
        — especially to explain a failure or suggest how to fix it.

        Args:
            feature_category: Optional substring to filter by category (e.g. "Attract").
            only_failed: If True, return only criteria that did NOT pass.

        Returns:
            JSON string containing list of per-criterion results with fields:
            - feature_category: the ABCD category
            - feature_name: the criterion name
            - feature_criteria: the criterion's definition
            - evaluation: True (passed) / False (failed)
            - llm_explanation: the evaluator's reasoning for the verdict
        """
        try:
            results = reader_db.get_parser_results(run_id)
            filtered = []
            for r in results:
                if only_failed and r.get("evaluation"):
                    continue
                if (
                    feature_category
                    and feature_category.lower()
                    not in str(r.get("feature_category", "")).lower()
                ):
                    continue
                filtered.append(
                    {
                        "feature_category": r.get("feature_category"),
                        "feature_name": r.get("feature_name"),
                        "feature_criteria": r.get("feature_criteria"),
                        "evaluation": r.get("evaluation"),
                        "llm_explanation": r.get("llm_explanation"),
                    }
                )
            return json.dumps(filtered, indent=2)
        except Exception as e:
            logger.error("query_parser_results failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

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
        # Everything, plus the ABCD verdicts — for the advisory chat agent (§13).
        "advisory": [
            query_audio_segments,
            query_text_events,
            query_visual_objects,
            query_scene_descriptions,
            query_global_stats,
            query_field_descriptions,
            query_parser_results,
        ],
    }

    if tool_group not in tool_map:
        raise ValueError(f"Unknown tool_group: {tool_group}")

    return tool_map[tool_group]


def build_job_tools(reader_db: ClipScribeReaderDB, runs: list[dict]) -> list:
    """Build tools for the job-level advisory agent (web-app-plan §6 extension).

    Unlike :func:`build_tools`, these span *every* completed run in a batch job,
    so the agent can compare videos and reason about aggregate hit rates. Each
    ``runs`` entry is ``{"run_id": ..., "video_name": ...}``. Per-run detail
    tools take an explicit ``run_id`` argument that is validated against this set
    server-side, so the agent physically cannot read runs outside the job.

    Args:
        reader_db: Database reader instance.
        runs: Completed runs in the job, in submission order.

    Returns:
        A list of LangGraph tools scoped to this job's runs.
    """
    valid_ids = {r["run_id"] for r in runs}
    name_of = {r["run_id"]: r.get("video_name") or r["run_id"] for r in runs}

    def _resolve(run_id: str) -> str | None:
        """Return an error message if ``run_id`` is not one of the job's runs."""
        if run_id not in valid_ids:
            known = ", ".join(f"{rid} ({name_of[rid]})" for rid in valid_ids)
            return f"Unknown run_id '{run_id}'. Runs in this job: {known}"
        return None

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

    @tool
    def query_job_scorecard() -> str:
        """
        Aggregate ABCD pass rates across every run in this job. Use this to
        answer overall/comparative questions (e.g. "what's the hit rate across
        the job", "which video scored best", "which category is weakest").

        Returns:
            JSON object with fields:
            - overall: {passed, total, pass_rate} across all runs
            - by_run: list of {run_id, video_name, passed, total, pass_rate}
            - by_category: list of {feature_category, passed, total, pass_rate}
              summed across all runs
        """
        try:
            by_run = []
            cat_passed: dict[str, int] = {}
            cat_total: dict[str, int] = {}
            grand_passed = grand_total = 0
            for r in runs:
                rid = r["run_id"]
                results = reader_db.get_parser_results(rid)
                passed = sum(1 for x in results if x.get("evaluation"))
                total = len(results)
                grand_passed += passed
                grand_total += total
                for x in results:
                    cat = str(x.get("feature_category") or "")
                    cat_total[cat] = cat_total.get(cat, 0) + 1
                    if x.get("evaluation"):
                        cat_passed[cat] = cat_passed.get(cat, 0) + 1
                by_run.append(
                    {
                        "run_id": rid,
                        "video_name": name_of[rid],
                        "passed": passed,
                        "total": total,
                        "pass_rate": round(passed / total, 4) if total else None,
                    }
                )
            by_category = [
                {
                    "feature_category": cat,
                    "passed": cat_passed.get(cat, 0),
                    "total": cat_total[cat],
                    "pass_rate": round(cat_passed.get(cat, 0) / cat_total[cat], 4),
                }
                for cat in cat_total
            ]
            return json.dumps(
                {
                    "overall": {
                        "passed": grand_passed,
                        "total": grand_total,
                        "pass_rate": (
                            round(grand_passed / grand_total, 4)
                            if grand_total
                            else None
                        ),
                    },
                    "by_run": by_run,
                    "by_category": by_category,
                },
                indent=2,
            )
        except Exception as e:
            logger.error("query_job_scorecard failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}"})

    @tool
    def query_parser_results(
        run_id: str | None = None,
        feature_category: str | None = None,
        only_failed: bool = False,
    ) -> str:
        """
        Query the ABCD evaluation verdicts for the job. Omit run_id to get every
        run's verdicts (each row tagged with run_id + video_name); pass a run_id
        to focus on one video.

        Args:
            run_id: Optional run to restrict to (one of the job's runs).
            feature_category: Optional substring to filter by category.
            only_failed: If True, return only criteria that did NOT pass.

        Returns:
            JSON list of per-criterion results with fields: run_id, video_name,
            feature_category, feature_name, feature_criteria, evaluation,
            llm_explanation.
        """
        if run_id is not None and (err := _resolve(run_id)):
            return json.dumps({"error": err})
        target_ids = [run_id] if run_id else [r["run_id"] for r in runs]
        try:
            out = []
            for rid in target_ids:
                for r in reader_db.get_parser_results(rid):
                    if only_failed and r.get("evaluation"):
                        continue
                    if (
                        feature_category
                        and feature_category.lower()
                        not in str(r.get("feature_category", "")).lower()
                    ):
                        continue
                    out.append(
                        {
                            "run_id": rid,
                            "video_name": name_of[rid],
                            "feature_category": r.get("feature_category"),
                            "feature_name": r.get("feature_name"),
                            "feature_criteria": r.get("feature_criteria"),
                            "evaluation": r.get("evaluation"),
                            "llm_explanation": r.get("llm_explanation"),
                        }
                    )
            return json.dumps(out, indent=2)
        except Exception as e:
            logger.error("job query_parser_results failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    def query_scene_descriptions(run_id: str) -> str:
        """
        Query GPT-generated scene descriptions for one run's shots.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON list of scene descriptions (shot_index, start_time, end_time,
            description).
        """
        if err := _resolve(run_id):
            return json.dumps({"error": err})
        try:
            return json.dumps(reader_db.get_scene_descriptions(run_id), indent=2)
        except Exception as e:
            logger.error("job query_scene_descriptions failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    def query_visual_objects(run_id: str, label_contains: str | None = None) -> str:
        """
        Query visual object occurrences for one run.

        Args:
            run_id: Which run (one of the job's runs).
            label_contains: Optional substring to filter object labels.

        Returns:
            JSON list of visual objects (label, shot_index, lifespan, coverage,
            motion, quadrant, ...).
        """
        if err := _resolve(run_id):
            return json.dumps({"error": err})
        try:
            return json.dumps(
                reader_db.get_visual_objects(run_id, label_contains), indent=2
            )
        except Exception as e:
            logger.error("job query_visual_objects failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    def query_audio_segments(run_id: str) -> str:
        """
        Query audio transcript segments for one run.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON list of audio segments (start_time, end_time, text, confidence).
        """
        if err := _resolve(run_id):
            return json.dumps({"error": err})
        try:
            return json.dumps(reader_db.get_audio_segments(run_id), indent=2)
        except Exception as e:
            logger.error("job query_audio_segments failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    def query_text_events(run_id: str) -> str:
        """
        Query OCR on-screen text events for one run.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON list of text events (second, line_index, text).
        """
        if err := _resolve(run_id):
            return json.dumps({"error": err})
        try:
            return json.dumps(reader_db.get_text_events(run_id), indent=2)
        except Exception as e:
            logger.error("job query_text_events failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    @tool
    def query_global_stats(run_id: str) -> str:
        """
        Query global video statistics (pacing, shot count, duration) for one run.

        Args:
            run_id: Which run (one of the job's runs).

        Returns:
            JSON object of global stats, or {} if none.
        """
        if err := _resolve(run_id):
            return json.dumps({"error": err})
        try:
            return json.dumps(reader_db.get_global_stats(run_id) or {}, indent=2)
        except Exception as e:
            logger.error("job query_global_stats failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": {}})

    @tool
    def query_field_descriptions(table_name: str | None = None) -> str:
        """
        Query field descriptions to understand what database columns mean.

        Args:
            table_name: Optional table to filter (e.g. "visual_object_occurrences").

        Returns:
            JSON list of {table_name, column_name, description}.
        """
        try:
            return json.dumps(reader_db.get_field_descriptions(table_name), indent=2)
        except Exception as e:
            logger.error("job query_field_descriptions failed: %s", e)
            return json.dumps({"error": f"Database query failed: {e}", "data": []})

    return [
        list_job_runs,
        query_job_scorecard,
        query_parser_results,
        query_scene_descriptions,
        query_visual_objects,
        query_audio_segments,
        query_text_events,
        query_global_stats,
        query_field_descriptions,
    ]
