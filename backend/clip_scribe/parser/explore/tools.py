"""LangChain tools for the exploratory run chat agent.

The agent reads the persisted ``ExploreReport`` first (cheap, already
summarized) and drops to the raw extracted data (the tools shared with ABCD, see
``parser/shared_tools.py``) only for detail the report does not carry, such as the
exact words spoken at a moment.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import BaseTool, tool

from clip_scribe.parser.explore.models import TextCategory
from clip_scribe.parser.shared_tools import (
    build_job_data_tools,
    build_run_data_tools,
    unknown_run_error,
)

if TYPE_CHECKING:
    from clip_scribe.db import ClipScribeAsyncReaderDB

logger = logging.getLogger("clip_scribe")

ReportSection = Literal[
    "overview", "metrics", "transcript", "on_screen_text", "visuals", "findings"
]

# Which ExploreReport fields each section returns. ``overview`` is the cheap
# entry point; the others are the full digests.
_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "overview": ("summary", "chapters"),
    "metrics": ("metrics",),
    "transcript": ("transcript",),
    "on_screen_text": ("on_screen_text",),
    "visuals": ("visuals",),
    "findings": ("findings",),
}


def _section(payload: dict, section: str) -> dict:
    """The ``ExploreReport`` fields that make up one report section."""
    return {key: payload.get(key) for key in _SECTION_KEYS[section]}


def _error(tool_name: str, exc: Exception) -> str:
    """JSON error payload for a failed tool call; also logs the failure."""
    logger.error("ClipScribe Explore: %s failed: %s", tool_name, exc)
    return json.dumps({"error": f"Database query failed: {exc}", "data": []})


def build_run_tools(reader_db: ClipScribeAsyncReaderDB, run_id: str) -> list[BaseTool]:
    """Build the read-only tools for one exploratory run.

    ``run_id`` is bound here, so the agent cannot read any other run.
    """

    @tool
    async def query_exploration_report(section: ReportSection = "overview") -> str:
        """
        Read the saved exploratory report for this video. Start here: it already
        summarizes everything and holds the numbers.

        Args:
            section: Which part to read:
                - "overview": overall summary and chapters (title, start/end seconds, summary)
                - "metrics": duration, shot count, average shot length, cuts per minute,
                  speech and on-screen-text coverage (0-1), words per minute, face coverage
                  (null if faces were not collected), and per-label object stats
                - "transcript": speech summary, topics, and key claims with their start seconds
                - "on_screen_text": summary and every distinct text card with its cleaned
                  text, raw OCR reading, category, and the time spans it was visible
                - "visuals": visual summary and tracked subjects with prominence,
                  screen time, and first-seen second
                - "findings": observations with their evidence timestamps (may be empty)

        Returns:
            JSON string with the requested part of the report. Fields that are null
            mean the video has no data of that kind (e.g. no speech).
        """
        try:
            row = await reader_db.get_exploration_result(run_id)
            if row is None:
                return json.dumps(
                    {"error": "No exploration report exists for this run."}
                )
            payload = row["payload"]
            return json.dumps(_section(payload, section), indent=2)
        except Exception as e:
            return _error("query_exploration_report", e)

    # Raw-data drill-down shared with the ABCD agent. Global stats and field docs
    # are left out: the report's metrics already cover them.
    data_tools = build_run_data_tools(reader_db, run_id)
    return [
        query_exploration_report,
        data_tools["query_audio_segments"],
        data_tools["query_text_events"],
        data_tools["query_visual_objects"],
        data_tools["query_scene_descriptions"],
    ]


def build_job_tools(
    reader_db: ClipScribeAsyncReaderDB, runs: list[dict]
) -> list[BaseTool]:
    """Build the read-only tools for the exploratory job-level agent.

    Unlike :func:`build_run_tools`, these span every completed run of a batch
    job so the agent can compare videos. ``runs`` is
    ``[{"run_id": ..., "video_name": ...}]``; any ``run_id`` the model passes is
    validated against it, so it cannot read outside the job. Cross-video numbers
    come straight from the saved reports; nothing here calls an LLM.
    """
    name_of = {r["run_id"]: r.get("video_name") or r["run_id"] for r in runs}

    async def payloads() -> dict[str, dict]:
        """Saved report of each run in the job that has one, keyed by run_id."""
        out: dict[str, dict] = {}
        for rid in name_of:
            row = await reader_db.get_exploration_result(rid)
            if row is not None:
                out[rid] = row["payload"]
        return out

    @tool
    async def query_job_overview() -> str:
        """
        Compare the videos in this job side by side: one row per video with its
        headline numbers and overview summary. Start here for any "which video
        is longest / fastest-paced / most talkative" or "what is each video
        about" question.

        Returns:
            JSON object with fields:
            - videos: list of {run_id, video_name, duration_sec, shot_count,
              avg_shot_sec, cuts_per_minute, speech_coverage, text_coverage
              (both 0-1 fractions of the runtime), words_per_minute (null if no
              speech), face_coverage (null if faces were not collected),
              chapters (count), summary}
            - runs_without_report: run_ids of runs that have no saved report
        """
        try:
            reports = await payloads()
            videos = []
            for rid, payload in reports.items():
                m = payload["metrics"]
                videos.append(
                    {
                        "run_id": rid,
                        "video_name": name_of[rid],
                        **{
                            key: m.get(key)
                            for key in (
                                "duration_sec",
                                "shot_count",
                                "avg_shot_sec",
                                "cuts_per_minute",
                                "speech_coverage",
                                "text_coverage",
                                "words_per_minute",
                                "face_coverage",
                            )
                        },
                        "chapters": len(payload.get("chapters") or []),
                        "summary": payload.get("summary"),
                    }
                )
            return json.dumps(
                {
                    "videos": videos,
                    "runs_without_report": [r for r in name_of if r not in reports],
                },
                indent=2,
            )
        except Exception as e:
            return _error("query_job_overview", e)

    @tool
    async def query_exploration_report(
        run_id: str, section: ReportSection = "overview"
    ) -> str:
        """
        Read one video's saved exploratory report.

        Args:
            run_id: Which run (one of the job's runs).
            section: Which part to read:
                - "overview": overall summary and chapters (title, start/end seconds, summary)
                - "metrics": duration, shot count, average shot length, cuts per minute,
                  speech and on-screen-text coverage (0-1), words per minute, face coverage
                  (null if faces were not collected), and per-label object stats
                - "transcript": speech summary, topics, and key claims with their start seconds
                - "on_screen_text": summary and every distinct text card with its cleaned
                  text, raw OCR reading, category, and the time spans it was visible
                - "visuals": visual summary and tracked subjects with prominence,
                  screen time, and first-seen second
                - "findings": observations with their evidence timestamps (may be empty)

        Returns:
            JSON string with the requested part of the report. Fields that are null
            mean the video has no data of that kind (e.g. no speech).
        """
        if err := unknown_run_error(runs, run_id):
            return json.dumps({"error": err})
        try:
            row = await reader_db.get_exploration_result(run_id)
            if row is None:
                return json.dumps(
                    {"error": "No exploration report exists for this run."}
                )
            return json.dumps(_section(row["payload"], section), indent=2)
        except Exception as e:
            return _error("query_exploration_report", e)

    @tool
    async def query_job_on_screen_text(
        category: TextCategory | None = None, contains: str | None = None
    ) -> str:
        """
        Search the on-screen text of every video in this job. Use it for "which
        videos show a price / CTA / URL" or "where does this phrase appear".

        Args:
            category: Optional text category to keep (e.g. "price", "cta", "url").
            contains: Optional case-insensitive substring of the cleaned text.

        Returns:
            JSON list of {run_id, video_name, text (cleaned), category,
            total_sec (seconds visible), first_seen_sec}, grouped by video.
        """
        try:
            wanted = contains.casefold() if contains else None
            out = []
            for rid, payload in (await payloads()).items():
                digest = payload.get("on_screen_text") or {}
                for card in digest.get("cards", []):
                    if category is not None and card["category"] != category.value:
                        continue
                    if wanted and wanted not in card["clean_text"].casefold():
                        continue
                    spans = card.get("spans") or []
                    out.append(
                        {
                            "run_id": rid,
                            "video_name": name_of[rid],
                            "text": card["clean_text"],
                            "category": card["category"],
                            "total_sec": card["total_sec"],
                            "first_seen_sec": spans[0]["start_sec"] if spans else None,
                        }
                    )
            return json.dumps(out, indent=2)
        except Exception as e:
            return _error("query_job_on_screen_text", e)

    @tool
    async def query_job_visual_subjects(label_contains: str | None = None) -> str:
        """
        Search the tracked visual subjects of every video in this job. Use it for
        "which videos feature a car / person" or to compare how prominent
        something is across videos. Empty for videos where objects were not tracked.

        Args:
            label_contains: Optional case-insensitive substring of the object label.

        Returns:
            JSON list of {run_id, video_name, label, prominence (high/medium/low),
            screen_time_sec, first_seen_sec}, grouped by video.
        """
        try:
            wanted = label_contains.casefold() if label_contains else None
            out = []
            for rid, payload in (await payloads()).items():
                digest = payload.get("visuals") or {}
                for subject in digest.get("subjects", []):
                    if wanted and wanted not in subject["label"].casefold():
                        continue
                    out.append(
                        {
                            "run_id": rid,
                            "video_name": name_of[rid],
                            **{
                                key: subject[key]
                                for key in (
                                    "label",
                                    "prominence",
                                    "screen_time_sec",
                                    "first_seen_sec",
                                )
                            },
                        }
                    )
            return json.dumps(out, indent=2)
        except Exception as e:
            return _error("query_job_visual_subjects", e)

    # Per-run raw-data drill-down shared with the ABCD job agent. Global stats and
    # field docs are left out: the reports' metrics already cover them.
    data = build_job_data_tools(reader_db, runs)
    return [
        data["list_job_runs"],
        query_job_overview,
        query_exploration_report,
        query_job_on_screen_text,
        query_job_visual_subjects,
        data["query_scene_descriptions"],
        data["query_visual_objects"],
        data["query_audio_segments"],
        data["query_text_events"],
    ]
