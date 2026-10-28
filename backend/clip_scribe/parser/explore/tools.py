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

from clip_scribe.parser.shared_tools import build_run_data_tools

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
            return json.dumps(
                {key: payload.get(key) for key in _SECTION_KEYS[section]}, indent=2
            )
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
