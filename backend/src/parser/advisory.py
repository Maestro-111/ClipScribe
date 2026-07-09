"""Advisory chat agent — post-run Q&A over a single video (web-app-plan §13).

Same LangGraph ReAct machinery as the evaluators (``agent.py`` / ``tools.py``),
with three differences: it gets the ``"advisory"`` tool group (every query tool
plus ``query_parser_results``), it is conversational rather than one-shot, and it
gives free-form guidance instead of a structured pass/fail.

The agent is read-only and strictly scoped to one ``run_id`` — every tool is
bound to that id server-side, so it physically cannot read another run's data.
It does only LLM calls + DB reads, so it runs in the API process (no worker, no
models, no torch).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.db import ClipScribeReaderDB
from src.parser.tools import build_tools

ADVISORY_SYSTEM_PROMPT = """You are a senior creative strategist helping a user \
understand and improve ONE advertising video that has already been analyzed and \
scored against the platform's ABCD criteria.

You have read-only tools to inspect this specific video's data:
- query_parser_results: the ABCD verdicts (pass/fail) and the evaluator's \
reasoning for each criterion. Start here when the user asks why something \
failed or how to improve.
- query_scene_descriptions, query_visual_objects, query_audio_segments, \
query_text_events, query_global_stats: the underlying extracted facts \
(shots, tracked objects, speech, on-screen text, pacing).
- query_field_descriptions: what a database field means, if you are unsure.

How to work:
1. Fetch the relevant data with tools before answering — never invent facts, \
verdicts, timings, or object labels that a tool did not return.
2. Ground every claim in specifics: cite the criterion, the verdict, the \
evaluator's stated reason, and concrete evidence (e.g. "the logo first appears \
at 6.2s", "no CTA text was detected").
3. When asked how to fix a failed criterion, give concrete, testable changes \
(what to add/move/cut and roughly when), tied to why the criterion failed.
4. Be concise and direct. If the data is insufficient to answer, say so rather \
than guessing.

You are limited to THIS video. If asked to compare to other videos or campaigns, \
explain that you only have access to this run's data."""


def build_advisory_agent(model: ChatOpenAI, reader_db: ClipScribeReaderDB, run_id: str):
    """Build a ReAct agent scoped to ``run_id`` with the advisory tool set."""
    tools = build_tools(reader_db, run_id, tool_group="advisory")
    return create_react_agent(model, tools)
