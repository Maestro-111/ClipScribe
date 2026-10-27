"""LangGraph agent construction, plus the one-shot runner for ABCD criteria."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError


from clip_scribe.parser.abcd.tools import (
    build_run_tools as abcd_build_async_tools,
    build_job_tools as abcd_build_job_tools,
)
from clip_scribe.parser.explore.tools import (
    build_run_tools as exploration_build_async_tools,
)

if TYPE_CHECKING:
    from clip_scribe.db import ClipScribeAsyncReaderDB
    from langgraph.graph.state import CompiledStateGraph
    from clip_scribe.parser.abcd.models import BaseAgentEvaluation

logger = logging.getLogger("clip_scribe")

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|neither|isn't|isn't|wasn't|aren't|don't)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_RE = re.compile(r"\b(?:true|yes)\b", re.IGNORECASE)


class AgentBuilder:
    """Builds ReAct agents: a generic builder and the chat agents for each
    analysis type (ABCD run/job, exploration run).
    """

    @staticmethod
    def build_agent(model: ChatOpenAI, tools: list) -> CompiledStateGraph:
        """
        Build a LangGraph ReAct agent with the specified model and tools.

        Args:
            model: ChatOpenAI model instance
            tools: List of LangGraph tool functions

        Returns:
            Compiled LangGraph agent
        """
        return create_react_agent(model, tools)

    @classmethod
    def build_abcd_run_agent(
        cls, model: ChatOpenAI, reader_db: ClipScribeAsyncReaderDB, run_id: str
    ) -> CompiledStateGraph:
        """Build a ReAct agent scoped to ``run_id`` with the advisory tool set."""
        tools = abcd_build_async_tools(reader_db, run_id, tool_group="advisory")
        return cls.build_agent(model, tools)

    @classmethod
    def build_abcd_job_agent(
        cls, model: ChatOpenAI, reader_db: ClipScribeAsyncReaderDB, runs: list[dict]
    ) -> CompiledStateGraph:
        """Build a ReAct agent spanning every completed run in one batch job.

        ``runs`` is ``[{"run_id": ..., "video_name": ...}]`` — the job's completed
        runs. Tools are bound to that set server-side, so the agent cannot read runs
        outside the job.
        """
        tools = abcd_build_job_tools(reader_db, runs)
        return cls.build_agent(model, tools)

    @classmethod
    def build_exploration_run_agent(
        cls, model: ChatOpenAI, reader_db: ClipScribeAsyncReaderDB, run_id: str
    ) -> CompiledStateGraph:
        """Build a ReAct agent scoped to one exploratory ``run_id``."""
        tools = exploration_build_async_tools(reader_db, run_id)
        return cls.build_agent(model, tools)


class AgentRunner:
    """Runs a built agent on one ABCD criterion and parses its answer into a
    structured evaluation.
    """

    def __init__(self, agent: CompiledStateGraph) -> None:
        """Args:
        agent: Compiled ReAct agent (tools already bound).
        """
        self.agent = agent

    def run_abcd_agent(
        self,
        question: str,
        instructions: str,
        *,
        agentic_eval: type[BaseAgentEvaluation],
        platform_context: str = "video criteria",
        time_scope: float | None = None,
        field_context: str | None = None,
        recursion_limit: int = 25,
        run_name: str | None = None,
    ) -> BaseAgentEvaluation:
        """
        Run the agent with a question and instructions, returning structured evaluation.

        Args:
            question: The evaluation question
            instructions: Additional instructions for the agent
            agentic_eval: The BaseAgentEvaluation subclass to construct the result with
            platform_context: Platform-specific context for the system prompt (default: "video criteria")
            time_scope: Optional time window in seconds. When set, the agent restricts
                         its evaluation to the first N seconds of the video.
            field_context: Optional pre-formatted string describing database fields available
                           to the agent. Injected into the system prompt so the agent
                           understands what each column means before querying.
            recursion_limit: how many reasoning step?
            run_name: Optional name for the LangSmith trace (e.g., the feature id).

        Returns:
            BaseAgentEvaluation with evaluation (bool) and explanation (str)
        """

        time_scope_text = ""
        if time_scope is not None:
            time_scope_text = f"""
    IMPORTANT — TIME SCOPE RESTRICTION:
    This evaluation applies ONLY to the first {time_scope} seconds of the video.
    You MUST use time filter parameters when calling tools:
    - query_audio_segments: set max_start_time={time_scope}
    - query_text_events: set max_second={int(time_scope)}
    - query_visual_objects: set max_lifespan_start={time_scope}
    - query_scene_descriptions: set max_start_time={time_scope}
    Ignore any data beyond the {time_scope}-second mark."""

        field_context_text = ""
        if field_context:
            field_context_text = f"""
    DATABASE FIELD REFERENCE:
    Use this reference to understand the meaning and value ranges of fields returned by your query tools.
    Analyze these fields explicitly when reasoning about the criteria.
    {field_context}"""

        system_message = f"""You are an expert video analyst evaluating {platform_context}.
    {time_scope_text}
    {field_context_text}
    Your task:
    1. Query the database using the provided tools to gather relevant information
    2. Analyze the results to answer the question — pay attention to quantitative fields and their values
    3. Return your evaluation as JSON with two fields:
       - "evaluation": true or false
       - "explanation": a concise explanation (1-3 sentences) of your reasoning, citing specific field values when relevant

    Question: {question}

    Instructions:
    {instructions}

    Use the tools to query the database, then provide your structured evaluation."""

        # Stream the agent so we always have access to the latest state,
        # even if the recursion limit is hit before the agent finishes.
        messages: list = []
        try:
            stream_config: RunnableConfig = {"recursion_limit": recursion_limit}
            if run_name:
                stream_config["run_name"] = run_name
            for chunk in self.agent.stream(
                {
                    "messages": [
                        {"role": "system", "content": system_message},
                    ]
                },
                stream_config,
                stream_mode="values",
            ):
                messages = chunk["messages"]
        except GraphRecursionError:
            logger.warning(
                "ClipScribe Parser: Agent hit recursion limit (%d). Using last collected "
                "message as final answer.",
                recursion_limit,
            )

        # Find the last AI text message (skip tool calls)
        final_message = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_message = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                break

        if not final_message:
            return agentic_eval(
                evaluation=False,
                explanation="Agent did not produce a final evaluation.",
            )

        return self._parse_abcd_agent_response(final_message, agentic_eval)

    @staticmethod
    def _parse_abcd_agent_response(
        final_message: str, agentic_eval: type[BaseAgentEvaluation]
    ) -> BaseAgentEvaluation:
        """Extract a structured evaluation from the agent's last text message."""
        try:
            if "{" in final_message and "}" in final_message:
                start = final_message.find("{")
                end = final_message.rfind("}") + 1
                parsed = json.loads(final_message[start:end])

                evaluation = bool(parsed.get("evaluation", False))
                explanation = str(parsed.get("explanation", "No explanation provided"))
                return agentic_eval(evaluation=evaluation, explanation=explanation)

            logger.info(
                "ClipScribe Parser: JSON parsing failed; falling back to heuristic text matching."
            )
            has_affirmative = bool(_AFFIRMATIVE_RE.search(final_message))
            has_negation = bool(_NEGATION_RE.search(final_message))
            evaluation = has_affirmative and not has_negation
            return agentic_eval(evaluation=evaluation, explanation=final_message[:500])

        except Exception as e:
            return agentic_eval(
                evaluation=False,
                explanation=f"Error parsing agent response: {str(e)}. Raw: {final_message[:200]}",
            )
