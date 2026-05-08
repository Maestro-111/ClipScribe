"""LangGraph ReAct agent for agentic feature evaluation."""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from src.parser.models import BaseAgentEvaluation


def _parse_agent_response(
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

        evaluation = "true" in final_message.lower() or "yes" in final_message.lower()
        return agentic_eval(evaluation=evaluation, explanation=final_message[:500])

    except Exception as e:
        return agentic_eval(
            evaluation=False,
            explanation=f"Error parsing agent response: {str(e)}. Raw: {final_message[:200]}",
        )


def build_agent(model: ChatOpenAI, tools: list):
    """
    Build a LangGraph ReAct agent with the specified model and tools.

    Args:
        model: ChatOpenAI model instance
        tools: List of LangGraph tool functions

    Returns:
        Compiled LangGraph agent
    """
    return create_react_agent(model, tools)


def run_agent(
    agent,
    question: str,
    instructions: str,
    agentic_eval: type[BaseAgentEvaluation],
    platform_context: str = "video criteria",
    time_scope: float | None = None,
    recursion_limit: int = 25,
) -> BaseAgentEvaluation:
    """
    Run the agent with a question and instructions, returning structured evaluation.

    Args:
        agent: Compiled LangGraph agent
        question: The evaluation question
        instructions: Additional instructions for the agent
        agentic_eval: The BaseAgentEvaluation subclass to construct the result with
        platform_context: Platform-specific context for the system prompt (default: "video criteria")
        time_scope: Optional time window in seconds. When set, the agent restricts
                     its evaluation to the first N seconds of the video.
        recursion_limit: how many reasoning step?

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

    system_message = f"""You are an expert video analyst evaluating {platform_context}.
{time_scope_text}
Your task:
1. Query the database using the provided tools to gather relevant information
2. Analyze the results to answer the question
3. Return your evaluation as JSON with two fields:
   - "evaluation": true or false
   - "explanation": a concise explanation (1-3 sentences) of your reasoning

Question: {question}

Instructions:
{instructions}

Use the tools to query the database, then provide your structured evaluation."""

    # Stream the agent so we always have access to the latest state,
    # even if the recursion limit is hit before the agent finishes.
    messages: list = []
    try:
        for chunk in agent.stream(
            {
                "messages": [
                    {"role": "system", "content": system_message},
                ]
            },
            {"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]
    except GraphRecursionError:
        pass  # fall through — we still have the messages collected so far

    # Find the last AI text message (skip tool calls)
    final_message = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_message = msg.content
            break

    if not final_message:
        return agentic_eval(
            evaluation=False,
            explanation="Agent did not produce a final evaluation.",
        )

    return _parse_agent_response(final_message, agentic_eval)
