"""Advisory chat service (web-app-plan §13) — API-side glue for post-run Q&A.

Builds the LLM + a ReAct agent (``clip_scribe/parser/agent_factory.py``), streams
answers as Server-Sent Events, and persists transcripts to ``chat_messages``.
There are two scopes: per-run inspector chat (one ``run_id``) and job-level chat
(one ``job_id`` spanning the job's completed runs). It does no pipeline model
loading and does not use the worker — just LLM calls + DB reads — so it lives in
the API process. Some LangChain/LangGraph imports may transitively import torch
in this environment, so routes lazy-import this service.

Conversation memory is the DB: each turn reloads the session's prior messages
and replays them into the agent, so history survives restarts and multiple API
replicas without a separate checkpointer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import yaml
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from clip_scribe.parser.agent_factory import AgentBuilder

from clip_scribe.parser.abcd.prompts import (
    JOB_SYSTEM_PROMPT as ABCD_JOB_SYSTEM_PROMPT,
    RUN_SYSTEM_PROMPT as ABCD_RUN_SYSTEM_PROMPT,
)

from clip_scribe.parser.explore.prompts import (
    RUN_SYSTEM_PROMPT as EXPLORE_RUN_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from app.settings import Settings
    from clip_scribe.db import ClipScribeAsyncReaderDB, ClipScribeAsyncWriterDB

logger = logging.getLogger("clip_scribe")

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "clip_scribe"
    / "clip_scribe_core"
    / "configs"
    / "clip_scribe.yaml"
)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _load_agent_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("clip_scribe", {}).get("parser", {}).get("agent", {})
    except FileNotFoundError:
        return {}


class ChatService:
    def __init__(
        self,
        reader: ClipScribeAsyncReaderDB,
        writer: ClipScribeAsyncWriterDB,
        settings: Settings,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.settings = settings
        agent_cfg = _load_agent_config()
        # Reusable across requests; scoped run/job agents are built per request.
        self.model = ChatOpenAI(
            model=agent_cfg.get("llm", "gpt-5.4"),
            temperature=agent_cfg.get("temperature", 0.2),
            timeout=60,
            max_completion_tokens=self.settings.CHAT_MAX_TOKENS,
            max_retries=agent_cfg.get("max_retries", 5),
        )

    # ── per-run inspector chat (scoped to one run) ──────────────────────────

    async def list_sessions(self, run_id: str) -> list[dict]:
        return await self.reader.get_chat_sessions(run_id)

    async def get_history(self, run_id: str, session_id: str) -> list[dict]:
        return await self.reader.get_chat_messages(run_id, session_id)

    async def delete_session(self, run_id: str, session_id: str) -> int:
        return await self.writer.delete_chat_session(run_id, session_id)

    async def stream_abcd_run(
        self, run_id: str, session_id: str, message: str
    ) -> AsyncIterator[str]:
        """Stream one abcd per-run chat turn (SSE frames) and persist the transcript."""

        history = await self.reader.get_chat_messages(run_id, session_id)
        agent = AgentBuilder.build_abcd_run_agent(self.model, self.reader, run_id)

        async for event in self._stream_turn(
            history=history,
            system_prompt=ABCD_RUN_SYSTEM_PROMPT,
            agent=agent,
            session_id=session_id,
            message=message,
            persist_scope={"run_id": run_id},
            log_label=f"run {run_id}",
        ):
            yield event

    async def stream_exploration_run(
        self, run_id: str, session_id: str, message: str
    ) -> AsyncIterator[str]:
        """Stream one exploration per-run chat turn (SSE frames) and persist the transcript."""
        history = await self.reader.get_chat_messages(run_id, session_id)
        agent = AgentBuilder.build_exploration_run_agent(
            self.model, self.reader, run_id
        )

        async for event in self._stream_turn(
            history=history,
            system_prompt=EXPLORE_RUN_SYSTEM_PROMPT,
            agent=agent,
            session_id=session_id,
            message=message,
            persist_scope={"run_id": run_id},
            log_label=f"run {run_id}",
        ):
            yield event

    # ── job-level chat (spans every completed run in a batch job) ────────────

    async def list_job_sessions(self, job_id: str) -> list[dict]:
        return await self.reader.get_job_chat_sessions(job_id)

    async def get_job_history(self, job_id: str, session_id: str) -> list[dict]:
        return await self.reader.get_job_chat_messages(job_id, session_id)

    async def delete_job_session(self, job_id: str, session_id: str) -> int:
        return await self.writer.delete_job_chat_session(job_id, session_id)

    async def resolve_job_runs(self, job_id: str) -> list[dict]:
        """Completed runs of a job as ``[{run_id, video_name}]``, oldest first.

        Walks the batch children (or the job itself for a single leaf job) and
        keeps only completed runs, which are the ones with inspectable data.
        """
        job = await self.reader.get_job(job_id)
        if job is None:
            return []
        children = await self.reader.get_child_jobs(job_id)
        candidates = children if children else [job]
        return [
            {"run_id": c["run_id"], "video_name": c.get("video_name") or c["run_id"]}
            for c in candidates
            if c.get("status") == "completed" and c.get("run_id")
        ]

    async def stream_abcd_job(
        self, job_id: str, session_id: str, message: str, runs: list[dict]
    ) -> AsyncIterator[str]:
        """Stream one job-level chat turn (SSE frames) and persist the transcript."""

        history = await self.reader.get_job_chat_messages(job_id, session_id)
        agent = AgentBuilder.build_abcd_job_agent(self.model, self.reader, runs)

        async for event in self._stream_turn(
            history=history,
            system_prompt=ABCD_JOB_SYSTEM_PROMPT,
            agent=agent,
            session_id=session_id,
            message=message,
            persist_scope={"job_id": job_id},
            log_label=f"job {job_id}",
        ):
            yield event

    # ── shared streaming machinery ──────────────────────────────────────────

    async def _stream_turn(
        self,
        *,
        history: list[dict],
        system_prompt: str,
        agent: Any,
        session_id: str,
        message: str,
        persist_scope: dict[str, str],
        log_label: str,
    ) -> AsyncIterator[str]:
        """Drive one agent turn: replay history, stream tokens/tools, persist.

        The async agent stream and async DB writes yield control while waiting,
        so each SSE frame can be forwarded without blocking the event loop.
        ``persist_scope`` is the keying for ``add_chat_message`` — ``{run_id}``
        for the per-run chat, ``{job_id}`` for the job chat.
        """
        # Rebuild conversation from the DB, then append the new turn.
        lc_messages: list = [SystemMessage(content=system_prompt)]
        for m in history:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
        lc_messages.append(HumanMessage(content=message))

        await self.writer.add_chat_message(
            session_id=session_id,
            role="user",
            content=message,
            run_id=persist_scope.get("run_id"),
            job_id=persist_scope.get("job_id"),
        )
        yield _sse({"type": "session", "session_id": session_id})

        answer_parts: list[str] = []
        seen_tools: set[str] = set()
        try:
            async for chunk, _meta in agent.astream(
                {"messages": lc_messages},
                {"recursion_limit": self.settings.RECURSION_LIMIT},
                stream_mode="messages",
            ):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                for tc in chunk.tool_call_chunks or []:
                    name = tc.get("name")
                    if name and name not in seen_tools:
                        seen_tools.add(name)
                        yield _sse({"type": "tool", "name": name})
                if chunk.content:
                    text = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else str(chunk.content)
                    )
                    answer_parts.append(text)
                    yield _sse({"type": "token", "content": text})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Advisory chat failed for %s", log_label)
            yield _sse({"type": "error", "message": str(exc)})

        answer = "".join(answer_parts).strip()
        if answer:
            await self.writer.add_chat_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                run_id=persist_scope.get("run_id"),
                job_id=persist_scope.get("job_id"),
                tool_calls=sorted(seen_tools) or None,
            )
        yield _sse({"type": "done"})
