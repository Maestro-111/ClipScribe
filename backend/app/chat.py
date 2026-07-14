"""Advisory chat service (web-app-plan §13) — API-side glue for post-run Q&A.

Builds the LLM + a ReAct agent (``backend/src/parser/advisory.py``), streams
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
from typing import TYPE_CHECKING, Any, Iterator

import yaml
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from src.parser.advisory import (
    ADVISORY_SYSTEM_PROMPT,
    JOB_ADVISORY_SYSTEM_PROMPT,
    build_advisory_agent,
    build_job_advisory_agent,
)

if TYPE_CHECKING:
    from app.settings import Settings
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB

logger = logging.getLogger("clip_scribe")

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "clip_scribe"
    / "configs"
    / "clip_scribe.yaml"
)

# Chat answers run longer than a one-line verdict, so we lift the token cap above
# the parser's default while reusing its model + temperature.
_CHAT_MAX_TOKENS = 1500
_RECURSION_LIMIT = 25


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
        reader: "ClipScribeReaderDB",
        writer: "ClipScribeWriterDB",
        settings: "Settings",
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.settings = settings
        agent_cfg = _load_agent_config()
        # Reusable across requests; scoped run/job agents are built per request.
        self.model = ChatOpenAI(
            model=agent_cfg.get("llm", "gpt-4o-mini"),
            temperature=agent_cfg.get("temperature", 0.2),
            timeout=60,
            max_completion_tokens=_CHAT_MAX_TOKENS,
            max_retries=agent_cfg.get("max_retries", 5),
        )

    # ── per-run inspector chat (scoped to one run) ──────────────────────────

    def list_sessions(self, run_id: str) -> list[dict]:
        return self.reader.get_chat_sessions(run_id)

    def get_history(self, run_id: str, session_id: str) -> list[dict]:
        return self.reader.get_chat_messages(run_id, session_id)

    def delete_session(self, run_id: str, session_id: str) -> int:
        return self.writer.delete_chat_session(run_id, session_id)

    def stream(self, run_id: str, session_id: str, message: str) -> Iterator[str]:
        """Stream one per-run chat turn (SSE frames) and persist the transcript."""
        history = self.reader.get_chat_messages(run_id, session_id)
        agent = build_advisory_agent(self.model, self.reader, run_id)
        yield from self._stream_turn(
            history=history,
            system_prompt=ADVISORY_SYSTEM_PROMPT,
            agent=agent,
            session_id=session_id,
            message=message,
            persist_scope={"run_id": run_id},
            log_label=f"run {run_id}",
        )

    # ── job-level chat (spans every completed run in a batch job) ────────────

    def list_job_sessions(self, job_id: str) -> list[dict]:
        return self.reader.get_job_chat_sessions(job_id)

    def get_job_history(self, job_id: str, session_id: str) -> list[dict]:
        return self.reader.get_job_chat_messages(job_id, session_id)

    def delete_job_session(self, job_id: str, session_id: str) -> int:
        return self.writer.delete_job_chat_session(job_id, session_id)

    def resolve_job_runs(self, job_id: str) -> list[dict]:
        """Completed runs of a job as ``[{run_id, video_name}]``, oldest first.

        Walks the batch children (or the job itself for a single leaf job) and
        keeps only completed runs, which are the ones with inspectable data.
        """
        job = self.reader.get_job(job_id)
        if job is None:
            return []
        children = self.reader.get_child_jobs(job_id)
        candidates = children if children else [job]
        return [
            {"run_id": c["run_id"], "video_name": c.get("video_name") or c["run_id"]}
            for c in candidates
            if c.get("status") == "completed" and c.get("run_id")
        ]

    def stream_job(
        self, job_id: str, session_id: str, message: str, runs: list[dict]
    ) -> Iterator[str]:
        """Stream one job-level chat turn (SSE frames) and persist the transcript."""
        history = self.reader.get_job_chat_messages(job_id, session_id)
        agent = build_job_advisory_agent(self.model, self.reader, runs)
        yield from self._stream_turn(
            history=history,
            system_prompt=JOB_ADVISORY_SYSTEM_PROMPT,
            agent=agent,
            session_id=session_id,
            message=message,
            persist_scope={"job_id": job_id},
            log_label=f"job {job_id}",
        )

    # ── shared streaming machinery ──────────────────────────────────────────

    def _stream_turn(
        self,
        *,
        history: list[dict],
        system_prompt: str,
        agent: Any,
        session_id: str,
        message: str,
        persist_scope: dict[str, str],
        log_label: str,
    ) -> Iterator[str]:
        """Drive one agent turn: replay history, stream tokens/tools, persist.

        Runs in a threadpool (StreamingResponse over a sync generator), so the
        blocking LLM stream + DB writes here never block the event loop.
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

        self.writer.add_chat_message(
            session_id=session_id, role="user", content=message, **persist_scope
        )
        yield _sse({"type": "session", "session_id": session_id})

        answer_parts: list[str] = []
        seen_tools: set[str] = set()
        try:
            for chunk, _meta in agent.stream(
                {"messages": lc_messages},
                {"recursion_limit": _RECURSION_LIMIT},
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
            self.writer.add_chat_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                tool_calls=sorted(seen_tools) or None,
                **persist_scope,
            )
        yield _sse({"type": "done"})
