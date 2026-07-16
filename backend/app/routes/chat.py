"""Advisory chat endpoints (web-app-plan §6, §13).

``POST /runs/{id}/chat`` streams per-run Q&A and ``POST /jobs/{id}/chat``
streams job-level Q&A across completed runs. The GET/DELETE routes manage
persisted session transcripts. Run routes 404 if the run does not exist; job
routes 404 if the job does not exist and 409 until at least one run is complete.
Agents are read-only and server-side scoped to ``run_id`` or ``job_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from app.deps import get_reader, get_writer
from app.errors import ProblemException
from app.models import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatSession,
    ChatSessionsResponse,
    JobChatHistoryResponse,
    JobChatSessionsResponse,
)
from src.utils.ids import new_ulid

if TYPE_CHECKING:
    from app.chat import ChatService
    from src.db import ClipScribeReaderDB

router = APIRouter(prefix="/runs", tags=["chat"])
job_router = APIRouter(prefix="/jobs", tags=["chat"])


def get_chat_service(request: Request) -> "ChatService":
    # Lazy import: app.chat pulls the LLM client (langchain/langgraph), which
    # transitively imports torch. Deferring it here keeps `import app.main`
    # light (fast startup, OpenAPI gen, tests) — torch loads only on the first
    # chat request. No models are ever loaded in the API (web-app-plan §13).
    from app.chat import ChatService

    state = request.app.state
    return ChatService(get_reader(request), get_writer(request), state.settings)


def _require_run(reader: "ClipScribeReaderDB", run_id: str) -> None:
    if reader.get_run(run_id) is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"run '{run_id}' not found"
        )


@router.post("/{run_id}/chat", summary="Ask the agent about the run (SSE stream)")
def post_run_chat(
    run_id: str,
    req: ChatRequest,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    _require_run(reader, run_id)
    session_id = req.session_id or new_ulid()
    return StreamingResponse(
        service.stream_run(run_id, session_id, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{run_id}/chat/sessions",
    response_model=ChatSessionsResponse,
    summary="List advisory-chat sessions for a run",
)
def list_sessions(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> ChatSessionsResponse:
    _require_run(reader, run_id)
    sessions = [ChatSession(**s) for s in service.list_sessions(run_id)]
    return ChatSessionsResponse(run_id=run_id, sessions=sessions)


@router.get(
    "/{run_id}/chat/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get one chat session's transcript",
)
def get_session(
    run_id: str,
    session_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> ChatHistoryResponse:
    _require_run(reader, run_id)
    messages = [ChatMessage(**m) for m in service.get_history(run_id, session_id)]
    return ChatHistoryResponse(run_id=run_id, session_id=session_id, messages=messages)


@router.delete(
    "/{run_id}/chat/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
def delete_session(
    run_id: str,
    session_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> None:
    _require_run(reader, run_id)
    service.delete_session(run_id, session_id)


# ── Job-level advisory chat (spans every completed run in a batch job) ───────


def _require_job(reader: "ClipScribeReaderDB", job_id: str) -> None:
    if reader.get_job(job_id) is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )


@job_router.post("/{job_id}/chat", summary="Ask the agent about the job (SSE)")
def post_job_chat(
    job_id: str,
    req: ChatRequest,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    _require_job(reader, job_id)
    runs = service.resolve_job_runs(job_id)
    if not runs:
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"job '{job_id}' has no completed runs to analyze yet",
        )
    session_id = req.session_id or new_ulid()
    return StreamingResponse(
        service.stream_job(job_id, session_id, req.message, runs),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@job_router.get(
    "/{job_id}/chat/sessions",
    response_model=JobChatSessionsResponse,
    summary="List job-level chat sessions",
)
def list_job_sessions(
    job_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> JobChatSessionsResponse:
    _require_job(reader, job_id)
    sessions = [ChatSession(**s) for s in service.list_job_sessions(job_id)]
    return JobChatSessionsResponse(job_id=job_id, sessions=sessions)


@job_router.get(
    "/{job_id}/chat/{session_id}",
    response_model=JobChatHistoryResponse,
    summary="Get one job-level chat session's transcript",
)
def get_job_session(
    job_id: str,
    session_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> JobChatHistoryResponse:
    _require_job(reader, job_id)
    messages = [ChatMessage(**m) for m in service.get_job_history(job_id, session_id)]
    return JobChatHistoryResponse(
        job_id=job_id, session_id=session_id, messages=messages
    )


@job_router.delete(
    "/{job_id}/chat/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a job-level chat session",
)
def delete_job_session(
    job_id: str,
    session_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    service: ChatService = Depends(get_chat_service),
) -> None:
    _require_job(reader, job_id)
    service.delete_job_session(job_id, session_id)
