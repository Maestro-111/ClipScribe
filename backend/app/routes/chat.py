"""Advisory chat endpoints (web-app-plan §6, §13).

``POST /runs/{id}/abcd/chat`` and ``/runs/{id}/exploration/chat`` stream per-run
Q&A; ``POST /jobs/{id}/abcd/chat`` and ``/jobs/{id}/exploration/chat`` stream
job-level Q&A across completed runs. Each 409s on the other analysis type. The GET/DELETE routes manage
persisted session transcripts. Run routes 404 if the run does not exist; job
routes 404 if the job does not exist and 409 until at least one run is complete.
Agents are read-only and server-side scoped to ``run_id`` or ``job_id``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from app.deps import (
    get_reader,
    get_writer,
    require_abcd_job,
    require_abcd_run,
    require_exploratory_job,
    require_exploratory_run,
    require_owned_job,
    require_owned_run,
)
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
from clip_scribe.utils.ids import new_ulid

if TYPE_CHECKING:
    from app.chat import ChatService

router = APIRouter(prefix="/runs", tags=["chat"])
job_router = APIRouter(prefix="/jobs", tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    # Lazy import: app.chat pulls the LLM client (langchain/langgraph), which
    # transitively imports torch. Deferring it here keeps `import app.main`
    # light (fast startup, OpenAPI gen, tests) — torch loads only on the first
    # chat request. No models are ever loaded in the API (web-app-plan §13).
    from app.chat import ChatService

    state = request.app.state
    return ChatService(get_reader(request), get_writer(request), state.settings)


@router.post(
    "/{run_id}/abcd/chat", summary="Ask the agent about the abcd run (SSE stream)"
)
async def post_run_abcd_chat(
    run_id: str,
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    _run: dict = Depends(require_abcd_run),
) -> StreamingResponse:
    session_id = req.session_id or new_ulid()
    return StreamingResponse(
        service.stream_abcd_run(run_id, session_id, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/{run_id}/exploration/chat",
    summary="Ask the agent about the exploration run (SSE stream)",
)
async def post_run_exploration_chat(
    run_id: str,
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    _run: dict = Depends(require_exploratory_run),
) -> StreamingResponse:
    session_id = req.session_id or new_ulid()
    return StreamingResponse(
        service.stream_exploration_run(run_id, session_id, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{run_id}/chat/sessions",
    response_model=ChatSessionsResponse,
    summary="List advisory-chat sessions for a run",
)
async def list_sessions(
    run_id: str,
    service: ChatService = Depends(get_chat_service),
    _run: dict = Depends(require_owned_run),
) -> ChatSessionsResponse:
    sessions = [ChatSession(**s) for s in await service.list_sessions(run_id)]
    return ChatSessionsResponse(run_id=run_id, sessions=sessions)


@router.get(
    "/{run_id}/chat/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get one chat session's transcript",
)
async def get_session(
    run_id: str,
    session_id: str,
    service: ChatService = Depends(get_chat_service),
    _run: dict = Depends(require_owned_run),
) -> ChatHistoryResponse:
    messages = [ChatMessage(**m) for m in await service.get_history(run_id, session_id)]
    return ChatHistoryResponse(run_id=run_id, session_id=session_id, messages=messages)


@router.delete(
    "/{run_id}/chat/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
async def delete_session(
    run_id: str,
    session_id: str,
    service: ChatService = Depends(get_chat_service),
    _run: dict = Depends(require_owned_run),
) -> None:
    await service.delete_session(run_id, session_id)


# ── Job-level advisory chat (spans every completed run in a batch job) ───────


async def _stream_job_chat(
    service: ChatService,
    job_id: str,
    req: ChatRequest,
    stream: Callable[[str, str, str, list[dict]], AsyncIterator[str]],
) -> StreamingResponse:
    """409 if the job has nothing completed yet, else stream one chat turn."""
    runs = await service.resolve_job_runs(job_id)
    if not runs:
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"job '{job_id}' has no completed runs to analyze yet",
        )
    session_id = req.session_id or new_ulid()
    return StreamingResponse(
        stream(job_id, session_id, req.message, runs),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@job_router.post(
    "/{job_id}/abcd/chat", summary="Ask the agent about the ABCD job (SSE)"
)
async def post_job_abcd_chat(
    job_id: str,
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    _job: dict = Depends(require_abcd_job),
) -> StreamingResponse:
    return await _stream_job_chat(service, job_id, req, service.stream_abcd_job)


@job_router.post(
    "/{job_id}/exploration/chat",
    summary="Ask the agent about the exploration job (SSE)",
)
async def post_job_exploration_chat(
    job_id: str,
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    _job: dict = Depends(require_exploratory_job),
) -> StreamingResponse:
    return await _stream_job_chat(service, job_id, req, service.stream_exploration_job)


@job_router.get(
    "/{job_id}/chat/sessions",
    response_model=JobChatSessionsResponse,
    summary="List job-level chat sessions",
)
async def list_job_sessions(
    job_id: str,
    service: ChatService = Depends(get_chat_service),
    _job: dict = Depends(require_owned_job),
) -> JobChatSessionsResponse:
    sessions = [ChatSession(**s) for s in await service.list_job_sessions(job_id)]
    return JobChatSessionsResponse(job_id=job_id, sessions=sessions)


@job_router.get(
    "/{job_id}/chat/{session_id}",
    response_model=JobChatHistoryResponse,
    summary="Get one job-level chat session's transcript",
)
async def get_job_session(
    job_id: str,
    session_id: str,
    service: ChatService = Depends(get_chat_service),
    _job: dict = Depends(require_owned_job),
) -> JobChatHistoryResponse:
    messages = [
        ChatMessage(**m) for m in await service.get_job_history(job_id, session_id)
    ]
    return JobChatHistoryResponse(
        job_id=job_id, session_id=session_id, messages=messages
    )


@job_router.delete(
    "/{job_id}/chat/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a job-level chat session",
)
async def delete_job_session(
    job_id: str,
    session_id: str,
    service: ChatService = Depends(get_chat_service),
    _job: dict = Depends(require_owned_job),
) -> None:
    await service.delete_job_session(job_id, session_id)
