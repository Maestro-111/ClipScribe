"""Job endpoints: create, dispatch, inspect, cancel, retry, delete, and stream.

``POST /jobs`` returns 202 with a parent job id immediately; each video becomes
one child run on the configured inline or Celery backend. Clients poll
``GET /jobs/{id}`` for parent/child state, use ``GET /jobs/{id}/progress`` for
coarse child progress, and open ``GET /jobs/{id}/events`` for a child job's
Redis Stream-backed SSE feed.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app import exports
from app.deps import current_user_id, get_reader, get_writer
from src.utils.clip_scribe_video_storage import make_video_storage
from app.errors import ProblemException
from app.events import started_key, stream_key, summarize_progress
from app.settings import get_settings
from app.job_runner import JobService, build_job_response
from app.models import (
    JobCreatedResponse,
    JobCreateRequest,
    JobListResponse,
    JobProgressResponse,
    JobResponse,
)

if TYPE_CHECKING:
    from src.db import ClipScribeReaderDB

router = APIRouter(prefix="/jobs", tags=["jobs"])

_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "canceled"})
# Block window for XREAD while tailing; on each timeout we emit an SSE comment to
# keep proxies from closing an idle connection and re-check the job row so a
# canceled/queued job (which may never emit a terminal event) still ends cleanly.
_XREAD_BLOCK_MS = 15000


def get_job_service(request: Request) -> JobService:
    """Assemble the job service from process-wide state (overridable in tests).

    DB reader/writer come from ``app.state`` (populated in lifespan from the
    builder inline, or a standalone engine in celery mode). The builder,
    executor, and futures are inline-only and stay ``None`` under celery.
    """
    state = request.app.state
    storage = make_video_storage(
        state.settings.storage_backend,
        state.settings.input_dir,
        state.settings.gcs_bucket,
    )
    return JobService(
        get_reader(request),
        get_writer(request),
        state.settings,
        storage,
        current_user_id(),
        builder=getattr(state, "builder", None),
        executor=getattr(state, "executor", None),
        futures=getattr(state, "futures", None),
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
    summary="Create and enqueue a batch job",
)
def create_job(
    req: JobCreateRequest,
    service: JobService = Depends(get_job_service),
) -> JobCreatedResponse:
    return service.create_job(req)


@router.get("", response_model=JobListResponse, summary="List jobs")
def list_jobs(
    job_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> JobListResponse:
    # Only top-level (parent/standalone) jobs are listed; each carries its
    # children and an aggregated status. The reader applies any status filter
    # before pagination using the same effective parent status.
    parents = reader.list_parent_jobs(limit=limit, offset=offset, status=job_status)
    jobs = [build_job_response(reader, p) for p in parents]
    return JobListResponse(jobs=jobs, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobResponse, summary="Get a job")
def get_job(
    job_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> JobResponse:
    job = reader.get_job(job_id)
    if job is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )
    return build_job_response(reader, job)


def _job_run_reports(
    reader: "ClipScribeReaderDB", job: dict
) -> list[exports.RunReport]:
    """Completed runs of a job, oldest first, each with its parser results.

    Works whether ``job`` is a batch parent (walk its children) or a single
    leaf/parse job (itself). Only completed children with a ``run_id`` carry
    inspectable data, so in-progress or failed siblings are skipped.
    """
    children = reader.get_child_jobs(job["job_id"])
    candidates = children if children else [job]
    reports: list[exports.RunReport] = []
    for child in candidates:
        run_id = child.get("run_id")
        if child.get("status") != "completed" or not run_id:
            continue
        reports.append(
            exports.RunReport(
                run_id=run_id,
                video_name=child.get("video_name") or run_id,
                rows=reader.get_parser_results(run_id),
            )
        )
    return reports


@router.get("/{job_id}/export", summary="Download every run's ABCD report")
def export_job(
    job_id: str,
    fmt: str = Query(default="xlsx", alias="format"),
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> Response:
    """Export all completed runs in a job as one CSV or XLSX download.

    XLSX carries a cross-run Summary sheet plus one Detail sheet per run; CSV
    stacks every run's criteria into one table with a leading Video column.
    """
    if fmt not in exports.VALID_FORMATS:
        raise ProblemException(
            status=400, title="Bad Request", detail=f"unsupported format '{fmt}'"
        )
    job = reader.get_job(job_id)
    if job is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )
    reports = _job_run_reports(reader, job)
    if not reports:
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"job '{job_id}' has no completed runs to export",
        )
    content = exports.job_csv(reports) if fmt == "csv" else exports.job_xlsx(reports)
    stem = f"{job.get('video_name') or 'job'}_abcd"
    filename = exports.export_filename(stem, fmt)
    return Response(
        content=content,
        media_type=exports.CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a job (canceling it first if still queued or running)",
)
def delete_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> None:
    service.delete_job(job_id)


def _sse_frame(fields: dict[str, str]) -> str:
    """Render one stream entry as an SSE ``data:`` frame (type + parsed payload)."""
    payload = {"type": fields["type"], "data": json.loads(fields["data"])}
    return f"data: {json.dumps(payload)}\n\n"


async def _job_event_stream(
    redis_url: str, job_id: str, reader: "ClipScribeReaderDB"
) -> AsyncIterator[str]:
    """Replay the job's Redis stream from the start, then tail live (§9, §16).

    Reading from id ``0`` means a client that connects mid-run — or after the
    job finished, while the stream is still within its TTL — gets the full
    history before live updates. The stream ends when a terminal event is read;
    as a backstop for jobs that never emit one (e.g. a queued job canceled before
    it ran, or Redis being down at run time), each idle tick re-checks the job
    row and closes on a terminal status.
    """
    import redis.asyncio as aioredis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    # socket_timeout must exceed the XREAD BLOCK window, or the client socket
    # read times out before the server's blocking read returns. Set it above the
    # block (and neutralize any shorter timeout baked into REDIS_URL) so an idle
    # tail waits the full window instead of raising; a truly hung server still
    # trips the timeout, which the loop below catches and retries.
    client = aioredis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_timeout=_XREAD_BLOCK_MS / 1000 + 5,
    )
    key = stream_key(job_id)
    last_id = "0"

    settings = get_settings()

    async def _job_is_terminal() -> bool:
        job = await run_in_threadpool(reader.get_job, job_id)
        return job is not None and job.get("status") in _TERMINAL_JOB_STATUSES

    try:
        # Replay everything already in the stream.
        entries = await client.xrange(key)
        # `job.started` seeds the client's phase tree but is the first entry
        # approximate MAXLEN trimming drops once a long, log-heavy run fills the
        # stream. If it has aged out, re-emit the snapshot the reporter persisted
        # so a client connecting mid-run still gets its phases — otherwise the live
        # view is stuck on "Waiting for the job to start…" while logs keep flowing.
        if not any(fields["type"] == "job.started" for _id, fields in entries):
            snapshot = await client.get(started_key(job_id))
            if snapshot is not None:
                yield _sse_frame(json.loads(snapshot))
        for entry_id, fields in entries:
            last_id = entry_id
            yield _sse_frame(fields)
            if fields["type"] in settings.TERMINAL_EVENTS:
                return

        # If the job is already terminal, the replay above is the whole story.
        if await _job_is_terminal():
            return

        # Otherwise tail for new entries.
        while True:
            try:
                resp = await client.xread(
                    {key: last_id}, block=_XREAD_BLOCK_MS, count=100
                )
            except RedisTimeoutError:
                # No new entries within the block window — treat as an idle tick.
                resp = None
            except RedisConnectionError:
                # Transient transport hiccup; back off before re-checking.
                await asyncio.sleep(1.0)
                resp = None
            if not resp:
                yield ": keepalive\n\n"
                if await _job_is_terminal():
                    return
                continue
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    last_id = entry_id
                    yield _sse_frame(fields)
                    if fields["type"] in settings.TERMINAL_EVENTS:
                        return
    except asyncio.CancelledError:  # client disconnected — let it propagate
        raise
    finally:
        await client.aclose()


@router.get("/{job_id}/events", summary="Live job progress (SSE)")
async def job_events(
    job_id: str,
    request: Request,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> StreamingResponse:
    """Stream a job's progress + log events as Server-Sent Events."""
    if reader.get_job(job_id) is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )
    redis_url = request.app.state.settings.redis_url
    return StreamingResponse(
        _job_event_stream(redis_url, job_id, reader),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{job_id}/progress",
    response_model=JobProgressResponse,
    summary="Coarse live progress (for the jobs list bar)",
)
def job_progress(
    job_id: str,
    request: Request,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> JobProgressResponse:
    """One-shot progress percent from the job's Redis stream (no SSE).

    Cheap enough to poll per running row: reads the stream once and reduces it.
    A completed job reports 100 without touching Redis; if the stream is gone or
    Redis is down, percent falls back to 0 (or 100 when already completed).
    """
    job = reader.get_job(job_id)
    if job is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )
    job_status = job.get("status", "queued")
    if job_status == "completed":
        return JobProgressResponse(job_id=job_id, status=job_status, percent=100.0)

    summary: dict = {
        "percent": 0.0,
        "phase": None,
        "shots_done": None,
        "total_shots": None,
    }
    try:
        import redis

        client = redis.Redis.from_url(
            request.app.state.settings.redis_url, decode_responses=True
        )
        entries = client.xrange(stream_key(job_id))
        events = [
            (f["type"], json.loads(f["data"]))
            for _id, f in entries
            if f.get("type") != "log"
        ]
        summary = summarize_progress(events)
    except Exception:  # noqa: BLE001 - progress is best-effort; never 500 the list
        pass

    return JobProgressResponse(job_id=job_id, status=job_status, **summary)


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a queued/running job or batch",
)
def cancel_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> None:
    service.cancel_job(job_id)


@router.post(
    "/{job_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
    summary="Retry a completed, failed, or canceled job/run",
)
def retry_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobCreatedResponse:
    return service.retry_job(job_id)
