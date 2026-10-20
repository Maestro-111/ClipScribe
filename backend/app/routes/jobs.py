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
from typing import TYPE_CHECKING, Any, AsyncIterator, cast

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app import exports, abcd_exports, metadata_exports
from app.deps import (
    current_user_id,
    get_reader,
    get_writer,
    require_abcd_job,
    require_owned_job,
    settings_dep,
)
from app.errors import ProblemException
from app.events import started_key, stream_key, summarize_progress
from app.job_runner import JobService, build_job_response
from app.models import (
    JobCreatedResponse,
    JobCreateRequest,
    JobListResponse,
    JobProgressResponse,
    JobResponse,
)
from app.settings import get_settings
from app.video_storage import make_video_storage

if TYPE_CHECKING:
    from app.settings import Settings
    from clip_scribe.db import ClipScribeAsyncReaderDB

router = APIRouter(prefix="/jobs", tags=["jobs"])

_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "canceled"})

type _RedisFields = dict[str, str]
type _RedisEntry = tuple[str, _RedisFields]
type _RedisReadResult = list[tuple[str, list[_RedisEntry]]]


def get_job_service(
    request: Request,
    user_id: str = Depends(current_user_id),
) -> JobService:
    """Assemble the job service from process-wide state (overridable in tests).

    Async DB reader/writer come from ``app.state`` in both execution modes.
    The synchronous builder, executor, and futures are inline-only and stay
    ``None`` under celery.
    ``user_id`` is injected (not called) so the real auth dependency resolves
    the bearer token; the service scopes every job it writes to it.
    """
    state = request.app.state
    return JobService(
        get_reader(request),
        get_writer(request),
        state.settings,
        lambda: make_video_storage(
            state.settings.storage_backend,
            state.settings.input_dir,
            state.settings.gcs_bucket,
        ),
        user_id,
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
async def create_job(
    req: JobCreateRequest,
    service: JobService = Depends(get_job_service),
) -> JobCreatedResponse:
    return await service.create_job(req)


@router.get("", response_model=JobListResponse, summary="List jobs")
async def list_jobs(
    job_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    user_id: str = Depends(current_user_id),
) -> JobListResponse:
    # Only top-level (parent/standalone) jobs are listed; each carries its
    # children and an aggregated status. Scoped to the caller (created_by) and
    # the reader applies any status filter before pagination using the same
    # effective parent status.
    parents = await reader.list_parent_jobs(
        user_id, limit=limit, offset=offset, status=job_status
    )
    jobs = [await build_job_response(reader, p) for p in parents]
    return JobListResponse(jobs=jobs, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobResponse, summary="Get a job")
async def get_job(
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    job: dict = Depends(require_owned_job),
) -> JobResponse:
    return await build_job_response(reader, job)


async def _job_run_abcd_reports(
    reader: ClipScribeAsyncReaderDB, job: dict
) -> list[abcd_exports.RunReport]:
    """Completed runs of a job, oldest first, each with its parser results.

    Works whether ``job`` is a batch parent (walk its children) or a single
    leaf/parse job (itself). Only completed children with a ``run_id`` carry
    inspectable data, so in-progress or failed siblings are skipped.
    """
    children = await reader.get_child_jobs(job["job_id"])
    candidates = children if children else [job]
    reports: list[abcd_exports.RunReport] = []
    for child in candidates:
        run_id = child.get("run_id")
        if child.get("status") != "completed" or not run_id:
            continue
        reports.append(
            abcd_exports.RunReport(
                run_id=run_id,
                video_name=child.get("video_name") or run_id,
                rows=await reader.get_abcd_results(run_id),
            )
        )
    return reports


async def _job_metadata_envelope(
    reader: ClipScribeAsyncReaderDB,
    job: dict,
    *,
    include_frame_detections: bool,
) -> metadata_exports.JobMetadataEnvelope:
    """Load every child plus canonical metadata for each persisted run."""
    children = await reader.get_child_jobs(job["job_id"])
    candidates = children if children else [job]

    async def load_entry(child: dict) -> metadata_exports.BatchRunEntry:
        run_id = child.get("run_id")
        report: metadata_exports.RunMetadataReport | None = None
        if run_id:
            run = await reader.get_run(run_id)
            summary = await reader.get_extraction_summary(
                run_id,
                include_frame_detections=include_frame_detections,
            )

            if run is not None and summary is not None:
                report = metadata_exports.RunMetadataReport(
                    run_id=run_id,
                    video_name=(
                        run.get("video_name") or child.get("video_name") or run_id
                    ),
                    video_type=run.get("video_type"),
                    created_at=run.get("created_at"),
                    extraction_mode=run.get("extraction_mode") or "with_objects",
                    summary=summary,
                )
        return metadata_exports.BatchRunEntry(
            job_id=child["job_id"],
            run_id=run_id,
            video_name=child.get("video_name") or run_id or child["job_id"],
            status=child.get("status") or "unknown",
            report=report,
        )

    # sequential instead of gather, concurrency on all might get out of db connection pool
    # maybe add a semaphore and preserve gather?

    entries = [await load_entry(child) for child in candidates]
    return metadata_exports.JobMetadataEnvelope(
        job_id=job["job_id"],
        job_name=job.get("video_name") or job["job_id"],
        created_at=job.get("created_at"),
        entries=entries,
    )


@router.get("/{job_id}/abcd/export", summary="Download every run's ABCD report")
async def export_abcd(
    job_id: str,
    fmt: str = Query(default="xlsx", alias="format"),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    job: dict = Depends(require_abcd_job),
) -> Response:
    """Export all completed runs in a job as one CSV or XLSX download.

    XLSX carries a cross-run Summary sheet plus one Detail sheet per run; CSV
    stacks every run's criteria into one table with a leading Video column.
    """
    if fmt not in abcd_exports.VALID_FORMATS:
        raise ProblemException(
            status=400, title="Bad Request", detail=f"unsupported format '{fmt}'"
        )
    reports = await _job_run_abcd_reports(reader, job)

    if not reports:
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"job '{job_id}' has no completed runs to export",
        )
    export = abcd_exports.job_csv if fmt == "csv" else abcd_exports.job_xlsx
    content = await run_in_threadpool(export, reports)
    stem = f"{job.get('video_name') or 'job'}_abcd"
    filename = exports.export_filename(stem, fmt)
    return Response(
        content=content,
        media_type=abcd_exports.CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{job_id}/metadata/export",
    summary="Download extracted video metadata for this job",
)
async def export_job_metadata(
    job_id: str,
    fmt: str = Query(default="xlsx", alias="format"),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    job: dict = Depends(require_owned_job),
) -> Response:
    """Export every available child run as XLSX or canonical JSON."""
    if fmt not in metadata_exports.VALID_FORMATS:
        raise ProblemException(
            status=400, title="Bad Request", detail=f"unsupported format '{fmt}'"
        )

    envelope = await _job_metadata_envelope(
        reader,
        job,
        include_frame_detections=fmt == "json",
    )

    if not any(entry.metadata_available for entry in envelope.entries):
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"job '{job_id}' has no persisted video metadata to export",
        )

    export = metadata_exports.job_json if fmt == "json" else metadata_exports.job_xlsx
    content = await run_in_threadpool(export, envelope)
    filename = exports.export_filename(f"{envelope.job_name}_metadata", fmt)
    return Response(
        content=content,
        media_type=metadata_exports.CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a job (canceling it first if still queued or running)",
)
async def delete_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> None:
    await service.delete_job(job_id)


def _sse_frame(fields: dict[str, str]) -> str:
    """Render one stream entry as an SSE ``data:`` frame (type + parsed payload)."""
    payload = {"type": fields["type"], "data": json.loads(fields["data"])}
    return f"data: {json.dumps(payload)}\n\n"


async def _job_event_stream(
    redis_url: str,
    job_id: str,
    reader: ClipScribeAsyncReaderDB,
    xread_block_ms: int,
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
        socket_timeout=xread_block_ms / 1000 + 5,
    )
    key = stream_key(job_id)
    last_id = "0"

    settings = get_settings()

    async def _job_is_terminal() -> bool:
        job = await reader.get_job(job_id)
        return job is not None and job.get("status") in _TERMINAL_JOB_STATUSES

    try:
        # Replay everything already in the stream.
        entries = cast(list[_RedisEntry], await client.xrange(key))
        # `job.started` seeds the client's phase tree but is the first entry
        # approximate MAXLEN trimming drops once a long, log-heavy run fills the
        # stream. If it has aged out, re-emit the snapshot the reporter persisted
        # so a client connecting mid-run still gets its phases — otherwise the live
        # view is stuck on "Waiting for the job to start…" while logs keep flowing.
        if not any(fields["type"] == "job.started" for _id, fields in entries):
            snapshot = cast(str | None, await client.get(started_key(job_id)))
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
            resp: _RedisReadResult | None
            try:
                resp = cast(
                    _RedisReadResult,
                    await client.xread({key: last_id}, block=xread_block_ms, count=100),
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
            for _stream, new_entries in resp:
                for entry_id, fields in new_entries:
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
    settings: Settings = Depends(settings_dep),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _job: dict = Depends(require_owned_job),
) -> StreamingResponse:
    """Stream a job's progress + log events as Server-Sent Events."""

    redis_url = settings.redis_url
    xread_block_ms = settings.XREAD_BLOCK_MS

    return StreamingResponse(
        _job_event_stream(redis_url, job_id, reader, xread_block_ms),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{job_id}/progress",
    response_model=JobProgressResponse,
    summary="Coarse live progress (for the jobs list bar)",
)
async def job_progress(
    job_id: str,
    request: Request,
    job: dict = Depends(require_owned_job),
) -> JobProgressResponse:
    """One-shot progress percent from the job's Redis stream (no SSE).

    Cheap enough to poll per running row: reads the stream once and reduces it.
    A completed job reports 100 without touching Redis; if the stream is gone or
    Redis is down, percent falls back to 0 (or 100 when already completed).
    """
    job_status = job.get("status", "queued")
    if job_status == "completed":
        return JobProgressResponse(job_id=job_id, status=job_status, percent=100.0)

    summary: dict[str, Any] = {
        "percent": 0.0,
        "phase": None,
        "shots_done": None,
        "total_shots": None,
    }
    client = None
    try:
        import redis.asyncio as aioredis

        client = aioredis.Redis.from_url(
            request.app.state.settings.redis_url, decode_responses=True
        )
        entries = cast(list[_RedisEntry], await client.xrange(stream_key(job_id)))
        events = [
            (f["type"], json.loads(f["data"]))
            for _id, f in entries
            if f.get("type") != "log"
        ]
        summary = summarize_progress(events)
    except Exception:  # noqa: BLE001 - progress is best-effort; never 500 the list
        pass
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - progress remains best-effort
                pass

    return JobProgressResponse(job_id=job_id, status=job_status, **summary)


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a queued/running job or batch",
)
async def cancel_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> None:
    await service.cancel_job(job_id)


@router.post(
    "/{job_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
    summary="Retry a completed, failed, or canceled job/run",
)
async def retry_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobCreatedResponse:
    return await service.retry_job(job_id)
