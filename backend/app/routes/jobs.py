"""Job endpoints: create + enqueue, list, and fetch (web-app-plan §6).

``POST /jobs`` returns 202 with a job id immediately; the run happens on the
executor. Clients poll ``GET /jobs/{id}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query, Request, status

from app.deps import get_reader, get_writer
from app.errors import ProblemException
from app.job_runner import JobService
from app.models import (
    JobCreatedResponse,
    JobCreateRequest,
    JobListResponse,
    JobResponse,
)

if TYPE_CHECKING:
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_job_service(request: Request) -> JobService:
    """Assemble the job service from process-wide state (overridable in tests).

    DB reader/writer come from ``app.state`` (populated in lifespan from the
    builder inline, or a standalone engine in celery mode). The builder,
    executor, and futures are inline-only and stay ``None`` under celery.
    """
    state = request.app.state
    return JobService(
        get_reader(request),
        get_writer(request),
        state.settings,
        builder=getattr(state, "builder", None),
        executor=getattr(state, "executor", None),
        futures=getattr(state, "futures", None),
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
    summary="Create and enqueue a job",
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
    jobs = reader.list_jobs(status=job_status, limit=limit, offset=offset)
    return JobListResponse(
        jobs=[JobResponse(**job) for job in jobs], limit=limit, offset=offset
    )


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
    return JobResponse(**job)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a completed, failed, or canceled job",
)
def delete_job(
    job_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    writer: "ClipScribeWriterDB" = Depends(get_writer),
) -> None:
    job = reader.get_job(job_id)
    if job is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )
    terminal = {"completed", "failed", "canceled"}
    if job["status"] not in terminal:
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"job '{job_id}' is '{job['status']}' — stop the job before deleting it",
        )
    writer.delete_job(job_id)


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a queued job",
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
    summary="Retry a failed or canceled job",
)
def retry_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobCreatedResponse:
    return service.retry_job(job_id)
