"""FastAPI dependency providers.

Each is a thin accessor over ``app.state`` (populated in ``main.lifespan``) so
routes stay decoupled from how the builder/executor are constructed, and tests
can swap them via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.errors import ProblemException
from app.models import ExplorationResponse, ParsingMode
from app.settings import Settings, get_settings
from clip_scribe.utils.clip_scribe_artifacts import (
    ArtifactUploader,
    make_artifact_uploader,
)
from app.video_storage import VideoStorage, make_video_storage

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from clip_scribe.clip_scribe_core.build_clip_scribe import ClipScribeBuilder
    from clip_scribe.db import ClipScribeAsyncWriterDB, ClipScribeAsyncReaderDB

# Single-tenant placeholder until auth lands. Everything user-scoped (uploads,
# the videos registry, the input picker) reads this, so wiring in real auth is
# "make current_user_id return the authenticated id" rather than a refactor.
DEFAULT_USER_ID = "local"
_bearer = HTTPBearer(
    auto_error=False
)  # auto_error=False → no creds isn't an instant 403


async def current_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    settings: Settings = get_settings()

    if settings.allow_anonymous_local:
        return DEFAULT_USER_ID

    if creds is None:
        raise ProblemException(
            status=401, title="Unauthorized", detail="authentication required"
        )
    return _verify_and_extract_sub(creds.credentials)


def _verify_and_extract_sub(credentials: str) -> str:
    raise ProblemException(
        status=401,
        title="Unauthorized",
        detail="bearer authentication is not configured",
    )


def settings_dep() -> Settings:
    return get_settings()


def video_storage_dep(settings: Settings = Depends(settings_dep)) -> VideoStorage:
    """The configured video storage backend (local disk or a GCS bucket)."""
    return make_video_storage(
        settings.storage_backend, settings.input_dir, settings.gcs_bucket
    )


def artifact_storage_dep(
    settings: Settings = Depends(settings_dep),
) -> ArtifactUploader:
    """The configured artifact storage backend, used to sign served artifacts.

    Same selector as video storage; only the read side (``tracked_video_url``)
    is exercised by the API — the write side runs in the worker's engine.
    """

    return make_artifact_uploader(settings.storage_backend, settings.gcs_bucket)


def get_builder(request: Request) -> ClipScribeBuilder:
    builder = getattr(request.app.state, "builder", None)
    if builder is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Models are not loaded; this endpoint is unavailable.",
        )
    return builder


def get_reader(request: Request) -> ClipScribeAsyncReaderDB:
    # The API always owns async DB adapters in app.state. In inline mode the
    # builder separately owns synchronous adapters for the worker thread.
    reader = getattr(request.app.state, "reader_db", None)
    if reader is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Database is not available; this endpoint is unavailable.",
        )
    return reader


def get_writer(request: Request) -> ClipScribeAsyncWriterDB:
    writer = getattr(request.app.state, "writer_db", None)
    if writer is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Database is not available; this endpoint is unavailable.",
        )
    return writer


def get_executor(request: Request) -> ThreadPoolExecutor:
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Job executor is not available; this endpoint is unavailable.",
        )
    return executor


def get_futures(request: Request) -> dict[str, Future[None]]:
    return getattr(request.app.state, "futures", {})


async def require_owned_run(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    user_id: str = Depends(current_user_id),
) -> dict:
    """Authorize a by-run-id read and return the run row.

    Run-scoped tables have no user column; ownership is derived by resolving the
    run to the job that produced it (``reader.run_owner``). A missing run *or* one
    owned by another user both 404 — returning 403 for the latter would leak the
    run's existence and let ids be enumerated. Depend on this in place of a bare
    existence check; the leaf reads that follow are then safe because the caller
    is known to own the run.
    """
    run = await reader.get_run(run_id)
    if run is None or await reader.run_owner(run_id) != user_id:
        raise ProblemException(
            status=404, title="Not Found", detail=f"run '{run_id}' not found"
        )
    return run


async def require_owned_job(
    job_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    user_id: str = Depends(current_user_id),
) -> dict:
    """Authorize a by-job-id read and return the job row.

    Ownership lives directly on ``jobs.created_by``. As with runs, a job the
    caller doesn't own 404s rather than 403s so existence isn't leaked.
    """
    job = await reader.get_job(job_id)
    if job is None or job.get("created_by") != user_id:
        raise ProblemException(
            status=404, title="Not Found", detail=f"job '{job_id}' not found"
        )
    return job


def _require_parsing_mode(row: dict, mode: ParsingMode, kind: str, row_id: str) -> None:
    actual = row.get("parsing_mode", ParsingMode.ABCD.value)
    if actual != mode.value:
        raise ProblemException(
            status=409,
            title="Conflict",
            detail=f"{kind} '{row_id}' was analyzed in {actual} mode; "
            f"this endpoint is only available for {mode.value} {kind}s",
        )


async def require_abcd_run(run: dict = Depends(require_owned_run)) -> dict:
    """``require_owned_run`` for ABCD-only routes: 409s on exploratory runs."""
    _require_parsing_mode(run, ParsingMode.ABCD, "run", run["run_id"])
    return run


async def require_abcd_job(job: dict = Depends(require_owned_job)) -> dict:
    """``require_owned_job`` for ABCD-only routes: 409s on exploratory jobs."""
    _require_parsing_mode(job, ParsingMode.ABCD, "job", job["job_id"])
    return job


async def require_exploratory_job(job: dict = Depends(require_owned_job)) -> dict:
    """``require_owned_job`` for exploration routes: 409s on ABCD jobs."""
    _require_parsing_mode(job, ParsingMode.EXPLORATORY, "job", job["job_id"])
    return job


async def require_exploratory_run(run: dict = Depends(require_owned_run)) -> dict:
    """``require_owned_run`` for exploration routes: 409s on ABCD runs."""
    _require_parsing_mode(run, ParsingMode.EXPLORATORY, "run", run["run_id"])
    return run


async def load_exploration(
    reader: ClipScribeAsyncReaderDB, run_id: str
) -> ExplorationResponse:
    """An exploratory run's persisted report, if one was written (single read)."""
    row = await reader.get_exploration_result(run_id)
    return ExplorationResponse(
        run_id=run_id,
        report=row["payload"] if row else None,
        model=row["model"] if row else None,
        created_at=row["created_at"] if row else None,
    )
