"""FastAPI dependency providers.

Each is a thin accessor over ``app.state`` (populated in ``main.lifespan``) so
routes stay decoupled from how the builder/executor are constructed, and tests
can swap them via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from app.errors import ProblemException
from app.settings import Settings, get_settings

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB


def settings_dep() -> Settings:
    return get_settings()


def get_builder(request: Request) -> "ClipScribeBuilder":
    builder = getattr(request.app.state, "builder", None)
    if builder is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Models are not loaded; this endpoint is unavailable.",
        )
    return builder


def get_reader(request: Request) -> "ClipScribeReaderDB":
    # Read from app.state, not the builder: in celery mode the API has a DB
    # connection but no builder (no models). lifespan populates reader_db from
    # the builder (inline) or a standalone engine (celery).
    reader = getattr(request.app.state, "reader_db", None)
    if reader is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Database is not available; this endpoint is unavailable.",
        )
    return reader


def get_writer(request: Request) -> "ClipScribeWriterDB":
    writer = getattr(request.app.state, "writer_db", None)
    if writer is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Database is not available; this endpoint is unavailable.",
        )
    return writer


def get_executor(request: Request) -> "ThreadPoolExecutor":
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="Job executor is not available; this endpoint is unavailable.",
        )
    return executor


def get_futures(request: Request) -> "dict[str, Future[None]]":
    return getattr(request.app.state, "futures", {})
