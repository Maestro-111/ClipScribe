"""FastAPI application entry point for the ClipScribe web API.

Step 5 of web-app-plan §10: the sync-path API. This process is intentionally a
monolith for now — it builds one long-lived :class:`ClipScribeBuilder` at
startup (mirroring the eventual Celery ``worker_process_init``) and runs jobs
in-process on a single-slot executor. Celery/Redis enqueue (step 8) and the
SSE bridge (step 9) replace the inline path later without changing the HTTP
contract.

Run locally from ``backend/``::

    uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.errors import register_error_handlers
from app.routes import artifacts, health, jobs, meta, runs, uploads
from app.settings import get_settings

if TYPE_CHECKING:
    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder

logger = logging.getLogger("clip_scribe")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.builder = None
    app.state.executor = None
    app.state.futures = {}  # job_id -> Future[None]

    settings.input_dir.mkdir(parents=True, exist_ok=True)

    if settings.load_models:
        # Heavy: loads SAM2/DINO/Whisper/DINOv2/MTCNN/PaddleOCR once. The
        # single-slot executor is the local stand-in for Celery concurrency=1
        # (shared GPU-resident models must not run two jobs at once).
        from src.clip_scribe.build_clip_scribe import ClipScribeBuilder

        logger.info("Loading ClipScribeBuilder (this can take 30-60s)...")
        app.state.builder = ClipScribeBuilder()
        app.state.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="clipscribe-job"
        )
        logger.info("ClipScribeBuilder ready; API accepting jobs.")
    else:
        logger.warning(
            "CLIPSCRIBE_API_LOAD_MODELS is off; builder not loaded. "
            "Job execution is disabled (read-only / test mode)."
        )

    try:
        yield
    finally:
        executor: ThreadPoolExecutor | None = app.state.executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        builder: ClipScribeBuilder | None = app.state.builder
        if builder is not None:
            try:
                # right now share the same connection pool, still close both explicitly
                builder.writer_db.close()
                builder.reader_db.close()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                logger.warning("Error closing DB on shutdown", exc_info=True)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ClipScribe API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(meta.router)
    app.include_router(jobs.router)
    app.include_router(runs.router)
    app.include_router(uploads.router)
    app.include_router(artifacts.router)

    return app


app = create_app()
