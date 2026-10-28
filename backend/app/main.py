"""FastAPI application entry point for the ClipScribe web API.

The app supports both execution modes from web-app-plan §10: ``inline`` builds
one long-lived :class:`ClipScribeBuilder` at startup and runs jobs in-process on
a single-slot executor, while ``celery`` keeps the API model-free and dispatches
jobs to a Redis-backed worker. Redis Streams back the live-progress SSE route in
both modes without changing the HTTP contract.

Run locally from ``backend/``::

    uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.errors import register_error_handlers
from app.routes import artifacts, chat, health, jobs, meta, runs, uploads
from app.settings import get_settings
from clip_scribe.db import (
    ClipScribeAsyncReaderDB,
    ClipScribeAsyncWriterDB,
    create_async_db_engine,
    resolve_database_url,
    resolve_pool_settings,
)
from clip_scribe.db.engine import ensure_sqlite_parent_directory
from clip_scribe.utils.clip_scribe_logging import configure_logging

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging({"api": settings.api_log_level})

    app.state.settings = settings
    app.state.builder = None
    app.state.executor = None
    app.state.futures = {}  # job_id -> Future[None]
    app.state.reader_db = None
    app.state.writer_db = None
    app.state.db_engine = None

    settings.input_dir.mkdir(parents=True, exist_ok=True)

    try:
        db_url = resolve_database_url()
        ensure_sqlite_parent_directory(db_url)
        pool_size, max_overflow = resolve_pool_settings()

        async_db_engine = create_async_db_engine(
            db_url, pool_size=pool_size, max_overflow=max_overflow
        )
        app.state.db_engine = async_db_engine
        app.state.reader_db = ClipScribeAsyncReaderDB(async_engine=async_db_engine)
        app.state.writer_db = ClipScribeAsyncWriterDB(async_engine=async_db_engine)

        if settings.job_backend == "celery":
            logger.info(
                "DB ready; ML stack not loaded in API; enqueuing jobs to %s",
                settings.redis_url,
            )
        elif settings.load_models:
            # Heavy: loads SAM2/DINO/Whisper/DINOv2/MTCNN/PaddleOCR once. The
            # single-slot executor is the local stand-in for Celery concurrency=1
            # (shared GPU-resident models must not run two jobs at once).

            logger.info("Job execution: inline thread")

            from clip_scribe.clip_scribe_core.build_clip_scribe import ClipScribeBuilder

            logger.info(
                "Loading ClipScribeBuilder and ML stack (this can take 30-60s)..."
            )

            builder = ClipScribeBuilder(device=settings.clip_scribe_device)

            app.state.builder = builder
            app.state.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="clipscribe-job"
            )
            logger.info("ClipScribeBuilder ready; API accepting jobs (inline).")
        else:
            logger.warning(
                "DB ready; inline mode is read-only because models are disabled"
            )

        yield

    finally:
        executor: ThreadPoolExecutor | None = app.state.executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        builder = app.state.builder
        if builder is not None:
            try:
                # The builder owns a separate synchronous engine from the API's
                # async engine. Its reader and writer share that sync engine.
                builder.writer_db.close()
                builder.reader_db.close()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                logger.warning("Error closing DB on shutdown", exc_info=True)
        db_engine = app.state.db_engine
        if db_engine is not None:
            try:
                await db_engine.dispose()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                logger.warning("Error disposing DB engine on shutdown", exc_info=True)


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
    app.include_router(chat.router)
    app.include_router(chat.job_router)
    app.include_router(uploads.router)
    app.include_router(artifacts.router)

    return app


app = create_app()
