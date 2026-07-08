"""Celery worker task module (web-app-plan §8, §10.8) — WORKER-ONLY.

This module imports the full ML stack via :class:`ClipScribeBuilder`. It is
imported only by the worker (through ``celery_app``'s ``include``), never by the
API, which dispatches by task name. See ``app/celery_app.py`` for the boundary.

Model loading happens once per worker process and is amortized across every job
that process handles:

- Under the default **prefork** pool, ``worker_process_init`` fires in each
  child after fork and pre-warms the builder there.
- Under ``--pool=solo`` (recommended on macOS/MPS, which can't fork the CV
  stack safely) that signal does not fire, so the first task lazy-loads via
  :func:`get_builder`. The idempotent guard means neither path double-loads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.job_execution import run_job_core

if TYPE_CHECKING:
    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder

logger = logging.getLogger("clip_scribe")

_BUILDER: "ClipScribeBuilder | None" = None


def get_builder() -> "ClipScribeBuilder":
    """Return the process-wide builder, loading it once on first use.

    Device is supplied from settings (which read ``CLIPSCRIBE_DEVICE`` from the
    env / .env loaded at ``app.settings`` import — same value the inline API
    passes), so the container can force ``cpu``/``cuda`` without the core builder
    ever touching the environment. ``.env`` was loaded in this same worker
    process at import, so ``os.environ`` (and thus the cached settings) already
    hold the value by the time a task runs.
    """
    global _BUILDER
    if _BUILDER is None:
        from app.settings import get_settings
        from src.clip_scribe.build_clip_scribe import ClipScribeBuilder

        device = get_settings().clip_scribe_device
        logger.info(
            "Worker booting ClipScribeBuilder (device=%s, can take 30-60s)...",
            device,
        )
        _BUILDER = ClipScribeBuilder(device=device)
        logger.info("Worker ClipScribeBuilder ready.")
    return _BUILDER


@worker_process_init.connect
def _boot_builder(**_: Any) -> None:
    """Pre-warm the builder in each prefork child so the first job isn't slow."""
    get_builder()


@celery_app.task(name="app.tasks.run_job")
def run_job(payload: dict[str, Any]) -> None:
    """Execute one job on this worker's long-lived, model-loaded builder."""
    run_job_core(get_builder(), payload)
