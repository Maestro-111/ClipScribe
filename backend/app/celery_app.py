"""Celery application — the thin, shared broker handle (web-app-plan §8, §10.8).

This module is deliberately lightweight: it imports only Celery and the API
settings, never torch or the pipeline. Both the API and the worker import it,
but for different reasons:

- The **API** imports it to call ``celery_app.send_task("app.tasks.run_job", …)``
  by name. ``send_task`` does not import the task module, so the ML stack never
  leaks into the slim API process — this is the import-boundary trick in §8.
- The **worker** is launched as ``celery -A app.celery_app worker`` and, at
  startup, imports the modules listed in ``include`` (``app.tasks``), which pull
  in ``ClipScribeBuilder`` and register the task.

The contract between the two processes is the string task name, nothing more.
"""

from __future__ import annotations

from celery import Celery

from app.settings import get_settings

_settings = get_settings()

celery_app = Celery(
    "clipscribe",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # One GPU/MPS worker handles one job at a time; a job can run for minutes.
    # Fetch a single task at a time so a busy worker leaves the backlog in Redis
    # for other workers/machines rather than hoarding it (web-app-plan §12).
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Results are advisory; job state lives in the ``jobs`` table. Expire the
    # Redis result entries so the broker doesn't grow unbounded.
    result_expires=3600,
)
