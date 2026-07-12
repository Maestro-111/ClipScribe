"""Live-progress transport: Redis-Streams reporter + log bridge (web-app-plan §5, §9, §16).

The pipeline core emits structured events through the abstract
:class:`~src.utils.progress.ProgressReporter` (``src/utils/progress.py``); this
module is the web-app sink that publishes them to a per-job Redis **stream** so
the SSE endpoint (``GET /jobs/{id}/events``) can both replay history to a late
subscriber and tail live updates.

A single stream (not pub/sub) is used deliberately: pub/sub drops messages when
no one is subscribed, so a user opening the live page mid-run would miss every
event emitted before they connected (§16). ``XADD`` + ``XREAD`` from id ``0``
gives replay *and* live tail with one mechanism.

This module is torch-free and safe to import in the slim API process (§8): it
depends only on ``redis`` and ``src.utils.progress``.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any, Mapping

import redis

from src.utils.progress import NullProgressReporter, ProgressReporter

logger = logging.getLogger("clip_scribe")

# Bound the stream length so a run's events don't grow without limit. A run emits
# a handful of events per shot plus one entry per INFO log record, so 2000 covers
# even long videos; approximate trimming (``~``) is cheaper for Redis.
STREAM_MAXLEN = 2000
# TTL applied once a terminal event is written, so finished jobs' streams expire
# instead of lingering forever — long enough for a late page-load to still replay.
STREAM_TTL_SECONDS = 24 * 3600

# Terminal events end the SSE stream. The engine emits completed/failed; a
# canceled job may never reach the engine, so the SSE endpoint also watches the
# job row (see routes/jobs.py) as a backstop.
TERMINAL_EVENTS = frozenset({"job.completed", "job.failed"})

# Set by run_job_core for the duration of a job so the log bridge can tag records
# with their job id without threading it through every logging call site (§9.10).
current_job_id: ContextVar[str | None] = ContextVar("current_job_id", default=None)


def stream_key(job_id: str) -> str:
    return f"job:{job_id}:stream"


# Per-phase share of overall progress, mirroring the frontend live page's
# weighting (web-app-plan §7). Kept here next to the events they summarize so the
# jobs-list progress bar (GET /jobs/{id}/progress) has a single server-side
# source and doesn't need a live SSE connection per row.
_PHASE_WEIGHT = {
    "scene_detection": 0.05,
    "audio": 0.15,
    "shot_processing": 0.7,
    "finalize": 0.1,
    "parse": 0.3,
}


def summarize_progress(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Reduce a job's stream events into a coarse progress summary.

    ``events`` is an ordered list of ``(type, data)`` pairs (log entries are
    ignored). Returns ``percent`` (0-100) plus the running phase and shot counts
    so the jobs list can render a bar without reconstructing full live state.
    """
    phase_order: list[str] = []
    phase_status: dict[str, str] = {}
    total_shots: int | None = None
    shots_done = 0
    current_phase: str | None = None
    terminal = False

    for event_type, data in events:
        if event_type == "job.started":
            phase_order = list(data.get("phases") or [])
            phase_status = {p: "pending" for p in phase_order}
        elif event_type == "phase.started":
            phase = data.get("phase")
            if phase:
                phase_status[phase] = "running"
                current_phase = phase
                if phase == "shot_processing" and data.get("total_shots"):
                    total_shots = int(data["total_shots"])
        elif event_type == "phase.completed":
            phase = data.get("phase")
            if phase:
                phase_status[phase] = "completed"
            if data.get("total_shots"):
                total_shots = int(data["total_shots"])
        elif event_type == "shot.completed":
            shots_done += 1
        elif event_type in TERMINAL_EVENTS:
            terminal = True

    if terminal:
        return {
            "percent": 100.0,
            "phase": current_phase,
            "shots_done": shots_done,
            "total_shots": total_shots,
        }

    total_weight = 0.0
    done_weight = 0.0
    for phase in phase_order:
        weight = _PHASE_WEIGHT.get(phase, 0.1)
        total_weight += weight
        status = phase_status.get(phase, "pending")
        if status == "completed":
            done_weight += weight
        elif status == "running":
            if phase == "shot_processing" and total_shots:
                done_weight += weight * min(shots_done / total_shots, 0.99)
            else:
                done_weight += weight * 0.1
    percent = round(100 * done_weight / total_weight, 1) if total_weight else 0.0
    return {
        "percent": percent,
        "phase": current_phase,
        "shots_done": shots_done,
        "total_shots": total_shots,
    }


def _entry(event_type: str, data: Mapping[str, Any] | None) -> dict[str, str]:
    """A stream entry is flat string fields: the type plus a JSON payload."""
    return {"type": event_type, "data": json.dumps(dict(data or {}))}


class RedisProgressReporter(ProgressReporter):
    """Publishes progress events to ``job:{job_id}:stream`` via ``XADD``.

    Bound to a single job. Per the :class:`ProgressReporter` contract, ``emit``
    must never raise — a Redis hiccup must not break a running job — so every
    Redis call is wrapped and failures are logged and swallowed.
    """

    def __init__(self, job_id: str, client: "redis.Redis") -> None:
        self._job_id = job_id
        self._key = stream_key(job_id)
        self._client = client

    def emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        try:
            self._client.xadd(
                self._key,
                _entry(event_type, data),
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            if event_type in TERMINAL_EVENTS:
                self._client.expire(self._key, STREAM_TTL_SECONDS)
        except Exception:  # noqa: BLE001 - reporting must never break the job
            logger.warning(
                "Failed to publish %s for job %s",
                event_type,
                self._job_id,
                exc_info=True,
            )


def make_reporter(redis_url: str, job_id: str) -> ProgressReporter:
    """Return a Redis reporter for ``job_id``, or Null if Redis is unreachable.

    Called by both execution paths — the inline API executor and the Celery
    worker. Falling back to :class:`NullProgressReporter` keeps a job runnable
    even when the live-progress transport is down.
    """
    try:
        client = redis.Redis.from_url(redis_url)
        client.ping()
    except Exception:  # noqa: BLE001 - degrade to no-op; the job still runs
        logger.warning(
            "Redis unavailable at %s; live progress disabled for job %s",
            redis_url,
            job_id,
        )
        return NullProgressReporter()
    return RedisProgressReporter(job_id, client)


class JobLogStreamHandler(logging.Handler):
    """Mirrors ``clip_scribe`` log records into the current job's stream (§9.10).

    Reads the job id from :data:`current_job_id` so no logging call site changes.
    Records emitted outside a job context (id unset) are dropped, so ambient
    startup logging never lands in a job stream. Attach once per process.
    """

    def __init__(self, client: "redis.Redis", level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._client = client

    def emit(self, record: logging.LogRecord) -> None:
        job_id = current_job_id.get()
        if job_id is None:
            return
        try:
            self._client.xadd(
                stream_key(job_id),
                _entry(
                    "log", {"level": record.levelname, "message": record.getMessage()}
                ),
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
        except Exception:  # noqa: BLE001 - logging must never break the job
            self.handleError(record)


def install_job_log_bridge(redis_url: str) -> None:
    """Attach the job-log stream handler to the ``clip_scribe`` logger once.

    Idempotent: repeated calls (one per job) won't double-attach. A no-op if
    Redis is unreachable — the job still runs, only its log tail is missing.
    """

    root = logging.getLogger("clip_scribe")

    if any(isinstance(h, JobLogStreamHandler) for h in root.handlers):
        return
    try:
        client = redis.Redis.from_url(redis_url)
        client.ping()
    except Exception:  # noqa: BLE001 - degrade to no-op
        logger.warning("Redis unavailable at %s; job log streaming disabled", redis_url)
        return
    root.addHandler(JobLogStreamHandler(client))
