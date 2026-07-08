"""Shared job execution — the one place that drives a job to a terminal state.

Both execution paths call :func:`run_job_core`:

- the **inline** single-slot executor (``JobService._run``, web-app-plan §10.5), and
- the **Celery** task (``app.tasks.run_job``, §10.8).

Keeping the lifecycle here means the two paths can never drift: they mark the
job ``running``, build the per-job engine off a long-lived
:class:`ClipScribeBuilder`, run it, and record ``completed`` / ``failed`` while
respecting an external cancel. The only difference between the paths is *how*
the work is dispatched, not what the work does.

The payload is a plain JSON-serializable dict so it survives the Celery broker
unchanged; the inline path builds the same dict for symmetry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.models import JobCreateRequest, JobStatus
from src.clip_scribe.build_clip_scribe_plalform import build_platform

if TYPE_CHECKING:
    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
    from src.db import ClipScribeReaderDB

logger = logging.getLogger("clip_scribe")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_canceled(reader: "ClipScribeReaderDB", job_id: str) -> bool:
    """Return True if the job was externally canceled while running.

    Cancellation marks the DB row immediately; the engine cannot yet be
    interrupted mid-run (cooperative cancel is web-app-plan §10.10), so both
    execution paths check this before writing a terminal status and never
    overwrite ``canceled`` with ``completed``/``failed``.
    """
    job = reader.get_job(job_id)
    return job is not None and job.get("status") == JobStatus.CANCELED.value


def build_task_payload(
    *,
    job_id: str,
    run_id: str,
    req: JobCreateRequest,
    video_name: str | None,
    video_path: str | None,
    video_type: str | None,
) -> dict[str, Any]:
    """Assemble the JSON-serializable payload handed to the executor / Celery."""
    return {
        "job_id": job_id,
        "run_id": run_id,
        "video_name": video_name,
        "video_path": video_path,
        "video_type": video_type,
        "req": req.model_dump(mode="json"),
    }


def run_job_core(builder: "ClipScribeBuilder", payload: dict[str, Any]) -> None:
    """Run one job to a terminal state, recording lifecycle on the ``jobs`` row.

    ``builder`` is the long-lived, model-loaded builder (the API's in inline
    mode, the worker's in celery mode). ``payload`` is what
    :func:`build_task_payload` produced.
    """
    writer = builder.writer_db
    reader = builder.reader_db

    job_id = payload["job_id"]
    run_id = payload["run_id"]
    req = JobCreateRequest.model_validate(payload["req"])

    writer.update_job(job_id, status=JobStatus.RUNNING.value, started_at=now_iso())
    try:
        platform_conf = build_platform(
            req.platform.value, **req.resolved_params.to_build_kwargs()
        )
        if platform_conf is None:
            raise ValueError(f"unsupported platform: {req.platform.value}")

        engine = builder.build_clip_scribe(
            video_name=payload["video_name"] or "",
            video_path=payload["video_path"] or "",
            video_type=payload["video_type"],
            clib_scribe_mode=req.mode.value,
            clib_scribe_platform_name=req.platform.value,
            clib_scribe_platform_conf=platform_conf,
            user_hints=req.user_hints,
            generate_hint_from_name=req.generate_hint_from_name,
        )
        engine.run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - recorded on the job row
        logger.exception("Job %s failed", job_id)
        if not is_canceled(reader, job_id):
            writer.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                error_text=str(exc),
                finished_at=now_iso(),
            )
    else:
        if not is_canceled(reader, job_id):
            writer.update_job(
                job_id, status=JobStatus.COMPLETED.value, finished_at=now_iso()
            )
