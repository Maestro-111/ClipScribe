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

logger = logging.getLogger("clip_scribe")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    Both paths publish live progress to the job's Redis stream via a
    :class:`RedisProgressReporter` (web-app-plan §9): the reporter is wired into
    the engine, and the ``current_job_id`` contextvar is set for the duration so
    the log bridge tags this job's log records. Both degrade to no-ops when Redis
    is down, so a job still runs — it just has no live tail.
    """
    from app.events import (
        current_job_id,
        install_job_log_bridge,
        make_canceller,
        make_reporter,
    )
    from app.settings import get_settings
    from src.utils.clip_scribe_cancel import JobCanceled
    from src.utils.clip_scribe_video_storage import make_video_storage

    writer = builder.writer_db
    job_id = payload["job_id"]
    run_id = payload["run_id"]

    # Set once the log bridge is installed; reset in the finally. None until then
    # so a setup failure before that point doesn't try to reset an unset token.
    token = None

    try:
        # Mark the job running FIRST, using only payload fields that can't fail,
        # so any error while building the per-job dependencies below is recorded
        # as a failure on a RUNNING row instead of leaving the job stuck QUEUED
        # (the failure handler is guarded to RUNNING). A queued job canceled or
        # deleted in the meantime is not RUNNING-eligible, so this returns.
        started = writer.update_job_if_status(
            job_id,
            allowed_statuses=(JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            status=JobStatus.RUNNING.value,
            started_at=now_iso(),
        )

        if not started:
            logger.info("Job %s was not startable; skipping execution", job_id)
            return

        req = JobCreateRequest.model_validate(payload["req"])
        settings = get_settings()
        redis_url = settings.redis_url
        # The payload carries a logical storage key, not a local path: the API
        # and worker may not share a filesystem (a cloud bucket + remote
        # worker). Bring the bytes local here, right before extraction, and
        # release after. Constructing the storage client can fail (e.g. bad GCS
        # credentials) — kept inside the try so that surfaces as a failed job.
        storage = make_video_storage(
            settings.storage_backend, settings.input_dir, settings.gcs_bucket
        )
        reporter = make_reporter(redis_url, job_id)
        canceller = make_canceller(redis_url, job_id)
        install_job_log_bridge(redis_url)
        token = current_job_id.set(job_id)

        platform_conf = build_platform(
            req.platform.value, **req.resolved_params.to_build_kwargs()
        )
        if platform_conf is None:
            raise ValueError(f"unsupported platform: {req.platform.value}")

        local_video = storage.materialize(payload["video_path"] or "")
        try:
            engine = builder.build_clip_scribe(
                video_name=payload["video_name"] or "",
                video_path=str(local_video),
                # Persist the logical storage key (not the ephemeral scratch
                # path) so the run can be served after the scratch is released.
                video_key=payload["video_path"] or "",
                video_type=payload["video_type"],
                clib_scribe_mode=req.mode.value,
                clib_scribe_platform_name=req.platform.value,
                clib_scribe_platform_conf=platform_conf,
                user_hints=req.user_hints,
                generate_hint_from_name=req.generate_hint_from_name,
                progress_reporter=reporter,
                cancel_token=canceller,
            )
            engine.run(run_id=run_id)
        finally:
            # Release any scratch copy the backend downloaded (no-op for local
            # storage, where the materialized path is the stored file itself).
            storage.release(local_video)
    except JobCanceled:
        # Cooperative cancel: the engine stopped at a checkpoint. cancel_job has
        # already flipped the row to 'canceled', so this guarded write is
        # normally a no-op; it also covers the defensive case where the flag was
        # set without the DB update. Either way the row stays 'canceled', not
        # 'failed', and no error_text is recorded.
        logger.info("Job %s canceled; pipeline stopped cooperatively", job_id)
        writer.update_job_if_status(
            job_id,
            allowed_statuses=(JobStatus.RUNNING.value,),
            status=JobStatus.CANCELED.value,
            finished_at=now_iso(),
        )
    except Exception as exc:  # noqa: BLE001 - recorded on the job row
        logger.exception("Job %s failed", job_id)
        writer.update_job_if_status(
            job_id,
            allowed_statuses=(JobStatus.RUNNING.value,),
            status=JobStatus.FAILED.value,
            error_text=str(exc),
            finished_at=now_iso(),
        )
    else:
        writer.update_job_if_status(
            job_id,
            allowed_statuses=(JobStatus.RUNNING.value,),
            status=JobStatus.COMPLETED.value,
            finished_at=now_iso(),
        )
    finally:
        # token is None only if setup failed before the contextvar was set.
        if token is not None:
            current_job_id.reset(token)
