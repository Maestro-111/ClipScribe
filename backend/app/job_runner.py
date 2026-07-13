"""Job orchestration: validate, persist, and dispatch (web-app-plan §10.5, §10.8).

``JobService`` validates a request, writes the ``jobs`` row, and dispatches the
run to one of two backends selected by ``settings.job_backend``:

- ``inline`` — submit to a single-slot ``ThreadPoolExecutor`` that mirrors
  Celery ``concurrency=1`` and runs the engine in this process (step 5).
- ``celery`` — ``send_task`` the job to a Redis-backed worker (step 8). The API
  loads no models in this mode; it only reads/writes the DB and dispatches.

The HTTP contract is identical either way: ``create_job`` returns immediately
with a job id and clients poll ``GET /jobs/{id}``. Both paths converge on
:func:`app.job_execution.run_job_core`, so lifecycle behavior never drifts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.errors import ProblemException
from app.job_execution import build_task_payload, now_iso
from app.models import JobCreatedResponse, JobCreateRequest, JobMode, JobStatus
from src.utils.ids import new_ulid

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from app.settings import Settings
    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB

logger = logging.getLogger("clip_scribe")

_TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELED.value}
)


class JobService:
    def __init__(
        self,
        reader: "ClipScribeReaderDB",
        writer: "ClipScribeWriterDB",
        settings: "Settings",
        *,
        builder: "ClipScribeBuilder | None" = None,
        executor: "ThreadPoolExecutor | None" = None,
        futures: "dict[str, Future[None]] | None" = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.settings = settings
        # Inline-mode only: the model-loaded builder + single-slot executor that
        # run the engine in-process. Both are None in celery mode.
        self.builder = builder
        self.executor = executor
        self.futures = futures if futures is not None else {}

    def create_job(self, req: JobCreateRequest) -> JobCreatedResponse:
        """Validate, persist a queued job, and dispatch it to the backend."""
        video_name = req.video_name
        video_path = req.video_path
        video_type = req.video_type
        video_path_abs: Path | None = None

        if req.mode == JobMode.PARSE:
            # run_id presence is enforced by the request model; existence is a
            # DB check (the API's contract: an enqueued parse job is valid).
            assert req.run_id is not None
            run = self.reader.get_run(req.run_id)
            if run is None:
                raise ProblemException(
                    status=404,
                    title="Not Found",
                    detail=f"run_id '{req.run_id}' not found",
                )
            run_id = req.run_id
            # Fall back to the existing run's video metadata for the parser.
            video_name = video_name or run.get("video_name")
            video_path = video_path or run.get("video_path")
            video_type = video_type or run.get("video_type")
        else:
            video_path_abs = self._resolve_input(req.video_path)
            run_id = new_ulid()

        job_id = new_ulid()

        self.writer.create_job(
            job_id=job_id,
            mode=req.mode.value,
            status=JobStatus.QUEUED.value,
            run_id=run_id,
            video_name=video_name,
            video_path=video_path,
            video_type=video_type,
            device=getattr(self.builder, "device", None),
            platform=req.platform.value,
            params_json=req.model_dump(mode="json"),
        )

        payload = build_task_payload(
            job_id=job_id,
            run_id=run_id,
            req=req,
            video_name=video_name,
            # Inline runs on this machine, so pass the resolved absolute path;
            # parse jobs (no local file) fall back to the request path.
            video_path=str(video_path_abs) if video_path_abs else req.video_path,
            video_type=video_type,
        )
        try:
            self._dispatch(job_id, payload)
        except Exception as exc:
            self.writer.update_job_if_status(
                job_id,
                allowed_statuses=(JobStatus.QUEUED.value,),
                status=JobStatus.FAILED.value,
                finished_at=now_iso(),
                error_text=str(exc),
            )
            raise

        return JobCreatedResponse(job_id=job_id, run_id=run_id, status=JobStatus.QUEUED)

    def _dispatch(self, job_id: str, payload: dict) -> None:
        """Hand the job off to the configured backend."""
        if self.settings.job_backend == "celery":
            from app.celery_app import celery_app

            result = celery_app.send_task("app.tasks.run_job", args=[payload])
            self.writer.update_job(job_id, celery_task_id=result.id)
            logger.info("Enqueued job %s to celery (task %s)", job_id, result.id)
        else:
            if self.executor is None or self.builder is None:
                raise ProblemException(
                    status=503,
                    title="Service Unavailable",
                    detail="Inline job execution is unavailable (models not loaded).",
                )
            future: "Future[None]" = self.executor.submit(self._run, payload)
            self.futures[job_id] = future

    def _run(self, payload: dict) -> None:
        """Inline executor callback: run one job on the API's own builder."""
        from app.job_execution import run_job_core

        try:
            assert self.builder is not None  # guarded in _dispatch
            run_job_core(self.builder, payload)
        finally:
            self.futures.pop(payload["job_id"], None)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or running job.

        Queued jobs are prevented from starting (``Future.cancel()`` inline, or
        ``revoke`` for celery). Running jobs cannot be interrupted mid-engine yet
        (cooperative cancel is web-app-plan §10.10); the DB is marked canceled
        immediately and :func:`run_job_core` respects it when the engine returns
        so the status is never overwritten to completed.
        """
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        cancellable = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
        if job["status"] not in cancellable:
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"job '{job_id}' is '{job['status']}' — "
                    "only queued or running jobs can be canceled"
                ),
            )

        # Signal a running engine to stop at its next checkpoint (cooperative
        # cancel). Set before the backend-specific handling below so the flag is
        # already visible by the time the running pipeline next polls it. This is
        # a no-op for a queued job (which the backend handling stops outright),
        # but harmless — the flag self-expires and a retry gets a fresh job_id.
        from app.events import signal_cancel

        signal_cancel(self.settings.redis_url, job_id)

        if self.settings.job_backend == "celery":
            task_id = job.get("celery_task_id")
            if task_id:
                from app.celery_app import celery_app

                # No terminate=True: hard-kill would leak files + GPU state and,
                # under the solo pool, take down the worker. Revoke stops a
                # still-queued task; a running one stops cooperatively at its
                # next check (the cancel flag above) and its terminal write is
                # suppressed because the row is already 'canceled'.
                celery_app.control.revoke(task_id)
        else:
            future = self.futures.get(job_id)
            if future is not None:
                # Succeeds only if still queued in the executor; a running job
                # can't be thread-killed, so it stops via the cancel flag above.
                future.cancel()

        canceled = self.writer.update_job_if_status(
            job_id,
            allowed_statuses=tuple(cancellable),
            status=JobStatus.CANCELED.value,
            finished_at=now_iso(),
        )
        if not canceled:
            latest = self.reader.get_job(job_id)
            latest_status = latest["status"] if latest else "missing"
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"job '{job_id}' is '{latest_status}' — "
                    "only queued or running jobs can be canceled"
                ),
            )

    def delete_job(self, job_id: str) -> None:
        """Delete a job, canceling it first if it is still queued or running.

        Terminal jobs are removed outright. A queued/running job is canceled
        (via :meth:`cancel_job`, so the queued task is revoked / de-queued and
        the DB is marked canceled) before its row is deleted. Note that under
        the current best-effort cancel, a *running* engine keeps executing until
        it returns — but its terminal write targets a row that no longer exists,
        so the delete is safe; the in-flight run simply finishes unrecorded.
        """
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        if job["status"] not in _TERMINAL_JOB_STATUSES:
            # Non-terminal ⇒ queued or running, both cancellable. Stop it first
            # so we don't orphan a live future / queued celery task.
            self.cancel_job(job_id)
        self.writer.delete_job(job_id)

    def retry_job(self, job_id: str) -> JobCreatedResponse:
        """Create a fresh job from the stored params of a failed/canceled job."""
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        retryable = {
            JobStatus.FAILED.value,
            JobStatus.CANCELED.value,
            JobStatus.COMPLETED.value,
        }
        if job["status"] not in retryable:
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"job '{job_id}' is '{job['status']}' — "
                    "only completed, failed, or canceled jobs can be retried"
                ),
            )
        params = job.get("params_json")
        if not params:
            raise ProblemException(
                status=422,
                title="Unprocessable Entity",
                detail=f"job '{job_id}' has no stored params and cannot be retried",
            )
        req = JobCreateRequest.model_validate(params)
        return self.create_job(req)

    def _resolve_input(self, rel_path: str | None) -> Path:
        """Resolve a request path under INPUT_DIR, guarding against traversal."""
        if not rel_path:
            raise ProblemException(
                status=400, title="Bad Request", detail="video_path is required"
            )
        base = self.settings.input_dir
        candidate = (base / rel_path).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            raise ProblemException(
                status=400,
                title="Bad Request",
                detail="video_path escapes the input directory",
            )
        if not candidate.is_file():
            raise ProblemException(
                status=404, title="Not Found", detail=f"video not found: {rel_path}"
            )
        return candidate
