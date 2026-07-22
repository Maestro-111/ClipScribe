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
from app.models import (
    JobChild,
    JobCreatedResponse,
    JobCreateRequest,
    JobMode,
    JobResponse,
    JobStatus,
)
from src.utils.ids import new_ulid

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future, ThreadPoolExecutor

    from app.models import VideoInput
    from app.settings import Settings
    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder
    from src.db import ClipScribeReaderDB, ClipScribeWriterDB
    from src.utils.clip_scribe_video_storage import VideoStorage

logger = logging.getLogger("clip_scribe")

_TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELED.value}
)
_CANCELLABLE_JOB_STATUSES = frozenset({JobStatus.QUEUED.value, JobStatus.RUNNING.value})


def aggregate_status(child_statuses: "Sequence[str]") -> str:
    """Derive a batch parent's status from its children (read-time, §2.1).

    - all children completed → ``completed``
    - all terminal, at least one failure → ``failed`` (else ``canceled``)
    - any child still queued/running → ``running`` if any progress or a running
      child, else ``queued`` (nothing started yet)
    """
    if not child_statuses:
        return JobStatus.QUEUED.value

    statuses = set(child_statuses)
    completed = JobStatus.COMPLETED.value
    if statuses == {completed}:
        return completed
    if statuses <= _TERMINAL_JOB_STATUSES:
        if JobStatus.FAILED.value in statuses:
            return JobStatus.FAILED.value
        if JobStatus.CANCELED.value in statuses:
            return JobStatus.CANCELED.value
        return completed
    # At least one child is still queued or running.
    if JobStatus.RUNNING.value in statuses or (statuses & _TERMINAL_JOB_STATUSES):
        return JobStatus.RUNNING.value
    return JobStatus.QUEUED.value


def _batch_label(videos: "Sequence[VideoInput]") -> str | None:
    """A human label for the parent row: the first video, plus a count if batched."""
    if not videos:
        return None
    if len(videos) == 1:
        return videos[0].video_name
    return f"{videos[0].video_name} (+{len(videos) - 1} more)"


def build_job_response(reader: "ClipScribeReaderDB", row: dict) -> JobResponse:
    """Assemble a :class:`JobResponse`, aggregating parent status from children.

    A parent row (``parent_job_id`` NULL) carries its children and an aggregated
    status; a child/leaf row is returned with its own status and no children.
    """
    children: list[JobChild] = []
    status = row.get("status", JobStatus.QUEUED.value)
    if row.get("parent_job_id") is None:
        child_rows = reader.get_child_jobs(row["job_id"])
        if child_rows:
            children = [JobChild(**cr) for cr in child_rows]
            status = aggregate_status([c.status for c in children])
    return JobResponse(**{**row, "status": status, "children": children})


class JobService:
    def __init__(
        self,
        reader: "ClipScribeReaderDB",
        writer: "ClipScribeWriterDB",
        settings: "Settings",
        storage: "VideoStorage",
        user_id: str,
        *,
        builder: "ClipScribeBuilder | None" = None,
        executor: "ThreadPoolExecutor | None" = None,
        futures: "dict[str, Future[None]] | None" = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.settings = settings
        # Source-video storage backend + requesting user, for validating that a
        # job's video keys still resolve to stored objects at dispatch time.
        self.storage = storage
        self.user_id = user_id
        # Inline-mode only: the model-loaded builder + single-slot executor that
        # run the engine in-process. Both are None in celery mode.
        self.builder = builder
        self.executor = executor
        self.futures = futures if futures is not None else {}

    def create_job(self, req: JobCreateRequest) -> JobCreatedResponse:
        """Fan a batch request out to a parent job + one child run per video.

        Every request writes one parent row (a container, never executed) and
        one child row + ``run_id`` per video, each dispatched to the backend
        exactly like a solo job (docs/deployment.md §2.1). The parent's status
        is derived from its children at read time, so the parent is never
        dispatched. Returns the parent job id (``run_id`` is null — children own
        the runs).
        """
        if req.mode == JobMode.PARSE or not req.videos:
            # The web API always sends full/extract with >=1 video; parse is
            # dev-only and runs via main.py, not here. Fail loud rather than
            # persist an empty parent.
            raise ProblemException(
                status=400,
                title="Bad Request",
                detail="a job requires at least one video "
                "(parse mode is not supported via the job API)",
            )
        # Surface an unavailable backend once, before any rows are written, so a
        # failed request never leaves a half-created parent behind.
        self._ensure_dispatchable()
        # Validate every input up front: a bad/missing key 404s/400s before we
        # persist. Returns the logical storage key, not a local path.
        resolved = [(v, self._validate_input(v.video_path)) for v in req.videos]

        device = getattr(self.builder, "device", None)
        parent_id = new_ulid()
        self.writer.create_job(
            job_id=parent_id,
            mode=req.mode.value,
            status=JobStatus.QUEUED.value,
            parent_job_id=None,
            run_id=None,
            video_name=_batch_label(req.videos),
            device=device,
            platform=req.platform.value,
            params_json=req.model_dump(mode="json"),
        )

        for video, video_key in resolved:
            child_id = new_ulid()
            run_id = new_ulid()
            # The child stores a single-video request so retrying it re-runs just
            # this video (retrying the parent re-fans the whole batch).
            child_req = req.model_copy(update={"videos": [video]})
            self.writer.create_job(
                job_id=child_id,
                mode=req.mode.value,
                status=JobStatus.QUEUED.value,
                parent_job_id=parent_id,
                run_id=run_id,
                video_name=video.video_name,
                video_path=video.video_path,
                video_type=video.video_type,
                device=device,
                platform=req.platform.value,
                params_json=child_req.model_dump(mode="json"),
            )
            payload = build_task_payload(
                job_id=child_id,
                run_id=run_id,
                req=child_req,
                video_name=video.video_name,
                # The logical storage key; the worker materializes it locally.
                video_path=video_key,
                video_type=video.video_type,
            )
            try:
                self._dispatch(child_id, payload)
            except Exception as exc:  # noqa: BLE001 - recorded on the child row
                # Best-effort: one child failing to dispatch doesn't abort the
                # batch; the parent's aggregate status reflects the failure.
                logger.exception("Failed to dispatch child job %s", child_id)
                self.writer.update_job_if_status(
                    child_id,
                    allowed_statuses=(JobStatus.QUEUED.value,),
                    status=JobStatus.FAILED.value,
                    finished_at=now_iso(),
                    error_text=str(exc),
                )

        return JobCreatedResponse(
            job_id=parent_id, run_id=None, status=JobStatus.QUEUED
        )

    def _ensure_dispatchable(self) -> None:
        """Fail fast (503) if the inline backend cannot run a job right now."""
        if self.settings.job_backend != "celery" and (
            self.executor is None or self.builder is None
        ):
            raise ProblemException(
                status=503,
                title="Service Unavailable",
                detail="Inline job execution is unavailable (models not loaded).",
            )

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

    def _effective_status(self, job: dict) -> str:
        """A job's user-facing status: aggregated from children for a parent."""
        if job.get("parent_job_id") is None:
            children = self.reader.get_child_jobs(job["job_id"])
            if children:
                return aggregate_status([c["status"] for c in children])
        return job.get("status", JobStatus.QUEUED.value)

    def _cancel_one(self, job: dict, *, strict: bool = True) -> None:
        """Cancel a single (leaf) job. ``strict`` raises 409 when not cancellable.

        Queued jobs are prevented from starting (``Future.cancel()`` inline, or
        ``revoke`` for celery). Running jobs are signaled through the Redis-backed
        cooperative cancel token and stop at the next safe checkpoint. The DB is
        marked canceled immediately and :func:`run_job_core` never overwrites it
        to completed. With ``strict=False`` (batch children) an already-terminal
        job is skipped silently.
        """
        job_id = job["job_id"]
        if job["status"] not in _CANCELLABLE_JOB_STATUSES:
            if strict:
                raise ProblemException(
                    status=409,
                    title="Conflict",
                    detail=(
                        f"job '{job_id}' is '{job['status']}' — "
                        "only queued or running jobs can be canceled"
                    ),
                )
            return

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
            allowed_statuses=tuple(_CANCELLABLE_JOB_STATUSES),
            status=JobStatus.CANCELED.value,
            finished_at=now_iso(),
        )
        if not canceled and strict:
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

    def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or running job.

        For a batch parent, cancels every still-cancellable child best-effort
        (the parent status is derived, so no parent row is written). For a leaf
        job, cancels it and 409s if it is already terminal.
        """
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        if job.get("parent_job_id") is None:
            children = self.reader.get_child_jobs(job_id)
            if children:
                cancellable = [
                    c for c in children if c["status"] in _CANCELLABLE_JOB_STATUSES
                ]
                if not cancellable:
                    raise ProblemException(
                        status=409,
                        title="Conflict",
                        detail=(
                            f"job '{job_id}' has no queued or running runs to cancel"
                        ),
                    )
                for child in cancellable:
                    self._cancel_one(child, strict=False)
                return
        self._cancel_one(job, strict=True)

    def delete_job(self, job_id: str) -> None:
        """Delete a job and all its associated run data.

        A parent's children are each torn down — canceled if still active, their
        run rows + artifacts purged, and their job row removed — before the
        parent row itself. Deleting a single child does the same for that one
        run. A running engine is signaled through the cooperative cancel token;
        if it still reaches a terminal write after the row is gone, the guarded
        update is a no-op and the deleted run is not resurrected.
        """
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        if job.get("parent_job_id") is None:
            for child in self.reader.get_child_jobs(job_id):
                self._delete_child(child)
            # A pure parent owns no run; a legacy standalone job might.
            if job["status"] not in _TERMINAL_JOB_STATUSES:
                self._cancel_one(job, strict=False)
            if job.get("run_id"):
                self._purge_run(job["run_id"])
            self.writer.delete_job(job_id)
        else:
            self._delete_child(job)

    def _delete_child(self, job: dict) -> None:
        """Cancel (if active), purge the run's data + artifacts, and delete the row."""
        if job["status"] not in _TERMINAL_JOB_STATUSES:
            self._cancel_one(job, strict=False)
        run_id = job.get("run_id")
        if run_id:
            self._purge_run(run_id)
        self.writer.delete_job(job["job_id"])

    def retry_job(self, job_id: str) -> JobCreatedResponse:
        """Retry a terminal job.

        A **child** run (part of a batch) is retried *in place*: the same job row
        is reset to queued with a fresh ``run_id`` and re-dispatched, so it stays
        in its parent batch and only that video re-runs. A **parent** (or
        standalone) job is re-created from its stored params, re-fanning the
        whole batch as a new job.
        """
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        if job.get("parent_job_id") is not None:
            return self._retry_child_in_place(job)

        retryable = {
            JobStatus.FAILED.value,
            JobStatus.CANCELED.value,
            JobStatus.COMPLETED.value,
        }
        status = self._effective_status(job)
        if status not in retryable:
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"job '{job_id}' is '{status}' — "
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

    def _retry_child_in_place(self, job: dict) -> JobCreatedResponse:
        """Re-run one child run under its existing ``job_id`` and parent batch."""
        job_id = job["job_id"]
        if job["status"] not in _TERMINAL_JOB_STATUSES:
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"run '{job_id}' is '{job['status']}' — "
                    "only completed, failed, or canceled runs can be retried"
                ),
            )
        params = job.get("params_json")
        if not params:
            raise ProblemException(
                status=422,
                title="Unprocessable Entity",
                detail=f"run '{job_id}' has no stored params and cannot be retried",
            )
        self._ensure_dispatchable()
        req = JobCreateRequest.model_validate(params)
        if not req.videos:
            raise ProblemException(
                status=422,
                title="Unprocessable Entity",
                detail=f"run '{job_id}' has no video to retry",
            )
        video = req.videos[0]
        video_key = self._validate_input(video.video_path)
        old_run_id = job.get("run_id")
        new_run_id = new_ulid()

        # Reset the row (guarded to terminal) before touching Redis/dispatch.
        if not self.writer.reset_job_for_retry(job_id, run_id=new_run_id):
            latest = self.reader.get_job(job_id)
            latest_status = latest["status"] if latest else "missing"
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"run '{job_id}' is '{latest_status}' — "
                    "only completed, failed, or canceled runs can be retried"
                ),
            )

        # The retry supersedes the old run: drop its persisted rows + artifacts
        # so it isn't left orphaned (the row now points at new_run_id).
        if old_run_id and old_run_id != new_run_id:
            self._purge_run(old_run_id)

        # Drop the previous run's stream so the SSE replay starts clean (the
        # job_id — and thus the stream key — is reused).
        from app.events import reset_stream

        reset_stream(self.settings.redis_url, job_id)

        payload = build_task_payload(
            job_id=job_id,
            run_id=new_run_id,
            req=req,
            video_name=video.video_name,
            video_path=video_key,
            video_type=video.video_type,
        )
        try:
            self._dispatch(job_id, payload)
        except Exception as exc:  # noqa: BLE001 - recorded on the row
            self.writer.update_job_if_status(
                job_id,
                allowed_statuses=(JobStatus.QUEUED.value,),
                status=JobStatus.FAILED.value,
                finished_at=now_iso(),
                error_text=str(exc),
            )
            raise
        return JobCreatedResponse(
            job_id=job_id, run_id=new_run_id, status=JobStatus.QUEUED
        )

    def _purge_run(self, run_id: str) -> None:
        """Delete a superseded run's DB rows and artifacts (best-effort)."""
        import shutil

        from app.settings import PROJECT_ROOT
        from src.utils.clip_scribe_artifacts import (
            make_artifact_uploader,
            run_artifact_dir,
        )

        self.writer.delete_run(run_id)
        art_dir = (PROJECT_ROOT / run_artifact_dir(run_id)).resolve()
        try:
            if art_dir.is_dir():
                shutil.rmtree(art_dir)
        except OSError:
            logger.warning(
                "Failed to remove artifact dir for run %s", run_id, exc_info=True
            )
        try:
            make_artifact_uploader(
                self.settings.storage_backend, self.settings.gcs_bucket
            ).delete_run_artifacts(run_id)
        except Exception:  # noqa: BLE001 - purge must not block delete/retry
            logger.warning(
                "Failed to remove remote artifacts for run %s", run_id, exc_info=True
            )

    def _validate_input(self, key: str | None) -> str:
        """Validate a job's video storage key and return it unchanged.

        The API never resolves the key to a local path — that would only be
        meaningful when the API and worker share a filesystem. Instead it checks
        the key is well-formed and still resolves to a stored object, then puts
        the *logical* key in the payload; the worker materializes it locally at
        run time (see app.job_execution.run_job_core).
        """
        if not key:
            raise ProblemException(
                status=400, title="Bad Request", detail="video_path is required"
            )
        # Keys are opaque but must be relative and free of traversal segments so
        # a crafted request can't reach outside the storage namespace.
        parts = Path(key).parts
        if Path(key).is_absolute() or ".." in parts:
            raise ProblemException(
                status=400,
                title="Bad Request",
                detail="video_path is not a valid storage key",
            )
        if self.reader.get_video_by_key(self.user_id, key) is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"video not found: {key}"
            )
        if not self.storage.exists(key):
            raise ProblemException(
                status=404, title="Not Found", detail=f"video not found: {key}"
            )
        return key
