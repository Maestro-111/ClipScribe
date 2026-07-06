"""Job orchestration for the sync path (web-app-plan §10 step 5).

``JobService`` validates a request, writes the ``jobs`` row, and submits the
run to a single-slot executor that mirrors Celery ``concurrency=1``. The
executor callback drives the job lifecycle (queued -> running -> completed /
failed) around ``ClipScribeEngine.run``.

The HTTP contract here is deliberately identical to the eventual async path:
``create_job`` returns immediately with a job id, and clients poll
``GET /jobs/{id}``. When Celery lands (step 8), only the submit call changes
(``executor.submit`` -> ``celery_app.send_task``); validation, the row shape,
and the response stay put.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.errors import ProblemException
from app.models import JobCreatedResponse, JobCreateRequest, JobMode, JobStatus
from src.clip_scribe.build_clip_scribe_plalform import build_platform
from src.utils.ids import new_ulid

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from app.settings import Settings
    from src.clip_scribe.build_clip_scribe import ClipScribeBuilder

logger = logging.getLogger("clip_scribe")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobService:
    def __init__(
        self,
        builder: "ClipScribeBuilder",
        executor: "ThreadPoolExecutor",
        settings: "Settings",
        futures: "dict[str, Future[None]]",
    ) -> None:
        self.builder = builder
        self.executor = executor
        self.settings = settings
        self.futures = futures
        self.reader = builder.reader_db
        self.writer = builder.writer_db

    def create_job(self, req: JobCreateRequest) -> JobCreatedResponse:
        """Validate, persist a queued job, and submit it to the executor."""
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

        future: Future[None] = self.executor.submit(
            self._run, job_id, run_id, req, video_name, video_path_abs, video_type
        )
        self.futures[job_id] = future

        return JobCreatedResponse(job_id=job_id, run_id=run_id, status=JobStatus.QUEUED)

    def _run(
        self,
        job_id: str,
        run_id: str,
        req: JobCreateRequest,
        video_name: str | None,
        video_path_abs: Path | None,
        video_type: str | None,
    ) -> None:
        """Executor callback: run one job and record its terminal state."""
        self.writer.update_job(
            job_id, status=JobStatus.RUNNING.value, started_at=_now_iso()
        )
        try:
            platform_conf = build_platform(
                req.platform.value, **req.resolved_params.to_build_kwargs()
            )
            if platform_conf is None:
                raise ValueError(f"unsupported platform: {req.platform.value}")

            engine = self.builder.build_clip_scribe(
                video_name=video_name or "",
                video_path=str(video_path_abs)
                if video_path_abs
                else (req.video_path or ""),
                video_type=video_type,
                clib_scribe_mode=req.mode.value,
                clib_scribe_platform_name=req.platform.value,
                clib_scribe_platform_conf=platform_conf,
                user_hints=req.user_hints,
                generate_hint_from_name=req.generate_hint_from_name,
            )
            engine.run(run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - recorded on the job row
            logger.exception("Job %s failed", job_id)
            if not self._is_canceled(job_id):
                self.writer.update_job(
                    job_id,
                    status=JobStatus.FAILED.value,
                    error_text=str(exc),
                    finished_at=_now_iso(),
                )
        else:
            if not self._is_canceled(job_id):
                self.writer.update_job(
                    job_id, status=JobStatus.COMPLETED.value, finished_at=_now_iso()
                )
        finally:
            self.futures.pop(job_id, None)

    def _is_canceled(self, job_id: str) -> bool:
        """Return True if the job was externally canceled while running."""
        job = self.reader.get_job(job_id)
        return job is not None and job.get("status") == JobStatus.CANCELED.value

    def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or running job.

        Queued jobs are prevented from starting via Future.cancel(). Running
        jobs cannot be interrupted mid-engine (plan step 10 adds cooperative
        cancel); the DB is marked canceled immediately and _run respects it
        when the engine returns so the status is never overwritten to completed.
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
        future = self.futures.get(job_id)
        if future is not None:
            future.cancel()  # no-op if already running, succeeds if still queued
        self.writer.update_job(
            job_id, status=JobStatus.CANCELED.value, finished_at=_now_iso()
        )

    def retry_job(self, job_id: str) -> JobCreatedResponse:
        """Create a fresh job from the stored params of a failed/canceled job."""
        job = self.reader.get_job(job_id)
        if job is None:
            raise ProblemException(
                status=404, title="Not Found", detail=f"job '{job_id}' not found"
            )
        retryable = {JobStatus.FAILED.value, JobStatus.CANCELED.value}
        if job["status"] not in retryable:
            raise ProblemException(
                status=409,
                title="Conflict",
                detail=(
                    f"job '{job_id}' is '{job['status']}' — "
                    "only failed or canceled jobs can be retried"
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
