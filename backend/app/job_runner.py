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
    ) -> None:
        self.builder = builder
        self.executor = executor
        self.settings = settings
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
            platform=req.platform,
            params_json=req.model_dump(mode="json"),
        )

        self.executor.submit(
            self._run, job_id, run_id, req, video_name, video_path_abs, video_type
        )

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
            platform_conf = build_platform(req.platform, **self._platform_kwargs(req))
            if platform_conf is None:
                raise ValueError(f"unsupported platform: {req.platform}")

            engine = self.builder.build_clip_scribe(
                video_name=video_name or "",
                video_path=str(video_path_abs)
                if video_path_abs
                else (req.video_path or ""),
                video_type=video_type,
                clib_scribe_mode=req.mode.value,
                clib_scribe_platform_name=req.platform,
                clib_scribe_platform_conf=platform_conf,
                user_hints=req.user_hints,
                generate_hint_from_name=req.generate_hint_from_name,
            )
            engine.run(run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - recorded on the job row
            logger.exception("Job %s failed", job_id)
            self.writer.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                error_text=str(exc),
                finished_at=_now_iso(),
            )
        else:
            self.writer.update_job(
                job_id, status=JobStatus.COMPLETED.value, finished_at=_now_iso()
            )

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

    @staticmethod
    def _platform_kwargs(req: JobCreateRequest) -> dict:
        if req.platform == "youtube":
            p = req.platform_params
            return {
                "youtube_brand_name": p.brand_name,
                "youtube_branded_products": p.branded_products,
                "youtube_branded_products_categories": p.branded_products_categories,
                "youtube_call_to_actions": p.call_to_actions,
            }
        return {}
