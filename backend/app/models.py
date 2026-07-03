"""Pydantic request/response schemas for the API (web-app-plan §6, §7).

These are the source of truth for the OpenAPI schema and the generated TS
client, so the shapes here are the Python↔TS contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class JobMode(str, Enum):
    FULL = "full"
    EXTRACT = "extract"
    PARSE = "parse"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class YouTubePlatformParams(BaseModel):
    """YouTube evaluation inputs (mirrors ``build_platform`` kwargs)."""

    brand_name: str = ""
    branded_products: list[str] = Field(default_factory=list)
    branded_products_categories: list[str] = Field(default_factory=list)
    call_to_actions: list[str] = Field(default_factory=list)


class JobCreateRequest(BaseModel):
    """Create + enqueue a job. Mirrors the ``main.py`` params, minus device.

    Device is not user-settable: the app uses the config value (the CLI uses
    ``--device``). Video is referenced by a server-side path under ``INPUT_DIR``
    (populated via ``POST /uploads`` or already present in ``input/``).
    """

    mode: JobMode
    platform: Literal["youtube"] = "youtube"
    # Relative to INPUT_DIR. Required for full/extract; ignored for parse.
    video_path: str | None = None
    video_name: str | None = None
    video_type: str | None = None
    platform_params: YouTubePlatformParams = Field(
        default_factory=YouTubePlatformParams
    )
    user_hints: list[str] | None = None
    generate_hint_from_name: bool = False
    # Required for parse; must reference an existing run (checked in the service).
    run_id: str | None = None

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> "JobCreateRequest":
        if self.mode == JobMode.PARSE:
            if not self.run_id:
                raise ValueError("run_id is required for mode 'parse'")
        else:
            if not self.video_path or not self.video_name:
                raise ValueError(
                    "video_path and video_name are required for mode "
                    f"'{self.mode.value}'"
                )
        return self


class JobCreatedResponse(BaseModel):
    """202 body for a freshly enqueued job."""

    job_id: str
    run_id: str | None
    status: JobStatus


class JobResponse(BaseModel):
    """Full orchestration state of a job."""

    job_id: str
    run_id: str | None = None
    status: str
    mode: str | None = None
    video_name: str | None = None
    video_path: str | None = None
    video_type: str | None = None
    device: str | None = None
    platform: str | None = None
    error_text: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    limit: int
    offset: int


# --------------------------------------------------------------------- metadata


class PlatformParamField(BaseModel):
    name: str
    type: str
    required: bool
    description: str


class PlatformInfo(BaseModel):
    name: str
    params: list[PlatformParamField]


class PlatformsResponse(BaseModel):
    platforms: list[PlatformInfo]


class InputVideo(BaseModel):
    name: str
    # Relative to INPUT_DIR, suitable for JobCreateRequest.video_path.
    path: str
    size_bytes: int


class InputsResponse(BaseModel):
    videos: list[InputVideo]


class UploadedVideo(BaseModel):
    name: str
    path: str
    size_bytes: int


class UploadResponse(BaseModel):
    uploaded: list[UploadedVideo]
