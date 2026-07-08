"""Pydantic request/response schemas for the API (web-app-plan §6, §7).

These are the source of truth for the OpenAPI schema and the generated TS
client, so the shapes here are the Python↔TS contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator


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


class PlatformName(str, Enum):
    """Supported evaluation platforms.

    Adding a platform = a new member here, a ``BasePlatformParams`` subclass,
    and an entry in ``PLATFORM_PARAMS_MODELS``. Requests for any value not in
    this enum are rejected at validation (422) — non-YouTube is blocked today.
    """

    YOUTUBE = "youtube"


class BasePlatformParams(BaseModel):
    """Base for per-platform evaluation params.

    Each platform's params own the mapping to ``build_platform`` kwargs so the
    job service stays platform-agnostic (it just calls ``to_build_kwargs()``).
    """

    def to_build_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError


class YouTubePlatformParams(BasePlatformParams):
    """YouTube evaluation inputs."""

    brand_name: str = ""
    branded_products: list[str] = Field(default_factory=list)
    branded_products_categories: list[str] = Field(default_factory=list)
    call_to_actions: list[str] = Field(default_factory=list)

    def to_build_kwargs(self) -> dict[str, Any]:
        return {
            "youtube_brand_name": self.brand_name,
            "youtube_branded_products": self.branded_products,
            "youtube_branded_products_categories": self.branded_products_categories,
            "youtube_call_to_actions": self.call_to_actions,
        }


# platform -> its params model. The single registry the request validator, the
# job service, and (future) the /platforms spec all key off of.
PLATFORM_PARAMS_MODELS: dict[PlatformName, type[BasePlatformParams]] = {
    PlatformName.YOUTUBE: YouTubePlatformParams,
}


class JobCreateRequest(BaseModel):
    """Create + enqueue a job. Mirrors the ``main.py`` params, minus device.

    Device is not user-settable: the app uses the config value (the CLI uses
    ``--device``). Video is referenced by a server-side path under ``INPUT_DIR``
    (populated via ``POST /uploads`` or already present in ``input/``).

    ``platform`` selects the evaluation platform; ``platform_params`` is a raw
    object validated against the selected platform's schema (see
    ``PLATFORM_PARAMS_MODELS``). The validated model is exposed via
    ``resolved_params``.
    """

    mode: JobMode
    platform: PlatformName = PlatformName.YOUTUBE
    # Relative to INPUT_DIR. Required for full/extract; ignored for parse.
    video_path: str | None = None
    video_name: str | None = None
    video_type: str | None = None
    # Shape depends on `platform`; parsed into the matching BasePlatformParams
    # subclass in the validator below.
    platform_params: dict[str, Any] = Field(default_factory=dict)
    user_hints: list[str] | None = None
    generate_hint_from_name: bool = False
    # Required for parse; must reference an existing run (checked in the service).
    run_id: str | None = None

    _resolved_params: BasePlatformParams = PrivateAttr()

    @model_validator(mode="after")
    def _validate(self) -> "JobCreateRequest":
        if self.mode == JobMode.PARSE:
            if not self.run_id:
                raise ValueError("run_id is required for mode 'parse'")
        else:
            if not self.video_path or not self.video_name:
                raise ValueError(
                    "video_path and video_name are required for mode "
                    f"'{self.mode.value}'"
                )

        model_cls = PLATFORM_PARAMS_MODELS.get(self.platform)
        if model_cls is None:
            raise ValueError(f"unsupported platform: {self.platform.value}")
        # Validate the raw params against the selected platform's schema.
        self._resolved_params = model_cls.model_validate(self.platform_params)
        return self

    @property
    def resolved_params(self) -> BasePlatformParams:
        """The platform_params parsed into the selected platform's model."""
        return self._resolved_params


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


# --------------------------------------------------------------------- run inspector


class RunResponse(BaseModel):
    """A completed extraction run."""

    run_id: str
    video_name: str | None = None
    video_path: str | None = None
    video_type: str | None = None
    created_at: str | None = None


class FrameDetection(BaseModel):
    """A single bounding-box detection in one video frame."""

    id: int
    run_id: str
    shot_index: int | None = None
    frame_idx: int | None = None
    timestamp_sec: float | None = None
    source: str | None = None  # dino | ocr | mtcnn | sam_mask
    label: str | None = None
    text: str | None = None
    box_x1: float | None = None
    box_y1: float | None = None
    box_x2: float | None = None
    box_y2: float | None = None
    confidence: float | None = None
    object_id: int | None = None


class ShotBoundary(BaseModel):
    """Temporal extent of a single shot."""

    id: int
    run_id: str
    shot_index: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    duration_sec: float | None = None


class GlobalStatsResponse(BaseModel):
    """Combined global stats + shot boundary list for a run."""

    global_stats: dict[str, Any] | None = None
    shot_boundaries: list[ShotBoundary]


class ParserResult(BaseModel):
    """Persisted per-criterion evaluation from the parser."""

    id: int
    run_id: str
    platform: str | None = None
    feature_category: str | None = None
    feature_name: str | None = None
    feature_criteria: str | None = None
    evaluation: bool | None = None
    llm_prompt: str | None = None
    llm_explanation: str | None = None
    langsmith_run_id: str | None = None
    created_at: str | None = None


class AudioSegment(BaseModel):
    """Whisper transcript segment."""

    id: int
    run_id: str
    start_time: float | None = None
    end_time: float | None = None
    text: str | None = None
    confidence: float | None = None


class TextEvent(BaseModel):
    """OCR text event at a specific second."""

    id: int
    run_id: str
    second: int | None = None
    line_index: int | None = None
    text: str | None = None
