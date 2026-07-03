"""Metadata endpoints: platform specs, config defaults, and input listing.

Feed the "New job" form (web-app-plan §6, §7): ``/platforms`` describes the
per-platform params so the form and validation stay in sync, ``/defaults``
exposes the read-only yaml config for pre-population, and ``/inputs`` lists
videos already available under ``INPUT_DIR``.
"""

from __future__ import annotations

from typing import Any

import yaml  # type: ignore
from fastapi import APIRouter, Depends

from app.models import (
    InputsResponse,
    InputVideo,
    PlatformInfo,
    PlatformParamField,
    PlatformsResponse,
)
from app.deps import settings_dep
from app.settings import PROJECT_ROOT, Settings

router = APIRouter(tags=["metadata"])

CONFIG_PATH = PROJECT_ROOT / "src" / "clip_scribe" / "configs" / "clip_scribe.yaml"

# Static per-platform param registry. Adding a platform here keeps the form,
# request validation, and build_platform kwargs aligned.
_PLATFORM_REGISTRY: dict[str, list[PlatformParamField]] = {
    "youtube": [
        PlatformParamField(
            name="brand_name",
            type="string",
            required=True,
            description="Brand being advertised, e.g. 'RAM'.",
        ),
        PlatformParamField(
            name="branded_products",
            type="string[]",
            required=False,
            description="Specific branded products mentioned or shown.",
        ),
        PlatformParamField(
            name="branded_products_categories",
            type="string[]",
            required=False,
            description="Category phrasings used to match the product.",
        ),
        PlatformParamField(
            name="call_to_actions",
            type="string[]",
            required=False,
            description="Call-to-action phrases to detect (e.g. 'learn more').",
        ),
    ],
}


@router.get("/platforms", response_model=PlatformsResponse, summary="List platforms")
def list_platforms() -> PlatformsResponse:
    return PlatformsResponse(
        platforms=[
            PlatformInfo(name=name, params=params)
            for name, params in _PLATFORM_REGISTRY.items()
        ]
    )


@router.get("/defaults", summary="Read-only config defaults")
def get_defaults() -> dict[str, Any]:
    """Return the current ``clip_scribe.yaml`` as-is so the form can pre-populate."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@router.get("/inputs", response_model=InputsResponse, summary="List input videos")
def list_inputs(settings: Settings = Depends(settings_dep)) -> InputsResponse:
    input_dir = settings.input_dir
    videos: list[InputVideo] = []
    if input_dir.is_dir():
        for entry in sorted(input_dir.iterdir()):
            if (
                entry.is_file()
                and entry.suffix.lower() in settings.allowed_video_suffixes
            ):
                videos.append(
                    InputVideo(
                        name=entry.name,
                        path=entry.name,
                        size_bytes=entry.stat().st_size,
                    )
                )
    return InputsResponse(videos=videos)
