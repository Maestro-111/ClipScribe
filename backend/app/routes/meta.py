"""Metadata endpoints: platform specs, config defaults, and input listing.

Feed the "New job" form (web-app-plan §6, §7): ``/platforms`` describes the
per-platform params so the form and validation stay in sync, ``/defaults``
exposes the read-only yaml config for pre-population, and ``/inputs`` lists
the videos a user has already uploaded (from the ``videos`` registry).
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
from app.deps import (
    current_user_id,
    get_reader,
    get_writer,
    video_storage_dep,
)
from app.settings import PROJECT_ROOT
from src.db import ClipScribeReaderDB, ClipScribeWriterDB
from src.utils.clip_scribe_video_storage import VideoStorage

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
def list_inputs(
    reader: ClipScribeReaderDB = Depends(get_reader),
    writer: ClipScribeWriterDB = Depends(get_writer),
    storage: VideoStorage = Depends(video_storage_dep),
    user_id: str = Depends(current_user_id),
) -> InputsResponse:
    """The user's uploaded videos, reconciled against storage at read time.

    A registry row whose object has been removed out-of-band (bucket cleanup,
    a dev deleting the file) is pruned and omitted here, so the picker can never
    offer a video that no longer exists. This is the only correctness check for
    the pick→run gap; a run keeps its own ``video_name`` snapshot, so pruning a
    row never affects run history.
    """
    videos: list[InputVideo] = []
    for row in reader.list_videos(user_id):
        stored_key = row["stored_key"]
        if storage.exists(stored_key):
            videos.append(
                InputVideo(
                    name=row["original_name"],
                    path=stored_key,
                    size_bytes=row["size_bytes"] or 0,
                )
            )
        else:
            writer.delete_video(user_id, stored_key)
    return InputsResponse(videos=videos)
