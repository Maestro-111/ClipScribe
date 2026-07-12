"""Video upload endpoint (web-app-plan §9.1, option a).

Bytes arrive over HTTP and are streamed to ``INPUT_DIR`` — a directory both the
API and the worker share (a bind-mount once containerized). This is what lets a
user upload from their local machine without the worker ever seeing the host
filesystem. Jobs then reference the returned server-side path.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.deps import settings_dep
from app.errors import ProblemException
from app.models import UploadedVideo, UploadResponse
from app.settings import Settings
from src.utils.ids import new_ulid

logger = logging.getLogger("clip_scribe")

router = APIRouter(tags=["uploads"])


@router.post("/uploads", response_model=UploadResponse, summary="Upload video(s)")
def upload_videos(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(settings_dep),
) -> UploadResponse:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    uploaded: list[UploadedVideo] = []

    for file in files:
        # Strip any client-supplied path; keep only a bare filename.
        name = Path(file.filename or "").name
        if not name:
            raise ProblemException(
                status=400, title="Bad Request", detail="a file is missing a name"
            )
        suffix = Path(name).suffix.lower()
        if suffix not in settings.allowed_video_suffixes:
            allowed = ", ".join(sorted(settings.allowed_video_suffixes))
            raise ProblemException(
                status=400,
                title="Bad Request",
                detail=f"unsupported file type '{suffix}'. Allowed: {allowed}",
            )

        stored_name = f"{new_ulid()}{suffix}"
        dest = settings.input_dir / stored_name
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)  # streamed, not buffered in memory

        uploaded.append(
            UploadedVideo(name=name, path=stored_name, size_bytes=dest.stat().st_size)
        )
        logger.info(
            "Uploaded %s as %s (%d bytes)", name, stored_name, dest.stat().st_size
        )

    return UploadResponse(uploaded=uploaded)
