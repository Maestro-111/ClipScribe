"""Video upload endpoint (web-app-plan §9.1, option a).

Bytes arrive over HTTP and are stored via the configured :class:`VideoStorage`
backend (local disk today, a cloud bucket later). Each upload is streamed to a
staging temp file while its content hash is computed, then deduplicated: the
same bytes uploaded again — in this session or a later one — reuse the existing
stored object instead of creating a second copy. The response returns the
storage key a job later references as ``video_path``; the original filename is
remembered in the ``videos`` registry so the picker can show it.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.deps import (
    current_user_id,
    get_reader,
    get_writer,
    settings_dep,
    video_storage_dep,
)
from app.errors import ProblemException
from app.models import UploadedVideo, UploadResponse
from app.settings import Settings
from src.db import ClipScribeReaderDB, ClipScribeWriterDB
from src.utils.clip_scribe_video_storage import VideoStorage

logger = logging.getLogger("clip_scribe")

router = APIRouter(tags=["uploads"])

# 1 MiB chunks: bounded memory while streaming arbitrarily large videos.
_CHUNK = 1024 * 1024


@router.post("/uploads", response_model=UploadResponse, summary="Upload video(s)")
def upload_videos(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(settings_dep),
    storage: VideoStorage = Depends(video_storage_dep),
    reader: ClipScribeReaderDB = Depends(get_reader),
    writer: ClipScribeWriterDB = Depends(get_writer),
    user_id: str = Depends(current_user_id),
) -> UploadResponse:
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

        # Stream to a staging temp file while hashing, so bytes are never fully
        # buffered in memory and the hash is ready before we decide to keep it.
        staged, content_hash, size = _stage_and_hash(storage, suffix, file)

        existing = reader.get_video_by_hash(user_id, content_hash)
        if existing is not None:
            # Dedup hit: drop the staged copy, reuse the stored object.
            staged.unlink(missing_ok=True)
            writer.touch_video(user_id, content_hash)
            stored_key = existing["stored_key"]
            original_name = existing["original_name"]
            logger.info("Deduplicated upload %s -> %s", name, stored_key)
        else:
            stored_key = storage.commit(user_id, staged, suffix)
            writer.insert_video(
                user_id=user_id,
                content_hash=content_hash,
                stored_key=stored_key,
                original_name=name,
                size_bytes=size,
            )
            original_name = name

        uploaded.append(
            UploadedVideo(name=original_name, path=stored_key, size_bytes=size)
        )

    return UploadResponse(uploaded=uploaded)


def _stage_and_hash(
    storage: VideoStorage, suffix: str, file: UploadFile
) -> tuple[Path, str, int]:
    """Stream ``file`` into a staging temp file, returning (path, sha256, size)."""
    hasher = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=storage.staging_dir(), suffix=suffix)
    staged = Path(tmp_name)
    try:
        with open(fd, "wb") as out:
            while chunk := file.file.read(_CHUNK):
                out.write(chunk)
                hasher.update(chunk)
                size += len(chunk)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged, hasher.hexdigest(), size
