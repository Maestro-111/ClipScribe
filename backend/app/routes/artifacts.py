"""Artifact serving (web-app-plan §6).

Serves the two files the frontend needs: the original input video and the baked
``tracked_output.mp4``. Behavior follows the single ``CLIPSCRIBE_STORAGE_BACKEND``
selector:

* **local** — stream the file with ``FileResponse``, which honors HTTP ``Range``
  (needed for ``<video>`` seeking). Paths resolve through the shared
  ``INPUT_DIR`` / ``run_artifact_dir`` conventions and are guarded against
  traversal.
* **gcs** — 302-redirect to a short-lived signed URL so the browser streams
  straight from the bucket (native Range, no media proxied through the API).

The per-frame visualization PNGs are intentionally not served: the inspector
draws overlays from DB detections, and under GCS the PNGs live inside the run's
archived ``artifacts.tar.gz`` bundle, not as loose objects.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, RedirectResponse

from app.deps import (
    artifact_storage_dep,
    require_owned_run,
    video_storage_dep,
)
from app.errors import ProblemException
from app.settings import PROJECT_ROOT, Settings, get_settings
from src.utils.clip_scribe_artifacts import ArtifactUploader, run_artifact_dir
from src.utils.clip_scribe_video_storage import VideoStorage

router = APIRouter(prefix="/runs", tags=["artifacts"])


def _artifact_dir(run_id: str) -> Path:
    return (PROJECT_ROOT / run_artifact_dir(run_id)).resolve()


def _file_or_404(path: Path, what: str) -> FileResponse:
    if not path.is_file():
        raise ProblemException(
            status=404, title="Not Found", detail=f"{what} not found"
        )
    return FileResponse(path)


@router.get(
    "/{run_id}/video",
    summary="Original input video (Range-aware)",
    response_model=None,  # returns FileResponse or a 302 redirect, not a model
)
def get_video(
    run: dict = Depends(require_owned_run),
    storage: VideoStorage = Depends(video_storage_dep),
) -> FileResponse | RedirectResponse:
    stored = run.get("video_path") or ""
    if not stored:
        raise ProblemException(
            status=404, title="Not Found", detail="run has no input video"
        )
    # GCS: hand the browser a signed URL for the stored key and let it stream
    # directly from the bucket.
    settings: Settings = get_settings()
    url = storage.signed_url(stored)
    if url is not None:
        return RedirectResponse(url)
    if settings.storage_backend == "gcs":
        raise ProblemException(
            status=404, title="Not Found", detail="input video not found"
        )
    # Local: stored paths may be "input/<name>" (CLI) or "<name>" (API); resolve
    # the basename under INPUT_DIR so both work and traversal is impossible.
    name = Path(stored).name
    if not name:
        raise ProblemException(
            status=404, title="Not Found", detail="run has no input video"
        )
    return _file_or_404(settings.input_dir / name, "input video")


@router.get(
    "/{run_id}/tracked-video",
    summary="Tracked output mp4 (Range-aware)",
    response_model=None,  # returns FileResponse or a 302 redirect, not a model
)
def get_tracked_video(
    run_id: str,
    artifacts: ArtifactUploader = Depends(artifact_storage_dep),
    _run: dict = Depends(require_owned_run),
) -> FileResponse | RedirectResponse:
    # GCS: 302 to the signed URL for the loose tracked_output.mp4 object.
    url = artifacts.tracked_video_url(run_id)
    settings: Settings = get_settings()
    if url is not None:
        return RedirectResponse(url)
    if settings.storage_backend == "gcs":
        raise ProblemException(
            status=404, title="Not Found", detail="tracked video not found"
        )
    return _file_or_404(_artifact_dir(run_id) / "tracked_output.mp4", "tracked video")
