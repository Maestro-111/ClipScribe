"""Filesystem-backed artifact serving (web-app-plan §6).

Serves the original input video, the baked ``tracked_output.mp4``, and the
per-frame visualization PNGs. All responses use ``FileResponse``, which honors
HTTP ``Range`` requests (needed for ``<video>`` seeking). Every path is
resolved through the shared ``INPUT_DIR`` / ``run_artifact_dir`` conventions and
guarded against traversal.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.deps import get_reader, settings_dep
from app.errors import ProblemException
from app.settings import PROJECT_ROOT, Settings
from src.utils.artifacts import run_artifact_dir

if TYPE_CHECKING:
    from src.db import ClipScribeReaderDB

router = APIRouter(prefix="/runs", tags=["artifacts"])


def _require_run(reader: "ClipScribeReaderDB", run_id: str) -> dict:
    run = reader.get_run(run_id)
    if run is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"run '{run_id}' not found"
        )
    return run


def _artifact_dir(run_id: str) -> Path:
    return (PROJECT_ROOT / run_artifact_dir(run_id)).resolve()


def _file_or_404(path: Path, what: str) -> FileResponse:
    if not path.is_file():
        raise ProblemException(
            status=404, title="Not Found", detail=f"{what} not found"
        )
    return FileResponse(path)


@router.get("/{run_id}/video", summary="Original input video (Range-aware)")
def get_video(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    settings: Settings = Depends(settings_dep),
) -> FileResponse:
    run = _require_run(reader, run_id)
    stored = run.get("video_path") or ""
    # Stored paths may be "input/<name>" (CLI) or "<name>" (API); resolve the
    # basename under INPUT_DIR so both work and traversal is impossible.
    name = Path(stored).name
    if not name:
        raise ProblemException(
            status=404, title="Not Found", detail="run has no input video"
        )
    return _file_or_404(settings.input_dir / name, "input video")


@router.get("/{run_id}/tracked-video", summary="Tracked output mp4 (Range-aware)")
def get_tracked_video(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> FileResponse:
    _require_run(reader, run_id)
    return _file_or_404(_artifact_dir(run_id) / "tracked_output.mp4", "tracked video")


@router.get("/{run_id}/png/{filename}", summary="Visualization PNG")
def get_png(
    run_id: str,
    filename: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> FileResponse:
    _require_run(reader, run_id)
    base = _artifact_dir(run_id)
    name = Path(filename).name
    if name != filename or Path(name).suffix.lower() != ".png":
        raise ProblemException(
            status=400, title="Bad Request", detail="invalid png filename"
        )
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ProblemException(
            status=400, title="Bad Request", detail="filename escapes the artifact dir"
        )
    return _file_or_404(candidate, "png")
