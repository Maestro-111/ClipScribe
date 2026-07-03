"""Read-only views of extractor + parser output (web-app-plan §6).

Thin wrappers over the reader DB. Every route 404s if the run does not exist,
so the frontend gets a consistent signal instead of empty payloads. Responses
are the reader's plain dict/list rows; tightening these into per-resource
Pydantic models can come with the inspector work (step 7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Query

from app.deps import get_reader
from app.errors import ProblemException

if TYPE_CHECKING:
    from src.db import ClipScribeReaderDB

router = APIRouter(prefix="/runs", tags=["runs"])


def _require_run(reader: "ClipScribeReaderDB", run_id: str) -> dict:
    run = reader.get_run(run_id)
    if run is None:
        raise ProblemException(
            status=404, title="Not Found", detail=f"run '{run_id}' not found"
        )
    return run


@router.get("/{run_id}", summary="Get a run")
def get_run(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> dict[str, Any]:
    return _require_run(reader, run_id)


@router.get("/{run_id}/global-stats", summary="Global stats + shot boundaries")
def get_global_stats(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> dict[str, Any]:
    _require_run(reader, run_id)
    return {
        "global_stats": reader.get_global_stats(run_id),
        "shot_boundaries": reader.get_shot_boundaries(run_id),
    }


@router.get("/{run_id}/objects", summary="Visual object occurrences")
def get_objects(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict[str, Any]]:
    _require_run(reader, run_id)
    return reader.get_visual_objects(run_id)


@router.get("/{run_id}/text-events", summary="OCR text events")
def get_text_events(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict[str, Any]]:
    _require_run(reader, run_id)
    return reader.get_text_events(run_id)


@router.get("/{run_id}/audio-segments", summary="Audio transcript segments")
def get_audio_segments(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict[str, Any]]:
    _require_run(reader, run_id)
    return reader.get_audio_segments(run_id)


@router.get("/{run_id}/scenes", summary="Scene descriptions")
def get_scenes(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict[str, Any]]:
    _require_run(reader, run_id)
    return reader.get_scene_descriptions(run_id)


@router.get("/{run_id}/frames", summary="Raw frame detections (overlay)")
def get_frames(
    run_id: str,
    from_sec: float | None = Query(default=None, alias="from"),
    to_sec: float | None = Query(default=None, alias="to"),
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> list[dict[str, Any]]:
    _require_run(reader, run_id)
    return reader.get_frame_detections(run_id, from_sec=from_sec, to_sec=to_sec)


@router.get("/{run_id}/parser", summary="Parser results")
def get_parser(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict[str, Any]]:
    _require_run(reader, run_id)
    return reader.get_parser_results(run_id)
