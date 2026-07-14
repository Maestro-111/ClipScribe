"""Read-only views of extractor + parser output (web-app-plan §6).

Thin wrappers over the reader DB. Every route 404s if the run does not exist,
so the frontend gets a consistent signal instead of empty payloads. Responses
are the reader's plain dict/list rows; tightening these into per-resource
Pydantic models can come with the inspector work (step 7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from app.deps import get_reader
from app.errors import ProblemException
from app.models import (
    AudioSegment,
    FrameDetection,
    GlobalStatsResponse,
    ParserResult,
    RunResponse,
    RunSibling,
    ShotBoundary,
    TextEvent,
)

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
) -> RunResponse:
    return RunResponse.model_validate(_require_run(reader, run_id))


@router.get("/{run_id}/siblings", summary="Runs sharing the same batch job")
def get_run_siblings(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[RunSibling]:
    """Runs in the same batch job (including this one), in submission order.

    Derived from the jobs graph, not the ``runs`` table, so it resolves even
    while a sibling is still processing (its run row isn't written yet). This is
    deliberately *not* guarded by ``_require_run``: the switcher must keep
    working when the current run hasn't been persisted yet, otherwise the user
    lands on an in-progress run and loses the way back. Empty when the run has no
    batch job (e.g. a CLI-produced run), read as "no siblings to switch between".
    """
    return [RunSibling.model_validate(r) for r in reader.get_run_siblings(run_id)]


@router.get("/{run_id}/global-stats", summary="Global stats + shot boundaries")
def get_global_stats(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> GlobalStatsResponse:
    _require_run(reader, run_id)
    return GlobalStatsResponse(
        global_stats=reader.get_global_stats(run_id),
        shot_boundaries=[
            ShotBoundary.model_validate(s) for s in reader.get_shot_boundaries(run_id)
        ],
    )


@router.get("/{run_id}/objects", summary="Visual object occurrences")
def get_objects(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict]:
    _require_run(reader, run_id)
    return reader.get_visual_objects(run_id)


@router.get("/{run_id}/text-events", summary="OCR text events")
def get_text_events(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[TextEvent]:
    _require_run(reader, run_id)
    return [TextEvent.model_validate(r) for r in reader.get_text_events(run_id)]


@router.get("/{run_id}/audio-segments", summary="Audio transcript segments")
def get_audio_segments(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[AudioSegment]:
    _require_run(reader, run_id)
    return [AudioSegment.model_validate(r) for r in reader.get_audio_segments(run_id)]


@router.get("/{run_id}/scenes", summary="Scene descriptions")
def get_scenes(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[dict]:
    _require_run(reader, run_id)
    return reader.get_scene_descriptions(run_id)


@router.get("/{run_id}/frames", summary="Raw frame detections (overlay)")
def get_frames(
    run_id: str,
    from_sec: float | None = Query(default=None, alias="from"),
    to_sec: float | None = Query(default=None, alias="to"),
    reader: "ClipScribeReaderDB" = Depends(get_reader),
) -> list[FrameDetection]:
    _require_run(reader, run_id)
    return [
        FrameDetection.model_validate(r)
        for r in reader.get_frame_detections(run_id, from_sec=from_sec, to_sec=to_sec)
    ]


@router.get("/{run_id}/parser", summary="Parser results")
def get_parser(
    run_id: str, reader: "ClipScribeReaderDB" = Depends(get_reader)
) -> list[ParserResult]:
    _require_run(reader, run_id)
    return [ParserResult.model_validate(r) for r in reader.get_parser_results(run_id)]
