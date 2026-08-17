"""Read-only views of extractor + parser output (web-app-plan §6).

Thin wrappers over the reader DB. Every route 404s if the run does not exist,
so the frontend gets a consistent signal instead of empty payloads. Responses
are the reader's plain dict/list rows; tightening these into per-resource
Pydantic models can come with the inspector work (step 7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app import exports
from app.deps import current_user_id, get_reader, require_owned_run
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


@router.get("/{run_id}", summary="Get a run")
def get_run(run: dict = Depends(require_owned_run)) -> RunResponse:
    return RunResponse.model_validate(run)


@router.get("/{run_id}/siblings", summary="Runs sharing the same batch job")
def get_run_siblings(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    user_id: str = Depends(current_user_id),
) -> list[RunSibling]:
    """Runs in the same batch job (including this one), in submission order.

    Derived from the jobs graph, not the ``runs`` table, so it resolves even
    while a sibling is still processing (its run row isn't written yet). This is
    deliberately *not* guarded by ``require_owned_run``: the switcher must keep
    working when the current run hasn't been persisted yet, otherwise the user
    lands on an in-progress run and loses the way back. Ownership is still
    enforced — ``get_run_siblings`` filters on ``created_by``, so a run the caller
    doesn't own (or a CLI-produced run) returns an empty list, read as "no
    siblings to switch between".
    """
    return [
        RunSibling.model_validate(r) for r in reader.get_run_siblings(run_id, user_id)
    ]


@router.get("/{run_id}/global-stats", summary="Global stats + shot boundaries")
def get_global_stats(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> GlobalStatsResponse:
    return GlobalStatsResponse(
        global_stats=reader.get_global_stats(run_id),
        shot_boundaries=[
            ShotBoundary.model_validate(s) for s in reader.get_shot_boundaries(run_id)
        ],
    )


@router.get("/{run_id}/objects", summary="Visual object occurrences")
def get_objects(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[dict]:
    return reader.get_visual_objects(run_id)


@router.get("/{run_id}/text-events", summary="OCR text events")
def get_text_events(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[TextEvent]:
    return [TextEvent.model_validate(r) for r in reader.get_text_events(run_id)]


@router.get("/{run_id}/audio-segments", summary="Audio transcript segments")
def get_audio_segments(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[AudioSegment]:
    return [AudioSegment.model_validate(r) for r in reader.get_audio_segments(run_id)]


@router.get("/{run_id}/scenes", summary="Scene descriptions")
def get_scenes(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[dict]:
    return reader.get_scene_descriptions(run_id)


@router.get("/{run_id}/frames", summary="Raw frame detections (overlay)")
def get_frames(
    run_id: str,
    from_sec: float | None = Query(default=None, alias="from"),
    to_sec: float | None = Query(default=None, alias="to"),
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[FrameDetection]:
    return [
        FrameDetection.model_validate(r)
        for r in reader.get_frame_detections(run_id, from_sec=from_sec, to_sec=to_sec)
    ]


@router.get("/{run_id}/parser", summary="Parser results")
def get_parser(
    run_id: str,
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[ParserResult]:
    return [ParserResult.model_validate(r) for r in reader.get_parser_results(run_id)]


@router.get("/{run_id}/parser/export", summary="Download this run's ABCD report")
def export_parser(
    run_id: str,
    fmt: str = Query(default="xlsx", alias="format"),
    reader: "ClipScribeReaderDB" = Depends(get_reader),
    run: dict = Depends(require_owned_run),
) -> Response:
    """Export one run's parser results as a CSV or XLSX download.

    XLSX carries a per-criterion Detail sheet plus a Scores summary; CSV is the
    flat Detail table (still opens in Excel, just without the extra tab).
    """
    if fmt not in exports.VALID_FORMATS:
        raise ProblemException(
            status=400, title="Bad Request", detail=f"unsupported format '{fmt}'"
        )
    rows = reader.get_parser_results(run_id)
    video_name = run.get("video_name") or run_id
    content = (
        exports.run_csv(rows) if fmt == "csv" else exports.run_xlsx(video_name, rows)
    )
    filename = exports.export_filename(f"{video_name}_abcd", fmt)
    return Response(
        content=content,
        media_type=exports.CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
