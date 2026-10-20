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
from starlette.concurrency import run_in_threadpool

from app import exports, abcd_exports, metadata_exports
from app.deps import (
    current_user_id,
    get_reader,
    require_abcd_run,
    require_exploratory_run,
    require_owned_run,
)
from app.errors import ProblemException
from clip_scribe.parser.explore.metrics import compute_run_metrics
from app.models import (
    AbcdResult,
    AudioSegment,
    ExplorationResponse,
    FrameDetection,
    GlobalStatsResponse,
    RunResponse,
    RunSibling,
    ShotBoundary,
    TextEvent,
)

if TYPE_CHECKING:
    from clip_scribe.db import ClipScribeAsyncReaderDB

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}", summary="Get a run")
async def get_run(run: dict = Depends(require_owned_run)) -> RunResponse:
    return RunResponse.model_validate(run)


@router.get("/{run_id}/siblings", summary="Runs sharing the same batch job")
async def get_run_siblings(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
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
        RunSibling.model_validate(r)
        for r in await reader.get_run_siblings(run_id, user_id)
    ]


@router.get("/{run_id}/global-stats", summary="Global stats + shot boundaries")
async def get_global_stats(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> GlobalStatsResponse:
    return GlobalStatsResponse(
        global_stats=await reader.get_global_stats(run_id),
        shot_boundaries=[
            ShotBoundary.model_validate(s)
            for s in await reader.get_shot_boundaries(run_id)
        ],
    )


@router.get("/{run_id}/objects", summary="Visual object occurrences")
async def get_objects(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[dict]:
    return await reader.get_visual_objects(run_id)


@router.get("/{run_id}/text-events", summary="OCR text events")
async def get_text_events(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[TextEvent]:
    return [TextEvent.model_validate(r) for r in await reader.get_text_events(run_id)]


@router.get("/{run_id}/audio-segments", summary="Audio transcript segments")
async def get_audio_segments(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[AudioSegment]:
    return [
        AudioSegment.model_validate(r) for r in await reader.get_audio_segments(run_id)
    ]


@router.get("/{run_id}/scenes", summary="Scene descriptions")
async def get_scenes(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[dict]:
    return await reader.get_scene_descriptions(run_id)


@router.get("/{run_id}/frames", summary="Raw frame detections (overlay)")
async def get_frames(
    run_id: str,
    from_sec: float | None = Query(default=None, alias="from"),
    to_sec: float | None = Query(default=None, alias="to"),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_owned_run),
) -> list[FrameDetection]:
    return [
        FrameDetection.model_validate(r)
        for r in await reader.get_frame_detections(
            run_id, from_sec=from_sec, to_sec=to_sec
        )
    ]


@router.get("/{run_id}/abcd", summary="ABCD results")
async def get_abcd(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    _run: dict = Depends(require_abcd_run),
) -> list[AbcdResult]:
    return [AbcdResult.model_validate(r) for r in await reader.get_abcd_results(run_id)]


@router.get("/{run_id}/exploration", summary="Exploratory report and metrics")
async def get_exploration(
    run_id: str,
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    run: dict = Depends(require_exploratory_run),
) -> ExplorationResponse:
    summary = await reader.get_extraction_summary(run_id)
    if summary is None:  # the run row was deleted after the ownership check
        raise ProblemException(
            status=404, title="Not Found", detail=f"run '{run_id}' not found"
        )
    metrics = await run_in_threadpool(
        compute_run_metrics,
        summary,
        objects_tracked=run.get("extraction_mode") == "with_objects",
    )
    row = await reader.get_exploration_result(run_id)
    return ExplorationResponse(
        run_id=run_id,
        metrics=metrics,
        report=row["payload"] if row else None,
        model=row["model"] if row else None,
        created_at=row["created_at"] if row else None,
    )


@router.get("/{run_id}/abcd/export", summary="Download this run's ABCD report")
async def export_abcd(
    run_id: str,
    fmt: str = Query(default="xlsx", alias="format"),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    run: dict = Depends(require_abcd_run),
) -> Response:
    """Export one run's ABCD results as a CSV or XLSX download.

    XLSX carries a per-criterion Detail sheet plus a Scores summary; CSV is the
    flat Detail table (still opens in Excel, just without the extra tab).
    """
    if fmt not in abcd_exports.VALID_FORMATS:
        raise ProblemException(
            status=400, title="Bad Request", detail=f"unsupported format '{fmt}'"
        )
    rows = await reader.get_abcd_results(run_id)
    video_name = run.get("video_name") or run_id
    if fmt == "csv":
        content = await run_in_threadpool(abcd_exports.run_csv, rows)
    else:
        content = await run_in_threadpool(abcd_exports.run_xlsx, video_name, rows)
    filename = exports.export_filename(f"{video_name}_abcd", fmt)
    return Response(
        content=content,
        media_type=abcd_exports.CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{run_id}/metadata/export",
    summary="Download this run's extracted video metadata",
)
async def export_metadata(
    run_id: str,
    fmt: str = Query(default="xlsx", alias="format"),
    reader: ClipScribeAsyncReaderDB = Depends(get_reader),
    run: dict = Depends(require_owned_run),
) -> Response:
    """Export human-readable XLSX or lossless canonical extraction JSON."""
    if fmt not in metadata_exports.VALID_FORMATS:
        raise ProblemException(
            status=400, title="Bad Request", detail=f"unsupported format '{fmt}'"
        )

    summary = await reader.get_extraction_summary(
        run_id,
        include_frame_detections=fmt == "json",
    )
    if summary is None:  # Defensive: require_owned_run already established it.
        raise ProblemException(
            status=404, title="Not Found", detail=f"run '{run_id}' not found"
        )

    video_name = run.get("video_name") or run_id
    report = metadata_exports.RunMetadataReport(
        run_id=run_id,
        video_name=video_name,
        video_type=run.get("video_type"),
        created_at=run.get("created_at"),
        extraction_mode=run.get("extraction_mode") or "with_objects",
        summary=summary,
    )
    export = metadata_exports.run_json if fmt == "json" else metadata_exports.run_xlsx
    content = await run_in_threadpool(export, report)
    filename = exports.export_filename(f"{video_name}_metadata", fmt)
    return Response(
        content=content,
        media_type=metadata_exports.CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
