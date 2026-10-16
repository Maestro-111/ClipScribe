"""Human-readable and lossless exports of persisted video metadata.

The XLSX builders intentionally present the extractor output as a set of
related, filterable tables.  Raw per-frame detections are excluded from the
workbook because they are overlay/debug primitives and can dwarf every other
section.  JSON is the lossless option and includes the complete canonical
``ExtractionSummary``, including ``frame_detections``.

This module is pure over already-loaded data.  Routes own authorization and DB
reads; they hand typed reports to these builders and receive response bytes.
The reader already returns every collection in display order, so rows are
emitted as-is.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.exports import XLSX_CONTENT_TYPE, autosize, cell_value, workbook_bytes

if TYPE_CHECKING:
    from clip_scribe.extractor.extractor_models import (
        ExtractionSummary,
        GlobalStats,
    )

SCHEMA_VERSION = 1
VALID_FORMATS = frozenset({"json", "xlsx"})
CONTENT_TYPES = {
    "json": "application/json",
    "xlsx": XLSX_CONTENT_TYPE,
}

_HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
_HEADER_FONT = Font(color="FFFFFF", bold=True)
_NOTE_FILL = PatternFill("solid", fgColor="D9EAF7")
_NOT_COLLECTED = "Not collected"

# Labels for ``_global_stats_values``, shared by the Overview and Global Stats
# sheets so the two always describe pacing identically.
_GLOBAL_STATS_HEADERS = (
    "Duration (seconds)",
    "Total shots",
    "Average shot duration (seconds)",
    "Dynamic start detected",
    "First shot duration (seconds)",
    "Dynamic start rule",
    "Quick pacing in first 5 seconds",
    "Shots starting in first 5 seconds",
    "Shot indices starting in first 5 seconds (0-based)",
    "Intro pacing rule",
    "Quick pacing detected anywhere",
    "Rapid-fire windows",
    "General pacing rule",
)

_JOB_SUMMARY_HEADERS = (
    "Video",
    "Run ID",
    "Status",
    "Metadata available",
    "Extraction mode",
    "Object tracking",
    "Duration (seconds)",
    "Shots",
    "Object identities",
    "Object occurrences",
    "Text timestamps",
    "Transcript segments",
)


@dataclass(frozen=True, slots=True)
class RunMetadataReport:
    """Canonical extraction output plus the run fields needed by an export."""

    run_id: str
    video_name: str
    summary: ExtractionSummary
    video_type: str | None = None
    created_at: str | None = None
    extraction_mode: str = "with_objects"

    @property
    def object_data_available(self) -> bool:
        return self.extraction_mode != "no_objects"


@dataclass(frozen=True, slots=True)
class BatchRunEntry:
    """One child in a batch, optionally carrying completed run metadata."""

    job_id: str
    video_name: str
    status: str
    run_id: str | None = None
    report: RunMetadataReport | None = None

    @property
    def metadata_available(self) -> bool:
        return self.report is not None


@dataclass(frozen=True, slots=True)
class JobMetadataEnvelope:
    """All children of a batch in submission order."""

    job_id: str
    job_name: str
    entries: Sequence[BatchRunEntry]
    created_at: str | None = None


def run_json(report: RunMetadataReport) -> bytes:
    """Serialize one complete run, including raw frame detections."""

    payload = _run_payload(report)
    payload["exported_at"] = _exported_at()
    return _json_bytes(payload)


def job_json(envelope: JobMetadataEnvelope) -> bytes:
    """Serialize every batch child, with metadata for completed runs."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _exported_at(),
        "job": {
            "job_id": envelope.job_id,
            "job_name": envelope.job_name,
            "created_at": envelope.created_at,
        },
        "runs": [_batch_entry_payload(entry) for entry in envelope.entries],
    }
    return _json_bytes(payload)


def run_xlsx(report: RunMetadataReport) -> bytes:
    """Build the human-readable workbook for one run.

    ``frame_detections`` is deliberately absent; callers should offer JSON when
    a lossless/raw export is required.
    """

    workbook = Workbook()
    overview = workbook.active
    overview.title = "Overview"
    _write_run_overview(overview, report)
    _write_run_detail_sheets(workbook, (report,), include_identity=False)
    _write_definitions(workbook.create_sheet("Definitions"))
    _set_workbook_metadata(workbook, f"Video metadata - {report.video_name}")
    return workbook_bytes(workbook)


def job_xlsx(envelope: JobMetadataEnvelope) -> bytes:
    """Build one section-oriented workbook for an entire batch."""

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    _write_job_summary(summary, envelope)
    reports = tuple(
        entry.report for entry in envelope.entries if entry.report is not None
    )
    global_stats = workbook.create_sheet("Global Stats")
    _write_global_stats(global_stats, reports)
    _write_run_detail_sheets(workbook, reports, include_identity=True)
    _write_definitions(workbook.create_sheet("Definitions"))
    _set_workbook_metadata(workbook, f"Batch metadata - {envelope.job_name}")
    return workbook_bytes(workbook)


def _run_payload(report: RunMetadataReport) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": report.run_id,
            "video_name": report.video_name,
            "video_type": report.video_type,
            "created_at": report.created_at,
            "extraction_mode": report.extraction_mode,
        },
        # Preserve the canonical extractor contract verbatim.  In particular,
        # this retains frame_detections even though XLSX omits them.
        "extraction_summary": report.summary,
    }


def _batch_entry_payload(entry: BatchRunEntry) -> dict[str, object]:
    report = entry.report
    return {
        "job_id": entry.job_id,
        "run_id": report.run_id if report is not None else entry.run_id,
        "video_name": report.video_name if report is not None else entry.video_name,
        "status": entry.status,
        "metadata_available": entry.metadata_available,
        "metadata": _run_payload(report) if report is not None else None,
    }


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _exported_at() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _global_stats_values(stats: GlobalStats) -> tuple[object, ...]:
    """One value per ``_GLOBAL_STATS_HEADERS`` entry (blank if never persisted)."""

    if not stats:
        return (None,) * len(_GLOBAL_STATS_HEADERS)
    dynamic = stats["dynamic_start"]
    intro = stats["quick_pacing_intro"]
    general = stats["quick_pacing_general"]
    return (
        stats["video_duration"],
        stats["total_shots"],
        stats["avg_shot_duration"],
        _yes_no(dynamic["detected"]),
        dynamic["first_shot_duration"],
        dynamic["criteria"],
        _yes_no(intro["detected"]),
        intro["shot_count"],
        _joined(intro["shots"]),
        intro["criteria"],
        _yes_no(general["detected"]),
        len(general["rapid_fire_segments"]),
        general["criteria"],
    )


def _object_counts(report: RunMetadataReport) -> tuple[object, object]:
    """(identities, occurrences), or a marker when tracking was skipped."""

    if not report.object_data_available:
        return _NOT_COLLECTED, _NOT_COLLECTED
    objects = report.summary["visual_objects"]
    return len(objects), sum(len(obj["occurrences"]) for obj in objects)


def _write_run_overview(ws: Worksheet, report: RunMetadataReport) -> None:
    summary = report.summary
    identities, occurrences = _object_counts(report)
    rows: list[tuple[str, object]] = [
        ("Video", report.video_name),
        ("Run ID", report.run_id),
        ("Video type", report.video_type),
        ("Created", report.created_at),
        ("Extraction mode", _display_mode(report.extraction_mode)),
        ("Object tracking", _tracking_status(report)),
        ("Object identities", identities),
        ("Object occurrences", occurrences),
        ("On-screen text timestamps", len(summary["text_events"])),
        (
            "On-screen text lines",
            sum(len(event["text"]) for event in summary["text_events"]),
        ),
        ("Transcript segments", len(summary["audio_segments"])),
        *zip(
            _GLOBAL_STATS_HEADERS,
            _global_stats_values(summary["global_stats"]),
            strict=True,
        ),
    ]
    _write_table(ws, ("Field", "Value"), rows, wrap_columns={2})


def _write_job_summary(ws: Worksheet, envelope: JobMetadataEnvelope) -> None:
    rows: list[tuple[object, ...]] = []
    for entry in envelope.entries:
        report = entry.report
        identity = (
            report.video_name if report is not None else entry.video_name,
            report.run_id if report is not None else entry.run_id,
            entry.status,
        )
        if report is None:
            padding = (None,) * (len(_JOB_SUMMARY_HEADERS) - len(identity) - 1)
            rows.append((*identity, "No", *padding))
            continue

        stats = report.summary["global_stats"]
        rows.append(
            (
                *identity,
                "Yes",
                _display_mode(report.extraction_mode),
                _tracking_status(report),
                stats.get("video_duration"),
                stats.get("total_shots"),
                *_object_counts(report),
                len(report.summary["text_events"]),
                len(report.summary["audio_segments"]),
            )
        )
    _write_table(ws, _JOB_SUMMARY_HEADERS, rows)


def _write_global_stats(
    ws: Worksheet,
    reports: Sequence[RunMetadataReport],
) -> None:
    """Write one complete pacing/statistics row per available batch run."""

    rows = [
        (
            report.video_name,
            report.run_id,
            *_global_stats_values(report.summary["global_stats"]),
        )
        for report in reports
    ]
    _write_table(
        ws,
        ("Video", "Run ID", *_GLOBAL_STATS_HEADERS),
        rows,
        wrap_columns={7, 10, 11, 14},
    )


def _write_definitions(ws: Worksheet) -> None:
    """Explain extractor terminology and workbook-specific omissions."""

    rows = (
        (
            "Raw frame detections",
            "Included in JSON only. They are omitted from XLSX because the per-frame "
            "rows can be very large and are intended for overlays/debugging.",
        ),
        (
            "Extraction mode",
            "With Objects runs object detection and tracking; No Objects skips both.",
        ),
        (
            "Object ID",
            "A resolved identity used to group the same tracked object across shots.",
        ),
        (
            "Object occurrence",
            "One continuous appearance of an object within a shot.",
        ),
        (
            "Screen coverage",
            "Largest bounding-box area during an occurrence as a fraction of frame area.",
        ),
        (
            "Average max coverage",
            "Mean of the per-occurrence screen coverage values for one object identity.",
        ),
        ("Velocity", "Average center-point movement in pixels per second."),
        (
            "Growth factor",
            "Ratio of the final bounding-box area to the initial bounding-box area.",
        ),
        (
            "Distance from centre",
            "Normalized distance from frame centre: 0 is centred and 1 is a corner.",
        ),
        (
            "Screen-time ratio",
            "Fraction of the full video duration for which the occurrence is visible.",
        ),
        ("Quadrant", "Average screen region occupied by the object."),
        (
            "Dynamic start",
            "Whether the first shot satisfies the extractor's configured duration rule.",
        ),
        (
            "Quick pacing",
            "Whether shot frequency satisfies the extractor's configured pacing rule.",
        ),
        ("Times and durations", "All workbook time values are in seconds."),
        ("Shot index", "Zero-based position of a shot in the video."),
    )
    _write_table(ws, ("Term", "Meaning"), rows, wrap_columns={2})


def _write_run_detail_sheets(
    workbook: Workbook,
    reports: Sequence[RunMetadataReport],
    *,
    include_identity: bool,
) -> None:
    identity_headers = ("Video", "Run ID") if include_identity else ()

    shots = workbook.create_sheet("Shots & Scenes")
    _write_table(
        shots,
        (
            *identity_headers,
            "Shot index (0-based)",
            "Start",
            "End",
            "Duration",
            "Description",
        ),
        _shot_rows(reports, include_identity),
        wrap_columns={len(identity_headers) + 5},
    )

    object_summary = workbook.create_sheet("Object Summary")
    _write_table(
        object_summary,
        (
            *identity_headers,
            "Object ID",
            "Label",
            "Occurrences",
            "Shots",
            "First seen",
            "Last seen",
            "Tracked duration",
            "Average max coverage",
            "Max coverage",
        ),
        _object_summary_rows(reports, include_identity),
        percentage_columns={len(identity_headers) + 8, len(identity_headers) + 9},
    )

    occurrences = workbook.create_sheet("Object Occurrences")
    _write_table(
        occurrences,
        (
            *identity_headers,
            "Object ID",
            "Label",
            "Shot index (0-based)",
            "Start",
            "End",
            "Duration",
            "Screen coverage",
            "Velocity (px/sec)",
            "Growth factor",
            "Direction",
            "Distance from centre",
            "Screen-time ratio",
            "Quadrant",
        ),
        _object_occurrence_rows(reports, include_identity),
        percentage_columns={
            len(identity_headers) + 7,
            len(identity_headers) + 11,
            len(identity_headers) + 12,
        },
    )

    text = workbook.create_sheet("On-screen Text")
    _write_table(
        text,
        (*identity_headers, "Second", "Line count", "Text"),
        _text_rows(reports, include_identity),
        wrap_columns={len(identity_headers) + 3},
    )

    transcript = workbook.create_sheet("Transcript")
    _write_table(
        transcript,
        (*identity_headers, "Start", "End", "Duration", "Confidence", "Text"),
        _audio_rows(reports, include_identity),
        wrap_columns={len(identity_headers) + 5},
        percentage_columns={len(identity_headers) + 4},
    )

    pacing = workbook.create_sheet("Pacing")
    _write_table(
        pacing,
        (
            *identity_headers,
            "Start",
            "End",
            "Duration",
            "Shot count",
            "Shot indices (0-based)",
        ),
        _pacing_rows(reports, include_identity),
        wrap_columns={len(identity_headers) + 5},
    )

    if not include_identity and reports and not reports[0].object_data_available:
        _add_note(object_summary, "Object tracking was not performed for this run.")
        _add_note(occurrences, "Object tracking was not performed for this run.")


def _shot_rows(
    reports: Sequence[RunMetadataReport], include_identity: bool
) -> Iterable[tuple[object, ...]]:
    for report in reports:
        shots = {
            shot["index"]: (shot["start"], shot["end"], shot["duration"])
            for shot in report.summary["shot_boundaries"]
        }
        descriptions: dict[int, list[str]] = {}
        for scene in report.summary["scene_descriptions"]:
            index = scene["shot_index"]
            descriptions.setdefault(index, []).append(scene["description"])
            # A shot without a boundary row falls back to its scene timing.
            start, end = scene["start_time"], scene["end_time"]
            shots.setdefault(index, (start, end, _duration(start, end)))

        for index in sorted(shots):
            row = (index, *shots[index], "\n\n".join(descriptions.get(index, [])))
            yield (*_identity(report, include_identity), *row)


def _object_summary_rows(
    reports: Sequence[RunMetadataReport], include_identity: bool
) -> Iterable[tuple[object, ...]]:
    for report in reports:
        for obj in report.summary["visual_objects"]:
            occurrences = obj["occurrences"]
            coverage = [occurrence["screen_coverage"] for occurrence in occurrences]
            row = (
                obj["global_id"],
                obj["label"],
                len(occurrences),
                _joined(
                    sorted({occurrence["shot_index"] for occurrence in occurrences})
                ),
                min(occurrence["lifespan"][0] for occurrence in occurrences),
                max(occurrence["lifespan"][1] for occurrence in occurrences),
                round(
                    sum(
                        _duration(*occurrence["lifespan"]) for occurrence in occurrences
                    ),
                    3,
                ),
                round(sum(coverage) / len(coverage), 6),
                max(coverage),
            )
            yield (*_identity(report, include_identity), *row)


def _object_occurrence_rows(
    reports: Sequence[RunMetadataReport], include_identity: bool
) -> Iterable[tuple[object, ...]]:
    for report in reports:
        for obj in report.summary["visual_objects"]:
            for occurrence in obj["occurrences"]:
                start, end = occurrence["lifespan"]
                row = (
                    obj["global_id"],
                    obj["label"],
                    occurrence["shot_index"],
                    start,
                    end,
                    _duration(start, end),
                    occurrence["screen_coverage"],
                    occurrence["velocity_px_sec"],
                    occurrence["growth_factor"],
                    occurrence["direction"],
                    occurrence["centrality_score"],
                    occurrence["screen_time_ratio"],
                    occurrence["quadrant"],
                )
                yield (*_identity(report, include_identity), *row)


def _text_rows(
    reports: Sequence[RunMetadataReport], include_identity: bool
) -> Iterable[tuple[object, ...]]:
    for report in reports:
        for event in report.summary["text_events"]:
            lines = event["text"]
            row = (event["second"], len(lines), "\n".join(lines))
            yield (*_identity(report, include_identity), *row)


def _audio_rows(
    reports: Sequence[RunMetadataReport], include_identity: bool
) -> Iterable[tuple[object, ...]]:
    for report in reports:
        for segment in report.summary["audio_segments"]:
            start, end = segment["start"], segment["end"]
            row = (
                start,
                end,
                _duration(start, end),
                segment["confidence"],
                segment["text"],
            )
            yield (*_identity(report, include_identity), *row)


def _pacing_rows(
    reports: Sequence[RunMetadataReport], include_identity: bool
) -> Iterable[tuple[object, ...]]:
    for report in reports:
        general = report.summary["global_stats"].get("quick_pacing_general")
        for interval in general["rapid_fire_segments"] if general else ():
            row = (
                interval["start_time"],
                interval["end_time"],
                interval["duration"],
                interval["shot_count"],
                _joined(interval["shot_indices"]),
            )
            yield (*_identity(report, include_identity), *row)


def _identity(report: RunMetadataReport, include_identity: bool) -> tuple[object, ...]:
    return (report.video_name, report.run_id) if include_identity else ()


def _write_table(
    ws: Worksheet,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    wrap_columns: set[int] | None = None,
    percentage_columns: set[int] | None = None,
) -> None:
    ws.append(list(headers))
    for cell in ws[1]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(vertical="top")

    for values in rows:
        ws.append([cell_value(value) for value in values])

    wrap_columns = wrap_columns or set()
    percentage_columns = percentage_columns or set()
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                vertical="top", wrap_text=cell.column in wrap_columns
            )
            if cell.column in percentage_columns and isinstance(
                cell.value, (int, float)
            ):
                cell.number_format = "0.0%"

    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
    autosize(ws, max_width=36, column_max_widths=dict.fromkeys(wrap_columns, 48))


def _add_note(ws: Worksheet, note: str) -> None:
    """Put a visible note beneath an otherwise-empty object table."""

    if ws.max_row != 1:
        return
    ws.append([note])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ws.max_column)
    cell = ws.cell(2, 1)
    cell.fill = _NOTE_FILL
    cell.font = Font(italic=True)
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.auto_filter.ref = None


def _set_workbook_metadata(workbook: Workbook, title: str) -> None:
    workbook.properties.creator = "ClipScribe"
    workbook.properties.title = title
    workbook.properties.subject = "Extracted video metadata"


def _tracking_status(report: RunMetadataReport) -> str:
    return "Performed" if report.object_data_available else "Not performed"


def _display_mode(mode: str) -> str:
    return mode.replace("_", " ").title()


def _joined(values: Iterable[int]) -> str:
    return ", ".join(str(value) for value in values)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _duration(start: float, end: float) -> float:
    return round(max(0.0, end - start), 3)
