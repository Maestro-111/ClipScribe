"""Exploratory report exporters.

Builds a human-readable XLSX or a lossless JSON from an exploratory run's
``ExplorationResponse`` (the persisted report, which carries its metrics). Like ``abcd_exports``
this is pure over already-loaded data: routes do the reads and hand the model
in, so it is unit-testable without a database.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.worksheet import Worksheet

from app.exports import XLSX_CONTENT_TYPE, autosize, cell_value, workbook_bytes
from app.models import ExplorationResponse
from clip_scribe.parser.explore.models import ExploreReport, TimeSpan

VALID_FORMATS = frozenset({"json", "xlsx"})
CONTENT_TYPES = {
    "json": "application/json",
    "xlsx": XLSX_CONTENT_TYPE,
}

_HEADER_FONT = Font(bold=True)
# Prose columns wrap and are capped so one long summary can't blow out a sheet.
_WRAP = Alignment(wrap_text=True, vertical="top")
_PROSE_WIDTH = 90


def run_json(response: ExplorationResponse) -> bytes:
    return response.model_dump_json(indent=2).encode("utf-8")


def run_xlsx(video_name: str, response: ExplorationResponse) -> bytes:
    """One run: Overview, Chapters, Transcript, On-screen text, Visuals, and
    Findings (only when the findings stage produced any)."""
    report = response.report
    if report is None:
        raise ValueError("run has no exploration report to export")

    wb = Workbook()
    overview = wb.active
    assert overview is not None
    overview.title = "Overview"
    _write_overview(overview, video_name, response, report)
    _write_chapters(wb.create_sheet("Chapters"), report)
    _write_transcript(wb.create_sheet("Transcript"), report)
    _write_on_screen_text(wb.create_sheet("On-screen text"), report)
    _write_visuals(wb.create_sheet("Visuals"), report)
    if report.findings:
        _write_findings(wb.create_sheet("Findings"), report)
    return workbook_bytes(wb)


def _table(
    ws: Worksheet,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    wrap_columns: Iterable[int] = (),
) -> None:
    ws.append(list(headers))
    for cell in ws[1]:
        cell.font = _HEADER_FONT
    for row in rows:
        ws.append([cell_value(v) for v in row])
    wrapped = set(wrap_columns)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.column in wrapped:
                cell.alignment = _WRAP
    ws.freeze_panes = "A2"
    autosize(ws, column_max_widths={c: _PROSE_WIDTH for c in wrapped})


def _clock(sec: float) -> str:
    minutes, seconds = divmod(int(sec), 60)
    return f"{minutes}:{seconds:02d}"


def _spans(spans: list[TimeSpan]) -> str:
    return ", ".join(f"{_clock(s.start_sec)}-{_clock(s.end_sec)}" for s in spans)


def _percent(fraction: float | None) -> str | None:
    return None if fraction is None else f"{fraction:.0%}"


def _write_overview(
    ws: Worksheet, video_name: str, response: ExplorationResponse, report: ExploreReport
) -> None:
    m = report.metrics
    rows: list[tuple[str, object]] = [
        ("Video", video_name),
        ("Run ID", response.run_id),
        ("Model", response.model),
        ("Created", response.created_at),
        ("Summary", report.summary),
        ("Duration (s)", m.duration_sec),
        ("Shots", m.shot_count),
        ("Average shot (s)", m.avg_shot_sec),
        ("Cuts per minute", m.cuts_per_minute),
        ("Speech coverage", _percent(m.speech_coverage)),
        ("On-screen text coverage", _percent(m.text_coverage)),
        ("Words per minute", m.words_per_minute),
        (
            "Face coverage",
            _percent(m.face_coverage) if m.objects_tracked else "Not collected",
        ),
    ]
    _table(ws, ("Field", "Value"), rows, wrap_columns={2})


def _write_chapters(ws: Worksheet, report: ExploreReport) -> None:
    _table(
        ws,
        ("#", "Title", "Start", "End", "Summary"),
        [
            (i, c.title, _clock(c.start_sec), _clock(c.end_sec), c.summary)
            for i, c in enumerate(report.chapters, start=1)
        ],
        wrap_columns={5},
    )


def _write_transcript(ws: Worksheet, report: ExploreReport) -> None:
    transcript = report.transcript
    if transcript is None:
        _table(ws, ("Section", "Value"), [("Status", "No speech in this video")])
        return
    rows: list[tuple[str, object, object]] = [
        ("Summary", None, transcript.summary),
        *(("Topic", None, topic) for topic in transcript.topics),
        *(("Key claim", _clock(c.start_sec), c.text) for c in transcript.key_claims),
    ]
    _table(ws, ("Section", "Time", "Text"), rows, wrap_columns={3})


def _write_on_screen_text(ws: Worksheet, report: ExploreReport) -> None:
    digest = report.on_screen_text
    if digest is None:
        _table(ws, ("Section", "Value"), [("Status", "No on-screen text detected")])
        return
    rows: list[Sequence[object]] = [
        ("Summary", digest.summary, None, None, None, None),
        *(
            (
                "Card",
                c.clean_text,
                c.text,
                c.category,
                _spans(c.spans),
                c.total_sec,
            )
            for c in digest.cards
        ),
    ]
    _table(
        ws,
        ("Section", "Text", "Raw OCR reading", "Category", "Visible", "Seconds"),
        rows,
        wrap_columns={2},
    )


def _write_visuals(ws: Worksheet, report: ExploreReport) -> None:
    digest = report.visuals
    if digest is None:
        _table(ws, ("Section", "Value"), [("Status", "No visual data")])
        return
    rows: list[Sequence[object]] = [
        ("Summary", digest.summary, None, None, None),
        *(
            (
                "Subject",
                s.label,
                s.prominence,
                s.screen_time_sec,
                _clock(s.first_seen_sec),
            )
            for s in digest.subjects
        ),
    ]
    _table(
        ws,
        ("Section", "Text / label", "Prominence", "Screen time (s)", "First seen"),
        rows,
        wrap_columns={2},
    )


def _write_findings(ws: Worksheet, report: ExploreReport) -> None:
    _table(
        ws,
        ("#", "Observation", "Evidence"),
        [
            (i, f.observation, ", ".join(_clock(s) for s in f.evidence_secs))
            for i, f in enumerate(report.findings, start=1)
        ],
        wrap_columns={2},
    )
