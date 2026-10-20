"""Tests for the ABCD report exporters and their routes.

Covers the pure CSV/XLSX builders in ``app.abcd_exports`` and the run-level and
job-level export endpoints, including the empty-job and bad-format guards.
Uses ``metadata.create_all`` directly (tests own their schema).
"""

import csv
import io

import openpyxl
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import abcd_exports, exports, metadata_exports
from app.deps import get_reader, get_writer
from app.main import app
from clip_scribe.db.schema import (
    frame_detections_table,
    metadata_obj,
    abcd_results_table,
    runs_table,
)
from clip_scribe.db.writer import ClipScribeWriterDB

PARENT = "job-parent"
CHILD1 = "job-child-1"
CHILD2 = "job-child-2"
RUN1 = "run-1"
RUN2 = "run-2"


def _rows(passed_first: bool) -> list[dict]:
    return [
        {
            "feature_category": "Attract",
            "feature_name": "Dynamic Start",
            "feature_criteria": "hooks in first 5s",
            "evaluation": passed_first,
            "llm_explanation": "fast cuts early",
            "llm_prompt": "prompt A",
        },
        {
            "feature_category": "Brand",
            "feature_name": "Brand Mention",
            "feature_criteria": "brand named",
            "evaluation": False,
            "llm_explanation": "no brand",
            "llm_prompt": "prompt B",
        },
    ]


# ── pure builders ────────────────────────────────────────────────────────────


def test_run_csv_header_and_verdicts():
    out = abcd_exports.run_csv(_rows(True)).decode()
    lines = out.splitlines()
    assert lines[0] == "Category,Feature,Criteria,Result,Explanation,Prompt"
    assert "Attract,Dynamic Start" in lines[1]
    assert lines[1].split(",")[3] == "PASS"
    assert lines[2].split(",")[3] == "FAIL"


def test_run_xlsx_has_detail_and_scores_sheets():
    wb = openpyxl.load_workbook(
        io.BytesIO(abcd_exports.run_xlsx("ad.mp4", _rows(True)))
    )
    assert wb.sheetnames == ["Detail", "Scores"]
    scores = wb["Scores"]
    # Header + Attract + Brand + Total.
    values = {row[0].value: (row[1].value, row[2].value) for row in scores.iter_rows()}
    assert values["Total"] == (1, 2)


def test_job_csv_prefixes_video_column():
    reports = [
        abcd_exports.RunReport(RUN1, "first.mp4", _rows(True)),
        abcd_exports.RunReport(RUN2, "second.mp4", _rows(False)),
    ]
    lines = abcd_exports.job_csv(reports).decode().splitlines()
    assert lines[0].startswith("Video,Category")
    assert lines[1].startswith("first.mp4,")
    assert any(line.startswith("second.mp4,") for line in lines)


def test_job_xlsx_summary_and_unique_sheets_per_run():
    reports = [
        abcd_exports.RunReport(RUN1, "same name.mp4", _rows(True)),
        abcd_exports.RunReport(RUN2, "same name.mp4", _rows(False)),
    ]
    wb = openpyxl.load_workbook(io.BytesIO(abcd_exports.job_xlsx(reports)))
    assert wb.sheetnames[0] == "Summary"
    # Two detail sheets, uniquely titled despite identical video names.
    assert len(wb.sheetnames) == 3
    assert len(set(wb.sheetnames)) == 3
    summary = wb["Summary"]
    totals = [row for row in summary.iter_rows(values_only=True) if row[0] == "Total"]
    assert totals[0][1:3] == (1, 4)  # 1 passed of 4 criteria across both runs


def test_export_filename_slugs_unsafe_chars():
    assert exports.export_filename("ad: name/v2_abcd", "xlsx") == "ad_name_v2_abcd.xlsx"
    assert exports.export_filename("", "csv") == "export.csv"


def test_exports_escape_formula_like_text_cells():
    rows = [
        {
            "feature_category": "=Category",
            "feature_name": "+Feature",
            "feature_criteria": "-Criteria",
            "evaluation": True,
            "llm_explanation": "@Explanation",
            "llm_prompt": "=Prompt",
        }
    ]
    report = abcd_exports.RunReport(RUN1, "=video.mp4", rows)

    run_csv_rows = list(csv.reader(io.StringIO(abcd_exports.run_csv(rows).decode())))
    assert run_csv_rows[1] == [
        "'=Category",
        "'+Feature",
        "'-Criteria",
        "PASS",
        "'@Explanation",
        "'=Prompt",
    ]

    job_csv_rows = list(
        csv.reader(io.StringIO(abcd_exports.job_csv([report]).decode()))
    )
    assert job_csv_rows[1][0] == "'=video.mp4"

    run_wb = openpyxl.load_workbook(io.BytesIO(abcd_exports.run_xlsx("ad.mp4", rows)))
    assert run_wb["Detail"]["A2"].value == "'=Category"
    assert run_wb["Scores"]["A2"].value == "'=Category"

    job_wb = openpyxl.load_workbook(io.BytesIO(abcd_exports.job_xlsx([report])))
    assert job_wb["Summary"]["A2"].value == "'=video.mp4"


def test_xlsx_strips_illegal_characters_and_caps_cell_length():
    rows = _rows(True)
    rows[0]["llm_explanation"] = "fast\x00 cuts\x1b early"
    rows[0]["llm_prompt"] = "p" * 40_000

    wb = openpyxl.load_workbook(io.BytesIO(abcd_exports.run_xlsx("ad.mp4", rows)))
    assert wb["Detail"]["E2"].value == "fast cuts early"
    assert len(wb["Detail"]["F2"].value) == 32_767


# ── routes ───────────────────────────────────────────────────────────────────


@pytest.fixture
def ctx(tmp_path, async_db_adapters):
    database_url = f"sqlite:///{tmp_path / 'exp.db'}"
    engine = create_engine(database_url)
    metadata_obj.create_all(engine)
    writer = ClipScribeWriterDB(engine=engine)
    async_reader, async_writer = async_db_adapters(database_url)

    with engine.begin() as conn:
        conn.execute(
            runs_table.insert(),
            [
                {"run_id": RUN1, "video_name": "first.mp4"},
                {"run_id": RUN2, "video_name": "second.mp4"},
            ],
        )
        for run_id, rows in ((RUN1, _rows(True)), (RUN2, _rows(False))):
            conn.execute(
                abcd_results_table.insert(),
                [{"run_id": run_id, "platform": "youtube", **r} for r in rows],
            )
            conn.execute(
                frame_detections_table.insert(),
                {
                    "run_id": run_id,
                    "shot_index": 0,
                    "frame_idx": 3,
                    "timestamp_sec": 0.1,
                    "source": "ocr",
                    "text": f"metadata-{run_id}",
                },
            )

    writer.create_job(
        job_id=PARENT, mode="full", video_name="batch", created_by="local"
    )
    writer.create_job(
        job_id=CHILD1,
        mode="full",
        parent_job_id=PARENT,
        run_id=RUN1,
        video_name="first.mp4",
        status="completed",
        created_by="local",
    )
    writer.create_job(
        job_id=CHILD2,
        mode="full",
        parent_job_id=PARENT,
        run_id=RUN2,
        video_name="second.mp4",
        status="completed",
        created_by="local",
    )

    app.dependency_overrides[get_reader] = lambda: async_reader
    app.dependency_overrides[get_writer] = lambda: async_writer
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_run_export_xlsx_route(ctx):
    resp = ctx.get(f"/runs/{RUN1}/abcd/export?format=xlsx")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == abcd_exports.CONTENT_TYPES["xlsx"]
    assert "attachment" in resp.headers["content-disposition"]
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert wb.sheetnames == ["Detail", "Scores"]


def test_run_export_csv_route(ctx):
    resp = ctx.get(f"/runs/{RUN1}/abcd/export?format=csv")
    assert resp.status_code == 200
    assert resp.text.splitlines()[0].startswith("Category,")


def test_job_export_xlsx_route(ctx):
    resp = ctx.get(f"/jobs/{PARENT}/abcd/export?format=xlsx")
    assert resp.status_code == 200
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert wb.sheetnames[0] == "Summary"
    assert len(wb.sheetnames) == 3  # Summary + one per run


def test_job_export_csv_stacks_runs(ctx):
    resp = ctx.get(f"/jobs/{PARENT}/abcd/export?format=csv")
    assert resp.status_code == 200
    body = resp.text
    assert "first.mp4," in body
    assert "second.mp4," in body


def test_run_metadata_export_xlsx_route(ctx):
    resp = ctx.get(f"/runs/{RUN1}/metadata/export?format=xlsx")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == metadata_exports.CONTENT_TYPES["xlsx"]
    assert 'filename="first.mp4_metadata.xlsx"' in resp.headers["content-disposition"]
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert wb.sheetnames == [
        "Overview",
        "Shots & Scenes",
        "Object Summary",
        "Object Occurrences",
        "On-screen Text",
        "Transcript",
        "Pacing",
        "Definitions",
    ]


def test_run_metadata_export_json_route_is_canonical(ctx):
    resp = ctx.get(f"/runs/{RUN1}/metadata/export?format=json")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == metadata_exports.CONTENT_TYPES["json"]
    assert 'filename="first.mp4_metadata.json"' in resp.headers["content-disposition"]
    payload = resp.json()
    assert payload["exported_at"].endswith("Z")
    assert payload["run"]["run_id"] == RUN1
    assert payload["run"]["video_name"] == "first.mp4"
    assert payload["run"]["video_type"] is None
    assert payload["run"]["extraction_mode"] == "with_objects"
    assert isinstance(payload["run"]["created_at"], str)
    detections = payload["extraction_summary"]["frame_detections"]
    assert detections[0]["text"] == f"metadata-{RUN1}"


def test_job_metadata_export_xlsx_route_stacks_runs(ctx):
    resp = ctx.get(f"/jobs/{PARENT}/metadata/export?format=xlsx")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == metadata_exports.CONTENT_TYPES["xlsx"]
    assert 'filename="batch_metadata.xlsx"' in resp.headers["content-disposition"]
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert wb.sheetnames[0] == "Summary"
    assert "Global Stats" in wb.sheetnames
    assert "Definitions" in wb.sheetnames
    summary_rows = list(wb["Summary"].iter_rows(values_only=True))
    assert [row[0] for row in summary_rows[1:]] == ["first.mp4", "second.mp4"]
    assert all(row[3] == "Yes" for row in summary_rows[1:])


def test_job_metadata_export_json_route_includes_each_child(ctx):
    resp = ctx.get(f"/jobs/{PARENT}/metadata/export?format=json")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["exported_at"].endswith("Z")
    assert payload["job"]["job_id"] == PARENT
    assert [run["job_id"] for run in payload["runs"]] == [CHILD1, CHILD2]
    assert [run["video_name"] for run in payload["runs"]] == [
        "first.mp4",
        "second.mp4",
    ]
    assert all(run["metadata_available"] for run in payload["runs"])
    assert [
        run["metadata"]["extraction_summary"]["frame_detections"][0]["text"]
        for run in payload["runs"]
    ] == [f"metadata-{RUN1}", f"metadata-{RUN2}"]


def test_export_bad_format_400(ctx):
    assert ctx.get(f"/runs/{RUN1}/abcd/export?format=pdf").status_code == 400
    assert ctx.get(f"/jobs/{PARENT}/abcd/export?format=pdf").status_code == 400
    assert ctx.get(f"/runs/{RUN1}/metadata/export?format=pdf").status_code == 400
    assert ctx.get(f"/jobs/{PARENT}/metadata/export?format=pdf").status_code == 400


def test_export_unknown_404(ctx):
    assert ctx.get("/runs/nope/abcd/export").status_code == 404
    assert ctx.get("/jobs/nope/export").status_code == 404
    assert ctx.get("/runs/nope/metadata/export").status_code == 404
    assert ctx.get("/jobs/nope/metadata/export").status_code == 404


def test_job_export_no_completed_runs_409(ctx, tmp_path, async_db_adapters):
    # A fresh parent whose only child is still running → nothing to export.
    database_url = f"sqlite:///{tmp_path / 'empty.db'}"
    engine = create_engine(database_url)
    metadata_obj.create_all(engine)
    writer = ClipScribeWriterDB(engine=engine)
    writer.create_job(job_id="p2", mode="full", video_name="batch", created_by="local")
    writer.create_job(
        job_id="c2",
        mode="full",
        parent_job_id="p2",
        status="running",
        created_by="local",
    )
    async_reader, _ = async_db_adapters(database_url)
    app.dependency_overrides[get_reader] = lambda: async_reader
    assert ctx.get("/jobs/p2/abcd/export").status_code == 409
    assert ctx.get("/jobs/p2/metadata/export").status_code == 409


def test_job_abcd_export_409_on_exploratory_job(ctx, tmp_path, async_db_adapters):
    database_url = f"sqlite:///{tmp_path / 'explore.db'}"
    engine = create_engine(database_url)
    metadata_obj.create_all(engine)
    writer = ClipScribeWriterDB(engine=engine)
    writer.create_job(
        job_id="p3",
        mode="full",
        parsing_mode="exploratory",
        video_name="batch",
        created_by="local",
    )
    async_reader, _ = async_db_adapters(database_url)
    app.dependency_overrides[get_reader] = lambda: async_reader
    assert ctx.get("/jobs/p3/abcd/export").status_code == 409
    assert ctx.post("/jobs/p3/chat", json={"message": "hi"}).status_code == 409
