"""Tests for the step-5 jobs-table CRUD and new read methods.

Covers ``ClipScribeWriterDB.create_job`` / ``update_job`` and the reader
additions backing the API (``get_job``, ``list_jobs``, ``list_runs``,
``get_shot_boundaries``, ``get_frame_detections``, ``get_parser_results``).
Uses metadata.create_all directly (tests own their schema; production schema
is owned by Alembic).
"""

import pytest
from sqlalchemy import create_engine

from src.db.schema import (
    metadata_obj,
    runs_table,
    shot_boundaries_table,
    frame_detections_table,
    parser_results_table,
)
from src.db.reader import ClipScribeReaderDB
from src.db.writer import ClipScribeWriterDB
from src.utils.ids import new_ulid


@pytest.fixture
def db(tmp_path):
    """Reader + writer sharing one throwaway file-backed SQLite DB."""
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    metadata_obj.create_all(engine)
    return ClipScribeWriterDB(engine=engine), ClipScribeReaderDB(engine=engine), engine


def test_create_job_defaults_to_queued(db):
    writer, reader, _ = db
    job_id = new_ulid()

    writer.create_job(
        job_id=job_id,
        mode="full",
        run_id="run-123",
        video_name="ad.mp4",
        video_path="input/ad.mp4",
        video_type="car ad",
        device="mps",
        platform="youtube",
        params_json={"user_hints": ["car"], "generate_hint_from_name": False},
    )

    job = reader.get_job(job_id)
    assert job is not None
    assert job["status"] == "queued"
    assert job["mode"] == "full"
    assert job["run_id"] == "run-123"
    assert job["created_at"] is not None
    # params_json round-trips as a dict, not a raw JSON string.
    assert job["params_json"] == {
        "user_hints": ["car"],
        "generate_hint_from_name": False,
    }


def test_get_job_missing_returns_none(db):
    _, reader, _ = db
    assert reader.get_job("does-not-exist") is None


def test_update_job_writes_only_provided_fields(db):
    writer, reader, _ = db
    job_id = new_ulid()
    writer.create_job(job_id=job_id, mode="extract")

    writer.update_job(job_id, status="running", started_at="2026-07-03T10:00:00")
    job = reader.get_job(job_id)
    assert job["status"] == "running"
    assert job["started_at"] == "2026-07-03T10:00:00"
    assert job["finished_at"] is None
    assert job["error_text"] is None

    # A later transition must not clobber previously-set fields.
    writer.update_job(job_id, status="failed", error_text="boom")
    job = reader.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error_text"] == "boom"
    assert job["started_at"] == "2026-07-03T10:00:00"


def test_update_job_noop_when_nothing_provided(db):
    writer, reader, _ = db
    job_id = new_ulid()
    writer.create_job(job_id=job_id, mode="parse", status="queued")
    writer.update_job(job_id)  # no fields -> no-op, no error
    assert reader.get_job(job_id)["status"] == "queued"


def test_update_job_if_status_respects_current_status(db):
    writer, reader, _ = db
    job_id = new_ulid()
    writer.create_job(job_id=job_id, mode="full", status="canceled")

    updated = writer.update_job_if_status(
        job_id,
        allowed_statuses=("queued", "running"),
        status="running",
        started_at="2026-07-03T10:00:00",
    )

    assert updated is False
    job = reader.get_job(job_id)
    assert job["status"] == "canceled"
    assert job["started_at"] is None

    updated = writer.update_job_if_status(
        job_id,
        allowed_statuses=("canceled",),
        error_text="kept canceled",
    )
    assert updated is True
    assert reader.get_job(job_id)["error_text"] == "kept canceled"


def test_list_jobs_orders_recent_first_and_filters_status(db):
    writer, reader, _ = db
    # ULIDs are monotonic, so a later create sorts first under DESC ordering.
    old = new_ulid()
    new = new_ulid()
    writer.create_job(job_id=old, mode="full", status="completed")
    writer.create_job(job_id=new, mode="full", status="queued")

    ids = [j["job_id"] for j in reader.list_jobs()]
    assert ids.index(new) < ids.index(old)

    queued = reader.list_jobs(status="queued")
    assert [j["job_id"] for j in queued] == [new]

    assert reader.list_jobs(limit=1) == reader.list_jobs(limit=1, offset=0)
    assert len(reader.list_jobs(limit=1)) == 1


def test_parent_children_and_siblings(db):
    writer, reader, _ = db
    parent = new_ulid()
    child_a = new_ulid()
    child_b = new_ulid()
    writer.create_job(job_id=parent, mode="full", status="queued")
    writer.create_job(
        job_id=child_a,
        mode="full",
        parent_job_id=parent,
        run_id="run-a",
        video_name="a.mp4",
    )
    writer.create_job(
        job_id=child_b,
        mode="full",
        parent_job_id=parent,
        run_id="run-b",
        video_name="b.mp4",
    )

    # Only the parent is top-level; children are excluded from the list.
    parents = reader.list_parent_jobs()
    assert [p["job_id"] for p in parents] == [parent]

    children = reader.get_child_jobs(parent)
    assert [c["job_id"] for c in children] == [child_a, child_b]
    assert children[0]["parent_job_id"] == parent

    # A run resolves to its siblings (both children), including itself.
    siblings = reader.get_run_siblings("run-a")
    assert {s["run_id"] for s in siblings} == {"run-a", "run-b"}
    # A run with no batch job has no siblings.
    assert reader.get_run_siblings("orphan") == []


def test_list_runs_most_recent_first(db):
    writer, reader, engine = db
    with engine.begin() as conn:
        conn.execute(
            runs_table.insert(),
            [
                {
                    "run_id": "r1",
                    "video_name": "a",
                    "created_at": "2026-01-01 00:00:00",
                },
                {
                    "run_id": "r2",
                    "video_name": "b",
                    "created_at": "2026-02-01 00:00:00",
                },
            ],
        )
    runs = reader.list_runs()
    assert [r["run_id"] for r in runs] == ["r2", "r1"]


def test_get_shot_boundaries_ordered(db):
    _, reader, engine = db
    with engine.begin() as conn:
        conn.execute(
            shot_boundaries_table.insert(),
            [
                {
                    "run_id": "r1",
                    "shot_index": 1,
                    "start_sec": 1.5,
                    "end_sec": 3.0,
                    "duration_sec": 1.5,
                },
                {
                    "run_id": "r1",
                    "shot_index": 0,
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "duration_sec": 1.5,
                },
            ],
        )
    rows = reader.get_shot_boundaries("r1")
    assert [r["shot_index"] for r in rows] == [0, 1]


def test_get_frame_detections_time_window(db):
    _, reader, engine = db
    with engine.begin() as conn:
        conn.execute(
            frame_detections_table.insert(),
            [
                {
                    "run_id": "r1",
                    "frame_idx": 0,
                    "timestamp_sec": 0.0,
                    "source": "dino",
                    "box_x1": 0,
                    "box_y1": 0,
                    "box_x2": 1,
                    "box_y2": 1,
                },
                {
                    "run_id": "r1",
                    "frame_idx": 30,
                    "timestamp_sec": 1.0,
                    "source": "dino",
                    "box_x1": 0,
                    "box_y1": 0,
                    "box_x2": 1,
                    "box_y2": 1,
                },
                {
                    "run_id": "r1",
                    "frame_idx": 60,
                    "timestamp_sec": 2.0,
                    "source": "ocr",
                    "box_x1": 0,
                    "box_y1": 0,
                    "box_x2": 1,
                    "box_y2": 1,
                },
            ],
        )
    assert len(reader.get_frame_detections("r1")) == 3
    windowed = reader.get_frame_detections("r1", from_sec=0.5, to_sec=1.5)
    assert [d["frame_idx"] for d in windowed] == [30]


def test_get_parser_results(db):
    _, reader, engine = db
    with engine.begin() as conn:
        conn.execute(
            parser_results_table.insert(),
            [
                {
                    "run_id": "r1",
                    "platform": "youtube",
                    "feature_category": "brand",
                    "feature_name": "logo",
                    "evaluation": True,
                },
            ],
        )
    rows = reader.get_parser_results("r1")
    assert len(rows) == 1
    assert rows[0]["evaluation"] in (True, 1)


def test_delete_run_clears_run_keyed_tables(db):
    writer, reader, engine = db
    with engine.begin() as conn:
        conn.execute(runs_table.insert(), {"run_id": "r1", "video_name": "a"})
        conn.execute(
            frame_detections_table.insert(),
            {
                "run_id": "r1",
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "source": "dino",
                "box_x1": 0,
                "box_y1": 0,
                "box_x2": 1,
                "box_y2": 1,
            },
        )
        conn.execute(
            parser_results_table.insert(),
            {"run_id": "r1", "platform": "youtube", "evaluation": True},
        )
        # A second run must survive the delete.
        conn.execute(runs_table.insert(), {"run_id": "r2", "video_name": "b"})

    writer.delete_run("r1")

    assert reader.get_run("r1") is None
    assert reader.get_frame_detections("r1") == []
    assert reader.get_parser_results("r1") == []
    assert reader.get_run("r2") is not None


def test_delete_job_clears_job_scoped_chat_only(db):
    writer, reader, _ = db
    writer.create_job(job_id="j1", mode="full")
    writer.create_job(job_id="j2", mode="full")
    writer.add_chat_message(
        job_id="j1", session_id="s1", role="user", content="job one"
    )
    writer.add_chat_message(
        job_id="j2", session_id="s1", role="user", content="job two"
    )
    writer.add_chat_message(
        run_id="r1", session_id="s1", role="user", content="run one"
    )

    assert writer.delete_job("j1") is True

    assert reader.get_job("j1") is None
    assert reader.get_job_chat_messages("j1", "s1") == []
    assert [m["content"] for m in reader.get_job_chat_messages("j2", "s1")] == [
        "job two"
    ]
    assert [m["content"] for m in reader.get_chat_messages("r1", "s1")] == ["run one"]
