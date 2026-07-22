"""Tests for step-3 raw-artifact persistence and supporting seams.

Covers the new writer paths (frame_detections, shot_boundaries via save_run;
parser_results via save_parser_results), the ULID id helper, and the artifact
uploader factory. The extractor's collection logic needs models to run, so it
is exercised end-to-end via main.py rather than here.
"""

import types

import pytest
from sqlalchemy import create_engine, select

from src.db.schema import (
    metadata_obj,
    frame_detections_table,
    shot_boundaries_table,
    parser_results_table,
    runs_table,
)
from src.db.writer import ClipScribeWriterDB
from src.utils.ids import new_ulid
from src.utils.clip_scribe_artifacts import (
    GCSArtifactUploader,
    NullArtifactUploader,
    make_artifact_uploader,
    run_artifact_dir,
)


@pytest.fixture
def writer(tmp_path):
    """A writer over a throwaway file-backed SQLite DB with the schema created.

    Uses metadata.create_all directly (tests own their schema); production
    schema is owned by Alembic.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'persist.db'}")
    metadata_obj.create_all(engine)
    return ClipScribeWriterDB(engine=engine), engine


def _rows(engine, table):
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(select(table))]


def test_save_run_persists_shot_boundaries_and_frame_detections(writer):
    db, engine = writer
    run_id = new_ulid()

    video_metadata = {
        "global_stats": {},
        "visual_objects": [],
        "text_events": [],
        "audio_segments": [],
        "scene_descriptions": [],
        "shot_boundaries": [
            {"index": 0, "start": 0.0, "end": 1.5, "duration": 1.5},
            {"index": 1, "start": 1.5, "end": 3.0, "duration": 1.5},
        ],
        "frame_detections": [
            {
                "shot_index": 0,
                "frame_idx": 9,
                "timestamp_sec": 0.3,
                "source": "dino",
                "label": "car",
                "text": None,
                "box_x1": 1.0,
                "box_y1": 2.0,
                "box_x2": 3.0,
                "box_y2": 4.0,
                "confidence": 0.8,
                "object_id": None,
            },
            {
                "shot_index": 0,
                "frame_idx": 9,
                "timestamp_sec": 0.3,
                "source": "sam_mask",
                "label": "car",
                "text": None,
                "box_x1": 1.0,
                "box_y1": 2.0,
                "box_x2": 3.0,
                "box_y2": 4.0,
                "confidence": None,
                "object_id": 7,
            },
        ],
    }

    returned = db.save_run(
        run_id=run_id,
        video_name="v.mp4",
        video_path="input/v.mp4",
        video_type="car ad",
        video_metadata=video_metadata,
        field_descriptions={},
    )

    # The caller-supplied run_id is honored (not regenerated) and keys the rows.
    assert returned == run_id
    assert _rows(engine, runs_table)[0]["run_id"] == run_id

    shots = _rows(engine, shot_boundaries_table)
    assert len(shots) == 2
    assert {s["shot_index"] for s in shots} == {0, 1}
    assert all(s["run_id"] == run_id for s in shots)

    dets = _rows(engine, frame_detections_table)
    assert len(dets) == 2
    assert {d["source"] for d in dets} == {"dino", "sam_mask"}
    sam = next(d for d in dets if d["source"] == "sam_mask")
    assert sam["object_id"] == 7
    assert all(d["run_id"] == run_id for d in dets)


def test_save_parser_results_maps_feature_fields(writer):
    db, engine = writer
    run_id = new_ulid()

    # One rich (YouTube-shaped) result, one bare result lacking feature fields.
    rich = types.SimpleNamespace(
        platform="youtube",
        feature_category="Brand",
        feature_name="Brand Mention (Speech)",
        feature_criteria="brand said aloud",
        evaluation=True,
        llm_prompt="prompt",
        llm_explanation="because",
    )
    bare = types.SimpleNamespace(platform="youtube", evaluation=False)

    db.save_parser_results(run_id, "youtube", [rich, bare])

    rows = sorted(_rows(engine, parser_results_table), key=lambda r: r["id"])
    assert len(rows) == 2
    assert rows[0]["feature_name"] == "Brand Mention (Speech)"
    assert rows[0]["evaluation"] is True
    assert rows[0]["run_id"] == run_id
    # Missing feature fields persist as NULL, common columns still set.
    assert rows[1]["feature_name"] is None
    assert rows[1]["evaluation"] is False
    assert rows[1]["platform"] == "youtube"


def test_save_parser_results_replaces_same_run_platform(writer):
    db, engine = writer
    run_id = new_ulid()

    old = types.SimpleNamespace(
        platform="youtube",
        feature_category="Brand",
        feature_name="Old",
        feature_criteria="old criteria",
        evaluation=True,
    )
    other_platform = types.SimpleNamespace(
        platform="instagram",
        feature_category="Brand",
        feature_name="Instagram",
        feature_criteria="instagram criteria",
        evaluation=True,
    )
    new = types.SimpleNamespace(
        platform="youtube",
        feature_category="Brand",
        feature_name="New",
        feature_criteria="new criteria",
        evaluation=False,
    )

    db.save_parser_results(run_id, "youtube", [old])
    db.save_parser_results(run_id, "instagram", [other_platform])
    db.save_parser_results(run_id, "youtube", [new])

    rows = sorted(
        _rows(engine, parser_results_table), key=lambda r: (r["platform"], r["id"])
    )
    assert [(r["platform"], r["feature_name"]) for r in rows] == [
        ("instagram", "Instagram"),
        ("youtube", "New"),
    ]


def test_new_ulid_is_sortable_and_26_chars():
    a = new_ulid()
    b = new_ulid()
    assert len(a) == 26 and len(b) == 26
    assert a != b
    # ULIDs are lexicographically time-ordered; later id sorts >= earlier.
    assert b >= a


def test_run_artifact_dir_keys_by_run_id():
    assert run_artifact_dir("01ABC") == "artifacts/01ABC"


def test_null_artifact_uploader_is_noop():
    # The local backend's uploader stores nothing and serves from disk (no URL).
    up = NullArtifactUploader()
    up.upload_run_artifacts("rid", "artifacts/rid")  # must not raise
    assert up.tracked_video_url("rid") is None
    up.delete_run_artifacts("rid")


def test_make_artifact_uploader_selects_backend():
    assert isinstance(make_artifact_uploader("local"), NullArtifactUploader)
    # gcs requires a bucket; a missing one is a fail-fast config error.
    with pytest.raises(ValueError):
        make_artifact_uploader("gcs")


def test_gcs_artifact_uploader_deletes_run_prefix():
    class FakeBlob:
        def __init__(self, name: str) -> None:
            self.name = name
            self.deleted = False

        def delete(self) -> None:
            self.deleted = True

    class FakeBucket:
        def blob(self, name: str):
            raise AssertionError(name)

    class FakeClient:
        def __init__(self) -> None:
            self.blobs = [
                FakeBlob("artifacts/rid/tracked_output.mp4"),
                FakeBlob("artifacts/rid/artifacts.tar.gz"),
                FakeBlob("artifacts/other/artifacts.tar.gz"),
            ]

        def bucket(self, name: str) -> FakeBucket:
            return FakeBucket()

        def list_blobs(self, bucket: str, *, prefix: str):
            assert bucket == "clipscribe"
            return [b for b in self.blobs if b.name.startswith(prefix)]

    client = FakeClient()

    GCSArtifactUploader("clipscribe", client=client).delete_run_artifacts("rid")

    assert [b.deleted for b in client.blobs] == [True, True, False]
