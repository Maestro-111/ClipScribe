"""API tests for the read-only /runs/* endpoints (web-app-plan §5, §11).

Seeds a run and its child rows in a temp SQLite DB, overrides get_reader, and
asserts response shapes plus 404 on an unknown run.
"""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import settings as settings_mod
from app.deps import get_reader
from app.main import app
from src.db.reader import ClipScribeReaderDB
from src.db.schema import (
    metadata_obj,
    runs_table,
    global_stats_table,
    shot_boundaries_table,
    visual_object_occurrences_table,
    text_events_table,
    audio_segments_table,
    scene_descriptions_table,
    frame_detections_table,
    parser_results_table,
)

RUN_ID = "r1"


@pytest.fixture
def client(tmp_path):
    os.environ["CLIPSCRIBE_API_LOAD_MODELS"] = "false"
    settings_mod.get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    metadata_obj.create_all(engine)
    with engine.begin() as conn:
        conn.execute(runs_table.insert(), {"run_id": RUN_ID, "video_name": "ad.mp4"})
        conn.execute(
            global_stats_table.insert(),
            {"run_id": RUN_ID, "total_shots": 2, "video_duration": 3.0},
        )
        conn.execute(
            shot_boundaries_table.insert(),
            [
                {
                    "run_id": RUN_ID,
                    "shot_index": 0,
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "duration_sec": 1.5,
                },
                {
                    "run_id": RUN_ID,
                    "shot_index": 1,
                    "start_sec": 1.5,
                    "end_sec": 3.0,
                    "duration_sec": 1.5,
                },
            ],
        )
        conn.execute(
            visual_object_occurrences_table.insert(),
            {
                "run_id": RUN_ID,
                "global_id": 1,
                "label": "car",
                "shot_index": 0,
                "lifespan_start": 0.0,
                "lifespan_end": 1.0,
            },
        )
        conn.execute(
            text_events_table.insert(),
            {"run_id": RUN_ID, "second": 0, "line_index": 0, "text": "RAM"},
        )
        conn.execute(
            audio_segments_table.insert(),
            {
                "run_id": RUN_ID,
                "start_time": 0.0,
                "end_time": 1.0,
                "text": "hi",
                "confidence": 0.9,
            },
        )
        conn.execute(
            scene_descriptions_table.insert(),
            {
                "run_id": RUN_ID,
                "shot_index": 0,
                "start_time": 0.0,
                "end_time": 1.5,
                "description": "a car",
            },
        )
        conn.execute(
            frame_detections_table.insert(),
            [
                {
                    "run_id": RUN_ID,
                    "frame_idx": 0,
                    "timestamp_sec": 0.0,
                    "source": "dino",
                    "box_x1": 0,
                    "box_y1": 0,
                    "box_x2": 1,
                    "box_y2": 1,
                },
                {
                    "run_id": RUN_ID,
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
        conn.execute(
            parser_results_table.insert(),
            {
                "run_id": RUN_ID,
                "platform": "youtube",
                "feature_name": "logo",
                "evaluation": True,
            },
        )

    reader = ClipScribeReaderDB(engine=engine)
    app.dependency_overrides[get_reader] = lambda: reader
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_get_run(client):
    r = client.get(f"/runs/{RUN_ID}")
    assert r.status_code == 200
    assert r.json()["video_name"] == "ad.mp4"


def test_get_run_404(client):
    r = client.get("/runs/missing")
    assert r.status_code == 404
    assert r.headers["content-type"] == "application/problem+json"


def test_global_stats_bundles_shot_boundaries(client):
    r = client.get(f"/runs/{RUN_ID}/global-stats")
    assert r.status_code == 200
    body = r.json()
    assert body["global_stats"]["total_shots"] == 2
    assert [s["shot_index"] for s in body["shot_boundaries"]] == [0, 1]


def test_objects_text_audio_scenes(client):
    assert client.get(f"/runs/{RUN_ID}/objects").json()[0]["label"] == "car"
    assert client.get(f"/runs/{RUN_ID}/text-events").json()[0]["text"] == "RAM"
    assert client.get(f"/runs/{RUN_ID}/audio-segments").json()[0]["text"] == "hi"
    assert client.get(f"/runs/{RUN_ID}/scenes").json()[0]["description"] == "a car"


def test_frames_window(client):
    assert len(client.get(f"/runs/{RUN_ID}/frames").json()) == 2
    windowed = client.get(
        f"/runs/{RUN_ID}/frames", params={"from": 1.0, "to": 3.0}
    ).json()
    assert [f["frame_idx"] for f in windowed] == [60]


def test_parser(client):
    body = client.get(f"/runs/{RUN_ID}/parser").json()
    assert body[0]["feature_name"] == "logo"


def test_subresource_404_on_unknown_run(client):
    assert client.get("/runs/missing/objects").status_code == 404
    assert client.get("/runs/missing/global-stats").status_code == 404
