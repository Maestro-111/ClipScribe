"""API tests for uploads and artifact serving (web-app-plan §5, §6, §11)."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import settings as settings_mod
from app.deps import (
    artifact_storage_dep,
    get_reader,
    get_writer,
    settings_dep,
    video_storage_dep,
)
from app.main import app
from app.settings import Settings
from src.db.reader import ClipScribeReaderDB
from src.db.schema import metadata_obj, runs_table
from src.db.writer import ClipScribeWriterDB

RUN_ID = "r1"


@pytest.fixture
def client(tmp_path):
    os.environ["CLIPSCRIBE_API_LOAD_MODELS"] = "false"
    settings_mod.get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{tmp_path / 'art.db'}")
    metadata_obj.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            runs_table.insert(),
            {"run_id": RUN_ID, "video_name": "ad.mp4", "video_path": "ad.mp4"},
        )
    reader = ClipScribeReaderDB(engine=engine)
    writer = ClipScribeWriterDB(engine=engine)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "ad.mp4").write_bytes(b"hello-video")

    artifact_dir = tmp_path / "artifacts" / RUN_ID
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "tracked_output.mp4").write_bytes(b"tracked")

    settings = Settings()
    settings.input_dir = input_dir.resolve()

    # Point artifact resolution at the temp tree.
    import app.routes.artifacts as art_mod

    orig_root = art_mod.PROJECT_ROOT
    art_mod.PROJECT_ROOT = tmp_path

    app.dependency_overrides[get_reader] = lambda: reader
    app.dependency_overrides[get_writer] = lambda: writer
    app.dependency_overrides[settings_dep] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    art_mod.PROJECT_ROOT = orig_root


# ---- uploads -------------------------------------------------------------


def test_upload_accepts_video(client):
    r = client.post("/uploads", files={"files": ("clip.mp4", b"data", "video/mp4")})
    assert r.status_code == 200
    body = r.json()
    assert body["uploaded"][0]["name"] == "clip.mp4"
    assert body["uploaded"][0]["path"] != "clip.mp4"
    assert body["uploaded"][0]["path"].endswith(".mp4")
    assert body["uploaded"][0]["size_bytes"] == 4


def test_upload_duplicate_names_get_distinct_paths(client):
    first = client.post(
        "/uploads", files={"files": ("clip.mp4", b"first", "video/mp4")}
    )
    second = client.post(
        "/uploads", files={"files": ("clip.mp4", b"second", "video/mp4")}
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["uploaded"][0]["name"] == "clip.mp4"
    assert second.json()["uploaded"][0]["name"] == "clip.mp4"
    assert first.json()["uploaded"][0]["path"] != second.json()["uploaded"][0]["path"]


def test_upload_same_bytes_dedup_to_one_key(client):
    # Identical content uploaded twice reuses the stored object and one row.
    first = client.post("/uploads", files={"files": ("a.mp4", b"same", "video/mp4")})
    second = client.post("/uploads", files={"files": ("b.mp4", b"same", "video/mp4")})
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["uploaded"][0]["path"] == second.json()["uploaded"][0]["path"]
    # The dedup keeps the original registered name, not the second upload's.
    assert second.json()["uploaded"][0]["name"] == "a.mp4"
    # And the picker lists it exactly once.
    listed = client.get("/inputs").json()["videos"]
    assert [v["name"] for v in listed] == ["a.mp4"]


def test_inputs_prunes_missing_object(client, tmp_path):
    up = client.post("/uploads", files={"files": ("c.mp4", b"bytes", "video/mp4")})
    key = up.json()["uploaded"][0]["path"]
    assert client.get("/inputs").json()["videos"][0]["name"] == "c.mp4"

    # Simulate an out-of-band removal (bucket cleanup / dev deletes the file).
    (tmp_path / "input" / key).unlink()

    # Reconcile-on-read drops it from the picker and prunes the registry row.
    assert client.get("/inputs").json()["videos"] == []
    assert client.get("/inputs").json()["videos"] == []


def test_upload_rejects_bad_extension(client):
    r = client.post(
        "/uploads", files={"files": ("evil.exe", b"x", "application/octet-stream")}
    )
    assert r.status_code == 400
    assert r.headers["content-type"] == "application/problem+json"


def test_upload_strips_path_from_filename(client):
    r = client.post("/uploads", files={"files": ("../../pwn.mp4", b"x", "video/mp4")})
    assert r.status_code == 200
    assert r.json()["uploaded"][0]["name"] == "pwn.mp4"
    assert r.json()["uploaded"][0]["path"].endswith(".mp4")


# ---- artifacts -----------------------------------------------------------


def test_get_video_range_aware(client):
    r = client.get(f"/runs/{RUN_ID}/video")
    assert r.status_code == 200
    assert r.headers.get("accept-ranges") == "bytes"
    assert r.content == b"hello-video"

    ranged = client.get(f"/runs/{RUN_ID}/video", headers={"Range": "bytes=0-4"})
    assert ranged.status_code == 206
    assert ranged.content == b"hello"


def test_get_tracked_video(client):
    r = client.get(f"/runs/{RUN_ID}/tracked-video")
    assert r.status_code == 200
    assert r.content == b"tracked"


def test_artifact_unknown_run_404(client):
    assert client.get("/runs/missing/tracked-video").status_code == 404
    assert client.get("/runs/missing/video").status_code == 404


def test_gcs_missing_source_video_404s_without_local_fallback(client):
    class MissingSignedUrlStorage:
        def signed_url(self, key: str) -> None:
            return None

    settings = client.app.dependency_overrides[settings_dep]()
    settings.storage_backend = "gcs"
    client.app.dependency_overrides[video_storage_dep] = (
        lambda: MissingSignedUrlStorage()
    )

    assert client.get(f"/runs/{RUN_ID}/video").status_code == 404


def test_gcs_missing_tracked_video_404s_without_local_fallback(client):
    class MissingTrackedArtifactStorage:
        def tracked_video_url(self, run_id: str) -> None:
            return None

    settings = client.app.dependency_overrides[settings_dep]()
    settings.storage_backend = "gcs"
    client.app.dependency_overrides[artifact_storage_dep] = (
        lambda: MissingTrackedArtifactStorage()
    )

    assert client.get(f"/runs/{RUN_ID}/tracked-video").status_code == 404
