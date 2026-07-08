"""API tests for the jobs endpoints and JobService (web-app-plan §5, §11).

No models are loaded: a fake builder (real reader/writer over temp SQLite, a
stub engine) and an inline executor stand in, wired via dependency_overrides.
"""

import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import settings as settings_mod
from app.deps import get_executor, get_reader
from app.job_runner import JobService
from app.main import app
from app.routes.jobs import get_job_service
from app.settings import Settings
from src.db.reader import ClipScribeReaderDB
from src.db.schema import metadata_obj
from src.db.writer import ClipScribeWriterDB


class InlineExecutor:
    """Runs submitted callables synchronously (or records them) for tests."""

    def __init__(self, run: bool = True):
        self.run = run
        self.calls: list = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        if self.run:
            fn(*args, **kwargs)
        return None


class FakeEngine:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.ran_with: str | None = None

    def run(self, run_id: str = "") -> None:
        self.ran_with = run_id
        if self.fail:
            raise RuntimeError("kaboom")


class FakeBuilder:
    def __init__(self, reader, writer, engine, device="cpu"):
        self.reader_db = reader
        self.writer_db = writer
        self.device = device
        self._engine = engine
        self.built: list[dict] = []

    def build_clip_scribe(self, **kwargs):
        self.built.append(kwargs)
        return self._engine


@pytest.fixture
def ctx(tmp_path):
    os.environ["CLIPSCRIBE_API_LOAD_MODELS"] = "false"
    # Pin the dispatch backend so an ambient .env (CLIPSCRIBE_JOB_BACKEND=celery)
    # can't push these tests onto the Redis path; they exercise the inline path.
    os.environ["CLIPSCRIBE_JOB_BACKEND"] = "inline"
    settings_mod.get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    metadata_obj.create_all(engine)
    reader = ClipScribeReaderDB(engine=engine)
    writer = ClipScribeWriterDB(engine=engine)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "ad.mp4").write_bytes(b"video-bytes")

    settings = Settings()
    settings.input_dir = input_dir.resolve()

    state = SimpleNamespace(
        reader=reader, writer=writer, settings=settings, input_dir=input_dir
    )

    app.dependency_overrides[get_reader] = lambda: reader
    app.dependency_overrides[get_executor] = lambda: InlineExecutor()

    def install_service(fake_engine=None, run=True, device="cpu"):
        fake_engine = fake_engine or FakeEngine()
        executor = InlineExecutor(run=run)
        builder = FakeBuilder(reader, writer, fake_engine, device=device)
        svc = JobService(reader, writer, settings, builder=builder, executor=executor)
        app.dependency_overrides[get_job_service] = lambda: svc
        state.svc = svc
        state.executor = executor
        state.builder = builder
        state.engine = fake_engine
        return svc

    state.install_service = install_service
    install_service()

    with TestClient(app) as client:
        yield client, state

    app.dependency_overrides.clear()


def _full_body(**over):
    body = {
        "mode": "full",
        "platform": "youtube",
        "video_path": "ad.mp4",
        "video_name": "ad.mp4",
        "video_type": "car ad",
        "platform_params": {"brand_name": "RAM"},
        "user_hints": ["car"],
    }
    body.update(over)
    return body


def test_create_job_returns_202_and_queues_row(ctx):
    client, state = ctx
    state.install_service(run=False)  # don't run the engine; row stays queued

    resp = client.post("/jobs", json=_full_body())
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["job_id"] and data["run_id"]

    row = state.reader.get_job(data["job_id"])
    assert row["status"] == "queued"
    assert row["mode"] == "full"
    assert row["run_id"] == data["run_id"]
    assert row["device"] == "cpu"
    # full request payload persisted for reproducibility
    assert row["params_json"]["platform_params"]["brand_name"] == "RAM"
    # one job was submitted to the executor
    assert len(state.executor.calls) == 1


def test_create_job_runs_to_completed(ctx):
    client, state = ctx
    state.install_service(run=True)  # inline success

    resp = client.post("/jobs", json=_full_body())
    job_id = resp.json()["job_id"]
    run_id = resp.json()["run_id"]

    row = state.reader.get_job(job_id)
    assert row["status"] == "completed"
    assert row["started_at"] and row["finished_at"]
    assert state.engine.ran_with == run_id


def test_create_job_records_failure(ctx):
    client, state = ctx
    state.install_service(fake_engine=FakeEngine(fail=True), run=True)

    resp = client.post("/jobs", json=_full_body())
    job_id = resp.json()["job_id"]

    row = state.reader.get_job(job_id)
    assert row["status"] == "failed"
    assert "kaboom" in row["error_text"]
    assert row["finished_at"]


def test_create_job_missing_video_is_404(ctx):
    client, state = ctx
    resp = client.post("/jobs", json=_full_body(video_path="nope.mp4"))
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"


def test_create_job_path_traversal_is_400(ctx):
    client, state = ctx
    resp = client.post("/jobs", json=_full_body(video_path="../secret.mp4"))
    assert resp.status_code == 400


def test_parse_requires_run_id_422(ctx):
    client, state = ctx
    resp = client.post("/jobs", json={"mode": "parse", "platform": "youtube"})
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"


def test_parse_unknown_run_id_404(ctx):
    client, state = ctx
    resp = client.post(
        "/jobs", json={"mode": "parse", "platform": "youtube", "run_id": "missing"}
    )
    assert resp.status_code == 404


def test_parse_existing_run_uses_run_metadata(ctx):
    client, state = ctx
    from src.db.schema import runs_table

    with state.reader._engine.begin() as conn:
        conn.execute(
            runs_table.insert(),
            {
                "run_id": "existing",
                "video_name": "old.mp4",
                "video_path": "input/old.mp4",
                "video_type": "ad",
            },
        )
    state.install_service(run=False)

    resp = client.post(
        "/jobs", json={"mode": "parse", "platform": "youtube", "run_id": "existing"}
    )
    assert resp.status_code == 202
    row = state.reader.get_job(resp.json()["job_id"])
    assert row["run_id"] == "existing"
    assert row["video_name"] == "old.mp4"


def test_unsupported_platform_is_422(ctx):
    client, state = ctx
    # Only youtube is registered; anything else is rejected at validation.
    resp = client.post("/jobs", json=_full_body(platform="tiktok"))
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"


def test_youtube_params_map_to_build_kwargs(ctx):
    from app.models import JobCreateRequest

    req = JobCreateRequest.model_validate(
        _full_body(
            platform_params={"brand_name": "RAM", "call_to_actions": ["buy now"]}
        )
    )
    assert req.resolved_params.to_build_kwargs() == {
        "youtube_brand_name": "RAM",
        "youtube_branded_products": [],
        "youtube_branded_products_categories": [],
        "youtube_call_to_actions": ["buy now"],
    }


def test_get_job_404(ctx):
    client, state = ctx
    assert client.get("/jobs/nope").status_code == 404


def test_list_jobs_filter_and_shape(ctx):
    client, state = ctx
    state.install_service(run=False)
    client.post("/jobs", json=_full_body())
    client.post("/jobs", json=_full_body())

    resp = client.get("/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 50 and body["offset"] == 0
    assert len(body["jobs"]) == 2

    queued = client.get("/jobs", params={"status": "queued"})
    assert len(queued.json()["jobs"]) == 2
    assert client.get("/jobs", params={"status": "completed"}).json()["jobs"] == []
