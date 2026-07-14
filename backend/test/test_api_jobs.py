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


def _full_body(videos=None, **over):
    body = {
        "mode": "full",
        "platform": "youtube",
        "videos": videos
        if videos is not None
        else [{"video_path": "ad.mp4", "video_name": "ad.mp4", "video_type": "car ad"}],
        "platform_params": {"brand_name": "RAM"},
        "user_hints": ["car"],
    }
    body.update(over)
    return body


def _only_child(state, parent_id):
    """The sole child run of a single-video batch job."""
    children = state.reader.get_child_jobs(parent_id)
    assert len(children) == 1
    return children[0]


def test_create_job_returns_202_and_queues_parent_plus_child(ctx):
    client, state = ctx
    state.install_service(run=False)  # don't run the engine; rows stay queued

    resp = client.post("/jobs", json=_full_body())
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["job_id"]
    # The parent owns no run; children do.
    assert data["run_id"] is None

    parent = state.reader.get_job(data["job_id"])
    assert parent["status"] == "queued"
    assert parent["mode"] == "full"
    assert parent["run_id"] is None
    assert parent["parent_job_id"] is None
    # full batch request persisted on the parent for reproducibility / retry
    assert parent["params_json"]["platform_params"]["brand_name"] == "RAM"

    child = _only_child(state, data["job_id"])
    assert child["status"] == "queued"
    assert child["run_id"]
    assert child["video_name"] == "ad.mp4"
    assert child["device"] == "cpu"
    # one child job was submitted to the executor
    assert len(state.executor.calls) == 1


def test_batch_creates_one_child_per_video(ctx):
    client, state = ctx
    state.install_service(run=False)

    resp = client.post(
        "/jobs",
        json=_full_body(
            videos=[
                {"video_path": "ad.mp4", "video_name": "ad.mp4"},
                {"video_path": "ad.mp4", "video_name": "second.mp4"},
            ]
        ),
    )
    assert resp.status_code == 202
    parent_id = resp.json()["job_id"]

    children = state.reader.get_child_jobs(parent_id)
    assert len(children) == 2
    assert {c["video_name"] for c in children} == {"ad.mp4", "second.mp4"}
    # distinct run ids, both dispatched
    assert len({c["run_id"] for c in children}) == 2
    assert len(state.executor.calls) == 2
    # parent aggregates its children (all queued) via the API
    assert client.get(f"/jobs/{parent_id}").json()["status"] == "queued"


def test_create_job_runs_to_completed(ctx):
    client, state = ctx
    state.install_service(run=True)  # inline success

    resp = client.post("/jobs", json=_full_body())
    parent_id = resp.json()["job_id"]

    child = _only_child(state, parent_id)
    assert child["status"] == "completed"
    assert child["started_at"] and child["finished_at"]
    assert state.engine.ran_with == child["run_id"]
    # parent aggregate reflects the completed child
    assert client.get(f"/jobs/{parent_id}").json()["status"] == "completed"


def test_create_job_records_failure(ctx):
    client, state = ctx
    state.install_service(fake_engine=FakeEngine(fail=True), run=True)

    resp = client.post("/jobs", json=_full_body())
    parent_id = resp.json()["job_id"]

    child = _only_child(state, parent_id)
    assert child["status"] == "failed"
    assert "kaboom" in child["error_text"]
    assert child["finished_at"]
    assert client.get(f"/jobs/{parent_id}").json()["status"] == "failed"


def test_dispatch_unavailable_creates_no_rows(ctx):
    client, state = ctx
    svc = JobService(state.reader, state.writer, state.settings)
    app.dependency_overrides[get_job_service] = lambda: svc

    resp = client.post("/jobs", json=_full_body())
    # Availability is checked before any row is written, so a 503 leaves no
    # half-created parent behind.
    assert resp.status_code == 503
    assert "Inline job execution is unavailable" in resp.json()["detail"]
    assert state.reader.list_jobs() == []


def test_canceled_job_does_not_start_if_task_later_runs(ctx):
    _, state = ctx
    from app.job_execution import build_task_payload, run_job_core
    from app.models import JobCreateRequest
    from src.utils.ids import new_ulid

    req = JobCreateRequest.model_validate(_full_body())
    video = req.videos[0]
    job_id = new_ulid()
    run_id = new_ulid()
    state.writer.create_job(
        job_id=job_id,
        mode=req.mode.value,
        status="canceled",
        run_id=run_id,
        video_name=video.video_name,
        video_path=video.video_path,
        platform=req.platform.value,
        params_json=req.model_dump(mode="json"),
    )

    run_job_core(
        state.builder,
        build_task_payload(
            job_id=job_id,
            run_id=run_id,
            req=req,
            video_name=video.video_name,
            video_path=video.video_path,
            video_type=video.video_type,
        ),
    )

    assert state.reader.get_job(job_id)["status"] == "canceled"
    assert state.builder.built == []
    assert state.engine.ran_with is None


def test_canceled_running_job_is_not_marked_completed(ctx):
    _, state = ctx
    from app.job_execution import build_task_payload, run_job_core
    from app.models import JobCreateRequest
    from src.utils.ids import new_ulid

    req = JobCreateRequest.model_validate(_full_body())
    video = req.videos[0]
    job_id = new_ulid()
    run_id = new_ulid()

    class CancelingEngine(FakeEngine):
        def run(self, run_id: str = "") -> None:
            self.ran_with = run_id
            state.writer.update_job(
                job_id,
                status="canceled",
                finished_at="2026-07-03T10:01:00",
            )

    state.install_service(fake_engine=CancelingEngine(), run=False)
    state.writer.create_job(
        job_id=job_id,
        mode=req.mode.value,
        status="queued",
        run_id=run_id,
        video_name=video.video_name,
        video_path=video.video_path,
        platform=req.platform.value,
        params_json=req.model_dump(mode="json"),
    )

    run_job_core(
        state.builder,
        build_task_payload(
            job_id=job_id,
            run_id=run_id,
            req=req,
            video_name=video.video_name,
            video_path=video.video_path,
            video_type=video.video_type,
        ),
    )

    row = state.reader.get_job(job_id)
    assert row["status"] == "canceled"
    assert row["finished_at"] == "2026-07-03T10:01:00"
    assert state.engine.ran_with == run_id


def test_create_job_missing_video_is_404(ctx):
    client, state = ctx
    resp = client.post(
        "/jobs",
        json=_full_body(videos=[{"video_path": "nope.mp4", "video_name": "nope.mp4"}]),
    )
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    # a failed resolve happens before any row is written
    assert state.reader.list_jobs() == []


def test_create_job_path_traversal_is_400(ctx):
    client, state = ctx
    resp = client.post(
        "/jobs",
        json=_full_body(
            videos=[{"video_path": "../secret.mp4", "video_name": "secret.mp4"}]
        ),
    )
    assert resp.status_code == 400


def test_create_job_requires_a_video_422(ctx):
    client, state = ctx
    resp = client.post("/jobs", json=_full_body(videos=[]))
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"


def test_parse_requires_run_id_422(ctx):
    client, state = ctx
    resp = client.post("/jobs", json={"mode": "parse", "platform": "youtube"})
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"


def test_parse_via_job_api_is_rejected(ctx):
    # Parse is a dev-only mode run via main.py, not the batch job API. A
    # model-valid parse request is rejected by the service before any row.
    client, state = ctx
    state.install_service(run=False)
    resp = client.post(
        "/jobs", json={"mode": "parse", "platform": "youtube", "run_id": "whatever"}
    )
    assert resp.status_code == 400
    assert state.reader.list_jobs() == []


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


def test_retry_child_runs_in_place(ctx):
    # Retrying a child re-runs just that video under the same job_id + parent,
    # with a fresh run_id — no new top-level job is spawned.
    client, state = ctx
    state.install_service(fake_engine=FakeEngine(fail=True), run=True)
    resp = client.post(
        "/jobs",
        json=_full_body(
            videos=[
                {"video_path": "ad.mp4", "video_name": "a.mp4"},
                {"video_path": "ad.mp4", "video_name": "b.mp4"},
            ]
        ),
    )
    parent_id = resp.json()["job_id"]
    child = state.reader.get_child_jobs(parent_id)[0]
    assert child["status"] == "failed"
    old_run = child["run_id"]

    r = client.post(f"/jobs/{child['job_id']}/retry")
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"] == child["job_id"]  # same child, in place
    assert body["run_id"] != old_run  # fresh run

    # Still exactly two children under the one parent — no new job created.
    assert len(state.reader.get_child_jobs(parent_id)) == 2
    assert len(state.reader.list_parent_jobs()) == 1


def test_delete_job_purges_child_runs(ctx):
    # Deleting a job must take its runs' data with it, not orphan them.
    client, state = ctx
    state.install_service(run=False)
    parent_id = client.post("/jobs", json=_full_body()).json()["job_id"]
    child = state.reader.get_child_jobs(parent_id)[0]
    rid = child["run_id"]

    # Simulate the extractor having written a run row for the child.
    from src.db.schema import runs_table

    with state.reader._engine.begin() as conn:
        conn.execute(runs_table.insert(), {"run_id": rid, "video_name": "ad.mp4"})
    assert state.reader.get_run(rid) is not None

    resp = client.delete(f"/jobs/{parent_id}")
    assert resp.status_code == 204
    assert state.reader.get_job(parent_id) is None
    assert state.reader.get_child_jobs(parent_id) == []
    assert state.reader.get_run(rid) is None  # run purged with the job


def test_run_siblings_resolve_before_runs_exist(ctx):
    # The inspector's run switcher must work while siblings are still
    # processing (their runs rows aren't written yet); siblings come from the
    # jobs graph, so the endpoint must not require the run row to exist.
    client, state = ctx
    state.install_service(run=False)
    resp = client.post(
        "/jobs",
        json=_full_body(
            videos=[
                {"video_path": "ad.mp4", "video_name": "a.mp4"},
                {"video_path": "ad.mp4", "video_name": "b.mp4"},
            ]
        ),
    )
    children = state.reader.get_child_jobs(resp.json()["job_id"])
    rid = children[0]["run_id"]

    sib = client.get(f"/runs/{rid}/siblings")
    assert sib.status_code == 200
    runs = sib.json()
    assert {s["video_name"] for s in runs} == {"a.mp4", "b.mp4"}
    assert all(s["status"] == "queued" for s in runs)


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
