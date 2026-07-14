"""Tests for the job-level advisory chat (spans every run in a batch job).

Covers the job_id-keyed transcript layer, the job tool set (list_job_runs,
query_job_scorecard, cross-run query_parser_results, run-id validation), and the
session GET/DELETE + guard-rail routes. The streaming POST is not exercised
end-to-end (needs an OpenAI call); we assert its 404 / 409 / 422 guards.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.deps import get_reader, get_writer
from app.main import app
from app.routes.chat import get_chat_service
from src.db.reader import ClipScribeReaderDB
from src.db.schema import metadata_obj, parser_results_table, runs_table
from src.db.writer import ClipScribeWriterDB
from src.parser.tools import build_job_tools

PARENT = "job-p"
CHILD1 = "job-c1"
CHILD2 = "job-c2"
RUN1 = "run-1"
RUN2 = "run-2"

RUNS = [
    {"run_id": RUN1, "video_name": "first.mp4"},
    {"run_id": RUN2, "video_name": "second.mp4"},
]


def _parser_rows(run_id: str, first_passes: bool) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "platform": "youtube",
            "feature_category": "Attract",
            "feature_name": "Dynamic Start",
            "feature_criteria": "hooks early",
            "evaluation": first_passes,
            "llm_explanation": "cuts",
        },
        {
            "run_id": run_id,
            "platform": "youtube",
            "feature_category": "Brand",
            "feature_name": "Brand Mention",
            "feature_criteria": "brand named",
            "evaluation": False,
            "llm_explanation": "no brand",
        },
    ]


@pytest.fixture
def ctx(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobchat.db'}")
    metadata_obj.create_all(engine)
    reader = ClipScribeReaderDB(engine=engine)
    writer = ClipScribeWriterDB(engine=engine)

    with engine.begin() as conn:
        conn.execute(
            runs_table.insert(),
            [
                {"run_id": RUN1, "video_name": "first.mp4"},
                {"run_id": RUN2, "video_name": "second.mp4"},
            ],
        )
        conn.execute(
            parser_results_table.insert(),
            [*_parser_rows(RUN1, True), *_parser_rows(RUN2, False)],
        )

    writer.create_job(job_id=PARENT, mode="full", video_name="batch")
    writer.create_job(
        job_id=CHILD1,
        mode="full",
        parent_job_id=PARENT,
        run_id=RUN1,
        video_name="first.mp4",
        status="completed",
    )
    writer.create_job(
        job_id=CHILD2,
        mode="full",
        parent_job_id=PARENT,
        run_id=RUN2,
        video_name="second.mp4",
        status="completed",
    )

    # Fake chat service so the GET/DELETE/POST-guard routes need no LLM client.
    class FakeChatService:
        def list_job_sessions(self, job_id):
            return reader.get_job_chat_sessions(job_id)

        def get_job_history(self, job_id, session_id):
            return reader.get_job_chat_messages(job_id, session_id)

        def delete_job_session(self, job_id, session_id):
            return writer.delete_job_chat_session(job_id, session_id)

        def resolve_job_runs(self, job_id):
            children = reader.get_child_jobs(job_id)
            return [
                {"run_id": c["run_id"], "video_name": c["video_name"]}
                for c in children
                if c.get("status") == "completed" and c.get("run_id")
            ]

        def stream_job(self, *a, **k):  # pragma: no cover - not hit by guard tests
            raise AssertionError("stream_job should not run in guard tests")

    app.dependency_overrides[get_reader] = lambda: reader
    app.dependency_overrides[get_writer] = lambda: writer
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    with TestClient(app) as client:
        yield client, reader, writer
    app.dependency_overrides.clear()


# ── transcript layer (job_id keyed, isolated from run chat) ──────────────────


def test_job_chat_roundtrip_and_isolation(ctx):
    _, reader, writer = ctx
    writer.add_chat_message(job_id=PARENT, session_id="s1", role="user", content="hi")
    writer.add_chat_message(
        job_id=PARENT,
        session_id="s1",
        role="assistant",
        content="yo",
        tool_calls=["query_job_scorecard"],
    )
    # A run-scoped message must NOT show up in the job transcript.
    writer.add_chat_message(
        run_id=RUN1, session_id="s1", role="user", content="run-only"
    )

    msgs = reader.get_job_chat_messages(PARENT, "s1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["tool_calls_json"] == ["query_job_scorecard"]

    sessions = reader.get_job_chat_sessions(PARENT)
    assert len(sessions) == 1 and sessions[0]["message_count"] == 2

    # The run chat sees only its own message.
    assert [m["content"] for m in reader.get_chat_messages(RUN1, "s1")] == ["run-only"]

    assert writer.delete_job_chat_session(PARENT, "s1") == 2
    assert reader.get_job_chat_messages(PARENT, "s1") == []


# ── job tools ────────────────────────────────────────────────────────────────


def _tool(reader, name):
    return next(t for t in build_job_tools(reader, RUNS) if t.name == name)


def test_list_job_runs(ctx):
    _, reader, _ = ctx
    out = json.loads(_tool(reader, "list_job_runs").invoke({}))
    assert {r["video_name"] for r in out} == {"first.mp4", "second.mp4"}


def test_job_scorecard_aggregates(ctx):
    _, reader, _ = ctx
    card = json.loads(_tool(reader, "query_job_scorecard").invoke({}))
    # 1 pass in run1, 0 in run2 → 1 of 4 overall.
    assert card["overall"] == {"passed": 1, "total": 4, "pass_rate": 0.25}
    by_run = {r["video_name"]: r["passed"] for r in card["by_run"]}
    assert by_run == {"first.mp4": 1, "second.mp4": 0}


def test_job_parser_results_across_and_single_run(ctx):
    _, reader, _ = ctx
    tool = _tool(reader, "query_parser_results")
    everything = json.loads(tool.invoke({}))
    assert len(everything) == 4
    assert {r["run_id"] for r in everything} == {RUN1, RUN2}

    only_one = json.loads(tool.invoke({"run_id": RUN1}))
    assert {r["run_id"] for r in only_one} == {RUN1}

    failed = json.loads(tool.invoke({"only_failed": True}))
    # SQLite returns booleans as 0/1, so test truthiness, not identity.
    assert failed and all(not r["evaluation"] for r in failed)


def test_job_tool_rejects_foreign_run_id(ctx):
    _, reader, _ = ctx
    res = json.loads(_tool(reader, "query_parser_results").invoke({"run_id": "other"}))
    assert "error" in res
    res2 = json.loads(_tool(reader, "query_global_stats").invoke({"run_id": "other"}))
    assert "error" in res2


# ── routes ───────────────────────────────────────────────────────────────────


def test_job_chat_session_endpoints(ctx):
    client, _, writer = ctx
    writer.add_chat_message(job_id=PARENT, session_id="s1", role="user", content="q1")
    writer.add_chat_message(
        job_id=PARENT, session_id="s1", role="assistant", content="a1"
    )

    sessions = client.get(f"/jobs/{PARENT}/chat/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["job_id"] == PARENT
    assert sessions.json()["sessions"][0]["session_id"] == "s1"

    hist = client.get(f"/jobs/{PARENT}/chat/s1")
    assert [m["content"] for m in hist.json()["messages"]] == ["q1", "a1"]

    assert client.request("DELETE", f"/jobs/{PARENT}/chat/s1").status_code == 204


def test_job_chat_404_unknown_job(ctx):
    client, _, _ = ctx
    assert client.get("/jobs/nope/chat/sessions").status_code == 404
    assert client.post("/jobs/nope/chat", json={"message": "hi"}).status_code == 404


def test_job_chat_post_empty_message_422(ctx):
    client, _, _ = ctx
    assert client.post(f"/jobs/{PARENT}/chat", json={"message": ""}).status_code == 422


def test_job_chat_409_when_no_completed_runs(ctx, tmp_path):
    client, _, _ = ctx
    # New parent with only a running child → nothing to analyze.
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    metadata_obj.create_all(engine)
    reader = ClipScribeReaderDB(engine=engine)
    writer = ClipScribeWriterDB(engine=engine)
    writer.create_job(job_id="p2", mode="full", video_name="batch")
    writer.create_job(job_id="c2", mode="full", parent_job_id="p2", status="running")
    app.dependency_overrides[get_reader] = lambda: reader

    class Svc:
        def resolve_job_runs(self, job_id):
            return []

    app.dependency_overrides[get_chat_service] = lambda: Svc()
    assert client.post("/jobs/p2/chat", json={"message": "hi"}).status_code == 409
