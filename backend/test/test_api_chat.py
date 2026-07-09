"""Tests for the advisory chat agent (web-app-plan §13).

Covers the DB transcript layer, the query_parser_results tool's filtering, and
the session GET/DELETE endpoints. The streaming POST is not exercised end-to-end
(that needs an OpenAI call); we assert its guard rails (404 / 422) instead.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import settings as settings_mod
from app.deps import get_reader, get_writer
from app.main import app
from app.routes.chat import get_chat_service
from src.db.reader import ClipScribeReaderDB
from src.db.schema import metadata_obj, parser_results_table, runs_table
from src.db.writer import ClipScribeWriterDB
from src.parser.tools import build_tools

RUN_ID = "r1"


@pytest.fixture
def ctx(tmp_path):
    os.environ["CLIPSCRIBE_API_LOAD_MODELS"] = "false"
    settings_mod.get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{tmp_path / 'chat.db'}")
    metadata_obj.create_all(engine)
    reader = ClipScribeReaderDB(engine=engine)
    writer = ClipScribeWriterDB(engine=engine)
    with engine.begin() as conn:
        conn.execute(runs_table.insert(), {"run_id": RUN_ID, "video_name": "ad.mp4"})
        conn.execute(
            parser_results_table.insert(),
            [
                {
                    "run_id": RUN_ID,
                    "platform": "youtube",
                    "feature_category": "Attract",
                    "feature_name": "Dynamic Start",
                    "feature_criteria": "hooks in first 5s",
                    "evaluation": True,
                    "llm_explanation": "fast cuts early",
                },
                {
                    "run_id": RUN_ID,
                    "platform": "youtube",
                    "feature_category": "Brand",
                    "feature_name": "Brand Mention",
                    "feature_criteria": "brand named",
                    "evaluation": False,
                    "llm_explanation": "no brand in speech or text",
                },
            ],
        )

    # A fake chat service so the GET/DELETE routes don't build an LLM client.
    class FakeChatService:
        def list_sessions(self, run_id):
            return reader.get_chat_sessions(run_id)

        def get_history(self, run_id, session_id):
            return reader.get_chat_messages(run_id, session_id)

        def delete_session(self, run_id, session_id):
            return writer.delete_chat_session(run_id, session_id)

    app.dependency_overrides[get_reader] = lambda: reader
    app.dependency_overrides[get_writer] = lambda: writer
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    with TestClient(app) as client:
        yield client, reader, writer

    app.dependency_overrides.clear()


def test_chat_message_db_roundtrip(ctx):
    _, reader, writer = ctx
    writer.add_chat_message(run_id=RUN_ID, session_id="s1", role="user", content="hi")
    writer.add_chat_message(
        run_id=RUN_ID,
        session_id="s1",
        role="assistant",
        content="hello",
        tool_calls=["query_parser_results"],
    )

    msgs = reader.get_chat_messages(RUN_ID, "s1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["tool_calls_json"] == ["query_parser_results"]

    sessions = reader.get_chat_sessions(RUN_ID)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["message_count"] == 2
    assert sessions[0]["title"] == "hi"

    assert writer.delete_chat_session(RUN_ID, "s1") == 2
    assert reader.get_chat_messages(RUN_ID, "s1") == []


def test_query_parser_results_tool_filters(ctx):
    _, reader, _ = ctx
    tools = build_tools(reader, RUN_ID, tool_group="advisory")
    tool = next(t for t in tools if t.name == "query_parser_results")

    all_results = json.loads(tool.invoke({}))
    assert len(all_results) == 2

    failed = json.loads(tool.invoke({"only_failed": True}))
    assert [r["feature_name"] for r in failed] == ["Brand Mention"]

    by_cat = json.loads(tool.invoke({"feature_category": "attract"}))
    assert [r["feature_name"] for r in by_cat] == ["Dynamic Start"]


def test_advisory_tool_group_includes_all_query_tools(ctx):
    _, reader, _ = ctx
    names = {t.name for t in build_tools(reader, RUN_ID, tool_group="advisory")}
    assert {
        "query_audio_segments",
        "query_text_events",
        "query_visual_objects",
        "query_scene_descriptions",
        "query_global_stats",
        "query_field_descriptions",
        "query_parser_results",
    } <= names


def test_list_and_get_sessions_endpoints(ctx):
    client, _, writer = ctx
    writer.add_chat_message(run_id=RUN_ID, session_id="s1", role="user", content="q1")
    writer.add_chat_message(
        run_id=RUN_ID, session_id="s1", role="assistant", content="a1"
    )

    resp = client.get(f"/runs/{RUN_ID}/chat/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == RUN_ID
    assert body["sessions"][0]["session_id"] == "s1"

    hist = client.get(f"/runs/{RUN_ID}/chat/s1")
    assert hist.status_code == 200
    assert [m["content"] for m in hist.json()["messages"]] == ["q1", "a1"]


def test_delete_session_endpoint(ctx):
    client, reader, writer = ctx
    writer.add_chat_message(run_id=RUN_ID, session_id="s1", role="user", content="q1")

    resp = client.request("DELETE", f"/runs/{RUN_ID}/chat/s1")
    assert resp.status_code == 204
    assert reader.get_chat_messages(RUN_ID, "s1") == []


def test_chat_endpoints_404_unknown_run(ctx):
    client, _, _ = ctx
    assert client.get("/runs/nope/chat/sessions").status_code == 404
    assert client.post("/runs/nope/chat", json={"message": "hi"}).status_code == 404


def test_chat_post_empty_message_422(ctx):
    client, _, _ = ctx
    resp = client.post(f"/runs/{RUN_ID}/chat", json={"message": ""})
    assert resp.status_code == 422
