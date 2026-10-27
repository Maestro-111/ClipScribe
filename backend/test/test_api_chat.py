"""Tests for the advisory chat agent (web-app-plan §13).

Covers the DB transcript layer, the query_parser_results tool's filtering, and
the session GET/DELETE endpoints. The streaming POST is not exercised end-to-end
(that needs an OpenAI call); we assert its guard rails (404 / 422) instead.
"""

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import settings as settings_mod
from app.deps import get_reader, get_writer
from app.main import app
from app.routes.chat import get_chat_service
from clip_scribe.db.reader import ClipScribeReaderDB
from clip_scribe.db.schema import (
    jobs_table,
    metadata_obj,
    abcd_results_table,
    runs_table,
)
from clip_scribe.db.writer import ClipScribeWriterDB
from clip_scribe.parser.abcd.tools import build_run_tools, build_tools

RUN_ID = "r1"


@pytest.fixture
def ctx(tmp_path, async_db_adapters):
    os.environ["CLIPSCRIBE_API_LOAD_MODELS"] = "false"
    settings_mod.get_settings.cache_clear()

    database_url = f"sqlite:///{tmp_path / 'chat.db'}"
    engine = create_engine(database_url)
    metadata_obj.create_all(engine)
    reader = ClipScribeReaderDB(engine=engine)
    writer = ClipScribeWriterDB(engine=engine)
    async_reader, async_writer = async_db_adapters(database_url)
    with engine.begin() as conn:
        # Owning job so run_owner() resolves to the test user ("local") and the
        # ownership guard admits the run-scoped chat routes.
        conn.execute(
            jobs_table.insert(),
            {
                "job_id": "j1",
                "run_id": RUN_ID,
                "status": "completed",
                "created_by": "local",
            },
        )
        conn.execute(runs_table.insert(), {"run_id": RUN_ID, "video_name": "ad.mp4"})
        conn.execute(
            abcd_results_table.insert(),
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
        async def list_sessions(self, run_id):
            return await async_reader.get_chat_sessions(run_id)

        async def get_history(self, run_id, session_id):
            return await async_reader.get_chat_messages(run_id, session_id)

        async def delete_session(self, run_id, session_id):
            return await async_writer.delete_chat_session(run_id, session_id)

    app.dependency_overrides[get_reader] = lambda: async_reader
    app.dependency_overrides[get_writer] = lambda: async_writer
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    with TestClient(app) as client:
        yield client, reader, writer, async_reader

    app.dependency_overrides.clear()


def test_chat_message_db_roundtrip(ctx):
    _, reader, writer, _ = ctx
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
    _, reader, _, _ = ctx
    tools = build_tools(reader, RUN_ID, tool_group="advisory")
    tool = next(t for t in tools if t.name == "query_parser_results")

    all_results = json.loads(tool.invoke({}))
    assert len(all_results) == 2

    failed = json.loads(tool.invoke({"only_failed": True}))
    assert [r["feature_name"] for r in failed] == ["Brand Mention"]

    by_cat = json.loads(tool.invoke({"feature_category": "attract"}))
    assert [r["feature_name"] for r in by_cat] == ["Dynamic Start"]


def test_async_query_parser_results_tool_filters(ctx):
    _, _, _, async_reader = ctx
    tools = build_run_tools(async_reader, RUN_ID, tool_group="advisory")
    tool = next(t for t in tools if t.name == "query_parser_results")

    failed = json.loads(asyncio.run(tool.ainvoke({"only_failed": True})))

    assert [r["feature_name"] for r in failed] == ["Brand Mention"]


def test_advisory_tool_group_includes_all_query_tools(ctx):
    _, reader, _, async_reader = ctx
    sync_names = {t.name for t in build_tools(reader, RUN_ID, tool_group="advisory")}
    async_names = {
        t.name for t in build_run_tools(async_reader, RUN_ID, tool_group="advisory")
    }
    expected = {
        "query_audio_segments",
        "query_text_events",
        "query_visual_objects",
        "query_scene_descriptions",
        "query_global_stats",
        "query_field_descriptions",
        "query_parser_results",
    }
    assert expected <= sync_names
    assert async_names == sync_names


def test_list_and_get_sessions_endpoints(ctx):
    client, _, writer, _ = ctx
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
    client, reader, writer, _ = ctx
    writer.add_chat_message(run_id=RUN_ID, session_id="s1", role="user", content="q1")

    resp = client.request("DELETE", f"/runs/{RUN_ID}/chat/s1")
    assert resp.status_code == 204
    assert reader.get_chat_messages(RUN_ID, "s1") == []


def test_chat_endpoints_404_unknown_run(ctx):
    client, _, _, _ = ctx
    assert client.get("/runs/nope/chat/sessions").status_code == 404
    assert (
        client.post("/runs/nope/abcd/chat", json={"message": "hi"}).status_code == 404
    )


def test_chat_post_empty_message_422(ctx):
    client, _, _, _ = ctx
    resp = client.post(f"/runs/{RUN_ID}/abcd/chat", json={"message": ""})
    assert resp.status_code == 422
