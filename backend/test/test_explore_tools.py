"""Exploratory chat tools: report sections and run scoping."""

import asyncio
import json

from clip_scribe.parser.explore.prompts import RUN_SYSTEM_PROMPT
from clip_scribe.parser.explore.tools import build_run_tools


class _FakeReader:
    def __init__(self, payload: dict | None) -> None:
        self.payload = payload
        self.calls: list[tuple] = []

    async def get_exploration_result(self, run_id: str) -> dict | None:
        self.calls.append(("report", run_id))
        return None if self.payload is None else {"payload": self.payload}

    async def get_audio_segments(self, run_id: str, max_start_time=None) -> list[dict]:
        self.calls.append(("audio", run_id, max_start_time))
        return [{"start_time": 0.0, "end_time": 1.0, "text": "hi"}]


PAYLOAD = {
    "summary": "an ad",
    "chapters": [{"title": "Intro"}],
    "metrics": {"duration_sec": 3.0},
    "transcript": None,
    "on_screen_text": {"summary": "s", "cards": []},
    "visuals": None,
    "findings": [],
}


def _tools(reader):
    return {t.name: t for t in build_run_tools(reader, "run-1")}


def test_report_sections_return_only_their_fields():
    tools = _tools(_FakeReader(PAYLOAD))
    run = lambda **kw: json.loads(  # noqa: E731
        asyncio.run(tools["query_exploration_report"].ainvoke(kw))
    )
    assert run() == {"summary": "an ad", "chapters": [{"title": "Intro"}]}
    assert run(section="metrics") == {"metrics": {"duration_sec": 3.0}}
    # No speech: the null is passed on so the agent can say so.
    assert run(section="transcript") == {"transcript": None}


def test_missing_report_is_an_error_not_a_crash():
    tools = _tools(_FakeReader(None))
    out = asyncio.run(tools["query_exploration_report"].ainvoke({}))
    assert "No exploration report" in json.loads(out)["error"]


def test_tools_are_bound_to_the_run():
    reader = _FakeReader(PAYLOAD)
    tools = _tools(reader)
    asyncio.run(tools["query_audio_segments"].ainvoke({"max_start_time": 5.0}))
    # The model cannot pass a run_id; the bound one is always used.
    assert "run_id" not in tools["query_audio_segments"].args
    assert reader.calls == [("audio", "run-1", 5.0)]


def test_prompt_names_every_tool():
    for name in _tools(_FakeReader(PAYLOAD)):
        assert name in RUN_SYSTEM_PROMPT


def test_chat_service_picks_agent_and_prompt_by_analysis_type(monkeypatch):
    """Exploratory chat gets the exploration agent and prompt, not the ABCD ones."""
    from app import chat
    from clip_scribe.parser.abcd.prompts import RUN_SYSTEM_PROMPT as ABCD_PROMPT

    built: list[str] = []
    turns: list[str] = []

    class _Builder:
        @staticmethod
        def build_abcd_run_agent(model, reader, run_id):
            built.append("abcd")
            return "abcd-agent"

        @staticmethod
        def build_exploration_run_agent(model, reader, run_id):
            built.append("exploration")
            return "exploration-agent"

    class _Reader:
        async def get_chat_messages(self, run_id, session_id):
            return []

    service = chat.ChatService.__new__(chat.ChatService)
    service.reader = _Reader()  # type: ignore[assignment]
    service.model = None  # type: ignore[assignment]

    async def fake_turn(**kw):
        turns.append(kw["system_prompt"])
        assert kw["persist_scope"] == {"run_id": "r1"}
        yield "frame"

    monkeypatch.setattr(chat, "AgentBuilder", _Builder)
    monkeypatch.setattr(service, "_stream_turn", fake_turn)

    async def drain(stream):
        return [f async for f in stream]

    asyncio.run(drain(service.stream_exploration_run("r1", "s", "hi")))
    asyncio.run(drain(service.stream_abcd_run("r1", "s", "hi")))

    assert built == ["exploration", "abcd"]
    assert turns == [RUN_SYSTEM_PROMPT, ABCD_PROMPT]
