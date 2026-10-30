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


def test_chat_service_job_chat_picks_agent_and_prompt_by_analysis_type(monkeypatch):
    from app import chat
    from clip_scribe.parser.abcd.prompts import JOB_SYSTEM_PROMPT as ABCD_JOB_PROMPT
    from clip_scribe.parser.explore.prompts import JOB_SYSTEM_PROMPT

    built: list[str] = []
    turns: list[str] = []

    class _Builder:
        @staticmethod
        def build_abcd_job_agent(model, reader, runs):
            built.append("abcd")
            return "agent"

        @staticmethod
        def build_exploration_job_agent(model, reader, runs):
            built.append("exploration")
            return "agent"

    class _Reader:
        async def get_job_chat_messages(self, job_id, session_id):
            return []

    service = chat.ChatService.__new__(chat.ChatService)
    service.reader = _Reader()  # type: ignore[assignment]
    service.model = None  # type: ignore[assignment]

    async def fake_turn(**kw):
        turns.append(kw["system_prompt"])
        assert kw["persist_scope"] == {"job_id": "j1"}
        yield "frame"

    monkeypatch.setattr(chat, "AgentBuilder", _Builder)
    monkeypatch.setattr(service, "_stream_turn", fake_turn)

    async def drain(stream):
        return [f async for f in stream]

    runs = [{"run_id": "r1", "video_name": "a.mp4"}]
    asyncio.run(drain(service.stream_exploration_job("j1", "s", "hi", runs)))
    asyncio.run(drain(service.stream_abcd_job("j1", "s", "hi", runs)))

    assert built == ["exploration", "abcd"]
    assert turns == [JOB_SYSTEM_PROMPT, ABCD_JOB_PROMPT]


# ── job-level tools ──────────────────────────────────────────────────────────


class _JobReader:
    """Reports for two runs; ``r3`` has none."""

    def __init__(self) -> None:
        self.reports = {
            "r1": _job_payload(
                "first ad",
                duration=30.0,
                cards=[
                    ("Buy now", "cta", 2.0, 5.0),
                    ("$99", "price", 4.0, 8.0),
                ],
                subjects=[("car", "high", 20.0, 1.0)],
            ),
            "r2": _job_payload(
                "second ad",
                duration=60.0,
                cards=[("Sale $49", "price", 3.0, 10.0)],
                subjects=[("Person", "medium", 12.0, 0.0)],
            ),
        }

    async def get_exploration_result(self, run_id: str) -> dict | None:
        payload = self.reports.get(run_id)
        return None if payload is None else {"payload": payload}

    async def get_audio_segments(self, run_id: str, max_start_time=None) -> list[dict]:
        return [{"start_time": 0.0, "end_time": 1.0, "text": f"speech of {run_id}"}]


def _job_payload(summary, *, duration, cards, subjects):
    return {
        "summary": summary,
        "chapters": [{"title": "Intro"}, {"title": "Outro"}],
        "metrics": {
            "duration_sec": duration,
            "shot_count": 4,
            "avg_shot_sec": duration / 4,
            "cuts_per_minute": 8.0,
            "speech_coverage": 0.5,
            "text_coverage": 0.25,
            "words_per_minute": 120.0,
            "face_coverage": None,
            "objects_tracked": True,
            "objects": [],
        },
        "transcript": None,
        "on_screen_text": {
            "summary": "s",
            "cards": [
                {
                    "text": text.upper(),
                    "clean_text": text,
                    "category": category,
                    "total_sec": total,
                    "spans": [{"start_sec": first, "end_sec": first + total}],
                }
                for text, category, total, first in cards
            ],
        },
        "visuals": {
            "summary": "v",
            "subjects": [
                {
                    "label": label,
                    "prominence": prominence,
                    "screen_time_sec": screen,
                    "first_seen_sec": first,
                }
                for label, prominence, screen, first in subjects
            ],
        },
        "findings": [],
    }


JOB_RUNS = [
    {"run_id": "r1", "video_name": "first.mp4"},
    {"run_id": "r2", "video_name": "second.mp4"},
    {"run_id": "r3", "video_name": "third.mp4"},
]


def _job(name: str, **args):
    from clip_scribe.parser.explore.tools import build_job_tools

    tools = {t.name: t for t in build_job_tools(_JobReader(), JOB_RUNS)}
    return json.loads(asyncio.run(tools[name].ainvoke(args)))


def test_job_overview_compares_videos_and_flags_missing_reports():
    out = _job("query_job_overview")
    by_run = {v["run_id"]: v for v in out["videos"]}
    assert set(by_run) == {"r1", "r2"}
    assert by_run["r1"]["video_name"] == "first.mp4"
    assert by_run["r2"]["duration_sec"] == 60.0
    assert by_run["r1"]["chapters"] == 2
    assert by_run["r1"]["summary"] == "first ad"
    # The run without a report is reported, not silently dropped.
    assert out["runs_without_report"] == ["r3"]


def test_job_report_tool_validates_run_id_and_picks_section():
    assert (
        _job("query_exploration_report", run_id="r2", section="metrics")["metrics"][
            "duration_sec"
        ]
        == 60.0
    )
    assert "Unknown run_id" in _job("query_exploration_report", run_id="other")["error"]
    # A run of the job that has no report.
    assert (
        "No exploration report"
        in _job("query_exploration_report", run_id="r3")["error"]
    )


def test_job_on_screen_text_search_filters_across_videos():
    prices = _job("query_job_on_screen_text", category="price")
    assert [(c["run_id"], c["text"]) for c in prices] == [
        ("r1", "$99"),
        ("r2", "Sale $49"),
    ]
    assert prices[0]["first_seen_sec"] == 8.0
    assert prices[0]["video_name"] == "first.mp4"

    only = _job("query_job_on_screen_text", contains="SALE")
    assert [c["run_id"] for c in only] == ["r2"]
    assert _job("query_job_on_screen_text", category="url") == []
    assert len(_job("query_job_on_screen_text")) == 3


def test_job_visual_subjects_search_is_case_insensitive():
    both = _job("query_job_visual_subjects")
    assert {(s["run_id"], s["label"]) for s in both} == {
        ("r1", "car"),
        ("r2", "Person"),
    }
    one = _job("query_job_visual_subjects", label_contains="person")
    assert [(s["run_id"], s["prominence"]) for s in one] == [("r2", "medium")]


def test_job_raw_data_tools_are_run_validated_and_shared():
    ok = _job("query_audio_segments", run_id="r2")
    assert ok[0]["text"] == "speech of r2"
    assert "error" in _job("query_audio_segments", run_id="other")


def test_job_prompt_names_every_job_tool():
    from clip_scribe.parser.explore.prompts import JOB_SYSTEM_PROMPT
    from clip_scribe.parser.explore.tools import build_job_tools

    for tool in build_job_tools(_JobReader(), JOB_RUNS):
        assert tool.name in JOB_SYSTEM_PROMPT
