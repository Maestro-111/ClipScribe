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
    limits: list[int] = []

    class _Builder:
        @staticmethod
        def build_abcd_job_agent(model, reader, runs):
            built.append("abcd")
            return "agent"

        @staticmethod
        def build_exploration_job_agent(model, reader, runs, *, search_results_num):
            built.append("exploration")
            limits.append(search_results_num)
            return "agent"

    class _Reader:
        async def get_job_chat_messages(self, job_id, session_id):
            return []

    service = chat.ChatService.__new__(chat.ChatService)
    service.reader = _Reader()  # type: ignore[assignment]
    service.model = None  # type: ignore[assignment]
    service.job_search_results_num = 7

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
    # The configured search cap reaches the exploration job agent.
    assert limits == [7]


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


def _job(name: str, *, limit: int = 50, **args):
    from clip_scribe.parser.explore.tools import build_job_tools

    tools = {
        t.name: t
        for t in build_job_tools(_JobReader(), JOB_RUNS, search_results_num=limit)
    }
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
    # Longest on screen first: "$99" (visible 4s) before "Sale $49" (3s).
    assert [(c["run_id"], c["text"]) for c in prices["items"]] == [
        ("r1", "$99"),
        ("r2", "Sale $49"),
    ]
    assert prices["items"][0]["first_seen_sec"] == 8.0
    assert prices["items"][0]["video_name"] == "first.mp4"
    assert prices["total_matches"] == 2 and prices["truncated"] is False

    only = _job("query_job_on_screen_text", contains="SALE")
    assert [c["run_id"] for c in only["items"]] == ["r2"]
    none = _job("query_job_on_screen_text", category="url")
    assert none["items"] == [] and none["total_matches"] == 0
    assert _job("query_job_on_screen_text")["total_matches"] == 3


def test_job_search_results_are_capped_but_counts_cover_every_match():
    # Three cards match; with a cap of 2 the longest-visible two come back.
    out = _job("query_job_on_screen_text", limit=2)
    assert out["total_matches"] == 3 and out["returned"] == 2
    assert out["truncated"] is True
    assert [c["text"] for c in out["items"]] == ["$99", "Sale $49"]
    # Counts include the row that was cut, so "which videos" is still answerable.
    assert out["matches_by_video"] == {"first.mp4": 2, "second.mp4": 1}

    subjects = _job("query_job_visual_subjects", limit=1)
    # Most screen time first: the car (20s) beats the person (12s).
    assert [s["label"] for s in subjects["items"]] == ["car"]
    assert subjects["truncated"] is True
    assert subjects["matches_by_video"] == {"first.mp4": 1, "second.mp4": 1}


def test_job_visual_subjects_search_is_case_insensitive():
    both = _job("query_job_visual_subjects")["items"]
    assert {(s["run_id"], s["label"]) for s in both} == {
        ("r1", "car"),
        ("r2", "Person"),
    }
    one = _job("query_job_visual_subjects", label_contains="person")["items"]
    assert [(s["run_id"], s["prominence"]) for s in one] == [("r2", "medium")]


def test_job_raw_data_tools_are_run_validated_and_shared():
    ok = _job("query_audio_segments", run_id="r2")
    assert ok[0]["text"] == "speech of r2"
    assert "error" in _job("query_audio_segments", run_id="other")


def test_job_prompt_names_every_job_tool():
    from clip_scribe.parser.explore.prompts import JOB_SYSTEM_PROMPT
    from clip_scribe.parser.explore.tools import build_job_tools

    for tool in build_job_tools(_JobReader(), JOB_RUNS, search_results_num=50):
        assert tool.name in JOB_SYSTEM_PROMPT


def test_chat_service_reads_the_nested_parser_config(monkeypatch):
    from types import SimpleNamespace

    from app import chat

    seen: dict = {}

    def _fake_make_chat_model(model, **kw):
        seen.update(kw, model=model)

    monkeypatch.setattr(chat, "make_chat_model", _fake_make_chat_model)
    monkeypatch.setattr(
        chat,
        "_load_parser_config",
        lambda: {
            "llm": "gpt-x",
            "agent": {"temperature": 0.7, "max_retries": 3},
            "explore": {"job_search_results_num": 12},
        },
    )
    service = chat.ChatService(None, None, SimpleNamespace(CHAT_MAX_TOKENS=99))  # type: ignore[arg-type]
    assert seen["model"] == "gpt-x"
    assert seen["temperature"] == 0.7 and seen["max_retries"] == 3
    assert seen["max_completion_tokens"] == 99
    assert service.job_search_results_num == 12

    # A missing section falls back to the defaults instead of failing.
    monkeypatch.setattr(chat, "_load_parser_config", lambda: {})
    service = chat.ChatService(None, None, SimpleNamespace(CHAT_MAX_TOKENS=99))  # type: ignore[arg-type]
    assert service.job_search_results_num == 50
