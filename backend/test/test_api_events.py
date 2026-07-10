"""Tests for the live-progress transport (web-app-plan §9, step 9).

No models and no real Redis: a ``fakeredis`` server backs both the sync writer
side (``RedisProgressReporter`` / the log bridge) and the async reader side (the
SSE generator), sharing one in-memory server so a write is visible to the read.
"""

import asyncio
import json
import logging

import fakeredis
import fakeredis.aioredis
import pytest

from app import events
from app.events import (
    JobLogStreamHandler,
    RedisProgressReporter,
    current_job_id,
    make_reporter,
    stream_key,
)
from app.events import summarize_progress
from app.routes.jobs import _job_event_stream, _sse_frame
from src.utils.progress import NullProgressReporter, ProgressEvent


# --- RedisProgressReporter -------------------------------------------------
def test_reporter_writes_ordered_entries_and_expires_on_terminal() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    reporter = RedisProgressReporter("job1", client)

    reporter.emit(ProgressEvent.JOB_STARTED, {"video_name": "ad"})
    reporter.emit(ProgressEvent.SHOT_STARTED, {"shot_idx": 0})
    reporter.emit(ProgressEvent.JOB_COMPLETED, {"run_id": "R1"})

    entries = client.xrange(stream_key("job1"))
    assert [e[1]["type"] for e in entries] == [
        "job.started",
        "shot.started",
        "job.completed",
    ]
    assert json.loads(entries[0][1]["data"]) == {"video_name": "ad"}
    # The terminal event set a TTL so finished streams eventually expire.
    assert client.ttl(stream_key("job1")) > 0


def test_reporter_never_raises_on_redis_error() -> None:
    class Boom:
        def xadd(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("redis down")

    reporter = RedisProgressReporter("j", Boom())  # type: ignore[arg-type]
    # Contract: emit must swallow transport errors, never break the job.
    reporter.emit(ProgressEvent.SHOT_STARTED, {"shot_idx": 0})


def test_make_reporter_falls_back_to_null_when_redis_unreachable(monkeypatch) -> None:
    def boom(*a, **k):  # noqa: ANN002, ANN003
        raise ConnectionError("no redis")

    monkeypatch.setattr(events.redis.Redis, "from_url", staticmethod(boom))
    assert isinstance(make_reporter("redis://x", "j"), NullProgressReporter)


# --- Log bridge ------------------------------------------------------------
def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("clip_scribe", logging.INFO, __file__, 1, msg, None, None)


def test_log_bridge_routes_only_within_a_job_context() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    handler = JobLogStreamHandler(client)

    # Outside a job context the record is dropped, not misrouted.
    handler.emit(_record("ambient"))
    assert client.xlen(stream_key("jobX")) == 0

    token = current_job_id.set("jobX")
    try:
        handler.emit(_record("hello"))
    finally:
        current_job_id.reset(token)

    entries = client.xrange(stream_key("jobX"))
    assert len(entries) == 1
    assert entries[0][1]["type"] == "log"
    assert json.loads(entries[0][1]["data"]) == {"level": "INFO", "message": "hello"}


# --- SSE generator ---------------------------------------------------------
def _sse_types(frames: list[str]) -> list[str]:
    return [
        json.loads(f[len("data: ") :])["type"] for f in frames if f.startswith("data: ")
    ]


async def _collect(agen) -> list[str]:  # noqa: ANN001
    return [frame async for frame in agen]


class _Reader:
    def __init__(self, status: str) -> None:
        self._status = status

    def get_job(self, job_id: str) -> dict:
        return {"status": self._status}


@pytest.fixture
def shared_redis(monkeypatch):
    """A fakeredis server shared by a sync writer and the async SSE reader."""
    server = fakeredis.FakeServer()
    sync = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    aio = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(
        "redis.asyncio.Redis.from_url",
        lambda *a, **k: aio,  # noqa: ARG005
    )
    return sync


def test_sse_replays_history_and_closes_on_terminal_event(shared_redis) -> None:
    reporter = RedisProgressReporter("j", shared_redis)
    reporter.emit(ProgressEvent.JOB_STARTED, {"video_name": "ad"})
    reporter.emit(ProgressEvent.JOB_COMPLETED, {"run_id": "R1"})

    frames = asyncio.run(
        _collect(_job_event_stream("redis://x", "j", _Reader("running")))
    )
    # Full replay for a late subscriber, then the stream ends on the terminal
    # event without blocking.
    assert _sse_types(frames) == ["job.started", "job.completed"]


def test_sse_closes_via_terminal_job_row_backstop(shared_redis) -> None:
    # A job canceled before it emitted a terminal event: only a start event is in
    # the stream, but the job row is terminal, so the generator must still close.
    reporter = RedisProgressReporter("j", shared_redis)
    reporter.emit(ProgressEvent.JOB_STARTED, {"video_name": "ad"})

    frames = asyncio.run(
        _collect(_job_event_stream("redis://x", "j", _Reader("canceled")))
    )
    assert _sse_types(frames) == ["job.started"]


# --- Progress summary -------------------------------------------------------
def test_summarize_progress_weights_phases_and_shots() -> None:
    events = [
        (
            "job.started",
            {"phases": ["scene_detection", "audio", "shot_processing", "finalize"]},
        ),
        ("phase.completed", {"phase": "scene_detection"}),  # 0.05
        ("phase.completed", {"phase": "audio"}),  # +0.15 = 0.20
        ("phase.started", {"phase": "shot_processing", "total_shots": 10}),
        ("shot.completed", {}),
        ("shot.completed", {}),  # 2/10 of the 0.70 shot weight = 0.14
    ]
    summary = summarize_progress(events)
    # 0.20 + 0.14 = 0.34 → 34%
    assert summary["percent"] == 34.0
    assert summary["phase"] == "shot_processing"
    assert summary["shots_done"] == 2
    assert summary["total_shots"] == 10


def test_summarize_progress_terminal_is_100() -> None:
    events = [
        ("job.started", {"phases": ["parse"]}),
        ("job.completed", {"run_id": "R1"}),
    ]
    assert summarize_progress(events)["percent"] == 100.0


def test_summarize_progress_empty_is_zero() -> None:
    assert summarize_progress([])["percent"] == 0.0


def test_sse_frame_shape() -> None:
    frame = _sse_frame({"type": "shot.started", "data": json.dumps({"shot_idx": 2})})
    assert frame == 'data: {"type": "shot.started", "data": {"shot_idx": 2}}\n\n'
