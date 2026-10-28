"""Logger levels: dynamic per-logger levels, env parsing, and the job-log bridge."""

import logging

import pytest

from app import events, settings
from clip_scribe.utils import clip_scribe_logging


@pytest.fixture
def restore_levels():
    """Undo level changes: loggers are process-global."""
    names = ("api", "clip_scribe")
    saved = {n: logging.getLogger(n).level for n in names}
    yield
    for n, level in saved.items():
        logging.getLogger(n).setLevel(level)


def test_levels_apply_on_every_call_and_leave_other_loggers_alone(
    monkeypatch, restore_levels
):
    # Skip the one-time handler setup; only the level handling is under test.
    monkeypatch.setattr(clip_scribe_logging, "_CONFIGURED", True)
    scribe_level = logging.getLogger("clip_scribe").level

    clip_scribe_logging.configure_logging({"api": logging.DEBUG})
    assert logging.getLogger("api").level == logging.DEBUG
    assert logging.getLogger("clip_scribe").level == scribe_level

    # A later call changes it again; None changes nothing.
    clip_scribe_logging.configure_logging({"api": logging.ERROR})
    clip_scribe_logging.configure_logging()
    assert logging.getLogger("api").level == logging.ERROR


def test_handlers_do_not_filter_below_the_logger_level():
    cfg = clip_scribe_logging.GENERAL_LOGGING_CONFIG
    # Otherwise lowering a logger to DEBUG would still print nothing.
    assert {h["level"] for h in cfg["handlers"].values()} == {"DEBUG"}
    assert set(cfg["loggers"]) == {"clip_scribe", "api"}
    # API records go to the console only; the log file is the pipeline's.
    assert cfg["loggers"]["api"]["handlers"] == ["console"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 20), ("", 20), ("10", 10), ("debug", 10), (" Warning ", 30)],
)
def test_log_level_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("X_LEVEL", raising=False)
    else:
        monkeypatch.setenv("X_LEVEL", raw)
    assert settings._log_level_env("X_LEVEL", logging.INFO) == expected


def test_log_level_env_rejects_unknown_names(monkeypatch):
    monkeypatch.setenv("X_LEVEL", "loud")
    with pytest.raises(ValueError, match="X_LEVEL"):
        settings._log_level_env("X_LEVEL", logging.INFO)


def test_job_log_bridge_covers_both_loggers_once(monkeypatch):
    class _Redis:
        def ping(self):
            return True

    monkeypatch.setattr(events.redis.Redis, "from_url", lambda url: _Redis())
    loggers = [logging.getLogger(n) for n in ("clip_scribe", "api")]
    before = {lg: list(lg.handlers) for lg in loggers}
    try:
        events.install_job_log_bridge("redis://x")
        events.install_job_log_bridge("redis://x")  # idempotent
        for lg in loggers:
            bridged = [
                h for h in lg.handlers if isinstance(h, events.JobLogStreamHandler)
            ]
            assert len(bridged) == 1
    finally:
        for lg in loggers:
            lg.handlers[:] = before[lg]


def test_every_handler_tags_records_with_their_source():
    cfg = clip_scribe_logging.GENERAL_LOGGING_CONFIG
    # The formatters print %(source)s, so a handler without the filter would
    # raise on every record.
    assert all("source" in h["filters"] for h in cfg["handlers"].values())

    def tag(name: str) -> str:
        record = logging.LogRecord(name, logging.INFO, "f.py", 1, "m", None, None)
        assert clip_scribe_logging.SourceFilter().filter(record) is True
        return record.source  # type: ignore[attr-defined]

    assert tag("api") == "api"
    assert tag("clip_scribe") == "core"
    assert tag("something.else") == "something.else"
