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


@pytest.fixture
def isolated_logging(monkeypatch, restore_levels):
    """Let ``configure_logging`` run its one-time setup, then undo it."""
    names = ("api", "clip_scribe")
    saved = {n: list(logging.getLogger(n).handlers) for n in names}
    monkeypatch.setattr(clip_scribe_logging, "_CONFIGURED", False)
    yield
    for n in names:
        lg = logging.getLogger(n)
        for h in lg.handlers:
            if h not in saved[n]:
                h.close()
        lg.handlers[:] = saved[n]


def test_api_debug_records_reach_the_console_tagged_as_api(
    capsys, monkeypatch, tmp_path, isolated_logging
):
    monkeypatch.setitem(
        clip_scribe_logging.GENERAL_LOGGING_CONFIG["handlers"]["file_main"],
        "filename",
        str(tmp_path / "core.log"),
    )
    monkeypatch.setattr(clip_scribe_logging, "LOG_DIR", str(tmp_path))

    clip_scribe_logging.configure_logging({"api": logging.DEBUG})
    logging.getLogger("api").debug("api-debug-line")
    logging.getLogger("clip_scribe").debug("core-hidden-line")
    logging.getLogger("clip_scribe").info("core-info-line")

    out = capsys.readouterr().out
    assert any("| api |" in ln and "api-debug-line" in ln for ln in out.splitlines())
    assert any("| core |" in ln and "core-info-line" in ln for ln in out.splitlines())
    assert "core-hidden-line" not in out
    # The log file belongs to the pipeline logger only.
    log_text = (tmp_path / "core.log").read_text(encoding="utf-8")
    assert "core-info-line" in log_text
    assert "api-debug-line" not in log_text


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
