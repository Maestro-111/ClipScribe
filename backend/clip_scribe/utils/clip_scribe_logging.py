import logging
import logging.config
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LOG_FILE_NAME = f"{datetime.now():%Y_%m_%d-%I_%M_%S_%p}_" + "_clip_scribe"
LOG_DIR = os.path.join(os.path.join(BASE_DIR, "logs"))
_CONFIGURED = False

# Short tag printed for each logger, so interleaved API and pipeline lines can be
# told apart. Unlisted loggers show their own name.
SOURCE_LABELS = {"api": "api", "clip_scribe": "core"}


class SourceFilter(logging.Filter):
    """Tags each record with ``record.source`` for the formatters; drops nothing."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.source = SOURCE_LABELS.get(record.name, record.name)
        return True


GENERAL_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(asctime)s | %(levelname)s | %(source)s | %(message)s",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        },
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(source)s | %(message)s",
        },
    },
    # Every handler needs it: the formatters above print ``%(source)s``.
    "filters": {"source": {"()": SourceFilter}},
    "handlers": {
        # Handlers pass everything; each logger's level is the only threshold,
        # so lowering a logger's level at runtime actually shows its records.
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "color",
            "filters": ["source"],
            "stream": "ext://sys.stdout",
        },
        "file_main": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filters": ["source"],
            "filename": f"{LOG_DIR}/{LOG_FILE_NAME}.log",
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "clip_scribe": {
            "handlers": ["console", "file_main"],
            "level": "INFO",
            "propagate": False,
        },
        # The web layer (``app/``); its level is set per process, see
        # ``configure_logging``. Console only: the log file is the pipeline's.
        "api": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def configure_logging(levels: Mapping[str, int] | None = None) -> None:
    """Configure ClipScribe logging, then apply per-logger ``levels``.

    Handlers and formatters are set up once per process; ``levels`` (logger name
    -> level, e.g. ``{"api": logging.DEBUG}``) is applied on every call, so a
    level can be changed later by calling again. Loggers not named keep their
    current level (INFO by default), and ``None`` changes nothing.
    """
    global _CONFIGURED

    if not _CONFIGURED:
        os.makedirs(LOG_DIR, exist_ok=True)
        logging.config.dictConfig(GENERAL_LOGGING_CONFIG)
        _CONFIGURED = True

    for name, level in (levels or {}).items():
        logging.getLogger(name).setLevel(level)
