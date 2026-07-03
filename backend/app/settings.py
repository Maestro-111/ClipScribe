"""Runtime configuration for the web API process.

Read from the environment so the same code runs as a native dev process, a
docker-compose service, or (later) a slim API container. Paths resolve against
the backend root the same way ``build_clip_scribe.PROJECT_ROOT`` does, so the
API, the CLI, and the worker all agree on where ``input/`` and ``artifacts/``
live even when the container split lands (web-app-plan §2, §8, §9).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# backend/ — app/ is a top-level package sibling of src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """Process-wide API settings, resolved once from the environment."""

    def __init__(self) -> None:
        # Directory the uploader writes to and jobs reference video paths under.
        # A single base so the container split is just a bind-mount line.
        self.input_dir: Path = (
            PROJECT_ROOT / os.environ.get("CLIPSCRIBE_INPUT_DIR", "input")
        ).resolve()

        # Build the heavy ClipScribeBuilder (loads all models, ~30-60s) at
        # startup. On by default because the sync path runs the engine in-process.
        # Tests set this False and inject a fake so no models load.
        self.load_models: bool = _bool_env("CLIPSCRIBE_API_LOAD_MODELS", True)

        # Vite dev server origin(s) for CORS; comma-separated.
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.environ.get(
                "CLIPSCRIBE_CORS_ORIGINS", "http://localhost:5173"
            ).split(",")
            if o.strip()
        ]

        # Allowed video upload extensions (lowercase, with dot).
        self.allowed_video_suffixes: frozenset[str] = frozenset(
            {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
