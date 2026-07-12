"""Runtime configuration for the web API process.

Read from the environment so the same code runs as a native dev process,
docker-compose service, or slim API container. Paths resolve against
the backend root the same way ``build_clip_scribe.PROJECT_ROOT`` does, so the
API, the CLI, and the worker all agree on where ``input/`` and ``artifacts/``
live in both native and containerized runs (web-app-plan §2, §8, §9).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load the repo-root .env before any setting is read. The API (celery mode) and
# the Celery worker both import this module at startup but never import
# build_clip_scribe (which has its own load_dotenv), so without this their env —
# REDIS_URL, CLIPSCRIBE_JOB_BACKEND, DB URLs — would come only from the real
# process environment. find_dotenv walks up from the CWD, so running from
# backend/ still finds the repo-root .env. override=False: a var already set in
# the real environment (e.g. compose `environment:` or a test) wins over .env.
load_dotenv(find_dotenv(filename=".env"), override=False)

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
        # Tests set this False and inject a fake so no models load. Ignored when
        # job_backend == "celery": that path enqueues to a worker and the API
        # loads only the DB, never the models.
        self.load_models: bool = _bool_env("CLIPSCRIBE_API_LOAD_MODELS", True)

        # How POST /jobs runs a job (web-app-plan §10 step 8):
        #   "inline" — run in-process on a single-slot executor (the step-5 path).
        #   "celery" — enqueue to a Redis-backed Celery worker; the API loads no
        #              models and only reads/writes the DB + dispatches tasks.
        self.job_backend: str = (
            os.environ.get("CLIPSCRIBE_JOB_BACKEND", "inline").strip().lower()
        )
        if self.job_backend not in ("inline", "celery"):
            raise ValueError(
                f"CLIPSCRIBE_JOB_BACKEND must be 'inline' or 'celery', "
                f"got {self.job_backend!r}"
            )

        # Redis URL used as both the Celery broker/result backend and the
        # live-progress Redis Streams store. Same value for API and worker when
        # co-located; see web-app-plan §12 for the container-networking split.
        self.redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.clip_scribe_device: str = os.environ.get("CLIPSCRIBE_DEVICE", "cpu")

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
