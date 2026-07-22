"""Runtime configuration for the web API process.

Read from the environment so the same code runs as a native dev process,
docker-compose service, or slim API container. Paths resolve against
the backend root the same way ``build_clip_scribe.PROJECT_ROOT`` does, so the
API, the CLI, and the worker all agree on where local source videos and
``artifacts/`` live in both native and containerized runs (web-app-plan §2,
§8, §9).
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
# Repo root: where .env and the dev service_account.json live.
REPO_ROOT = PROJECT_ROOT.parent


def _absolutize_gcs_credentials() -> None:
    """Anchor a relative GOOGLE_APPLICATION_CREDENTIALS at the repo root.

    The google SDK resolves this env var against the process CWD, but the API
    and worker run from ``backend/`` while the file (and the repo-root ``.env``
    that points at it) live at the repo root. A bare ``service_account.json``
    would therefore be looked up at ``backend/service_account.json`` and miss.
    Rewriting a relative value to an absolute path makes it CWD-independent;
    absolute paths (e.g. a container-mounted secret) are left untouched.
    """
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if raw and not Path(raw).is_absolute():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((REPO_ROOT / raw).resolve())


_absolutize_gcs_credentials()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:

    """Process-wide API settings, resolved once from the environment."""

    # Bound the stream length so a run's events don't grow without limit. A run emits
    # a handful of events per shot plus one entry per INFO log record, so 2000 covers
    # even long videos; approximate trimming (``~``) is cheaper for Redis.
    STREAM_MAXLEN = 2000
    # TTL applied once a terminal event is written, so finished jobs' streams expire
    # instead of lingering forever — long enough for a late page-load to still replay.
    STREAM_TTL_SECONDS = 24 * 3600

    # Terminal events end the SSE stream. The engine emits completed/failed, and
    # canceled once cooperative cancel stops a running pipeline; a job canceled
    # before it reaches the engine may emit none of these, so the SSE endpoint also
    # watches the job row (see routes/jobs.py) as a backstop.
    TERMINAL_EVENTS = frozenset({"job.completed", "job.failed", "job.canceled"})

    # TTL on the per-job cancel flag. Longer than any realistic single run so a
    # long-running job still sees a late cancel, after which the key self-expires
    # (a retried job gets a fresh job_id / key, so no stale flag can leak across).
    CANCEL_FLAG_TTL_SECONDS = 24 * 3600

    def __init__(self) -> None:
        # Local source-video storage root. The API stores opaque keys here when
        # CLIPSCRIBE_STORAGE_BACKEND=local; the gcs backend uses it only for
        # local staging/materialization scratch.
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

        # The single storage selector, governing BOTH source videos and run
        # artifacts (see src/utils/clip_scribe_video_storage.py and
        # clip_scribe_artifacts.py):
        #   "local" — files under input_dir / artifacts_dir (default; dev).
        #   "gcs"   — a cloud bucket; videos and artifacts share one bucket,
        #             separated by the videos/ and artifacts/ prefixes.
        self.storage_backend: str = (
            os.environ.get("CLIPSCRIBE_STORAGE_BACKEND", "local").strip().lower()
        )
        if self.storage_backend not in ("local", "gcs"):
            raise ValueError(
                f"CLIPSCRIBE_STORAGE_BACKEND must be 'local' or 'gcs', "
                f"got {self.storage_backend!r}"
            )

        # The GCS bucket for videos + artifacts. Required when the backend is
        # gcs; unused for local. Credentials come from the environment
        # (GOOGLE_APPLICATION_CREDENTIALS in dev; attached identity in prod).
        self.gcs_bucket: str | None = os.environ.get("CLIPSCRIBE_GCS_BUCKET")
        if self.storage_backend == "gcs" and not self.gcs_bucket:
            raise ValueError(
                "CLIPSCRIBE_GCS_BUCKET is required when "
                "CLIPSCRIBE_STORAGE_BACKEND=gcs"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
