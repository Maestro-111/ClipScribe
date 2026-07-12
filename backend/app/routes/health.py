"""Liveness and readiness probes (web-app-plan §6).

``/healthz`` is pure liveness — it never touches models, the DB, or Redis.
``/readyz`` reports whether the database and Redis are reachable and, in inline
mode, whether the heavy builder finished loading. Models are *not* required in
celery mode, where the API deliberately loads none (web-app-plan §8, §10.8).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.errors import ProblemException

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readyz(request: Request) -> dict[str, object]:
    state = request.app.state
    settings = state.settings

    builder = getattr(state, "builder", None)
    models_loaded = builder is not None

    # DB reachability via app.state (populated in both inline and celery modes).
    database_ok = False
    reader = getattr(state, "reader_db", None)
    if reader is not None:
        try:
            # get_latest_run() is a public read that hits the DB and returns
            # None on an empty DB without raising — a cheap reachability ping.
            reader.get_latest_run()
            database_ok = True
        except Exception:
            database_ok = False

    # Redis reachability — the live-progress transport (step 9) and, in celery
    # mode, the Celery broker.
    redis_ok = False
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url)
        redis_ok = bool(client.ping())
    except Exception:
        redis_ok = False

    # Inline mode runs the engine in-process, so it must have models loaded;
    # celery mode enqueues to a worker and is ready without them.
    models_ok = models_loaded or settings.job_backend == "celery"
    ready = database_ok and redis_ok and models_ok
    checks: dict[str, object] = {
        "models_loaded": models_loaded,
        "database": database_ok,
        "redis": redis_ok,
    }
    if not ready:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="API is not ready to serve requests.",
            extra=checks,
        )

    return {"status": "ready", **checks}
