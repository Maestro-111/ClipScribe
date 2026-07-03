"""Liveness and readiness probes (web-app-plan §6).

``/healthz`` is pure liveness — it never touches models or the DB. ``/readyz``
reports whether the heavy builder finished loading and whether the database is
reachable. The Redis reachability check is added with the pub/sub bridge in
step 9.
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
    builder = getattr(request.app.state, "builder", None)
    models_loaded = builder is not None

    database_ok = False
    if builder is not None:
        try:
            # get_latest_run() is a public read that hits the DB and returns
            # None on an empty DB without raising — a cheap reachability ping.
            builder.reader_db.get_latest_run()
            database_ok = True
        except Exception:
            database_ok = False

    ready = models_loaded and database_ok
    if not ready:
        raise ProblemException(
            status=503,
            title="Service Unavailable",
            detail="API is not ready to serve requests.",
            extra={"models_loaded": models_loaded, "database": database_ok},
        )

    return {"status": "ready", "models_loaded": models_loaded, "database": database_ok}
