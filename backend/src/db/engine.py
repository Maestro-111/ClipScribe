"""Engine factory, upsert helper, and base DB class."""

import functools
import logging
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from sqlalchemy import Engine, Table, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

logger = logging.getLogger("clip_scribe")

# backend/ — matches PROJECT_ROOT in build_clip_scribe.py (parents[2] there too).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Defaults for the connection pool when the env vars are unset. Mirror
# SQLAlchemy's own defaults so behaviour is unchanged out of the box.
_DEFAULT_POOL_SIZE = 5
_DEFAULT_MAX_OVERFLOW = 10


def resolve_database_url() -> str:
    """Resolve the active database URL from the environment.

    Single source of truth shared by the builder's ``_assemble_db`` and the
    Alembic ``env.py`` so the two never drift. The backend is selected solely by
    the ``CLIPSCRIBE_DB_BACKEND`` env var (default ``sqlite`` for local CLI dev);
    the Docker/compose stack and prod set it to ``postgresql``. All DB tuning is
    env-driven — the backend selector here and the pool knobs in
    :func:`resolve_pool_settings` — so ``clip_scribe.yaml`` carries no database
    config at all and there is one configuration mechanism, not two. For sqlite
    use ``SQLITE_URL`` (default ``sqlite:///data/clip_scribe.db``) resolved
    against the project root; for postgresql require ``POSTGRESQL_URL``.
    """
    backend = os.environ.get("CLIPSCRIBE_DB_BACKEND", "sqlite")

    if backend == "sqlite":
        db_url = os.environ.get("SQLITE_URL", "sqlite:///data/clip_scribe.db")
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
            relative_path = db_url[len("sqlite:///") :]
            db_url = f"sqlite:///{_PROJECT_ROOT / relative_path}"
        return db_url

    return os.environ["POSTGRESQL_URL"]


def _int_env(name: str, default: int) -> int:
    """Read a non-negative int env var, falling back on unset/invalid values."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%d is negative; using default %d", name, value, default)
        return default
    return value


def resolve_pool_settings() -> tuple[int, int]:
    """Resolve ``(pool_size, max_overflow)`` from the environment.

    Env-driven so the API (both modes), the Celery worker, and the CLI builder
    all agree without a config file. Only the PostgreSQL engine applies these;
    the SQLite dev backend ignores them (see :func:`create_db_engine`).
    """
    return (
        _int_env("CLIPSCRIBE_DB_POOL_SIZE", _DEFAULT_POOL_SIZE),
        _int_env("CLIPSCRIBE_DB_MAX_OVERFLOW", _DEFAULT_MAX_OVERFLOW),
    )


_T = TypeVar("_T")

# How many times to attempt a transient-failing DB operation before giving up.
_RETRY_MAX_ATTEMPTS = 3
# Exponential backoff base and ceiling (seconds). Kept small: these run in the
# FastAPI threadpool, so a long sleep would pin a worker thread.
_RETRY_BASE_DELAY = 0.05
_RETRY_MAX_DELAY = 1.0


def retry_transient(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Retry a whole-transaction DB operation on transient errors.

    Wraps a method that opens exactly one ``engine.begin()`` transaction. On a
    transient failure — a Postgres deadlock/serialization abort or a dropped
    connection (both surface as :class:`OperationalError`), or SQLite's
    "database is locked" — the transaction has already rolled back, so simply
    re-running the whole method is safe. Non-transient errors (integrity,
    programming) are not caught and propagate immediately.

    Only apply to methods whose entire body is one atomic transaction; do NOT
    apply to a helper that runs inside a caller's transaction, since retrying it
    alone cannot recover the aborted outer transaction.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> _T:
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except OperationalError as exc:
                attempt += 1
                if attempt >= _RETRY_MAX_ATTEMPTS:
                    raise
                delay = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                logger.warning(
                    "Transient DB error in %s (attempt %d/%d): %s; retrying in %.2fs",
                    fn.__name__,
                    attempt,
                    _RETRY_MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)

    return wrapper


def ensure_sqlite_parent_directory(database_url: str) -> None:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite"):
        return

    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """Create a SQLAlchemy engine with appropriate settings for the dialect.

    For SQLite: enables WAL journal mode and foreign keys via connect events.
    For PostgreSQL: enables connection pool pre-ping and configurable pool size.
    """
    is_sqlite = database_url.startswith("sqlite")

    database_backend = "sqlite" if is_sqlite else "postgresql"
    logger.info(f"Creating database engine for {database_backend}")

    if is_sqlite:
        ensure_sqlite_parent_directory(database_url)
        engine = create_engine(database_url)
    else:
        engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )

    # SQLite-specific PRAGMAs
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    # Schema is owned by Alembic migrations (run `alembic upgrade head`), not
    # auto-created here. This keeps a single source of truth and avoids the
    # CREATE-collision that would occur if both create_all and a migration
    # tried to build the same tables.
    return engine


def _upsert_ignore(
    engine: Engine, table: Table, rows: list[dict], conflict_columns: list[str]
):
    """Build an INSERT ... ON CONFLICT DO NOTHING statement for the engine dialect."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return (
            sqlite_insert(table)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=conflict_columns,
            )
        )

    from sqlalchemy.dialects.postgresql import insert as postgresql_insert

    return (
        postgresql_insert(table)
        .values(rows)
        .on_conflict_do_nothing(
            index_elements=conflict_columns,
        )
    )


class ClipScribeBaseDB:
    def __init__(self, engine: Engine):
        self._engine = engine

    def close(self) -> None:
        self._engine.dispose()
        logger.info("ClipScribeDB engine disposed.")
