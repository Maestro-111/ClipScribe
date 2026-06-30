"""Engine factory, upsert helper, and base DB class."""

import logging
import os
from pathlib import Path

import yaml  # type: ignore
from sqlalchemy import Engine, Table, create_engine, event
from sqlalchemy.engine import make_url

logger = logging.getLogger("clip_scribe")

# backend/ — matches PROJECT_ROOT in build_clip_scribe.py (parents[2] there too).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "clip_scribe" / "configs" / "clip_scribe.yaml"
)


def resolve_database_url() -> str:
    """Resolve the active database URL from config + environment.

    Single source of truth shared by the builder's ``_assemble_db`` and the
    Alembic ``env.py`` so the two never drift. Mirrors the original inline
    logic: read ``database.backend`` from ``clip_scribe.yaml``; for sqlite use
    ``SQLITE_URL`` (default ``sqlite:///data/clip_scribe.db``) resolved against
    the project root; for postgresql require ``POSTGRESQL_URL``.
    """
    backend = "sqlite"
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        backend = cfg.get("database", {}).get("backend", "sqlite")
    except FileNotFoundError:
        pass

    if backend == "sqlite":
        db_url = os.environ.get("SQLITE_URL", "sqlite:///data/clip_scribe.db")
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
            relative_path = db_url[len("sqlite:///") :]
            db_url = f"sqlite:///{_PROJECT_ROOT / relative_path}"
        return db_url

    return os.environ["POSTGRESQL_URL"]


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
