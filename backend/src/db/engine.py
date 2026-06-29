"""Engine factory, upsert helper, and base DB class."""

import logging
from pathlib import Path

from sqlalchemy import Engine, Table, create_engine, event

from .schema import metadata_obj


def create_db_engine(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    logger: logging.Logger | None = None,
) -> Engine:
    """Create a SQLAlchemy engine with appropriate settings for the dialect.

    For SQLite: enables WAL journal mode and foreign keys via connect events.
    For PostgreSQL: enables connection pool pre-ping and configurable pool size.
    """
    is_sqlite = database_url.startswith("sqlite")

    if logger:
        logger.info(
            f"Creating database engine for {"sqlite" if is_sqlite else "postgresql"}"
        )

    if is_sqlite:
        # Ensure parent directory exists for SQLite file
        # sqlite:///relative/path  or  sqlite:////absolute/path
        url_path = database_url.split("///", 1)[-1]
        if url_path:
            db_file = Path(url_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

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

    metadata_obj.create_all(engine)
    return engine


def _upsert_ignore(
    engine: Engine, table: Table, rows: list[dict], conflict_columns: list[str]
):
    """Build an INSERT ... ON CONFLICT DO NOTHING statement for the engine dialect."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        from sqlalchemy.dialects.postgresql import insert

    stmt = (
        insert(table)
        .values(rows)
        .on_conflict_do_nothing(
            index_elements=conflict_columns,
        )
    )
    return stmt


class ClipScribeBaseDB:
    def __init__(self, engine: Engine, logger: logging.Logger):
        self._engine = engine
        self.logger = logger

    def close(self) -> None:
        self._engine.dispose()
        self.logger.info("ClipScribeDB engine disposed.")
