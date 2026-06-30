from logging.config import fileConfig

from sqlalchemy import create_engine
from sqlalchemy import pool

from alembic import context

from src.db.engine import ensure_sqlite_parent_directory, resolve_database_url
from src.db.schema import metadata_obj

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Schema metadata for 'autogenerate' support. Single source of truth shared
# with the runtime engine via src.db.schema.
target_metadata = metadata_obj

# Resolve the live database URL from clip_scribe.yaml + env vars rather than
# the static alembic.ini placeholder, so migrations always target the same DB
# the application uses (sqlite in dev, postgresql in deployment).
DATABASE_URL = resolve_database_url()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL, no DBAPI needed)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=DATABASE_URL.startswith("sqlite"),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    ensure_sqlite_parent_directory(DATABASE_URL)
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # batch mode lets SQLite emulate ALTER TABLE for future migrations.
            render_as_batch=DATABASE_URL.startswith("sqlite"),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
