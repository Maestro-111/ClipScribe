"""Focused coverage for whole-transaction database retry policy."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

import clip_scribe.db.engine as db_engine


class _PostgresDriverError(Exception):
    """Minimal stand-in for asyncpg/psycopg errors carrying a SQLSTATE."""

    def __init__(self, sqlstate: str, attribute: str = "sqlstate") -> None:
        super().__init__(f"PostgreSQL error {sqlstate}")
        setattr(self, attribute, sqlstate)


def _postgres_error(sqlstate: str, attribute: str = "sqlstate") -> DBAPIError:
    return DBAPIError("INSERT", {}, _PostgresDriverError(sqlstate, attribute))


def _sqlite_operational_error(error_code: int, message: str) -> OperationalError:
    original = sqlite3.OperationalError(message)
    original.sqlite_errorcode = error_code
    return OperationalError("INSERT", {}, original)


@pytest.mark.parametrize(
    ("sqlstate", "attribute"),
    [("40001", "sqlstate"), ("40P01", "sqlstate"), ("40P01", "pgcode")],
)
def test_sync_retry_replays_postgres_transaction_abort(
    monkeypatch: pytest.MonkeyPatch, sqlstate: str, attribute: str
) -> None:
    sleep = Mock()
    monkeypatch.setattr(db_engine.time, "sleep", sleep)
    attempts = 0

    @db_engine.retry_transient
    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _postgres_error(sqlstate, attribute)
        return "committed"

    assert operation() == "committed"
    assert attempts == 2
    sleep.assert_called_once_with(0.05)


@pytest.mark.parametrize(
    "error_code",
    [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY | 0x100],
)
def test_sync_retry_replays_sqlite_lock_contention(
    monkeypatch: pytest.MonkeyPatch, error_code: int
) -> None:
    sleep = Mock()
    monkeypatch.setattr(db_engine.time, "sleep", sleep)
    attempts = 0

    @db_engine.retry_transient
    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _sqlite_operational_error(error_code, "database is locked")
        return "committed"

    assert operation() == "committed"
    assert attempts == 2
    sleep.assert_called_once_with(0.05)


@pytest.mark.parametrize(
    "error",
    [
        _postgres_error("23505"),
        IntegrityError("INSERT", {}, _PostgresDriverError("23505")),
        _sqlite_operational_error(sqlite3.SQLITE_IOERR, "disk I/O error"),
        OperationalError(
            "INSERT",
            {},
            OSError("connection lost during commit"),
            connection_invalidated=True,
        ),
    ],
)
def test_sync_retry_propagates_nonretryable_errors_immediately(
    monkeypatch: pytest.MonkeyPatch, error: DBAPIError
) -> None:
    sleep = Mock()
    monkeypatch.setattr(db_engine.time, "sleep", sleep)
    attempts = 0

    @db_engine.retry_transient
    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(DBAPIError) as raised:
        operation()

    assert raised.value is error
    assert attempts == 1
    sleep.assert_not_called()


def test_sync_retry_stops_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = Mock()
    monkeypatch.setattr(db_engine.time, "sleep", sleep)
    error = _postgres_error("40001")
    attempts = 0

    @db_engine.retry_transient
    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(DBAPIError) as raised:
        operation()

    assert raised.value is error
    assert attempts == 3
    assert [call.args for call in sleep.call_args_list] == [(0.05,), (0.1,)]


def test_async_retry_replays_generic_dbapi_serialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(db_engine.asyncio, "sleep", sleep)
    attempts = 0

    @db_engine.retry_transient_async
    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # asyncpg's adapter maps this SQLSTATE to SQLAlchemy DBAPIError,
            # not OperationalError.
            raise _postgres_error("40001")
        return "committed"

    assert asyncio.run(operation()) == "committed"
    assert attempts == 2
    sleep.assert_awaited_once_with(0.05)


def test_async_retry_propagates_connection_loss_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(db_engine.asyncio, "sleep", sleep)
    error = OperationalError(
        "COMMIT",
        {},
        OSError("connection lost during commit"),
        connection_invalidated=True,
    )
    attempts = 0

    @db_engine.retry_transient_async
    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    async def exercise() -> None:
        with pytest.raises(DBAPIError) as raised:
            await operation()
        assert raised.value is error

    asyncio.run(exercise())
    assert attempts == 1
    sleep.assert_not_awaited()
