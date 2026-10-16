"""Shared PostgreSQL plumbing for the durable stores (archive catalog, inspector traces).

Stores keep writing portable SQL with ``?`` placeholders; :class:`PgConnection`
translates them to psycopg's ``%s`` style so one statement serves both SQLite and
PostgreSQL. Each store owns a dedicated schema (``search_path``) so generic table
names cannot collide, and records its schema version in that schema's
``schema_version`` table so later migrations can be applied incrementally.

The connection URL may contain credentials: it is never logged or returned. Use
:func:`redact_database_url` for anything user-visible.
"""

from __future__ import annotations

import atexit
import hashlib
import re
import time
import weakref
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def is_postgres_url(url: str | None) -> bool:
    """True for postgres:// or postgresql:// (optionally +driver) URLs."""
    if not url:
        return False
    scheme = url.split(":", 1)[0].lower()
    return scheme in {"postgres", "postgresql"} or scheme.startswith("postgresql+")


def redact_database_url(url: str) -> str:
    """Return ``scheme://user@host:port/db`` without password or query parameters."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = host + (f":{parts.port}" if parts.port else "")
        if parts.username:
            netloc = f"{parts.username}@{netloc}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        return "postgresql://[REDACTED]"


def _close_pool_at_exit(reference: "weakref.ref") -> None:
    pool = reference()
    if pool is not None and not pool.closed:
        pool.close(timeout=1.0)


def advisory_key(name: str) -> int:
    """Stable signed 64-bit key for pg_advisory_* locks."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big", signed=True)


class Row(dict):
    """Row addressable by column name or position, like ``sqlite3.Row``."""

    __slots__ = ("_values",)

    def __init__(self, names: Sequence[str], values: Sequence[Any]) -> None:
        super().__init__(zip(names, values))
        self._values = tuple(values)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)

    def __iter__(self):  # tuple-style unpacking: ``count, observed = row``
        return iter(self._values)


def _row_factory(cursor: Any):
    names = [column.name for column in cursor.description] if cursor.description else []

    def make(values: Sequence[Any]) -> Row:
        return Row(names, values)

    return make


class PgConnection:
    """Thin wrapper giving a psycopg connection the sqlite3 ``execute`` surface."""

    dialect = "postgres"

    def __init__(self, connection: Any) -> None:
        self.raw = connection

    @staticmethod
    def translate(sql: str) -> str:
        return sql.replace("%", "%%").replace("?", "%s")

    def execute(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Any:
        cursor = self.raw.cursor(row_factory=_row_factory)
        cursor.execute(self.translate(sql), params or None)
        return cursor

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def lock(self, name: str) -> None:
        """Transaction-scoped exclusive advisory lock (released at commit/rollback)."""
        self.raw.execute("SELECT pg_advisory_xact_lock(%s)", (advisory_key(name),))


class Database:
    """A PostgreSQL connection pool bound to one schema of one database."""

    backend = "postgres"

    def __init__(
        self,
        url: str,
        *,
        schema: str,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
    ) -> None:
        if not is_postgres_url(url):
            raise ValueError("DATABASE_URL must be a postgresql:// URL")
        if not _SCHEMA_NAME.match(schema):
            raise ValueError("Invalid schema name")
        from psycopg_pool import ConnectionPool

        self.url = url
        self.schema = schema
        self.location = redact_database_url(url) + f"#{schema}"
        conninfo = re.sub(r"^postgres(ql)?\+[a-z0-9]+://", "postgresql://", url, flags=re.I)
        self._conninfo = conninfo
        self._connect_kwargs = {"options": f"-c search_path={schema}"}
        self.pool = ConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max(min_size, max_size),
            timeout=timeout,
            kwargs=self._connect_kwargs,
            open=False,
            name=f"dpl-{schema}",
        )
        try:
            self.pool.open(wait=True, timeout=timeout)
        except Exception:
            self.pool.close()
            # The driver message can echo the conninfo; never surface it.
            raise RuntimeError(f"Could not connect to PostgreSQL at {redact_database_url(url)}") from None
        # Safety net: an unclosed pool otherwise stalls interpreter exit on its worker threads.
        atexit.register(_close_pool_at_exit, weakref.ref(self.pool))

    @contextmanager
    def connection(self) -> Iterator[PgConnection]:
        """One transaction: committed on success, rolled back on error."""
        with self.pool.connection() as raw:
            yield PgConnection(raw)

    def dedicated_connection(self) -> Any:
        """A non-pooled autocommit connection (for session-scoped locks)."""
        import psycopg

        return psycopg.connect(self._conninfo, autocommit=True, **self._connect_kwargs)

    def ensure_schema(self, migrations: Mapping[int, Sequence[str]]) -> int:
        """Idempotently create the schema and apply versions above the recorded one."""
        with self.connection() as db:
            db.lock(f"schema:{self.schema}")
            db.raw.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            db.execute(
                """CREATE TABLE IF NOT EXISTS schema_version (
                       version INTEGER PRIMARY KEY, applied_at DOUBLE PRECISION NOT NULL)"""
            )
            row = db.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
            current = int(row[0])
            for version in sorted(migrations):
                if version <= current:
                    continue
                for statement in migrations[version]:
                    db.raw.execute(statement)
                db.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?)", (version, time.time()))
                current = version
            return current

    def close(self) -> None:
        self.pool.close()


def open_database(url: str, *, schema: str, min_size: int = 1, max_size: int = 10) -> Database:
    return Database(url, schema=schema, min_size=min_size, max_size=max_size)
