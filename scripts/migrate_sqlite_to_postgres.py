"""One-shot copy of the SQLite archive catalog and inspector traces into PostgreSQL.

Idempotent: every row is inserted with ``ON CONFLICT DO NOTHING``, so re-running it (or
running it after the app already wrote to Postgres) never overwrites newer rows. Stop the
API first so the SQLite files are not changing while they are copied.

Usage::

    DATABASE_URL=postgresql://user:pass@host:5432/db uv run python scripts/migrate_sqlite_to_postgres.py \\
        [--catalog data/archive/catalog.sqlite3] [--traces data/workspaces/.inspect/traces.sqlite3]

Paths default to ARTIFACT_ARCHIVE_CATALOG_PATH and <WORKSPACE_ROOT_BASE>/.inspect/traces.sqlite3.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings  # noqa: E402
from core.storage import archive as archive_module  # noqa: E402
from core.storage.database import Database, redact_database_url  # noqa: E402
from fe_inspect import store as trace_module  # noqa: E402

# (table, columns, conflict target); order respects the archive foreign key.
CATALOG_TABLES = [
    ("archive_sessions", ["session_id", "owner_id", "state", "current_revision", "updated_at", "error", "metadata_json"]),
    ("archive_revisions", ["session_id", "revision", "object_key", "sha256", "size_bytes", "version_id", "status", "archived_at", "error"]),
]
TRACE_TABLES = [
    ("tombstones", ["session_id", "deleted_at"]),
    ("revisions", ["session_id", "revision"]),
    ("archive_metadata", ["session_id", "key", "body"]),
    ("spans", ["span_id", "session_id", "parent_span_id", "kind", "status", "process_id", "sequence", "revision", "body"]),
    ("events", ["session_id", "event_id", "sequence", "body"]),
    ("states", ["session_id", "body"]),
    ("instances", ["instance_id", "session_id", "process_id", "ended_at", "body"]),
]


def copy_tables(source: Path, database: Database, tables: list[tuple[str, list[str]]], order_by: dict[str, str]) -> None:
    if not source.is_file():
        print(f"skip: {source} does not exist")
        return
    reader = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        with database.connection() as target:
            for table, columns in tables:
                available = {row[1] for row in reader.execute(f"PRAGMA table_info({table})")}
                if not available:
                    print(f"skip: {table} not present in {source.name}")
                    continue
                selected = [column for column in columns if column in available]
                rows = reader.execute(
                    f"SELECT {', '.join(selected)} FROM {table} ORDER BY {order_by.get(table, 'rowid')}"
                ).fetchall()
                placeholders = ", ".join("?" for _ in selected)
                inserted = 0
                for row in rows:
                    cursor = target.execute(
                        f"INSERT INTO {table} ({', '.join(selected)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                        tuple(row),
                    )
                    inserted += cursor.rowcount
                print(f"{table}: {inserted} inserted, {len(rows) - inserted} already present")
    finally:
        reader.close()


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", type=Path, default=Path(settings.ARTIFACT_ARCHIVE_CATALOG_PATH))
    parser.add_argument("--traces", type=Path, default=Path(settings.WORKSPACE_ROOT_BASE) / ".inspect" / "traces.sqlite3")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL") or settings.DATABASE_URL)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL (or --database-url) is required")
    print(f"target: {redact_database_url(args.database_url)}")

    catalog_db = Database(args.database_url, schema="artifact_archive", max_size=2)
    trace_db = Database(args.database_url, schema="inspect_trace", max_size=2)
    try:
        catalog_db.ensure_schema(archive_module.POSTGRES_MIGRATIONS)
        trace_db.ensure_schema(trace_module.POSTGRES_MIGRATIONS)
        copy_tables(args.catalog, catalog_db, CATALOG_TABLES, {"archive_revisions": "session_id, revision"})
        copy_tables(args.traces, trace_db, TRACE_TABLES, {})
    finally:
        catalog_db.close()
        trace_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
