"""Transactional snapshots, tenant-owned sessions, and PII-minimized traces."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class RAGStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots(version TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS active(singleton INTEGER PRIMARY KEY CHECK(singleton=1), version TEXT NOT NULL REFERENCES snapshots(version));
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS traces(id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL, expires REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS session_tenant ON sessions(tenant);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def publish(self, version: str, payload: dict):
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES(?,?,?)",
                (version, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            db.execute("INSERT OR REPLACE INTO active VALUES(1,?)", (version,))

    def activate(self, version: str):
        with self.connection() as db:
            if not db.execute(
                "SELECT 1 FROM snapshots WHERE version=?", (version,)
            ).fetchone():
                raise KeyError("Unknown index version")
            db.execute("INSERT OR REPLACE INTO active VALUES(1,?)", (version,))

    def load(self, version: str | None = None) -> tuple[str, dict]:
        with self.connection() as db:
            row = (
                db.execute(
                    "SELECT version,payload FROM snapshots WHERE version=?", (version,)
                ).fetchone()
                if version
                else db.execute(
                    "SELECT s.version,s.payload FROM snapshots s JOIN active a ON s.version=a.version"
                ).fetchone()
            )
        if not row:
            raise KeyError("No active index. Run python -m rag index --provider lsa")
        return row[0], json.loads(row[1])

    def save_session(
        self, session_id: str, tenant: str, payload: dict, ttl: int = 86400
    ):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT tenant FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if row and row[0] != tenant:
                raise PermissionError("Session does not belong to tenant")
            db.execute(
                "INSERT OR REPLACE INTO sessions VALUES(?,?,?,?)",
                (
                    session_id,
                    tenant,
                    json.dumps(payload, ensure_ascii=False),
                    time.time() + ttl,
                ),
            )

    def session(self, session_id: str, tenant: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM sessions WHERE id=? AND tenant=? AND expires>?",
                (session_id, tenant, time.time()),
            ).fetchone()
        if not row:
            raise KeyError("Session unavailable")
        return json.loads(row[0])

    def save_trace(self, request_id: str, tenant: str, payload: dict):
        with self.connection() as db:
            db.execute(
                "INSERT INTO traces VALUES(?,?,?,?)",
                (
                    request_id,
                    tenant,
                    json.dumps(payload, ensure_ascii=False),
                    time.time() + 86400,
                ),
            )

    def trace(self, request_id: str, tenant: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM traces WHERE id=? AND tenant=? AND expires>?",
                (request_id, tenant, time.time()),
            ).fetchone()
        if not row:
            raise KeyError("Trace unavailable")
        return json.loads(row[0])

    def purge(self):
        with self.connection() as db:
            for table in ("sessions", "traces"):
                db.execute(f"DELETE FROM {table} WHERE expires<=?", (time.time(),))

    def delete_session(self, session_id: str, tenant: str):
        with self.connection() as db:
            db.execute(
                "DELETE FROM sessions WHERE id=? AND tenant=?", (session_id, tenant)
            )
