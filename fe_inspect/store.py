"""Durable, content-safe execution records for the inspector.

The database (SQLite by default, PostgreSQL when a ``Database`` is supplied) is
authoritative: every mutation commits before returning, writers are serialized,
readers see one transactional snapshot, and deleted sessions cannot be recreated
by late callbacks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from core.storage.database import Database

_PROCESS_ID = f"{os.getpid()}:{uuid.uuid4().hex}"
_WRITER_HANDLES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_WRITER_GUARD = threading.Lock()
_MAX_STRING = 256_000
_MAX_ITEMS = 2_000
_MAX_DEPTH = 24
_REASONING_KEYS = {"reasoning", "reasoning_summary", "reasoning_content", "reasoning_text", "thinking",
                   "thinking_content", "chain_of_thought", "chainofthought", "cot"}
_SECRET_PARTS = {
    "api_key", "apikey", "authorization", "password", "passwd", "secret", "secret_key",
    "private_key", "credential", "credentials", "access_token", "refresh_token", "id_token",
    "service_token", "auth_token", "bearer_token", "client_secret", "cookie", "set_cookie",
}
_CREDENTIAL_ASSIGNMENT = re.compile(
    r'''(?i)(["']?(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|service[_ -]?token|'''
    r'''auth[_ -]?token|authorization|password|passwd|secret|client[_ -]?secret|token)["']?'''
    r'''\s*(?:[:=]|\bis\b)\s*)(?:\[REDACTED\]|"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;&}\]\r\n]+)'''
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_URL_CREDENTIAL = re.compile(r"(https?://)[^\s/@:]+:[^\s/@]+@", re.I)
_TERMINAL = {
    "REPORT_READY": "completed", "FAILED": "failed", "CANCELLED": "cancelled",
    "ASSESSMENT_FAILED": "failed", "EXPIRED": "expired",
}


def _safe_string(value: str) -> str:
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _API_KEY.sub("[REDACTED]", value)
    value = _JWT.sub("[REDACTED]", value)
    value = _URL_CREDENTIAL.sub(r"\1[REDACTED]@", value)
    return _CREDENTIAL_ASSIGNMENT.sub(lambda match: match.group(1) + "[REDACTED]", value)


def _secret_key(key: str) -> bool:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    # Preserve token accounting even when metric names contain 'token'.
    if key == "token" or key in _SECRET_PARTS:
        return True
    return any(key.endswith("_" + part) for part in _SECRET_PARTS)


def sanitize_payload(value: Any) -> Any:
    """Convert runtime data to bounded JSON without credentials or reasoning text.

    Unknown objects never use repr/str, which can contain private implementation
    state or secrets. Result/handle-like objects expose only common public fields.
    Truncation is represented explicitly rather than silently dropping content.
    """
    ancestors: set[int] = set()

    def convert(item: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            return {"_truncated": True, "reason": "maximum_depth", "depth_limit": _MAX_DEPTH}
        if isinstance(item, Enum):
            return convert(item.value, depth + 1)
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, str):
            clean = _safe_string(item)
            if len(clean) > _MAX_STRING:
                return {"text": clean[:_MAX_STRING], "_truncated": True,
                        "original_characters": len(clean), "character_limit": _MAX_STRING}
            return clean
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if isinstance(item, Path):
            return convert(str(item), depth + 1)
        if isinstance(item, bytes):
            return {"_type": "bytes", "length": len(item)}
        identity = id(item)
        if identity in ancestors:
            return {"_truncated": True, "reason": "recursive_reference"}
        ancestors.add(identity)
        try:
            if isinstance(item, Mapping):
                result: dict[str, Any] = {}
                for index, (key, child) in enumerate(item.items()):
                    if index >= _MAX_ITEMS:
                        result["_truncated"] = True
                        result["_omitted_items"] = len(item) - _MAX_ITEMS
                        break
                    safe_key = _safe_string(str(key)) if isinstance(key, (str, int, float)) else type(key).__name__
                    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", safe_key)
                    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
                    if normalized in _REASONING_KEYS:
                        continue
                    result[safe_key] = "[REDACTED]" if _secret_key(safe_key) else convert(child, depth + 1)
                return result
            if isinstance(item, (list, tuple, set, frozenset)):
                result = [convert(child, depth + 1) for child in list(item)[:_MAX_ITEMS]]
                if len(item) > _MAX_ITEMS:
                    result.append({"_truncated": True, "_omitted_items": len(item) - _MAX_ITEMS})
                return result
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                return convert({field.name: getattr(item, field.name) for field in dataclasses.fields(item)}, depth + 1)
            if callable(getattr(item, "model_dump", None)):
                try:
                    return convert(item.model_dump(mode="python"), depth + 1)
                except Exception:
                    return {"_type": type(item).__name__}
            result = {"_type": type(item).__name__}
            for name in ("status", "result", "output", "outputs", "text", "content", "error",
                         "usage", "instance_id", "session_id", "turn_id", "job_id", "task_id",
                         "handle_id", "worker_id", "agent", "name", "started_at", "ended_at"):
                # Read stored fields only: inspecting a property must not execute code.
                fields = getattr(item, "__dict__", {})
                if isinstance(fields, dict) and name in fields:
                    result[name] = convert(fields[name], depth + 1)
            return result
        finally:
            ancestors.remove(identity)

    return convert(value, 0)


def _json(value: Any) -> str:
    return json.dumps(sanitize_payload(value), ensure_ascii=False, separators=(",", ":"))


def _timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (float, int)):
        return float(value) if math.isfinite(value) else None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return stamp.replace(tzinfo=timezone.utc).timestamp() if stamp.tzinfo is None else stamp.timestamp()
        except (TypeError, ValueError):
            return None


#: PostgreSQL schema (in its own ``inspect_trace`` schema), versioned via schema_version.
POSTGRES_MIGRATIONS: dict[int, list[str]] = {
    1: [
        "CREATE TABLE IF NOT EXISTS revisions (session_id TEXT PRIMARY KEY, revision BIGINT NOT NULL DEFAULT 0)",
        "CREATE TABLE IF NOT EXISTS tombstones (session_id TEXT PRIMARY KEY, deleted_at DOUBLE PRECISION NOT NULL)",
        """CREATE TABLE IF NOT EXISTS archive_metadata (
               session_id TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL,
               PRIMARY KEY (session_id, key))""",
        """CREATE TABLE IF NOT EXISTS spans (
               span_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
               parent_span_id TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
               process_id TEXT NOT NULL, sequence BIGINT NOT NULL,
               revision BIGINT NOT NULL, body TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS spans_session ON spans(session_id, sequence)",
        "CREATE INDEX IF NOT EXISTS spans_instance ON spans(((body::json)->>'instance_id'))",
        "CREATE INDEX IF NOT EXISTS spans_open ON spans(process_id) WHERE status IN ('running','queued')",
        """CREATE TABLE IF NOT EXISTS events (
               session_id TEXT NOT NULL, event_id TEXT NOT NULL,
               sequence BIGINT NOT NULL, body TEXT NOT NULL,
               PRIMARY KEY (session_id, event_id), UNIQUE (session_id, sequence))""",
        "CREATE TABLE IF NOT EXISTS states (session_id TEXT PRIMARY KEY, body TEXT NOT NULL)",
        """CREATE TABLE IF NOT EXISTS instances (
               row_order BIGINT GENERATED ALWAYS AS IDENTITY,
               instance_id TEXT PRIMARY KEY, session_id TEXT, process_id TEXT NOT NULL,
               ended_at DOUBLE PRECISION, body TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS instances_session ON instances(session_id)",
    ],
}
_WRITE_LOCK = "inspect_trace:write"
_OWNER_LOCK = "inspect_trace:owner"


class _PgWriterHandle:
    """Holds the session-level advisory lock that marks the single owner process."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def __del__(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass


class SessionDeletedError(ValueError):
    """A tombstoned session rejects late writes permanently."""


class TraceStore:
    """Persist observed session/agent/model/tool lifecycles in one SQLite database."""

    def __init__(self, root: Path | str, database: "Database | None" = None) -> None:
        self.database = database
        self.backend = "postgres" if database is not None else "sqlite"
        if database is not None:
            self.path = None
            self.location = database.location
            self._claim_pg_writer()
            self.process_id = _PROCESS_ID
            database.ensure_schema(POSTGRES_MIGRATIONS)
            with self._write() as db:
                self._recover(db)
            return
        self.path = Path(root).resolve() / ".inspect" / "traces.sqlite3"
        self.location = str(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._claim_writer()
        self.process_id = _PROCESS_ID
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS revisions (
                    session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS tombstones (
                    session_id TEXT PRIMARY KEY, deleted_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS archive_metadata (
                    session_id TEXT NOT NULL, key TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(session_id,key)
                );
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    parent_span_id TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
                    process_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    revision INTEGER NOT NULL, body TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS spans_session ON spans(session_id, sequence);
                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_id), UNIQUE (session_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS states (
                    session_id TEXT PRIMARY KEY, body TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS instances (
                    instance_id TEXT PRIMARY KEY, session_id TEXT, process_id TEXT NOT NULL,
                    ended_at REAL, body TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS instances_session ON instances(session_id);
            """)
            db.execute("BEGIN IMMEDIATE")
            self._recover(db)
            db.commit()

    def _recover(self, db: Any) -> None:
        """Mark lifetimes left open by a previous (dead) owner process as abandoned."""
        for row in db.execute("SELECT * FROM spans WHERE status IN ('running','queued') AND kind!='run' AND process_id!=?", (self.process_id,)).fetchall():
            body = json.loads(row["body"])
            body.update(status="abandoned", ended_at=None, duration_ms=None,
                        error="Recorder process restarted; the actual execution end time was not observed.")
            self._write_span(db, body)
        for row in db.execute("SELECT * FROM instances WHERE ended_at IS NULL AND process_id!=?", (self.process_id,)).fetchall():
            body = json.loads(row["body"])
            if body.get("status") != "abandoned":
                body.update(status="abandoned", end_reason="process_restart", end_time_known=False)
                db.execute("UPDATE instances SET body=? WHERE instance_id=?", (_json(body), body["instance_id"]))
                self._revise_instance_sessions(db, body["instance_id"], body.get("session_id"))

    def _claim_pg_writer(self) -> None:
        """PostgreSQL equivalent of the writer file lock: a session-level advisory lock.

        It is held on a dedicated connection for this process's lifetime and released by
        the server when the process (or its connection) dies, so recovery never marks a
        still-live owner's work abandoned.
        """
        from core.storage.database import advisory_key

        key = "postgres:" + self.database.location
        with _WRITER_GUARD:
            if key in _WRITER_HANDLES:
                self._writer_handle = _WRITER_HANDLES[key]
                return
            connection = self.database.dedicated_connection()
            try:
                acquired = connection.execute("SELECT pg_try_advisory_lock(%s)", (advisory_key(f"{_OWNER_LOCK}:{self.database.schema}"),)).fetchone()[0]
            except Exception:
                connection.close()
                raise
            if not acquired:
                connection.close()
                raise RuntimeError("Trace archive already has an active core process; run one API process per trace database")
            handle = _PgWriterHandle(connection)
            _WRITER_HANDLES[key] = handle
            self._writer_handle = handle

    def _claim_writer(self) -> None:
        """Enforce this core's single-process deployment before recovering lifetimes.

        The OS releases the lock after a crash. Another live process must not mark
        our still-running agents abandoned merely because its process ID differs.
        Multiple handles/threads inside the owner process may share the archive.
        """
        key = str(self.path)
        with _WRITER_GUARD:
            if key in _WRITER_HANDLES:
                self._writer_handle = _WRITER_HANDLES[key]
                return
            handle = (self.path.parent / "writer.lock").open("a+b")
            try:
                if os.name == "nt":
                    import msvcrt
                    if handle.seek(0, os.SEEK_END) == 0:
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as exc:
                handle.close()
                raise RuntimeError("Trace archive already has an active core process; run one API process per workspace volume") from exc
            _WRITER_HANDLES[key] = handle
            self._writer_handle = handle

    def get_metadata(self, session_id: str, key: str) -> Any:
        with self._connect() as db:
            row = db.execute("SELECT body FROM archive_metadata WHERE session_id=? AND key=?", (session_id, key)).fetchone()
            return json.loads(row[0]) if row else None

    def set_metadata(self, session_id: str, key: str, value: Any) -> None:
        with self._write() as db:
            if self._deleted(db, session_id):
                return
            db.execute("INSERT INTO archive_metadata(session_id,key,body) VALUES (?,?,?) ON CONFLICT(session_id,key) DO UPDATE SET body=excluded.body",
                       (session_id, key, _json(value)))

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.database is not None:
            with self.database.connection() as pg:
                yield pg
            return
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=30000")
            yield db
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def _write(self) -> Iterator[Any]:
        with self._connect() as db:
            if self.database is not None:
                # BEGIN IMMEDIATE equivalent: one writer at a time across all processes.
                db.lock(f"{_WRITE_LOCK}:{self.database.schema}")
            else:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()

    @staticmethod
    def _deleted(db: Any, session_id: str) -> bool:
        return db.execute("SELECT 1 FROM tombstones WHERE session_id=?", (session_id,)).fetchone() is not None

    @staticmethod
    def _bump(db: Any, session_id: str) -> int:
        db.execute("INSERT INTO revisions(session_id,revision) VALUES (?,1) ON CONFLICT(session_id) DO UPDATE SET revision=revisions.revision+1", (session_id,))
        return int(db.execute("SELECT revision FROM revisions WHERE session_id=?", (session_id,)).fetchone()[0])

    def _write_span(self, db: Any, body: dict) -> dict:
        body["revision"] = self._bump(db, body["session_id"])
        clean = sanitize_payload(body)
        db.execute("""INSERT INTO spans(span_id,session_id,parent_span_id,kind,status,process_id,sequence,revision,body)
            VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(span_id) DO UPDATE SET
            status=excluded.status, revision=excluded.revision, body=excluded.body""",
                   (clean["span_id"], clean["session_id"], clean["parent_span_id"], clean["kind"],
                    clean["status"], clean["process_id"], clean["sequence"], clean["revision"], _json(clean)))
        return clean

    def _root(self, db: Any, session_id: str, started_at: float | None = None) -> dict:
        row = db.execute("SELECT body FROM spans WHERE span_id=?", (f"run:{session_id}",)).fetchone()
        if row:
            return json.loads(row[0])
        return self._write_span(db, {
            "span_id": f"run:{session_id}", "parent_span_id": None, "session_id": session_id,
            "name": "Session", "kind": "run", "agent": None, "instance_id": None,
            "lifetime_scope": "session", "scheduled_at": None, "started_at": started_at,
            "ended_at": None, "duration_ms": None, "status": "running", "inputs": None,
            "outputs": None, "attributes": {}, "error": None, "sequence": 1,
            "process_id": self.process_id,
        })

    def start_span(self, session_id: str, name: str, kind: str, parent_span_id: str | None = None,
                   agent: str | None = None, instance_id: str | None = None,
                   lifetime_scope: str | None = None, inputs: Any = None,
                   attributes: Any = None, scheduled_at: Any = None,
                   span_id: str | None = None, started_at: Any = None) -> dict:
        span_id = span_id or f"span_{uuid.uuid4().hex}"
        with self._write() as db:
            if self._deleted(db, session_id):
                raise SessionDeletedError(f"Session '{session_id}' has been deleted")
            existing = db.execute("SELECT body FROM spans WHERE span_id=?", (span_id,)).fetchone()
            if existing:
                body = json.loads(existing[0])
                if body["session_id"] != session_id:
                    raise ValueError("A span identifier cannot be shared by different sessions")
                return body
            root = self._root(db, session_id)
            if span_id == root["span_id"]:
                return root
            parent_span_id = parent_span_id or root["span_id"]
            if not db.execute("SELECT 1 FROM spans WHERE span_id=? AND session_id=?", (parent_span_id, session_id)).fetchone():
                raise ValueError("Parent span must already exist in the same session")
            sequence = db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM spans WHERE session_id=?", (session_id,)).fetchone()[0]
            return self._write_span(db, {
                "span_id": span_id, "parent_span_id": parent_span_id, "session_id": session_id,
                "name": name, "kind": kind, "agent": agent, "instance_id": instance_id,
                "lifetime_scope": lifetime_scope, "scheduled_at": _timestamp(scheduled_at),
                "started_at": _timestamp(started_at) if started_at is not None else time.time(),
                "ended_at": None, "duration_ms": None, "status": "running", "inputs": inputs,
                "outputs": None, "attributes": attributes or {}, "error": None,
                "sequence": sequence, "process_id": self.process_id,
            })

    def finish_span(self, span_id: str, status: str = "completed", outputs: Any = None,
                    error: Any = None, attributes: Any = None) -> dict | None:
        with self._write() as db:
            row = db.execute("SELECT body FROM spans WHERE span_id=?", (span_id,)).fetchone()
            if not row:
                return None
            body = json.loads(row[0])
            if body["status"] not in {"running", "queued"}:
                return body
            ended_at = time.time()
            body.update(status=status, ended_at=ended_at, error=error)
            if outputs is not None:
                body["outputs"] = outputs
            if body.get("started_at") is not None:
                body["duration_ms"] = round(max(0, ended_at - body["started_at"]) * 1000, 3)
            if attributes:
                body["attributes"].update(sanitize_payload(attributes))
            return self._write_span(db, body)

    def schedule_span(self, session_id: str, name: str, parent_span_id: str | None = None,
                      agent: str | None = None, inputs: Any = None,
                      attributes: Any = None) -> dict:
        """Record scheduling without claiming that the task has started executing."""
        scheduled_at = time.time()
        with self._write() as db:
            if self._deleted(db, session_id):
                raise SessionDeletedError(f"Session '{session_id}' has been deleted")
            root = self._root(db, session_id)
            parent_span_id = parent_span_id or root["span_id"]
            if not db.execute("SELECT 1 FROM spans WHERE span_id=? AND session_id=?", (parent_span_id, session_id)).fetchone():
                raise ValueError("Parent span must already exist in the same session")
            sequence = db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM spans WHERE session_id=?", (session_id,)).fetchone()[0]
            return self._write_span(db, {
                "span_id": f"span_{uuid.uuid4().hex}", "parent_span_id": parent_span_id,
                "session_id": session_id, "name": name, "kind": "task", "agent": agent,
                "instance_id": None, "lifetime_scope": "task", "scheduled_at": scheduled_at,
                "started_at": None, "ended_at": None, "duration_ms": None, "status": "queued",
                "inputs": inputs, "outputs": None, "attributes": attributes or {}, "error": None,
                "sequence": sequence, "process_id": self.process_id,
            })

    def begin_span(self, span_id: str) -> dict | None:
        """Transition the same scheduled task record to observed execution."""
        with self._write() as db:
            row = db.execute("SELECT body FROM spans WHERE span_id=?", (span_id,)).fetchone()
            if not row:
                return None
            body = json.loads(row[0])
            if body["status"] != "queued":
                return body
            body.update(started_at=time.time(), status="running")
            return self._write_span(db, body)

    def update_output(self, span_id: str, outputs: Any) -> dict | None:
        with self._write() as db:
            row = db.execute("SELECT body FROM spans WHERE span_id=?", (span_id,)).fetchone()
            if not row:
                return None
            body = json.loads(row[0])
            body["outputs"] = outputs
            return self._write_span(db, body)

    def read_spans(self, session_id: str, after_revision: int = 0) -> dict:
        """Return a coherent full graph snapshot, even when polling by revision.

        after_revision is deliberately not a row filter: removing unchanged parents
        would make a realtime graph structurally incomplete.
        """
        with self._connect() as db:
            if self.database is not None:
                db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            else:
                db.execute("BEGIN")
            revision = db.execute("SELECT revision FROM revisions WHERE session_id=?", (session_id,)).fetchone()
            spans = [json.loads(row[0]) for row in db.execute("SELECT body FROM spans WHERE session_id=? ORDER BY sequence", (session_id,))]
            instances = [json.loads(row[0]) for row in db.execute(f"""SELECT body FROM instances WHERE session_id=? OR instance_id IN
                (SELECT {self._instance_expr} FROM spans WHERE session_id=?) ORDER BY {self._instance_order}""", (session_id, session_id))]
            db.commit()
        return {"session_id": session_id, "spans": spans, "revision": int(revision[0]) if revision else 0,
                "coverage": "recorded" if any(span["kind"] != "run" for span in spans) else "not_recorded",
                "storage": self.backend, "instances": instances}

    @property
    def _instance_expr(self) -> str:
        return "((body::json)->>'instance_id')" if self.database is not None else "json_extract(body,'$.instance_id')"

    @property
    def _instance_order(self) -> str:
        return "row_order" if self.database is not None else "rowid"

    def span_summary(self, session_id: str) -> dict:
        with self._connect() as db:
            count, observed = db.execute("SELECT COUNT(*),SUM(CASE WHEN kind!='run' THEN 1 ELSE 0 END) FROM spans WHERE session_id=?", (session_id,)).fetchone()
            return {"span_count": count, "coverage": "recorded" if observed else "not_recorded"}

    def save_event(self, event: dict) -> dict | None:
        clean = sanitize_payload(event)
        if clean.get("type") == "reasoning_summary":
            return None
        # In this event schema 'token' is generated response text, not an auth
        # credential. Preserve only that field in this explicit context; its
        # content still passes through string credential sanitization.
        if clean.get("type") == "response_token" and isinstance(event.get("payload"), Mapping):
            response_text = event["payload"].get("token")
            if isinstance(response_text, str):
                clean.setdefault("payload", {})["token"] = sanitize_payload(response_text)
        session_id = str(clean["session_id"])
        event_id = str(clean.get("event_id") or "evt_" + hashlib.sha256(json.dumps(clean, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
        with self._write() as db:
            if self._deleted(db, session_id):
                return None
            existing = db.execute("SELECT body FROM events WHERE session_id=? AND event_id=?", (session_id, event_id)).fetchone()
            if existing:
                return json.loads(existing[0])
            sequence = int(clean.get("sequence") or 0)
            if sequence <= 0:
                sequence = db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM events WHERE session_id=?", (session_id,)).fetchone()[0]
            clean.update(event_id=event_id, session_id=session_id, sequence=sequence)
            # Different IDs at the same source sequence indicate corrupt input;
            # fail explicitly rather than silently losing one event.
            db.execute("INSERT INTO events(session_id,event_id,sequence,body) VALUES (?,?,?,?)", (session_id, event_id, sequence, json.dumps(clean, ensure_ascii=False, separators=(",", ":"))))
            self._bump(db, session_id)
            return clean

    def read_events(self, session_id: str, after_sequence: int = 0, limit: int = 5000) -> list[dict]:
        with self._connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT body FROM events WHERE session_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                                                           (session_id, after_sequence, max(0, min(limit, 5000))))]

    def count_events(self, session_id: str) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM events WHERE session_id=?", (session_id,)).fetchone()[0])

    def save_state(self, state: dict) -> dict | None:
        clean = sanitize_payload(state)
        session_id = str(clean["session_id"])
        with self._write() as db:
            if self._deleted(db, session_id):
                return None
            existing = db.execute("SELECT body FROM states WHERE session_id=?", (session_id,)).fetchone()
            if existing:
                old = json.loads(existing[0])
                if clean.get("state_version", 0) < old.get("state_version", 0):
                    return old
                if clean.get("state_version", 0) == old.get("state_version", 0) and (_timestamp(clean.get("updated_at")) or 0) < (_timestamp(old.get("updated_at")) or 0):
                    return old
            db.execute("INSERT INTO states(session_id,body) VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET body=excluded.body", (session_id, _json(clean)))
            root = self._root(db, session_id, _timestamp(clean.get("created_at")))
            # The state source owns these timestamps; a recorder's startup time
            # cannot substitute for a session's actual creation or completion.
            root["started_at"] = _timestamp(clean.get("created_at"))
            raw_status = clean.get("status")
            root["status"] = _TERMINAL.get(raw_status, "running")
            root["ended_at"] = _timestamp(clean.get("updated_at")) if raw_status in _TERMINAL else None
            root["duration_ms"] = round(max(0, root["ended_at"] - root["started_at"]) * 1000, 3) if root["ended_at"] is not None and root["started_at"] is not None else None
            root["attributes"].update(lifecycle_status=raw_status, state_version=clean.get("state_version"),
                                      created_at=clean.get("created_at"), updated_at=clean.get("updated_at"))
            self._write_span(db, root)
            return clean

    def read_state(self, session_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT body FROM states WHERE session_id=?", (session_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def register_instance(self, agent: str, class_name: str, lifetime_scope: str,
                          session_id: str | None = None) -> dict:
        body = sanitize_payload({"instance_id": f"inst_{uuid.uuid4().hex}", "agent": agent,
                                 "class_name": class_name, "lifetime_scope": lifetime_scope,
                                 "observed_at": time.time(), "ended_at": None,
                                 "process_id": self.process_id, "session_id": session_id,
                                 "status": "active"})
        with self._write() as db:
            if session_id and self._deleted(db, session_id):
                raise SessionDeletedError(f"Session '{session_id}' has been deleted")
            db.execute("INSERT INTO instances(instance_id,session_id,process_id,ended_at,body) VALUES (?,?,?,?,?)",
                       (body["instance_id"], session_id, self.process_id, None, _json(body)))
            if session_id:
                self._bump(db, session_id)
        return body

    def end_instance(self, instance_id: str, reason: str = "released") -> dict | None:
        with self._write() as db:
            row = db.execute("SELECT body FROM instances WHERE instance_id=?", (instance_id,)).fetchone()
            if not row:
                return None
            body = json.loads(row[0])
            if body.get("ended_at") is not None or body.get("status") == "abandoned":
                return body
            body.update(ended_at=time.time(), end_reason=reason, status="released", end_time_known=True)
            db.execute("UPDATE instances SET ended_at=?, body=? WHERE instance_id=?", (body["ended_at"], _json(body), instance_id))
            self._revise_instance_sessions(db, instance_id, body.get("session_id"))
            return sanitize_payload(body)

    def _revise_instance_sessions(self, db: Any, instance_id: str, session_id: str | None) -> None:
        sessions = {row[0] for row in db.execute(f"SELECT DISTINCT session_id FROM spans WHERE {self._instance_expr}=?", (instance_id,))}
        if session_id:
            sessions.add(session_id)
        for observed_session in sessions:
            self._bump(db, observed_session)

    def close_process(self) -> None:
        """Record graceful recorder shutdown; do not invent an agent completion."""
        with self._write() as db:
            for row in db.execute("SELECT body FROM spans WHERE process_id=? AND status IN ('running','queued') AND kind!='run'", (self.process_id,)).fetchall():
                body = json.loads(row[0])
                body.update(status="abandoned", ended_at=None, duration_ms=None,
                            error="Recorder stopped before execution completion was observed; actual end time unknown.")
                self._write_span(db, body)
            for row in db.execute("SELECT body FROM instances WHERE process_id=? AND ended_at IS NULL", (self.process_id,)).fetchall():
                body = json.loads(row[0])
                if body.get("status") == "abandoned":
                    continue
                body.update(ended_at=time.time(), end_reason="process_shutdown", status="released", end_time_known=True)
                db.execute("UPDATE instances SET ended_at=?,body=? WHERE instance_id=?", (body["ended_at"], _json(body), body["instance_id"]))
                self._revise_instance_sessions(db, body["instance_id"], body.get("session_id"))

    def purge_session(self, session_id: str) -> None:
        with self._write() as db:
            db.execute("INSERT INTO tombstones(session_id,deleted_at) VALUES (?,?) ON CONFLICT(session_id) DO NOTHING", (session_id, time.time()))
            for table in ("spans", "events", "states", "instances", "revisions", "archive_metadata"):
                db.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
