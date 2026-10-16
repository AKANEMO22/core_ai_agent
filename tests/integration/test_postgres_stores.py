"""Durable store behaviour on PostgreSQL (gated by TEST_DATABASE_URL) and parity with SQLite.

Run against a throwaway server, e.g.::

    docker run -d --rm --name dpl-pg-test -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:17-alpine
    TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres uv run pytest tests/integration/test_postgres_stores.py

Every test uses its own schema, dropped afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.storage.archive import ArchiveError, ArchiveNotFound, ArtifactArchive
from core.storage.artifacts import StoredObject
from fe_inspect import store as store_module
from fe_inspect.store import SessionDeletedError, TraceStore

PG_URL = os.environ.get("TEST_DATABASE_URL", "")
requires_pg = pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def pg_schema():
    from core.storage.database import Database

    schema = f"t_{uuid.uuid4().hex[:16]}"
    opened: list[Database] = []

    def open_db() -> Database:
        database = Database(PG_URL, schema=schema, min_size=1, max_size=8)
        opened.append(database)
        return database

    yield schema, open_db
    key = f"postgres:{opened[0].location}" if opened else None
    if key and key in store_module._WRITER_HANDLES:
        handle = store_module._WRITER_HANDLES.pop(key)
        handle.connection.close()
    if opened:
        with opened[0].connection() as db:
            db.raw.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    for database in opened:
        database.close()


# --------------------------------------------------------------------------- trace store


@pytest.fixture
def trace_factory(tmp_path, pg_schema):
    _, open_db = pg_schema
    database = open_db()
    return lambda: TraceStore(tmp_path, database=database)


@requires_pg
def test_pg_reopen_retains_graph_state_and_rejects_stale_state(trace_factory):
    store = trace_factory()
    store.save_state({"session_id": "durable", "status": "INITIALIZED", "state_version": 1,
                      "created_at": 123.0, "updated_at": 125.0})
    turn = store.start_span("durable", "Speaker", "agent", started_at=130.0)
    model = store.start_span("durable", "model", "model", parent_span_id=turn["span_id"])
    store.finish_span(model["span_id"], outputs={"text": "actual output", "usage": {"total_tokens": 17}})
    store.finish_span(turn["span_id"])
    store.save_state({"session_id": "durable", "status": "REPORT_READY", "state_version": 2,
                      "created_at": 123.0, "updated_at": 200.0})
    reopened = trace_factory()
    snapshot = reopened.read_spans("durable", after_revision=999999)
    assert snapshot["storage"] == "postgres" and snapshot["coverage"] == "recorded"
    assert len(snapshot["spans"]) == 3
    assert snapshot["spans"][0]["started_at"] == 123.0
    assert snapshot["spans"][0]["ended_at"] == 200.0
    assert snapshot["spans"][0]["duration_ms"] == 77000
    assert snapshot["spans"][2]["parent_span_id"] == turn["span_id"]
    assert snapshot["spans"][2]["outputs"]["text"] == "actual output"
    assert reopened.read_state("durable")["status"] == "REPORT_READY"
    reopened.save_state({"session_id": "durable", "status": "INITIALIZED", "state_version": 1})
    assert reopened.read_state("durable")["state_version"] == 2
    assert reopened.span_summary("durable") == {"span_count": 3, "coverage": "recorded"}


@requires_pg
def test_pg_concurrent_writers_create_one_root_unique_sequences_and_monotonic_revision(trace_factory):
    store = trace_factory()
    parent = store.start_span("parallel", "fanout", "agent")

    def record(index):
        handle = trace_factory()
        span = handle.start_span("parallel", f"child-{index}", "tool", parent_span_id=parent["span_id"])
        return handle.finish_span(span["span_id"], outputs={"index": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(record, range(32)))
    snapshot = store.read_spans("parallel")
    assert len(snapshot["spans"]) == 34
    assert len({row["sequence"] for row in snapshot["spans"]}) == 34
    assert sum(1 for row in snapshot["spans"] if row["kind"] == "run") == 1
    assert all(row["parent_span_id"] == parent["span_id"] for row in result)
    assert all(row["status"] == "completed" for row in result)
    # 1 root + 1 parent + 32 starts + 32 finishes, every one a distinct revision.
    assert snapshot["revision"] == 66
    assert max(row["revision"] for row in snapshot["spans"]) == snapshot["revision"]
    with pytest.raises(ValueError, match="same session"):
        store.start_span("other", "wrong parent", "tool", parent_span_id=parent["span_id"])


@requires_pg
def test_pg_concurrent_event_sequences_and_revision_bumps_never_collide(trace_factory):
    store = trace_factory()
    barrier = threading.Barrier(2)

    def writer(prefix):
        barrier.wait()
        return [store.save_event({"session_id": "bump", "event_id": f"{prefix}-{i}", "type": "progress"})
                for i in range(25)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [row for batch in pool.map(writer, ["a", "b"]) for row in batch]
    assert sorted(row["sequence"] for row in rows) == list(range(1, 51))
    assert store.count_events("bump") == 50
    assert store.read_spans("bump")["revision"] == 50


@requires_pg
def test_pg_recovery_marks_other_process_work_abandoned(trace_factory, monkeypatch):
    first = trace_factory()
    instance = first.register_instance("speaker", "Speaker", "process", session_id="recover")
    running = first.start_span("recover", "speaker", "agent", instance_id=instance["instance_id"])
    queued = first.schedule_span("recover", "queued job")
    finished = first.start_span("recover", "done", "tool")
    first.finish_span(finished["span_id"])
    revision_before = first.read_spans("recover")["revision"]
    monkeypatch.setattr(store_module, "_PROCESS_ID", "new-real-process-identity")
    reopened = trace_factory()
    snapshot = reopened.read_spans("recover")
    rows = {row["span_id"]: row for row in snapshot["spans"]}
    for record in (running, queued):
        assert rows[record["span_id"]]["status"] == "abandoned"
        assert rows[record["span_id"]]["ended_at"] is None
        assert "not observed" in rows[record["span_id"]]["error"]
    assert rows[finished["span_id"]]["status"] == "completed"
    assert snapshot["instances"][0]["status"] == "abandoned" and snapshot["instances"][0]["ended_at"] is None
    assert snapshot["revision"] > revision_before


@requires_pg
def test_pg_events_deduplicate_order_page_and_redact(trace_factory, pg_schema):
    store = trace_factory()
    event = {"session_id": "events", "event_id": "second", "sequence": 2,
             "type": "response_token", "payload": {"token": "visible text", "api_key": "do-not-store"}}
    store.save_event(event)
    store.save_event({**event, "payload": {"token": "duplicate must not replace original"}})
    store.save_event({"session_id": "events", "event_id": "first", "sequence": 1,
                      "type": "progress", "payload": {"token": "auth-secret"}})
    store.save_event({"session_id": "events", "event_id": "reasoning", "sequence": 3,
                      "type": "reasoning_summary", "payload": {"summary": "private"}})
    events = trace_factory().read_events("events")
    assert [row["sequence"] for row in events] == [1, 2]
    assert events[0]["payload"]["token"] == "[REDACTED]"
    assert events[1]["payload"]["token"] == "visible text"
    assert events[1]["payload"]["api_key"] == "[REDACTED]"
    assert len(store.read_events("events", after_sequence=1, limit=1)) == 1
    with store.database.connection() as db:
        stored = " ".join(row[0] for row in db.execute("SELECT body FROM events"))
    assert "do-not-store" not in stored and "auth-secret" not in stored


@requires_pg
def test_pg_purge_tombstone_blocks_late_writes_across_reopen(trace_factory):
    store = trace_factory()
    span = store.start_span("deleted", "speaker", "agent")
    store.register_instance("speaker", "Speaker", "session", "deleted")
    store.save_event({"session_id": "deleted", "event_id": "evt", "sequence": 1, "type": "progress"})
    store.save_state({"session_id": "deleted", "status": "INITIALIZED", "state_version": 1})
    store.set_metadata("deleted", "k", {"v": 1})
    store.purge_session("deleted")
    store.purge_session("deleted")  # idempotent tombstone (ON CONFLICT DO NOTHING)
    reopened = trace_factory()
    assert reopened.read_spans("deleted")["spans"] == []
    assert reopened.read_spans("deleted")["instances"] == []
    assert reopened.read_state("deleted") is None
    assert reopened.get_metadata("deleted", "k") is None
    assert reopened.count_events("deleted") == 0
    assert reopened.finish_span(span["span_id"]) is None
    assert reopened.save_state({"session_id": "deleted"}) is None
    assert reopened.save_event({"session_id": "deleted", "sequence": 2}) is None
    reopened.set_metadata("deleted", "k", {"v": 2})
    assert reopened.get_metadata("deleted", "k") is None
    with pytest.raises(SessionDeletedError):
        reopened.start_span("deleted", "late", "tool")
    with pytest.raises(SessionDeletedError):
        reopened.register_instance("speaker", "Speaker", "session", "deleted")
    with pytest.raises(SessionDeletedError):
        reopened.schedule_span("deleted", "late queued")


@requires_pg
def test_pg_schedule_begin_and_shutdown_lifecycle(trace_factory):
    store = trace_factory()
    queued = store.schedule_span("schedule", "delegation", inputs={"question": "q"})
    assert queued["status"] == "queued" and queued["started_at"] is None
    began = store.begin_span(queued["span_id"])
    assert began["started_at"] >= queued["scheduled_at"] and began["status"] == "running"
    assert store.begin_span(queued["span_id"])["started_at"] == began["started_at"]
    assert store.finish_span(queued["span_id"])["duration_ms"] is not None
    instance = store.register_instance("worker", "Worker", "process")
    active = store.start_span("shutdown", "worker", "agent", instance_id=instance["instance_id"])
    store.close_process()
    snapshot = trace_factory().read_spans("shutdown")
    row = next(row for row in snapshot["spans"] if row["span_id"] == active["span_id"])
    assert row["status"] == "abandoned" and row["ended_at"] is None
    assert snapshot["instances"][0]["end_reason"] == "process_shutdown"


@requires_pg
def test_pg_second_live_process_cannot_claim_trace_database(trace_factory, pg_schema):
    schema, _ = pg_schema
    store = trace_factory()
    running = store.start_span("owned", "current operation", "agent")
    code = (
        "import sys; from core.storage.database import Database; from fe_inspect.store import TraceStore; "
        "TraceStore('.', database=Database(sys.argv[1], schema=sys.argv[2]))"
    )
    attempt = subprocess.run([sys.executable, "-c", code, PG_URL, schema],
                             capture_output=True, text=True, timeout=30,
                             cwd=str(Path(__file__).resolve().parents[2]))
    assert attempt.returncode != 0
    assert "already has an active core process" in attempt.stderr
    assert "test@" not in attempt.stderr  # password never echoed
    current = next(row for row in store.read_spans("owned")["spans"] if row["span_id"] == running["span_id"])
    assert current["status"] == "running"


@requires_pg
def test_pg_schema_bootstrap_is_idempotent_and_versioned(pg_schema):
    _, open_db = pg_schema
    database = open_db()
    assert database.ensure_schema(store_module.POSTGRES_MIGRATIONS) == 1
    assert database.ensure_schema(store_module.POSTGRES_MIGRATIONS) == 1
    with database.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


# --------------------------------------------------------------------------- archive catalog


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_uploads = False
        self.lock = threading.Lock()

    def ensure_bucket(self, bucket, *, versioning=True):
        pass

    def check_bucket(self, bucket):
        pass

    def _stored(self, key):
        data = self.objects[key]
        return StoredObject(key, len(data), hashlib.sha256(data).hexdigest(), f"v-{len(data)}")

    def put_file(self, bucket, key, path, *, sha256):
        if self.fail_uploads:
            raise OSError("simulated upload failure")
        with self.lock:
            self.objects[key] = Path(path).read_bytes()
            return self._stored(key)

    def get_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[key])
        return self._stored(key)

    def head_object(self, bucket, key):
        return self._stored(key)

    def delete_prefix(self, bucket, prefix):
        with self.lock:
            for key in [key for key in self.objects if key.startswith(prefix)]:
                del self.objects[key]


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=requires_pg)])
def archive_factory(request, tmp_path):
    objects = MemoryObjectStore()
    if request.param == "sqlite":
        yield lambda: ArtifactArchive(object_store=objects, bucket="b", catalog_path=tmp_path / "catalog.sqlite3")
        return
    schema_fixture = request.getfixturevalue("pg_schema")
    database = schema_fixture[1]()
    yield lambda: ArtifactArchive(object_store=objects, bucket="b", database=database)


def _workspace(root: Path, session_id: str, note: str = "v1") -> Path:
    workspace = root / f"ws-{session_id}"
    (workspace / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace / "inputs" / "form_input.json").write_text(json.dumps(
        {"session_id": session_id, "candidate_name": "Ada", "target_role": "Engineer"}), encoding="utf-8")
    (workspace / "notes.txt").write_text(note, encoding="utf-8")
    return workspace


def test_archive_revisions_dedupe_restore_and_listing(archive_factory, tmp_path):
    archive = archive_factory()
    workspace = _workspace(tmp_path, "s1")
    first = archive.archive_session("s1", workspace, owner_id="owner")
    assert first["state"] == "archived" and first["revision"] == 1
    assert archive.archive_session("s1", workspace, owner_id="owner")["revision"] == 1  # identical content
    (workspace / "notes.txt").write_text("v2", encoding="utf-8")
    second = archive_factory().archive_session("s1", workspace, owner_id="owner")
    assert second["revision"] == 2 and second["revision_status"] == "archived"
    assert archive.is_archived("s1", "owner") and not archive.is_archived("s1", "intruder")
    listed = archive.list_archived_sessions()
    assert listed[0]["session_id"] == "s1" and listed[0]["candidate_name"] == "Ada"
    with pytest.raises(ArchiveError, match="owner"):
        archive.archive_session("s1", workspace, owner_id="intruder")
    shutil.rmtree(workspace)
    restored = archive.restore_session("s1", tmp_path / "restored", owner_id="owner")
    assert (restored / "notes.txt").read_text(encoding="utf-8") == "v2"


def test_archive_failed_upload_marks_failed_then_stale(archive_factory, tmp_path):
    archive = archive_factory()
    workspace = _workspace(tmp_path, "s2")
    archive.object_store.fail_uploads = True
    with pytest.raises(ArchiveError):
        archive.archive_session("s2", workspace)
    assert archive.status("s2")["state"] == "failed"
    archive.object_store.fail_uploads = False
    assert archive.archive_session("s2", workspace)["revision"] == 2
    (workspace / "notes.txt").write_text("changed", encoding="utf-8")
    archive.object_store.fail_uploads = True
    with pytest.raises(ArchiveError):
        archive.archive_session("s2", workspace)
    status = archive.status("s2")
    assert status["state"] == "stale" and status["revision"] == 2  # current row kept
    archive.object_store.fail_uploads = False
    (workspace / "notes.txt").write_text("v1", encoding="utf-8")
    assert archive.archive_session("s2", workspace)["state"] == "archived"


def test_archive_delete_keeps_tombstone_and_blocks_reuse(archive_factory, tmp_path):
    archive = archive_factory()
    workspace = _workspace(tmp_path, "s3")
    archive.archive_session("s3", workspace)
    archive.delete_session("s3")
    assert not archive.has_session("s3") and archive.status("s3")["state"] == "deleted"
    assert archive.object_store.objects == {}
    reopened = archive_factory()
    with pytest.raises(ArchiveError, match="deleted"):
        reopened.archive_session("s3", workspace)
    with pytest.raises(ArchiveNotFound):
        reopened.restore_session("s3", tmp_path / "nope")
    reopened.delete_session("never-archived")
    assert reopened.status("never-archived")["state"] == "deleted"
    assert reopened.list_archived_sessions() == []


@requires_pg
def test_pg_concurrent_archive_handles_allocate_unique_revisions(pg_schema, tmp_path):
    database = pg_schema[1]()
    objects = MemoryObjectStore()
    handles = [ArtifactArchive(object_store=objects, bucket="b", database=database) for _ in range(4)]
    barrier = threading.Barrier(4)

    def run(index):
        workspace = _workspace(tmp_path / f"h{index}", "race", note=f"content-{index}")
        barrier.wait()
        # Separate handles have separate in-process locks: only the DB lock serializes them.
        return handles[index].archive_session("race", workspace)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))
    with database.connection() as db:
        revisions = [row[0] for row in db.execute(
            "SELECT revision FROM archive_revisions WHERE session_id=? ORDER BY revision", ("race",))]
    assert revisions == [1, 2, 3, 4]
    assert handles[0].status("race")["state"] == "archived"


@requires_pg
def test_pg_app_reports_postgres_and_readiness_checks_database(tmp_path):
    import fakeredis.aioredis
    from fastapi.testclient import TestClient

    from api.main import create_app
    from tests.integration.test_mock_interview_flow import _llm, _settings

    settings = _settings().model_copy(update={"DATABASE_URL": PG_URL, "INSPECTOR_ENABLED": True, "ASSESSOR_POOL_SIZE": 0})
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
                     llm_client=_llm(), settings=settings)
    with TestClient(app) as client:
        ready = client.get("/health/ready").json()
        assert ready["details"]["database"] == "connected", ready
        config = client.get("/api/v1/inspect/config").json()
        assert config["storage"] == "postgres"
        assert "test@" not in str(config) and ":test" not in str(config)
