"""Persistence and integrity checks; temporary fixtures never seed runtime data."""

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

import pytest

from fe_inspect import store as store_module
from fe_inspect.store import SessionDeletedError, TraceStore, sanitize_payload


def test_reopen_retains_graph_state_outputs_and_actual_session_times(tmp_path):
    store = TraceStore(tmp_path)
    store.save_state({"session_id": "durable", "status": "INITIALIZED", "state_version": 1,
                      "created_at": 123.0, "updated_at": 125.0})
    turn = store.start_span("durable", "Speaker", "agent", started_at=130.0)
    model = store.start_span("durable", "model", "model", parent_span_id=turn["span_id"])
    store.finish_span(model["span_id"], outputs={"text": "actual output", "usage": {"total_tokens": 17}})
    store.finish_span(turn["span_id"])
    store.save_state({"session_id": "durable", "status": "REPORT_READY", "state_version": 2,
                      "created_at": 123.0, "updated_at": 200.0})
    reopened = TraceStore(tmp_path)
    snapshot = reopened.read_spans("durable", after_revision=999999)
    assert snapshot["storage"] == "sqlite" and snapshot["coverage"] == "recorded"
    assert len(snapshot["spans"]) == 3
    assert snapshot["spans"][0]["started_at"] == 123.0
    assert snapshot["spans"][0]["ended_at"] == 200.0
    assert snapshot["spans"][0]["duration_ms"] == 77000
    assert snapshot["spans"][2]["parent_span_id"] == turn["span_id"]
    assert snapshot["spans"][2]["outputs"]["text"] == "actual output"
    assert reopened.read_state("durable")["status"] == "REPORT_READY"
    with sqlite3.connect(reopened.path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    # Replaying stale state must not replace a newer archived version.
    reopened.save_state({"session_id": "durable", "status": "INITIALIZED", "state_version": 1})
    assert reopened.read_state("durable")["state_version"] == 2


def test_concurrent_connections_create_one_root_and_correct_parent_graph(tmp_path):
    store = TraceStore(tmp_path)
    parent = store.start_span("parallel", "fanout", "agent")

    def record(index):
        connection = TraceStore(tmp_path)
        span = connection.start_span("parallel", f"child-{index}", "tool", parent_span_id=parent["span_id"])
        return connection.finish_span(span["span_id"], outputs={"index": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(record, range(32)))
    snapshot = store.read_spans("parallel")
    assert len(snapshot["spans"]) == 34
    assert len({row["sequence"] for row in snapshot["spans"]}) == 34
    assert all(row["parent_span_id"] == parent["span_id"] for row in result)
    assert all(row["status"] == "completed" for row in result)
    assert max(row["revision"] for row in snapshot["spans"]) == snapshot["revision"]
    with pytest.raises(ValueError, match="same session"):
        store.start_span("other", "wrong parent", "tool", parent_span_id=parent["span_id"])


def test_recovery_marks_unobserved_end_unknown_and_keeps_completed_records(tmp_path, monkeypatch):
    first = TraceStore(tmp_path)
    instance = first.register_instance("speaker", "Speaker", "process", session_id="recover")
    running = first.start_span("recover", "speaker", "agent", instance_id=instance["instance_id"])
    queued = first.schedule_span("recover", "queued job")
    finished = first.start_span("recover", "done", "tool")
    first.finish_span(finished["span_id"])
    monkeypatch.setattr(store_module, "_PROCESS_ID", "new-real-process-identity")
    reopened = TraceStore(tmp_path)
    rows = {row["span_id"]: row for row in reopened.read_spans("recover")["spans"]}
    for record in (running, queued):
        assert rows[record["span_id"]]["status"] == "abandoned"
        assert rows[record["span_id"]]["ended_at"] is None
        assert "not observed" in rows[record["span_id"]]["error"]
    assert rows[finished["span_id"]]["status"] == "completed"
    instance_row = reopened.read_spans("recover")["instances"][0]
    assert instance_row["status"] == "abandoned" and instance_row["ended_at"] is None


def test_events_deduplicate_order_page_and_preserve_response_text(tmp_path):
    store = TraceStore(tmp_path)
    event = {"session_id": "events", "event_id": "second", "sequence": 2,
             "type": "response_token", "payload": {"token": "visible text", "api_key": "do-not-store"}}
    store.save_event(event)
    store.save_event({**event, "payload": {"token": "duplicate must not replace original"}})
    store.save_event({"session_id": "events", "event_id": "first", "sequence": 1,
                      "type": "progress", "payload": {"token": "auth-secret"}})
    store.save_event({"session_id": "events", "event_id": "reasoning", "sequence": 3,
                      "type": "reasoning_summary", "payload": {"summary": "private"}})
    reopened = TraceStore(tmp_path)
    events = reopened.read_events("events")
    assert [row["sequence"] for row in events] == [1, 2]
    assert events[0]["payload"]["token"] == "[REDACTED]"
    assert events[1]["payload"]["token"] == "visible text"
    assert events[1]["payload"]["api_key"] == "[REDACTED]"
    assert reopened.count_events("events") == 2
    assert len(reopened.read_events("events", after_sequence=1, limit=1)) == 1
    assert "do-not-store" not in reopened.path.read_bytes().decode("utf-8", errors="ignore")


def test_purge_tombstone_blocks_all_late_session_writes_across_reopen(tmp_path):
    store = TraceStore(tmp_path)
    span = store.start_span("deleted", "speaker", "agent")
    store.register_instance("speaker", "Speaker", "session", "deleted")
    store.save_event({"session_id": "deleted", "event_id": "evt", "sequence": 1, "type": "progress"})
    store.save_state({"session_id": "deleted", "status": "INITIALIZED", "state_version": 1})
    store.purge_session("deleted")
    reopened = TraceStore(tmp_path)
    assert reopened.read_spans("deleted")["spans"] == []
    assert reopened.read_spans("deleted")["instances"] == []
    assert reopened.read_state("deleted") is None
    assert reopened.count_events("deleted") == 0
    assert reopened.finish_span(span["span_id"]) is None
    assert reopened.save_state({"session_id": "deleted"}) is None
    assert reopened.save_event({"session_id": "deleted", "sequence": 2}) is None
    with pytest.raises(SessionDeletedError):
        reopened.start_span("deleted", "late", "tool")
    with pytest.raises(SessionDeletedError):
        reopened.register_instance("speaker", "Speaker", "session", "deleted")
    with pytest.raises(SessionDeletedError):
        reopened.schedule_span("deleted", "late queued")


def test_scheduled_span_does_not_claim_execution_until_begin(tmp_path):
    store = TraceStore(tmp_path)
    queued = store.schedule_span("schedule", "delegation", inputs={"question": "q"})
    assert queued["status"] == "queued" and queued["started_at"] is None
    assert queued["scheduled_at"] is not None
    began = store.begin_span(queued["span_id"])
    assert began["span_id"] == queued["span_id"]
    assert began["started_at"] >= queued["scheduled_at"] and began["status"] == "running"
    assert store.begin_span(queued["span_id"])["started_at"] == began["started_at"]
    assert store.finish_span(queued["span_id"])["duration_ms"] is not None


def test_sanitization_redacts_credentials_reasoning_and_preserves_token_metrics(tmp_path):
    class Status(Enum):
        RUNNING = "running"

    @dataclass
    class Result:
        status: Status
        content: str
        password: str

    class PrivateObject:
        def __repr__(self):
            raise AssertionError("No private repr access")

    data = {"apiKey": "api-secret", "authorization": "Bearer token-secret", "password": "pw-secret",
            "response": "api_key=embedded-secret authorization=Bearer real-secret",
            "usage": {"input_tokens": 3, "output_tokens": 4, "token_count": 7},
            "reasoning_summary": "private", "reasoningSummary": "private",
            "nested": {"chain_of_thought": "private"},
            "object": PrivateObject(), "result": Result(Status.RUNNING, "safe", "private")}
    clean = sanitize_payload(data)
    rendered = json.dumps(clean)
    assert all(secret not in rendered for secret in ("api-secret", "token-secret", "pw-secret", "embedded-secret", "real-secret", "private"))
    assert clean["usage"] == {"input_tokens": 3, "output_tokens": 4, "token_count": 7}
    assert clean["object"] == {"_type": "PrivateObject"}
    assert clean["result"]["status"] == "running"
    assert sanitize_payload(clean) == clean
    bounded = sanitize_payload("a" * 300_000)
    assert bounded["_truncated"] is True and bounded["original_characters"] == 300_000
    recursive = {}
    recursive["self"] = recursive
    assert sanitize_payload(recursive)["self"]["reason"] == "recursive_reference"


def test_process_shutdown_persists_instance_end_and_abandoned_span(tmp_path):
    store = TraceStore(tmp_path)
    instance = store.register_instance("worker", "Worker", "process")
    active = store.start_span("shutdown", "worker", "agent", instance_id=instance["instance_id"])
    store.close_process()
    snapshot = TraceStore(tmp_path).read_spans("shutdown")
    row = next(row for row in snapshot["spans"] if row["span_id"] == active["span_id"])
    assert row["status"] == "abandoned" and row["ended_at"] is None
    assert snapshot["instances"][0]["ended_at"] is not None
    assert snapshot["instances"][0]["end_reason"] == "process_shutdown"


def test_second_live_process_cannot_abandon_current_writer(tmp_path):
    store = TraceStore(tmp_path)
    running = store.start_span("owned", "current operation", "agent")
    attempt = subprocess.run(
        [sys.executable, "-c", "from fe_inspect.store import TraceStore; import sys; TraceStore(sys.argv[1])", str(tmp_path)],
        capture_output=True, text=True, timeout=10,
    )
    assert attempt.returncode != 0
    assert "already has an active core process" in attempt.stderr
    assert next(row for row in store.read_spans("owned")["spans"] if row["span_id"] == running["span_id"])["status"] == "running"
