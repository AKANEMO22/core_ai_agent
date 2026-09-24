"""Integration tests for crash recovery, event replay, and idle job claiming.

Covers:
- Process restart / crash recovery mid-session
- Resumption of event streams using last_sequence or last_event_id
- Replay of pending assessment jobs via Redis Streams consumer groups (XAUTOCLAIM / claim idle jobs)
- Prevention of double-execution or state corruption on restarted workers
- Retention cleanup validation
"""

import asyncio
import os
import time
from pathlib import Path
import pytest
import fakeredis.aioredis
from fastapi.testclient import TestClient

from api.auth import create_service_token, create_session_token
from api.main import create_app
from api.websocket import ConnectionManager
from config.settings import get_settings
from core.assessor.queue import RedisAssessmentQueue
from core.assessor.worker import AssessorWorker
from core.runtime.providers.mock import MockLLMClient
from core.runtime.types import LoopEvent, LoopEventType
from core.session.events import SessionEventStore
from core.session.state import LifecycleStatus, SessionStateManager
from core.workspace.cleanup import WorkspaceCleanupManager
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager


@pytest.mark.asyncio
async def test_process_crash_and_event_replay_recovery(tmp_path: Path):
    """Test process restart recovery: new process loads history from Redis and resumes monotonically."""
    settings = get_settings()
    workspace_base = tmp_path / "workspaces"
    workspace_base.mkdir(parents=True, exist_ok=True)

    fake_r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    sess_id = "sess_crash_rec_001"

    # Step 1: In "Process 1", record events into Redis event store
    store1 = SessionEventStore(redis_client=fake_r)
    mgr1 = ConnectionManager(event_store=store1)

    for i in range(1, 6):
        evt = LoopEvent(
            schema_version="1.0",
            event_id=f"evt_{i:03d}",
            session_id=sess_id,
            turn_id="turn_001",
            sequence=i,
            timestamp=time.time(),
            type=LoopEventType.RESPONSE_TOKEN if i < 5 else LoopEventType.COMPLETE,
            payload={"token": f"word_{i}"} if i < 5 else {"finish_reason": "stop"},
        )
        await mgr1.record_event_async(sess_id, evt)

    # Verify events are stored in Redis
    redis_events = await fake_r.lrange(f"session:events:{sess_id}", 0, -1)
    assert len(redis_events) == 5

    # Step 2: Simulate "Process 1" crash -> store1 and mgr1 are discarded
    del store1
    del mgr1

    # Step 3: "Process 2" boots up fresh with same Redis and workspace
    store2 = SessionEventStore(redis_client=fake_r)
    mgr2 = ConnectionManager(event_store=store2)

    # Initially in-memory is empty
    assert len(store2._events.get(sess_id, [])) == 0

    # Query events strictly after sequence 3
    replayed = await mgr2.get_events_after_async(sess_id, last_sequence=3)
    assert len(replayed) == 2
    assert [e.sequence for e in replayed] == [4, 5]

    # Query events by last_event_id
    replayed_by_id = await mgr2.get_events_after_async(sess_id, last_event_id="evt_002")
    assert len(replayed_by_id) == 3
    assert [e.event_id for e in replayed_by_id] == ["evt_003", "evt_004", "evt_005"]

    # Step 4: Resume session in Process 2 and record turn 2 event
    new_evt = LoopEvent(
        schema_version="1.0",
        event_id="evt_turn2_001",
        session_id=sess_id,
        turn_id="turn_002",
        sequence=0,  # auto-increment
        timestamp=time.time(),
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "resumed"},
    )
    recorded = await mgr2.record_event_async(sess_id, new_evt)
    # Must strictly continue monotonically from sequence 5 -> 6
    assert recorded.sequence == 6


@pytest.mark.asyncio
async def test_assessor_worker_crash_and_xautoclaim_recovery(tmp_path: Path):
    """Test worker crash mid-job and recovery via XAUTOCLAIM claiming idle jobs."""
    fake_r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis_client=fake_r, stream_key="stream:test_crash_recovery")
    await queue.setup_consumer_group(group_name="workers_recovery")

    ws = SessionWorkspace(session_id="sess_worker_crash", root_path=tmp_path / "sess_worker_crash")
    ws.initialize()

    job_data = {
        "job_id": "sess_worker_crash:turn_001:rubric_v1",
        "session_id": "sess_worker_crash",
        "turn_id": "turn_001",
        "stage_id": "tech",
        "question": "Explain Paxos vs Raft.",
        "candidate_text": "Raft is leader-based and decomposes consensus into leader election and log replication.",
        "vad_metrics": {"speech_duration_ms": 6000, "pause_count": 1, "words_per_minute": 130},
        "rubric_version": "rubric_v1",
        "rubric_target": ["Consensus", "Distributed Systems"],
    }
    await queue.enqueue(job_data)

    mock_llm = MockLLMClient(canned_json_responses={
        "assessment": {
            "content_score": 10,
            "communication_score": 9,
            "strengths": ["Excellent structured breakdown"],
            "improvements": [],
        }
    })

    # Worker 1 reads job from stream but crashes before evaluating or ACKing
    jobs = await queue.read_jobs(group_name="workers_recovery", consumer_name="worker_crashed", count=1)
    assert len(jobs) == 1
    msg_id, read_job = jobs[0]
    assert read_job["job_id"] == job_data["job_id"]

    # Evaluation artifact does NOT exist yet
    assert not (ws.root_path / "evaluations" / "turn_001_eval.json").exists()
    assert await fake_r.scard("session:sess_worker_crash:completed_assessments") == 0

    # Worker 2 boots up and claims idle unacknowledged jobs
    worker2 = AssessorWorker(
        redis_client=fake_r,
        queue=queue,
        llm_client=mock_llm,
        workspace_resolver=lambda sid: ws,
    )

    claimed_count = await worker2.recover_and_process_idle_jobs(
        group_name="workers_recovery",
        consumer_name="worker_survivor",
        min_idle_time_ms=0,
        count=10,
    )
    assert claimed_count == 1

    # Verify evaluation artifact now persisted
    eval_file = ws.root_path / "evaluations" / "turn_001_eval.json"
    assert eval_file.exists()
    eval_data = ws.read_json("evaluations/turn_001_eval.json")
    payload = eval_data.get("payload", eval_data)
    assert payload["content_score"] == 10
    assert payload["communication_score"] == 9

    # Verify Redis set cardinality recorded
    assert await fake_r.scard("session:sess_worker_crash:completed_assessments") == 1

    # Step 5: Duplicate execution prevention: re-delivering job hits idempotency
    await queue.enqueue(job_data)
    initial_llm_calls = mock_llm.call_count
    processed_again = await worker2.process_next_job(group_name="workers_recovery", consumer_name="worker_survivor")
    assert processed_again is True
    # LLM must NOT be called again
    assert mock_llm.call_count == initial_llm_calls
    # Completed set cardinality remains exactly 1
    assert await fake_r.scard("session:sess_worker_crash:completed_assessments") == 1


def test_workspace_retention_cleanup_after_session_expiry(tmp_path: Path):
    """Test retention cleanup removes expired workspaces and preserves active workspaces."""
    cleanup_mgr = WorkspaceCleanupManager(root_base=tmp_path / "workspaces", retention_hours=24)
    ws_base = tmp_path / "workspaces"
    ws_base.mkdir(parents=True, exist_ok=True)

    # 1. Create an active workspace (recent mtime)
    active_dir = ws_base / "sess_active_01"
    active_dir.mkdir(parents=True, exist_ok=True)
    (active_dir / "marker.txt").write_text("active session", encoding="utf-8")

    # 2. Create an expired workspace (simulated mtime 48 hours in past)
    expired_dir = ws_base / "sess_expired_01"
    expired_dir.mkdir(parents=True, exist_ok=True)
    expired_file = expired_dir / "marker.txt"
    expired_file.write_text("expired session", encoding="utf-8")

    past_time = time.time() - (48 * 3600)
    os.utime(expired_file, (past_time, past_time))
    os.utime(expired_dir, (past_time, past_time))

    # Run cleanup
    cleaned = cleanup_mgr.cleanup_expired_workspaces()
    assert "sess_expired_01" in cleaned
    assert not expired_dir.exists()
    assert active_dir.exists()

    # Idempotent cleanup again: nothing more to clean
    assert cleanup_mgr.cleanup_expired_workspaces() == []

    # Manual idempotent delete of active workspace
    assert cleanup_mgr.delete_workspace("sess_active_01") is True
    assert not active_dir.exists()
    assert cleanup_mgr.delete_workspace("sess_active_01") is False
