"""Unit tests for session state, CAS lifecycle transitions, replayable event store, and idempotency."""

import asyncio
import time
import pytest

from core.runtime.types import LoopEvent, LoopEventType
from core.session.state import (
    LifecycleStatus,
    SessionState,
    SessionStateManager,
    InvalidStateTransitionError,
)
from core.session.events import (
    SessionEventStore,
    EventReplayer,
    EventNotFoundError,
    SequenceOrderingError,
)
from core.session.idempotency import (
    IdempotencyManager,
    IdempotencyConflictError,
    TokenBucket,
)


# --- Step 1 Required Tests from Brief ---

def test_cas_state_transition():
    """Verify valid CAS state transition increments version and stale version raises error."""
    state = SessionState(session_id="s1", status=LifecycleStatus.INITIALIZED, state_version=1)
    mgr = SessionStateManager()
    # Valid transition with matching version
    updated = mgr.transition(state, expected_version=1, new_status=LifecycleStatus.PROFILING)
    assert updated.status == LifecycleStatus.PROFILING
    assert updated.state_version == 2

    # Stale transition should fail
    with pytest.raises(InvalidStateTransitionError):
        mgr.transition(updated, expected_version=1, new_status=LifecycleStatus.READY)


def test_idempotency_cache():
    """Verify basic idempotency recording and retrieval."""
    idemp = IdempotencyManager()
    idemp.record("msg_123", {"result": "turn_001_complete"})
    cached = idemp.get("msg_123")
    assert cached is not None
    assert cached["result"] == "turn_001_complete"
    assert idemp.get("unknown") is None


# --- Comprehensive Lifecycle & CAS FSM Tests ---

def test_fsm_invalid_lifecycle_transition():
    """Verify FSM disallows illegal status jumps (e.g. INITIALIZED directly to STAGE_ACTIVE)."""
    state = SessionState(session_id="s_fsm_1", status=LifecycleStatus.INITIALIZED, state_version=1)
    mgr = SessionStateManager()

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        mgr.transition(state, expected_version=1, new_status=LifecycleStatus.STAGE_ACTIVE)
    assert "Invalid transition" in str(exc_info.value)


def test_fsm_terminal_state_lock():
    """Verify terminal states (REPORT_READY, FAILED, CANCELLED, EXPIRED) reject any further transitions."""
    mgr = SessionStateManager()
    terminal_statuses = [
        LifecycleStatus.REPORT_READY,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.EXPIRED,
    ]

    for term_status in terminal_statuses:
        state = SessionState(session_id="s_term", status=term_status, state_version=5)
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            mgr.transition(state, expected_version=5, new_status=LifecycleStatus.STAGE_ACTIVE)
        assert "terminal" in str(exc_info.value).lower() or "invalid transition" in str(exc_info.value).lower()


def test_fsm_full_lifecycle_path():
    """Verify standard happy-path progression through all interview lifecycle stages."""
    mgr = SessionStateManager()
    state = SessionState(session_id="s_full", status=LifecycleStatus.INITIALIZED, state_version=1)

    # INITIALIZED -> PROFILING
    state = mgr.transition(state, 1, LifecycleStatus.PROFILING)
    assert state.status == LifecycleStatus.PROFILING
    assert state.state_version == 2

    # PROFILING -> READY
    state = mgr.transition(state, 2, LifecycleStatus.READY)
    assert state.status == LifecycleStatus.READY
    assert state.state_version == 3

    # READY -> STAGE_ACTIVE
    state = mgr.transition(state, 3, LifecycleStatus.STAGE_ACTIVE, current_stage_id="warmup")
    assert state.status == LifecycleStatus.STAGE_ACTIVE
    assert state.current_stage_id == "warmup"
    assert state.state_version == 4

    # STAGE_ACTIVE -> STAGE_TRANSITIONING
    state = mgr.transition(state, 4, LifecycleStatus.STAGE_TRANSITIONING, is_stage_complete=True)
    assert state.status == LifecycleStatus.STAGE_TRANSITIONING
    assert state.is_stage_complete is True
    assert state.state_version == 5

    # STAGE_TRANSITIONING -> STAGE_ACTIVE (next stage)
    state = mgr.transition(
        state, 5, LifecycleStatus.STAGE_ACTIVE, current_stage_id="tech_deep_dive", is_stage_complete=False
    )
    assert state.status == LifecycleStatus.STAGE_ACTIVE
    assert state.current_stage_id == "tech_deep_dive"
    assert state.state_version == 6

    # STAGE_ACTIVE -> DIALOGUE_COMPLETED
    state = mgr.transition(state, 6, LifecycleStatus.DIALOGUE_COMPLETED)
    assert state.status == LifecycleStatus.DIALOGUE_COMPLETED
    assert state.state_version == 7

    # DIALOGUE_COMPLETED -> ASSESSING
    state = mgr.transition(state, 7, LifecycleStatus.ASSESSING)
    assert state.status == LifecycleStatus.ASSESSING
    assert state.state_version == 8

    # ASSESSING -> REPORT_READY
    state = mgr.transition(state, 8, LifecycleStatus.REPORT_READY)
    assert state.status == LifecycleStatus.REPORT_READY
    assert state.state_version == 9


def test_session_state_serialization_and_fields():
    """Verify SessionState dictionary conversion, field defaults, and deserialization."""
    state = SessionState(
        session_id="s_ser_1",
        status=LifecycleStatus.STAGE_ACTIVE,
        state_version=3,
        current_stage_id="stage_arch",
        is_stage_complete=False,
        current_turn=4,
        metadata={"candidate": "Nguyen", "score": 95},
    )
    data = state.to_dict()
    assert data["session_id"] == "s_ser_1"
    assert data["status"] == "STAGE_ACTIVE"
    assert data["state_version"] == 3
    assert data["current_stage_id"] == "stage_arch"
    assert data["current_turn"] == 4
    assert data["metadata"]["score"] == 95

    restored = SessionState.from_dict(data)
    assert restored.session_id == state.session_id
    assert restored.status == LifecycleStatus.STAGE_ACTIVE
    assert restored.state_version == 3
    assert restored.current_stage_id == "stage_arch"
    assert restored.metadata == state.metadata


def test_session_state_manager_registry():
    """Verify SessionStateManager registers, retrieves, and updates sessions by ID."""
    mgr = SessionStateManager()
    initial = mgr.create_session("sess_reg_1", metadata={"role": "Backend Lead"})
    assert initial.session_id == "sess_reg_1"
    assert initial.status == LifecycleStatus.INITIALIZED
    assert initial.state_version == 1

    fetched = mgr.get_state("sess_reg_1")
    assert fetched is not None
    assert fetched.session_id == "sess_reg_1"

    updated = mgr.transition_session("sess_reg_1", expected_version=1, new_status=LifecycleStatus.PROFILING)
    assert updated.status == LifecycleStatus.PROFILING
    assert updated.state_version == 2
    assert mgr.get_state("sess_reg_1").state_version == 2


# --- Event Store & Monotonic Ordering Tests ---

def test_session_event_store_monotonic_sequence():
    """Verify events appended to store receive strictly monotonic sequence numbers."""
    store = SessionEventStore()
    
    evt1 = store.append(
        session_id="sess_evt_1",
        event=LoopEvent(
            session_id="sess_evt_1",
            turn_id="turn_01",
            sequence=1,
            type=LoopEventType.PROGRESS,
            payload={"step": "started"},
        ),
    )
    assert evt1.sequence == 1

    # Auto-assigned or validated sequence
    evt2 = store.append_event(
        session_id="sess_evt_1",
        turn_id="turn_01",
        event_type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "Hello"},
    )
    assert evt2.sequence == 2
    assert evt2.event_id.startswith("evt_")

    evt3 = store.append_event(
        session_id="sess_evt_1",
        turn_id="turn_01",
        event_type=LoopEventType.COMPLETE,
        payload={"finish_reason": "stop"},
    )
    assert evt3.sequence == 3

    events = store.get_events("sess_evt_1")
    assert len(events) == 3
    assert [e.sequence for e in events] == [1, 2, 3]


def test_session_event_store_rejects_non_monotonic_sequence():
    """Verify store raises SequenceOrderingError if explicit sequence violates monotonic order."""
    store = SessionEventStore()
    store.append(
        session_id="sess_bad_seq",
        event=LoopEvent(
            session_id="sess_bad_seq",
            turn_id="turn_01",
            sequence=5,
            type=LoopEventType.PROGRESS,
            payload={},
        ),
    )

    # Attempting to insert a duplicate or lower sequence must fail
    with pytest.raises(SequenceOrderingError):
        store.append(
            session_id="sess_bad_seq",
            event=LoopEvent(
                session_id="sess_bad_seq",
                turn_id="turn_01",
                sequence=4,
                type=LoopEventType.PROGRESS,
                payload={},
            ),
        )


def test_session_event_store_multi_session_isolation():
    """Verify multiple sessions maintain independent sequence counters and event logs."""
    store = SessionEventStore()
    e_a1 = store.append_event("session_A", "t1", LoopEventType.PROGRESS, {"agent": "A"})
    e_b1 = store.append_event("session_B", "t1", LoopEventType.PROGRESS, {"agent": "B"})
    e_a2 = store.append_event("session_A", "t1", LoopEventType.RESPONSE_TOKEN, {"token": "A2"})

    assert e_a1.sequence == 1
    assert e_b1.sequence == 1
    assert e_a2.sequence == 2

    assert len(store.get_events("session_A")) == 2
    assert len(store.get_events("session_B")) == 1


# --- Event Replayer Tests ---

def test_event_replayer_slice_from_event_id():
    """Verify EventReplayer resumes streaming strictly after last_event_id (SSE standard)."""
    store = SessionEventStore()
    e1 = store.append_event("s_replay", "t1", LoopEventType.PROGRESS, {"step": 1})
    e2 = store.append_event("s_replay", "t1", LoopEventType.RESPONSE_TOKEN, {"step": 2})
    e3 = store.append_event("s_replay", "t1", LoopEventType.RESPONSE_TOKEN, {"step": 3})
    e4 = store.append_event("s_replay", "t1", LoopEventType.COMPLETE, {"step": 4})

    replayer = EventReplayer(store)

    # Slicing after e2 should return [e3, e4]
    replayed = replayer.replay("s_replay", from_event_id=e2.event_id)
    assert len(replayed) == 2
    assert replayed[0].event_id == e3.event_id
    assert replayed[1].event_id == e4.event_id

    # Slicing after the last event should return empty list
    assert len(replayer.replay("s_replay", from_event_id=e4.event_id)) == 0

    # Non-existent event_id raises EventNotFoundError
    with pytest.raises(EventNotFoundError):
        replayer.replay("s_replay", from_event_id="evt_non_existent")


def test_event_replayer_slice_by_sequence_and_filters():
    """Verify EventReplayer slices by sequence range and filters by event type."""
    store = SessionEventStore()
    store.append_event("s_rep_seq", "t1", LoopEventType.PROGRESS, {})
    store.append_event("s_rep_seq", "t1", LoopEventType.RESPONSE_TOKEN, {"t": 1})
    store.append_event("s_rep_seq", "t1", LoopEventType.TOOL_CALL_START, {})
    store.append_event("s_rep_seq", "t1", LoopEventType.RESPONSE_TOKEN, {"t": 2})
    store.append_event("s_rep_seq", "t1", LoopEventType.COMPLETE, {})

    replayer = EventReplayer(store)

    # Slice sequences 2 to 4
    sliced = replayer.replay("s_rep_seq", from_sequence=2, to_sequence=4)
    assert len(sliced) == 3
    assert [e.sequence for e in sliced] == [2, 3, 4]

    # Filter only RESPONSE_TOKEN
    tokens = replayer.replay("s_rep_seq", event_types=[LoopEventType.RESPONSE_TOKEN])
    assert len(tokens) == 2
    assert all(e.type == LoopEventType.RESPONSE_TOKEN for e in tokens)


# --- Idempotency & Conflict Detection Tests ---

def test_idempotency_payload_hash_conflict_detection():
    """Verify 409 conflict error when same idempotency key is reused with different payload."""
    idemp = IdempotencyManager()
    key = "turn_req_999"
    payload_a = {"user_input": "Tell me about Paxos", "mode": "deep"}
    payload_b = {"user_input": "Tell me about Raft", "mode": "fast"}

    # First record with payload_a succeeds
    idemp.record(key, {"answer": "Paxos explanation"}, payload=payload_a)

    # Re-reading or recording same payload succeeds idempotently
    cached = idemp.get(key, payload=payload_a)
    assert cached["answer"] == "Paxos explanation"

    # Using same key with differing payload raises IdempotencyConflictError (409)
    with pytest.raises(IdempotencyConflictError) as exc_info:
        idemp.record(key, {"answer": "Raft explanation"}, payload=payload_b)
    assert "payload mismatch" in str(exc_info.value).lower() or "conflict" in str(exc_info.value).lower()

    with pytest.raises(IdempotencyConflictError):
        idemp.get(key, payload=payload_b)


@pytest.mark.asyncio
async def test_idempotency_per_session_async_lock():
    """Verify async lock serializes operations within the same session but allows parallel across sessions."""
    idemp = IdempotencyManager()
    execution_order = []

    async def session_task(session_id: str, task_name: str, delay: float):
        async with idemp.session_lock(session_id):
            execution_order.append(f"{task_name}_start")
            await asyncio.sleep(delay)
            execution_order.append(f"{task_name}_end")

    # Launch two tasks on session_1 concurrently
    t1 = asyncio.create_task(session_task("session_1", "T1", 0.05))
    t2 = asyncio.create_task(session_task("session_1", "T2", 0.01))

    # Launch a task on session_2 concurrently (should not be blocked by session_1)
    t3 = asyncio.create_task(session_task("session_2", "T3", 0.02))

    await asyncio.gather(t1, t2, t3)

    # T1 and T2 must NOT interleave (T1_start -> T1_end -> T2_start -> T2_end)
    t1_start_idx = execution_order.index("T1_start")
    t1_end_idx = execution_order.index("T1_end")
    t2_start_idx = execution_order.index("T2_start")
    t2_end_idx = execution_order.index("T2_end")

    assert t1_end_idx < t2_start_idx, f"Tasks in session_1 interleaved improperly: {execution_order}"


def test_token_bucket_rate_limiter():
    """Verify token bucket rate limiter restricts bursts beyond capacity and replenishes over time."""
    bucket = TokenBucket(rate=10.0, capacity=3.0)  # 10 tokens/sec, max 3 tokens burst

    # Consume available capacity
    assert bucket.acquire("sess_tb", tokens=1.0) is True
    assert bucket.acquire("sess_tb", tokens=1.0) is True
    assert bucket.acquire("sess_tb", tokens=1.0) is True

    # Next attempt immediately exceeds capacity and should fail
    assert bucket.acquire("sess_tb", tokens=1.0) is False

    # After short delay (e.g. 0.15s), should replenish ~1.5 tokens
    time.sleep(0.15)
    assert bucket.acquire("sess_tb", tokens=1.0) is True


def test_security_scrubbing_in_idempotency_and_state_errors():
    """Verify secrets and API keys are scrubbed from any error messages produced by session components."""
    mgr = SessionStateManager()
    secret_key = "sk-1234567890abcdef1234567890"

    state = SessionState(session_id=f"sess_{secret_key}", status=LifecycleStatus.INITIALIZED, state_version=1)

    # Trigger CAS error
    try:
        mgr.transition(state, expected_version=99, new_status=LifecycleStatus.PROFILING)
    except InvalidStateTransitionError as e:
        err_msg = str(e)
        assert secret_key not in err_msg
        assert "[REDACTED_API_KEY]" in err_msg
