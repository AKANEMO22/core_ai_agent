"""Unit tests for Redis-backed session primitives: state store, idempotency, ledger, lease, events."""

import asyncio

import fakeredis.aioredis
import pytest

from core.runtime.types import LoopEvent, LoopEventType
from core.session.cancellation import CancellationRegistry
from core.session.events import SequenceOrderingError, SessionEventStore
from core.session.idempotency import (
    IdempotencyConflictError,
    IdempotencyInFlightError,
    IdempotencyManager,
    MessageLedger,
    RedisIdempotencyStore,
    RedisLease,
)
from core.session.state import (
    InvalidStateTransitionError,
    LifecycleStatus,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionStateManager,
)
from core.session.store import RedisSessionStateStore


def _redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return SessionStateManager()
    return RedisSessionStateStore(_redis(), ttl_seconds=3600)


@pytest.mark.asyncio
async def test_store_create_get_transition_and_index(store):
    created = await store.acreate("s1", metadata={"k": "v"})
    assert created.state_version == 1
    with pytest.raises(SessionAlreadyExistsError):
        await store.acreate("s1")

    profiling = await store.atransition(created, 1, LifecycleStatus.PROFILING)
    assert profiling.state_version == 2
    fetched = await store.aget("s1")
    assert fetched.status == LifecycleStatus.PROFILING and fetched.metadata == {"k": "v"}
    assert await store.alist_by_status(LifecycleStatus.PROFILING) == ["s1"]
    assert await store.alist_by_status(LifecycleStatus.INITIALIZED) == []

    with pytest.raises(InvalidStateTransitionError):
        await store.atransition(created, 1, LifecycleStatus.PROFILING)  # stale version
    with pytest.raises(InvalidStateTransitionError):
        await store.atransition(profiling, 2, LifecycleStatus.REPORT_READY)  # illegal FSM edge

    await store.adelete("s1")
    assert await store.aget("s1") is None
    with pytest.raises(SessionNotFoundError):
        await store.atransition(profiling, 2, LifecycleStatus.READY)


@pytest.mark.asyncio
async def test_redis_store_concurrent_transitions_single_winner():
    store = RedisSessionStateStore(_redis())
    state = await store.acreate("race")
    results = await asyncio.gather(
        *[store.atransition(state, 1, LifecycleStatus.PROFILING) for _ in range(8)],
        return_exceptions=True,
    )
    winners = [r for r in results if not isinstance(r, Exception)]
    assert len(winners) == 1
    assert all(isinstance(r, InvalidStateTransitionError) for r in results if isinstance(r, Exception))


@pytest.mark.asyncio
async def test_redis_store_ttl_and_survives_new_instance():
    redis = _redis()
    first = RedisSessionStateStore(redis, ttl_seconds=120)
    await first.acreate("persist")
    assert 0 < await redis.ttl("session:persist:state") <= 120
    second = RedisSessionStateStore(redis, ttl_seconds=120)  # simulated restart
    assert (await second.aget("persist")).status == LifecycleStatus.INITIALIZED


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["memory", "redis"])
async def test_idempotency_begin_record_conflict_and_inflight(impl):
    store = IdempotencyManager() if impl == "memory" else RedisIdempotencyStore(_redis())
    claimed, cached = await store.abegin("k1", payload={"a": 1})
    assert claimed and cached is None
    with pytest.raises(IdempotencyInFlightError):
        await store.abegin("k1", payload={"a": 1})
    await store.arecord("k1", result={"session_id": "x"}, payload={"a": 1})
    claimed, cached = await store.abegin("k1", payload={"a": 1})
    assert not claimed and cached == {"session_id": "x"}
    with pytest.raises(IdempotencyConflictError):
        await store.abegin("k1", payload={"a": 2})

    claimed, _ = await store.abegin("k2", payload={"b": 1})
    await store.arelease("k2")
    claimed_again, _ = await store.abegin("k2", payload={"b": 1})
    assert claimed and claimed_again


@pytest.mark.asyncio
async def test_message_ledger_lifecycle():
    ledger = MessageLedger(_redis(), ttl_seconds=60)
    assert await ledger.begin("s", "m1", "turn_001", "h1") is None
    existing = await ledger.begin("s", "m1", "turn_001", "h1")
    assert existing.status == "in_progress" and existing.turn_id == "turn_001"
    with pytest.raises(IdempotencyConflictError):
        await ledger.begin("s", "m1", "turn_001", "other-hash")
    await ledger.complete("s", "m1")
    assert (await ledger.get("s", "m1")).status == "completed"
    await ledger.discard("s", "m1")
    assert await ledger.get("s", "m1") is None


@pytest.mark.asyncio
async def test_redis_lease_exclusive_and_token_checked_release():
    redis = _redis()
    lease = RedisLease(redis)
    token = await lease.acquire("lock", 5000)
    assert token is not None
    assert await lease.acquire("lock", 5000) is None
    assert await lease.release("lock", "wrong-token") is False
    assert await lease.release("lock", token) is True
    assert await lease.acquire("lock", 5000) is not None


def _evt(session_id: str, turn_id: str = "turn_001") -> LoopEvent:
    return LoopEvent(session_id=session_id, turn_id=turn_id, sequence=0, type=LoopEventType.PROGRESS)


@pytest.mark.asyncio
async def test_event_store_sequences_survive_restart_and_are_retained():
    redis = _redis()
    store = SessionEventStore(redis_client=redis, ttl_seconds=600, max_length=3)
    for _ in range(5):
        await store.append_async("s", _evt("s"))
    assert [e.sequence for e in store.get_events("s")] == [3, 4, 5]
    assert await redis.llen("session:events:s") == 3
    assert 0 < await redis.ttl("session:events:s") <= 600
    assert 0 < await redis.ttl("session:events:s:seq") <= 600

    restarted = SessionEventStore(redis_client=redis, ttl_seconds=600, max_length=3)
    nxt = await restarted.append_async("s", _evt("s", "turn_002"))
    assert nxt.sequence == 6
    assert [e.sequence for e in await restarted.get_turn_events("s", "turn_002")] == [6]

    with pytest.raises(SequenceOrderingError):
        await restarted.append_async(
            "s", LoopEvent(session_id="s", turn_id="t", sequence=2, type=LoopEventType.PROGRESS)
        )

    await restarted.purge("s")
    assert await redis.exists("session:events:s", "session:events:s:seq") == 0


@pytest.mark.asyncio
async def test_event_store_lru_eviction_reloads_from_redis():
    redis = _redis()
    store = SessionEventStore(redis_client=redis, memory_max_sessions=2)
    for sid in ("a", "b", "c"):
        await store.append_async(sid, _evt(sid))
    assert store.get_events("a") == []  # evicted from memory
    assert len(await store.get_events_async("a")) == 1  # reloaded


@pytest.mark.asyncio
async def test_replay_requests_resync_for_trimmed_cursor():
    from api.websocket import ConnectionManager

    redis = _redis()
    mgr = ConnectionManager(SessionEventStore(redis_client=redis, max_length=3))
    for _ in range(6):
        await mgr.record_event_async("s", _evt("s"))
    assert (await mgr.replay("s", last_sequence=1)).resync_required is True
    ok = await mgr.replay("s", last_sequence=4)
    assert not ok.resync_required and [e.sequence for e in ok.events] == [5, 6]
    assert (await mgr.replay("s", last_event_id="evt_unknown")).resync_required is True


@pytest.mark.asyncio
async def test_cancellation_registry():
    registry = CancellationRegistry()
    token = registry.token_for("s")
    assert registry.cancel_generation("s") is True and token.is_set()
    assert registry.cancel_generation("s") is False
    fresh = registry.token_for("s")
    assert fresh is not token and not fresh.is_set()

    task = asyncio.create_task(asyncio.sleep(10))
    registry.register_task("s", task)
    assert registry.cancel_session("s") == 1
    await asyncio.sleep(0)
    assert task.cancelled()
    assert registry.token_for("s").is_set()  # cancelled sessions refuse new generations
    registry.forget("s")
    assert not registry.is_session_cancelled("s")
