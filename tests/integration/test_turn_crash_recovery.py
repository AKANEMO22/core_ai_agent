"""Retry recovery across the execution-lease and durable message-ledger boundary."""

import asyncio
from types import SimpleNamespace

import fakeredis.aioredis
import pytest

from core.interview.context import ContextAssembler
from core.interview.controller import InterviewSessionController, RunnerRegistry
from core.interview.director import StageDirector
from core.interview.runner import InteractiveSubagentRunner
from core.runtime.agent import RuntimeAgent
from core.runtime.providers.mock import MockLLMClient
from core.runtime.types import LoopEventType
from core.session.cancellation import CancellationRegistry
from core.session.events import SessionEventStore
from core.session.idempotency import MessageLedger, RedisLease, compute_payload_hash
from core.session.state import LifecycleStatus
from core.session.store import RedisSessionStateStore
from core.workspace.session import SessionWorkspace


SESSION = "crashed-session"
MESSAGE = "answer-1"
ANSWER = "I use transactional idempotency keys."


async def _setup(tmp_path, *, target_turns=10):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    state_store = RedisSessionStateStore(redis)
    await state_store.acreate(
        SESSION, initial_status=LifecycleStatus.STAGE_ACTIVE, current_stage_id="technical"
    )
    workspace = SessionWorkspace(SESSION, tmp_path / SESSION)
    workspace.initialize()
    provider = MockLLMClient(canned_tokens=["How do you handle retries?"])
    stages = [{"stage_id": "technical", "target_turns": target_turns}]

    async def factory(session_id):
        runner_workspace = SessionWorkspace(session_id, workspace.root_path)
        return InteractiveSubagentRunner(
            workspace=runner_workspace,
            runtime_agent=RuntimeAgent(provider, workspace=runner_workspace),
            context_assembler=ContextAssembler(),
            stage_director=StageDirector(stages, state_store),
        )

    event_store = SessionEventStore(redis_client=redis)
    publisher = SimpleNamespace(event_store=event_store, publish=event_store.append_async)
    controller = InterviewSessionController(
        state_store, publisher, RunnerRegistry(factory), MessageLedger(redis),
        RedisLease(redis), CancellationRegistry(), asyncio.Semaphore(1),
    )
    return SimpleNamespace(
        redis=redis, state_store=state_store, workspace=workspace, provider=provider,
        controller=controller, event_store=event_store,
    )


async def _abandon(rig):
    controller = rig.controller
    token = await controller.lease.acquire(controller._lock_key(SESSION), 60000)
    assert token is not None
    await controller.ledger.begin(
        SESSION, MESSAGE, "turn_001", compute_payload_hash({"answer": ANSWER})
    )


async def _expire_execution(rig):
    # Redis expiry models a dead owner's lease disappearing; its ledger stays durable.
    await rig.redis.pexpire(rig.controller._lock_key(SESSION), 0)
    assert await rig.redis.exists(rig.controller._lock_key(SESSION)) == 0
    assert (await rig.controller.ledger.get(SESSION, MESSAGE)).status == "in_progress"
    # Recreate process-local caches while keeping Redis and durable workspace files.
    rig.event_store = SessionEventStore(redis_client=rig.redis)
    publisher = SimpleNamespace(event_store=rig.event_store, publish=rig.event_store.append_async)
    rig.controller = InterviewSessionController(
        RedisSessionStateStore(rig.redis), publisher, RunnerRegistry(rig.controller.runners.factory),
        MessageLedger(rig.redis), RedisLease(rig.redis), CancellationRegistry(), asyncio.Semaphore(1),
    )


@pytest.mark.asyncio
async def test_abandoned_execution_retries_same_turn_after_lease_expires(tmp_path):
    rig = await _setup(tmp_path)
    await _abandon(rig)

    held = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER, expected_turn=1)
    assert held.status == "busy"
    assert held.error_code == "TURN_IN_PROGRESS"
    assert held.retry_after_ms is not None and held.retry_after_ms > 0
    assert rig.provider.call_count == 0

    await _expire_execution(rig)
    recovered = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER, expected_turn=1)
    assert recovered.status == "completed"
    assert recovered.turn_id == "turn_001"
    assert (await rig.state_store.aget(SESSION)).current_turn == 1
    assert rig.provider.call_count == 1

    replay = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER, expected_turn=1)
    assert replay.status == "replayed"
    assert replay.turn_id == "turn_001"
    assert any(e.type == LoopEventType.TURN_COMPLETE for e in replay.replay_events)
    assert rig.provider.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_terminal_event", [True, False])
@pytest.mark.parametrize("target_turns", [1, 10])
async def test_committed_turn_recovers_stale_ledger_without_regeneration(
    tmp_path, retain_terminal_event, target_turns
):
    rig = await _setup(tmp_path, target_turns=target_turns)
    await _abandon(rig)
    runner = await rig.controller.runners.get(SESSION)
    async for event in runner.run_turn(SESSION, ANSWER, message_id=MESSAGE):
        if retain_terminal_event or event.type != LoopEventType.TURN_COMPLETE:
            await rig.controller.publisher.publish(SESSION, event)
    assert rig.provider.call_count == 1
    await _expire_execution(rig)

    replay = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER, expected_turn=1)
    assert replay.status == "replayed"
    assert replay.turn_id == "turn_001"
    assert any(e.type == LoopEventType.TURN_COMPLETE for e in replay.replay_events)
    assert (await rig.controller.ledger.get(SESSION, MESSAGE)).status == "completed"
    assert (await rig.state_store.aget(SESSION)).current_turn == 1
    assert rig.provider.call_count == 1


@pytest.mark.asyncio
async def test_expired_lease_does_not_allow_payload_change(tmp_path):
    rig = await _setup(tmp_path)
    await _abandon(rig)
    await _expire_execution(rig)
    conflict = await rig.controller.submit_answer(SESSION, MESSAGE, "Different answer")
    assert conflict.error_code == "MESSAGE_ID_CONFLICT"
    assert rig.provider.call_count == 0
    assert (await rig.controller.ledger.get(SESSION, MESSAGE)).status == "in_progress"


@pytest.mark.asyncio
async def test_advanced_turn_without_matching_artifacts_requires_recovery(tmp_path):
    rig = await _setup(tmp_path)
    await _abandon(rig)
    state = await rig.state_store.aget(SESSION)
    await rig.state_store.atransition(state, state.state_version, current_turn=1)
    await _expire_execution(rig)

    result = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER)
    assert result.error_code == "TURN_RECOVERY_REQUIRED"
    assert result.turn_id == "turn_001"
    assert (await rig.state_store.aget(SESSION)).current_turn == 1
    assert rig.provider.call_count == 0
    assert (await rig.controller.ledger.get(SESSION, MESSAGE)).status == "in_progress"


@pytest.mark.asyncio
async def test_generated_but_uncommitted_turn_is_not_regenerated(tmp_path, monkeypatch):
    rig = await _setup(tmp_path)
    await _abandon(rig)
    runner = await rig.controller.runners.get(SESSION)

    async def crash_before_commit(*args, **kwargs):
        raise RuntimeError("process crashed before committing the turn")

    monkeypatch.setattr(runner.stage_director, "record_turn", crash_before_commit)
    with pytest.raises(RuntimeError, match="process crashed"):
        async for event in runner.run_turn(SESSION, ANSWER, message_id=MESSAGE):
            await rig.controller.publisher.publish(SESSION, event)
    assert rig.workspace.exists("turns/turn_1_interviewer.json")
    await _expire_execution(rig)

    result = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER, expected_turn=1)
    assert result.error_code == "TURN_RECOVERY_REQUIRED"
    assert result.turn_id == "turn_001"
    assert (await rig.state_store.aget(SESSION)).current_turn == 0
    assert rig.provider.call_count == 1


@pytest.mark.asyncio
async def test_commit_without_stage_advancement_requires_reconciliation(tmp_path, monkeypatch):
    rig = await _setup(tmp_path, target_turns=1)
    await _abandon(rig)
    runner = await rig.controller.runners.get(SESSION)

    async def crash_before_stage_advancement(*args, **kwargs):
        raise RuntimeError("process crashed before stage advancement")

    monkeypatch.setattr(runner.stage_director, "advance_stage", crash_before_stage_advancement)
    with pytest.raises(RuntimeError, match="process crashed"):
        async for event in runner.run_turn(SESSION, ANSWER, message_id=MESSAGE):
            await rig.controller.publisher.publish(SESSION, event)
    state = await rig.state_store.aget(SESSION)
    assert state.current_turn == 1 and state.is_stage_complete
    assert state.status == LifecycleStatus.STAGE_ACTIVE
    await _expire_execution(rig)

    result = await rig.controller.submit_answer(SESSION, MESSAGE, ANSWER, expected_turn=1)
    assert result.error_code == "TURN_RECOVERY_REQUIRED"
    assert (await rig.controller.ledger.get(SESSION, MESSAGE)).status == "in_progress"
    assert rig.provider.call_count == 1
