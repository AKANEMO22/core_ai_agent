"""Integration tests for Phase 0 hardening: recovery, cancellation, backpressure, deletion, config."""

import asyncio
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Tuple

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from api.auth import create_service_token
from api.main import create_app
from config.settings import Settings, get_settings, validate_settings
from core.runtime.providers.mock import MockLLMClient
from tests.fakes import ScriptedInterviewLLM
from core.session.state import LifecycleStatus

STAGES = {
    "stages": [
        {"stage_id": "warmup", "name": "Warmup", "target_turns": 1},
        {"stage_id": "technical", "name": "Technical Depth", "target_turns": 2},
    ]
}
ANSWERS = [
    "I lead the payments platform team and own the ledger service.",
    "We guarantee exactly-once payments with idempotency keys stored transactionally.",
    "To avoid stampedes we use request coalescing and jittered expirations.",
]


def _llm(stream_delay: float = 0.0) -> MockLLMClient:
    return ScriptedInterviewLLM(
        speaker_tokens=["Thanks.", " Could", " you", " go", " deeper", " on", " that?"],
        stream_delay=stream_delay,
        stage_plan=STAGES,
    )


def _settings(**overrides: Any) -> Settings:
    base = {"RETRIEVAL_BACKEND": "mock", "ASSESSOR_POOL_SIZE": 0, "ANSWER_BURST": 100.0, "ORCHESTRATOR_JOIN_MS": 300}
    base.update(overrides)
    return get_settings().model_copy(update=base)


def _headers(settings: Settings) -> Dict[str, str]:
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    return {"Authorization": f"Bearer {token}"}


async def _create(app: Any, settings: Settings) -> Tuple[str, str]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        res = await client.post(
            "/api/v1/sessions",
            json={"candidate_name": "Tester", "target_company": "Acme Systems", "target_role": "Senior Backend Engineer"},
            headers=_headers(settings),
        )
        assert res.status_code == 202, res.text
        return res.json()["session_id"], res.json()["session_token"]


@pytest.mark.asyncio
async def test_restart_resumes_session_without_duplicate_turns(tmp_path: Path) -> None:
    """G2: a new process on the same Redis + workspace continues the interview and finishes it."""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    workspaces = tmp_path / "ws"

    llm_a = _llm()
    app_a = create_app(workspace_base=str(workspaces), redis_client=redis, llm_client=llm_a, settings=settings)
    session_id, _ = await _create(app_a, settings)
    assert (await app_a.state.controller.start_interview(session_id)).status == "completed"
    first = await app_a.state.controller.submit_answer(session_id, "m1", ANSWERS[0])
    assert first.status == "completed"

    # --- process restart ---
    llm_b = _llm()
    app_b = create_app(workspace_base=str(workspaces), redis_client=redis, llm_client=llm_b, settings=settings)
    controller = app_b.state.controller

    duplicate = await controller.submit_answer(session_id, "m1", ANSWERS[0])
    assert duplicate.status == "replayed" and duplicate.turn_id == "turn_001"
    assert duplicate.replay_events and llm_b.call_count == 0

    opening_again = await controller.start_interview(session_id)
    assert opening_again.status == "replayed" and llm_b.call_count == 0

    second = await controller.submit_answer(session_id, "m2", ANSWERS[1], expected_turn=2)
    third = await controller.submit_answer(session_id, "m3", ANSWERS[2], expected_turn=3)
    assert (second.turn_id, third.turn_id) == ("turn_002", "turn_003")
    await controller.drain()

    state = await app_b.state.state_mgr.aget(session_id)
    assert state.status == LifecycleStatus.ASSESSING
    assert state.metadata["expected_assessments"] == 3

    # Workers of the new process evaluate everything; the callback synthesizes the report
    queue = app_b.state.assessment_queue
    await queue.setup_consumer_group(group_name="g", start_id="0")
    while await app_b.state.assessor_worker.process_next_job(group_name="g", consumer_name="c"):
        pass
    assert (await app_b.state.state_mgr.aget(session_id)).status == LifecycleStatus.REPORT_READY

    events = await app_b.state.event_store.get_events_async(session_id)
    sequences = [e.sequence for e in events]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)


@pytest.mark.asyncio
async def test_sweeper_recovers_crash_between_barrier_and_synthesis(tmp_path: Path) -> None:
    """A session left in ASSESSING with every evaluation done is finished by the sweeper."""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    session_id, _ = await _create(app, settings)
    controller = app.state.controller
    await controller.start_interview(session_id)
    for i, answer in enumerate(ANSWERS, start=1):
        await controller.submit_answer(session_id, f"m{i}", answer)

    # Evaluate without the completion callback (simulates a crash before synthesis)
    worker = app.state.assessor_worker
    worker.on_assessment_completed = None
    await app.state.assessment_queue.setup_consumer_group(group_name="g", start_id="0")
    while await worker.process_next_job(group_name="g", consumer_name="c"):
        pass
    # Simulate the dialogue-completed hook never running either
    await controller.drain()
    assert (await app.state.state_mgr.aget(session_id)).status in (
        LifecycleStatus.ASSESSING,
        LifecycleStatus.DIALOGUE_COMPLETED,
    )

    touched = await app.state.pipeline.sweep()
    assert touched >= 1
    assert (await app.state.state_mgr.aget(session_id)).status == LifecycleStatus.REPORT_READY


@pytest.mark.asyncio
async def test_zero_turn_dialogue_produces_report(tmp_path: Path) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    store = app.state.state_mgr
    state = await store.acreate("empty", initial_status=LifecycleStatus.STAGE_ACTIVE)
    await store.atransition(state, 1, LifecycleStatus.DIALOGUE_COMPLETED)
    app.state.workspace_mgr.get_or_create_workspace("empty")
    await app.state.pipeline.on_dialogue_completed("empty")
    final = await store.aget("empty")
    assert final.status == LifecycleStatus.REPORT_READY
    assert final.metadata["expected_assessments"] == 0


@pytest.mark.asyncio
async def test_backpressure_returns_busy_and_releases_message(tmp_path: Path) -> None:
    settings = _settings(MAX_INTERACTIVE_GENERATIONS=1, MAX_INTERACTIVE_QUEUE_MS=50)
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=_llm(),
        settings=settings,
    )
    session_id, _ = await _create(app, settings)
    controller = app.state.controller
    await controller.start_interview(session_id)

    await controller.generation_slots.acquire()  # saturate capacity
    try:
        busy = await controller.submit_answer(session_id, "m1", ANSWERS[0])
    finally:
        controller.generation_slots.release()
    assert busy.status == "busy" and busy.error_code == "GENERATION_BUSY" and busy.retry_after_ms
    assert busy.to_event(session_id).type.value == "busy"

    # The message was not consumed: the same message_id succeeds once capacity frees up
    retried = await controller.submit_answer(session_id, "m1", ANSWERS[0])
    assert retried.status == "completed"
    assert app.state.metrics.counter_value("busy_total") == 1


@pytest.mark.asyncio
async def test_answer_rate_limit(tmp_path: Path) -> None:
    settings = _settings(ANSWER_BURST=1.0, ANSWER_RATE_PER_SECOND=0.001)
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=_llm(),
        settings=settings,
    )
    session_id, _ = await _create(app, settings)
    controller = app.state.controller
    await controller.start_interview(session_id)
    assert (await controller.submit_answer(session_id, "m1", ANSWERS[0])).status == "completed"
    limited = await controller.submit_answer(session_id, "m2", ANSWERS[1])
    assert limited.status == "busy" and limited.error_code == "RATE_LIMITED"


def test_barge_in_cancels_generation_and_persists_partial_turn(tmp_path: Path) -> None:
    settings = _settings()
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=_llm(stream_delay=0.05),
        settings=settings,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/sessions", json={"candidate_name": "Barge"}, headers=_headers(settings)
        ).json()
        with client.websocket_connect(
            f"/ws/sessions/{created['session_id']}/stream",
            headers={"Authorization": f"Bearer {created['session_token']}"},
        ) as ws:
            ws.send_json({"type": "start_interview"})
            while ws.receive_json()["type"] != "turn_complete":
                pass

            ws.send_json({"type": "candidate_answer", "message_id": "b1", "candidate_answer": ANSWERS[0]})
            while ws.receive_json()["type"] != "response_token":
                pass
            ws.send_json({"type": "cancel_generation"})
            frames = []
            while not frames or frames[-1]["type"] != "turn_complete":
                frames.append(ws.receive_json())
            types = [f["type"] for f in frames]
            assert "generation_cancelled" in types
            assert frames[-1]["payload"]["interrupted"] is True

            ws.send_json({"type": "cancel_generation"})
            # Background control-plane events (delegations, directives) may arrive first
            frame = ws.receive_json()
            while frame["type"] != "error":
                frame = ws.receive_json()
            assert frame["payload"]["code"] == "NO_ACTIVE_GENERATION"

        workspace = app.state.workspace_mgr.get_or_create_workspace(created["session_id"])
        turn = workspace.read_json("turns/turn_1_interviewer.json")
        assert turn["interrupted"] is True
        assert turn["content"] and len(turn["content"].split()) < 7
        status_res = client.get(f"/api/v1/sessions/{created['session_id']}", headers=_headers(settings)).json()
        assert status_res["current_turn"] == 1


@pytest.mark.asyncio
async def test_delete_purges_redis_and_revokes_tokens(tmp_path: Path) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    session_id, session_token = await _create(app, settings)
    controller = app.state.controller
    await controller.start_interview(session_id)
    await controller.submit_answer(session_id, "m1", ANSWERS[0])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        res = await client.delete(f"/api/v1/sessions/{session_id}", headers=_headers(settings))
        assert res.status_code == 200
        leftover = [k async for k in redis.scan_iter(match=f"*{session_id}*")]
        assert leftover == [f"session:{session_id}:revoked"]
        assert await redis.xlen("audit:session_deletions") == 1
        assert not app.state.workspace_mgr.root_base.joinpath(
            app.state.workspace_mgr.get_or_create_workspace(session_id).normalized_session_id, "turns"
        ).joinpath("turn_1_candidate.json").exists()

        candidate = {"Authorization": f"Bearer {session_token}"}
        assert (await client.get(f"/api/v1/sessions/{session_id}", headers=candidate)).status_code == 401
        assert (await client.delete(f"/api/v1/sessions/{session_id}", headers=_headers(settings))).status_code in (200, 404)

    refused = await controller.submit_answer(session_id, "m2", ANSWERS[1])
    assert refused.status == "rejected"


@pytest.mark.asyncio
async def test_delete_removes_assessment_payloads_and_pending_entries(tmp_path: Path) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    session_id, _ = await _create(app, settings)
    queue = app.state.assessment_queue
    await queue.setup_consumer_group("g", "0")
    job = {"session_id": session_id, "turn_id": "turn_001", "candidate_text": "private answer"}
    other = {**job, "session_id": "other-session"}
    await queue.enqueue(job)
    await queue.read_jobs("g", "c")
    await queue.enqueue(other)
    await queue.schedule_retry(job, 3600)
    await queue.schedule_retry(other, 3600)
    await queue.send_to_dead_letter(job, "error")
    await queue.send_to_dead_letter(other, "error")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.delete(f"/api/v1/sessions/{session_id}", headers=_headers(settings))
    assert response.status_code == 200
    for stream in (queue.stream_key, queue.dead_letter_stream_key):
        entries = await redis.xrange(stream)
        assert len(entries) == 1
        assert entries[0][1]["session_id"] == "other-session"
    assert (await redis.xpending(queue.stream_key, "g"))["pending"] == 0
    delayed = await redis.zrange(queue.delayed_key, 0, -1)
    assert len(delayed) == 1 and session_id not in delayed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation_fails", [False, True])
async def test_delete_fences_inflight_assessment(tmp_path: Path, evaluation_fails: bool) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    session_id, _ = await _create(app, settings)
    workspace_path = app.state.workspace_mgr.get_or_create_workspace(session_id).root_path
    entered, release = asyncio.Event(), asyncio.Event()

    class PausedEvaluator:
        async def evaluate_turn(self, **kwargs):
            entered.set()
            await release.wait()
            if evaluation_fails:
                raise RuntimeError("provider failed after deletion")
            return {"content_score": 7}

    queue = app.state.assessment_queue
    await queue.setup_consumer_group("g", "0")
    await queue.enqueue({"session_id": session_id, "turn_id": "turn_001", "candidate_text": "private answer"})
    app.state.assessor_worker.evaluator = PausedEvaluator()
    task = asyncio.create_task(app.state.assessor_worker.process_next_job("g", "c"))
    await asyncio.wait_for(entered.wait(), 2)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.delete(f"/api/v1/sessions/{session_id}", headers=_headers(settings))
        assert response.status_code == 200
    finally:
        release.set()
        await task
    assert not workspace_path.exists()
    assert [key async for key in redis.scan_iter(match=f"*{session_id}*")] == [f"session:{session_id}:revoked"]
    assert await redis.xlen(queue.stream_key) == 0
    assert await redis.zcard(queue.delayed_key) == 0
    assert await redis.xlen(queue.dead_letter_stream_key) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_worker", [False, True])
async def test_delete_waits_for_assessment_artifact_commit(tmp_path: Path, monkeypatch, cancel_worker: bool) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    session_id, _ = await _create(app, settings)
    workspace = app.state.workspace_mgr.get_or_create_workspace(session_id)
    entered, release = threading.Event(), threading.Event()
    original_write = workspace.write_artifact

    def paused_write(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(workspace, "write_artifact", paused_write)
    queue = app.state.assessment_queue
    await queue.setup_consumer_group("g", "0")
    await queue.enqueue({"session_id": session_id, "turn_id": "turn_001", "candidate_text": "private answer"})
    worker_task = asyncio.create_task(app.state.assessor_worker.process_next_job("g", "c"))
    assert await asyncio.to_thread(entered.wait, 2)
    if cancel_worker:
        worker_task.cancel()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        delete_task = asyncio.create_task(client.delete(f"/api/v1/sessions/{session_id}", headers=_headers(settings)))
        try:
            for _ in range(100):
                if await redis.exists(f"session:{session_id}:revoked"):
                    break
                await asyncio.sleep(0.01)
            assert await redis.exists(f"session:{session_id}:revoked")
            await asyncio.sleep(0.05)
            assert not delete_task.done()
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await worker_task
            response = await delete_task
    assert response.status_code == 200
    assert not workspace.root_path.exists()
    assert [key async for key in redis.scan_iter(match=f"*{session_id}*")] == [f"session:{session_id}:revoked"]


@pytest.mark.asyncio
async def test_assessment_callback_after_delete_does_not_recreate_report(tmp_path: Path) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path / "ws"), redis_client=redis, llm_client=_llm(), settings=settings)
    session_id, _ = await _create(app, settings)
    controller = app.state.controller
    await controller.start_interview(session_id)
    for index, answer in enumerate(ANSWERS, 1):
        await controller.submit_answer(session_id, f"m{index}", answer)
    await controller.drain()
    assert (await app.state.state_mgr.aget(session_id)).status == LifecycleStatus.ASSESSING
    workspace_path = app.state.workspace_mgr.get_or_create_workspace(session_id).root_path
    queue, worker = app.state.assessment_queue, app.state.assessor_worker
    await queue.setup_consumer_group("g", "0")
    await worker.process_next_job("g", "c")
    await worker.process_next_job("g", "c")
    entered, release = asyncio.Event(), asyncio.Event()
    original_callback = worker.on_assessment_completed

    async def delayed_callback(session_id):
        entered.set()
        await release.wait()
        await original_callback(session_id)

    worker.on_assessment_completed = delayed_callback
    task = asyncio.create_task(worker.process_next_job("g", "c"))
    await asyncio.wait_for(entered.wait(), 2)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.delete(f"/api/v1/sessions/{session_id}", headers=_headers(settings))
        assert response.status_code == 200
    finally:
        release.set()
        await task
    assert not workspace_path.exists()
    assert [key async for key in redis.scan_iter(match=f"*{session_id}*")] == [f"session:{session_id}:revoked"]


@pytest.mark.asyncio
async def test_profiling_failure_and_timeout_mark_session_failed(tmp_path: Path) -> None:
    settings = _settings(PROFILING_DEADLINE_SECONDS=0.2)
    failing = MockLLMClient(raise_error=RuntimeError("provider down"))
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=failing,
        settings=settings,
    )
    session_id, _ = await _create(app, settings)
    state = await app.state.state_mgr.aget(session_id)
    assert state.status == LifecycleStatus.FAILED
    assert state.metadata["failure_code"] == "PROFILING_FAILED"

    slow = _llm(stream_delay=1.0)
    app_slow = create_app(
        workspace_base=str(tmp_path / "ws2"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=slow,
        settings=settings,
    )
    slow_id, _ = await _create(app_slow, settings)
    slow_state = await app_slow.state.state_mgr.aget(slow_id)
    assert slow_state.status == LifecycleStatus.FAILED
    assert slow_state.metadata["failure_code"] == "PROFILING_TIMEOUT"

    async with AsyncClient(transport=ASGITransport(app=app_slow), base_url="http://t") as client:
        body = (await client.get(f"/api/v1/sessions/{slow_id}", headers=_headers(settings))).json()
        assert body["failure_code"] == "PROFILING_TIMEOUT" and body["failure_message"]
        assert (await client.get(f"/api/v1/sessions/{slow_id}/report", headers=_headers(settings))).status_code == 409


@pytest.mark.asyncio
async def test_idempotent_session_creation_via_redis(tmp_path: Path) -> None:
    settings = _settings()
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=_llm(),
        settings=settings,
    )
    headers = {**_headers(settings), "Idempotency-Key": "same-key"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        first = await client.post("/api/v1/sessions", json={"candidate_name": "A"}, headers=headers)
        second = await client.post("/api/v1/sessions", json={"candidate_name": "A"}, headers=headers)
        conflict = await client.post("/api/v1/sessions", json={"candidate_name": "B"}, headers=headers)
    assert first.json()["session_id"] == second.json()["session_id"]
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_metrics_and_readiness(tmp_path: Path) -> None:
    settings = _settings()
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=_llm(),
        settings=settings,
    )
    session_id, _ = await _create(app, settings)
    await app.state.controller.start_interview(session_id)
    await app.state.controller.submit_answer(session_id, "m1", ANSWERS[0])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        assert (await client.get("/metrics")).status_code == 401
        body = (await client.get("/metrics", headers=_headers(settings))).text
        assert "ai_interview_ttft_seconds_bucket" in body
        assert "ai_interview_turn_duration_seconds_count" in body
        ready = await client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["details"]["assessor_pool"] == "not_running"
        assert ready.headers.get("x-correlation-id")


def test_production_config_validation_blocks_unsafe_settings(tmp_path: Path) -> None:
    unsafe = get_settings().model_copy(
        update={"APP_ENV": "production", "RETRIEVAL_BACKEND": "mock"}
    )
    errors = validate_settings(unsafe)
    assert any("JWT_SECRET" in e for e in errors)
    assert any("CORS" in e for e in errors)
    assert any("RETRIEVAL_BACKEND" in e for e in errors)
    with pytest.raises(RuntimeError, match="Unsafe production configuration"):
        create_app(workspace_base=str(tmp_path / "ws"), fake_redis=True, settings=unsafe)

    safe = unsafe.model_copy(
        update={
            "JWT_SECRET": "x" * 48,
            "CORS_ALLOWED_ORIGINS": ["https://app.example.org"],
            "OPENAI_API_KEY": "sk-configured-key-for-validation",
            "ALLOW_MOCK_RETRIEVAL": True,
        }
    )
    assert validate_settings(safe) == []
    assert validate_settings(get_settings()) == []  # development defaults stay usable
