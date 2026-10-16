"""Explicit ending bypasses incomplete assessments without inventing a verdict."""
from pathlib import Path

import fakeredis.aioredis
from fastapi.testclient import TestClient

from api.main import create_app
from tests.integration.test_mock_interview_flow import _settings, _llm, _create_session
from api.auth import create_service_token
import asyncio
import pytest
from core.assessor.pipeline import AssessmentPipeline
from core.assessor.queue import RedisAssessmentQueue
from core.assessor.worker import AssessorWorker
from core.assessor.synthesizer import ReportSynthesizer, AssessmentBarrierError
from core.runtime.providers.mock import MockLLMClient
from core.session.state import SessionStateManager, LifecycleStatus
from core.workspace.session import SessionWorkspace


def test_end_ready_session_generates_partial_report_and_is_idempotent(tmp_path: Path):
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True), llm_client=_llm(), settings=settings)
    headers = {"Authorization": "Bearer " + create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)}
    with TestClient(app) as client:
        created = _create_session(client, headers)
        sid = created["session_id"]
        headers = {"Authorization": "Bearer " + created["session_token"]}
        ws = app.state.workspace_mgr.get_or_create_workspace(sid)
        ws.write_text("turns/turn_1_candidate.json", '{"content": "Unassessed evidence about Redis streams"}')
        response = client.post(f"/api/v1/sessions/{sid}/end", headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "REPORT_READY"
        report = client.get(f"/api/v1/sessions/{sid}/report", headers=headers).json()
        assert report["decision"]["verdict"] == "NEEDS_REVIEW"
        # Default interview language is Vietnamese, so the report is written in Vietnamese.
        assert "phiên kết thúc sớm" in report["report_markdown"].lower()
        assert "Unassessed evidence" in report["report_markdown"]
        version = client.get(f"/api/v1/sessions/{sid}", headers=headers).json()["state_version"]
        assert client.post(f"/api/v1/sessions/{sid}/end", headers=headers).status_code == 200
        assert client.get(f"/api/v1/sessions/{sid}", headers=headers).json()["state_version"] == version
        assert client.post(f"/api/v1/sessions/{sid}/start", headers=headers).status_code in (404, 409)


@pytest.mark.asyncio
async def test_end_cancels_remote_assessment_without_killing_consumer_or_writing(tmp_path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis)
    await queue.setup_consumer_group(start_id="0")
    ws = SessionWorkspace(session_id="remote", root_path=tmp_path / "remote")
    ws.initialize()
    started, canceled = asyncio.Event(), asyncio.Event()

    class BlockedEvaluator:
        async def evaluate_turn(self, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()

    worker = AssessorWorker(redis, queue, MockLLMClient(), lambda sid: ws, evaluator=BlockedEvaluator())
    await queue.enqueue({"session_id": "remote", "turn_id": "turn_1", "candidate_text": "answer"})
    consumer = asyncio.create_task(worker.process_next_job())
    await asyncio.wait_for(started.wait(), 2)
    await redis.set(queue.stopped_key("remote"), "1")
    assert await asyncio.wait_for(consumer, 2) is True
    assert canceled.is_set()
    assert not ws.exists("evaluations/turn_1_eval.json")
    assert await queue.enqueue({"session_id": "remote", "turn_id": "turn_2"}) == ""
    # Same consumer operation remains usable for another session.
    await queue.enqueue({"session_id": "other", "turn_id": "turn_1"})
    worker.evaluator = type("Evaluator", (), {"evaluate_turn": lambda self, **kw: asyncio.sleep(0, result={"content_score": 5})})()
    assert await worker.process_next_job() is True


@pytest.mark.asyncio
async def test_end_concurrency_and_restart_recover_without_assessment_barrier(tmp_path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    states = SessionStateManager()
    states.create_session("end", initial_status=LifecycleStatus.ASSESSING, metadata={"expected_assessments": 5})
    ws = SessionWorkspace(session_id="end", root_path=tmp_path / "end")
    ws.initialize()
    synthesizer = ReportSynthesizer(redis, lambda sid: ws, state_manager=states)
    pipeline = AssessmentPipeline(redis, states, synthesizer)
    started, release = asyncio.Event(), asyncio.Event()
    original = synthesizer.synthesize
    calls = []

    async def blocked(sid, **kwargs):
        calls.append(kwargs)
        started.set()
        await release.wait()
        return await original(sid, **kwargs)

    synthesizer.synthesize = blocked
    first = asyncio.create_task(pipeline.end_session("end"))
    await started.wait()
    assert await pipeline.end_session("end") is False
    assert (await states.aget("end")).status == LifecycleStatus.ENDING
    first.cancel()  # Process/request interruption leaves durable intent for recovery.
    await asyncio.gather(first, return_exceptions=True)
    release.set()
    restarted = AssessmentPipeline(redis, states, synthesizer)
    assert await restarted.sweep() == 1
    assert (await states.aget("end")).status == LifecycleStatus.REPORT_READY
    assert calls == [{"early_end": True}, {"early_end": True}]
    assert await restarted.end_session("end") is True


@pytest.mark.asyncio
async def test_normal_synthesis_inflight_cannot_complete_early_ended_session(tmp_path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    states = SessionStateManager()
    state = states.create_session("race", initial_status=LifecycleStatus.ASSESSING, metadata={"expected_assessments": 0})
    ws = SessionWorkspace(session_id="race", root_path=tmp_path / "race")
    ws.initialize()
    synth = ReportSynthesizer(redis, lambda sid: ws, state_manager=states)
    started, release = asyncio.Event(), asyncio.Event()

    class Narrative:
        async def write(self, *args, **kwargs):
            started.set()
            await release.wait()
            return {"strengths": [], "improvements": [], "study_plan": []}

    synth.narrative_writer = Narrative()
    task = asyncio.create_task(synth.synthesize("race"))
    await started.wait()
    await states.atransition(state, state.state_version, LifecycleStatus.ENDING)
    release.set()
    with pytest.raises(AssessmentBarrierError, match="superseded"):
        await task
    assert not ws.exists("reports/final_assessment.json")
    assert (await states.aget("race")).status == LifecycleStatus.ENDING


def test_end_during_inflight_turn_retries_cas_race_and_returns_pending(tmp_path: Path):
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True), llm_client=_llm(), settings=settings)
    headers = {"Authorization": "Bearer " + create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)}
    with TestClient(app) as client:
        created = _create_session(client, headers)
        sid = created["session_id"]
        headers = {"Authorization": "Bearer " + created["session_token"]}
        store = app.state.state_mgr
        state = client.portal.call(store.aget, sid)

        async def activate():
            return await store.atransition(state, state.state_version, LifecycleStatus.STAGE_ACTIVE, current_stage_id="warmup")

        client.portal.call(activate)
        original = store.atransition
        bumped = []

        async def racing(state, expected_version, new_status=None, **updates):
            if new_status == LifecycleStatus.ENDING and not bumped:
                # A turn commits first: same status, newer version.
                bumped.append(await original(state, expected_version, current_turn=state.current_turn + 1))
            return await original(state, expected_version, new_status, **updates)

        async def synthesis_elsewhere(session_id):
            return False

        store.atransition = racing
        app.state.pipeline.finish_ending = synthesis_elsewhere
        response = client.post(f"/api/v1/sessions/{sid}/end", headers=headers)
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "ENDING"
        assert bumped and bumped[0].status == LifecycleStatus.STAGE_ACTIVE


def test_end_returns_5xx_when_ending_intent_was_not_persisted(tmp_path: Path):
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True), llm_client=_llm(), settings=settings)
    headers = {"Authorization": "Bearer " + create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)}
    with TestClient(app) as client:
        created = _create_session(client, headers)
        sid = created["session_id"]
        headers = {"Authorization": "Bearer " + created["session_token"]}
        store = app.state.state_mgr
        before = client.portal.call(store.aget, sid).status

        async def unavailable(*args, **kwargs):
            raise ConnectionError("state store unavailable")

        store.atransition = unavailable
        response = client.post(f"/api/v1/sessions/{sid}/end", headers=headers)
        assert response.status_code == 503, response.text
        assert client.portal.call(store.aget, sid).status == before


@pytest.mark.asyncio
async def test_failing_early_end_report_is_bounded_and_fence_expires(tmp_path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    states = SessionStateManager()
    states.create_session("wedge", initial_status=LifecycleStatus.STAGE_ACTIVE)
    ws = SessionWorkspace(session_id="wedge", root_path=tmp_path / "wedge")
    ws.initialize()
    synthesizer = ReportSynthesizer(redis, lambda sid: ws, state_manager=states)
    pipeline = AssessmentPipeline(redis, states, synthesizer, counter_ttl_seconds=3600, max_end_attempts=2)
    finished = []

    async def broken(sid, **kwargs):
        raise RuntimeError("narrative provider down")

    async def on_finished(sid):
        finished.append(sid)

    synthesizer.synthesize = broken
    pipeline.end_finished = on_finished
    with pytest.raises(RuntimeError):
        await pipeline.end_session("wedge")
    assert 0 < await redis.ttl("session:wedge:work_stopped") <= 3600
    await pipeline.sweep()  # attempt 2 fails
    assert (await states.aget("wedge")).status == LifecycleStatus.ENDING
    await pipeline.sweep()  # attempts exhausted
    state = await states.aget("wedge")
    assert state.status == LifecycleStatus.ASSESSMENT_FAILED
    assert state.metadata["failure_code"] == "EARLY_END_REPORT_FAILED"
    assert finished == ["wedge"]
    assert await pipeline.sweep() == 0


@pytest.mark.asyncio
async def test_lost_synthesis_lease_stops_early_end_work(tmp_path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    states = SessionStateManager()
    states.create_session("lease", initial_status=LifecycleStatus.STAGE_ACTIVE)
    ws = SessionWorkspace(session_id="lease", root_path=tmp_path / "lease")
    ws.initialize()
    synthesizer = ReportSynthesizer(redis, lambda sid: ws, state_manager=states)
    pipeline = AssessmentPipeline(redis, states, synthesizer, synthesis_lock_ttl_seconds=1)
    started, stopped = asyncio.Event(), asyncio.Event()

    async def slow(sid, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    synthesizer.synthesize = slow
    task = asyncio.create_task(pipeline.end_session("lease"))
    await asyncio.wait_for(started.wait(), 2)
    await redis.set("session:lease:synth_lock", "other-process")  # lease expired and was taken over
    assert await asyncio.wait_for(task, 3) is False
    assert stopped.is_set()
    assert await redis.get("session:lease:synth_lock") == "other-process"
    assert (await states.aget("lease")).status == LifecycleStatus.ENDING


@pytest.mark.asyncio
async def test_turn_stopped_by_end_fence_reports_session_ended():
    from types import SimpleNamespace
    from core.interview.controller import InterviewSessionController
    from core.runtime.types import LoopEvent, LoopEventType

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    published = []

    class Publisher:
        async def publish(self, sid, event):
            published.append(event)

    fake = SimpleNamespace(lease=SimpleNamespace(redis=redis), publisher=Publisher(), metrics=None)

    async def events():
        for i in range(3):
            if i == 1:
                await redis.set("session:s:work_stopped", "1")
            yield LoopEvent(session_id="s", turn_id="turn_1", sequence=i, type=LoopEventType.STAGE_TRANSITION, payload={})

    summary = {"status": "failed", "dialogue_completed": False, "turn_number": None}
    await InterviewSessionController._consume(fake, "s", events(), 0.0, summary)
    assert summary["status"] == "ended"
    assert len(published) == 1
    assert InterviewSessionController._ended("turn_1").to_event("s").payload["code"] == "SESSION_ENDED"


@pytest.mark.asyncio
async def test_requeue_after_failed_early_end_retries_the_end_report(tmp_path):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    states = SessionStateManager()
    states.create_session("retry", initial_status=LifecycleStatus.STAGE_ACTIVE)
    ws = SessionWorkspace(session_id="retry", root_path=tmp_path / "retry")
    ws.initialize()
    synthesizer = ReportSynthesizer(redis, lambda sid: ws, state_manager=states)
    pipeline = AssessmentPipeline(redis, states, synthesizer, counter_ttl_seconds=3600, max_end_attempts=1)
    queue = RedisAssessmentQueue(redis)
    calls = []

    async def broken(sid, **kwargs):
        raise RuntimeError("narrative provider down")

    synthesizer.synthesize = broken
    with pytest.raises(RuntimeError):
        await pipeline.end_session("retry")
    await pipeline.sweep()
    assert (await states.aget("retry")).status == LifecycleStatus.ASSESSMENT_FAILED

    async def recovering(sid, **kwargs):
        calls.append(kwargs)

    synthesizer.synthesize = recovering
    assert await pipeline.requeue_failed("retry", queue) == 0
    state = await states.aget("retry")
    # Requeue must not strand the session in ASSESSING behind the work fence.
    assert state.status != LifecycleStatus.ASSESSING
    assert calls == [{"early_end": True}]
    assert state.metadata["end_attempts"] == 1


def test_cancel_fences_other_processes_and_releases_local_markers(tmp_path: Path):
    settings = _settings()
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app = create_app(workspace_base=str(tmp_path), redis_client=redis, llm_client=_llm(), settings=settings)
    headers = {"Authorization": "Bearer " + create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)}
    with TestClient(app) as client:
        created = _create_session(client, headers)
        sid = created["session_id"]
        headers = {"Authorization": "Bearer " + created["session_token"]}
        registry = app.state.controller.cancellations
        response = client.post(f"/api/v1/sessions/{sid}/cancel", headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "CANCELLED"
        assert client.portal.call(redis.exists, f"session:{sid}:work_stopped") == 1
        assert client.portal.call(redis.ttl, f"session:{sid}:work_stopped") > 0
        assert not registry.is_session_cancelled(sid)
        assert sid not in registry._tasks and sid not in registry._tokens
        assert client.post(f"/api/v1/sessions/{sid}/cancel", headers=headers).json()["status"] == "CANCELLED"


@pytest.mark.asyncio
async def test_fence_observed_by_remote_process_is_released_when_idle():
    from core.session.cancellation import CancellationRegistry

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    registry = CancellationRegistry()
    await redis.set("session:remote:work_stopped", "1")

    async def slow():
        await asyncio.sleep(5)

    with pytest.raises(asyncio.CancelledError):
        await registry.run("remote", slow(), redis)
    assert not registry.is_session_cancelled("remote")
    assert "remote" not in registry._tasks
