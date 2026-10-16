"""Semantic regressions for shared start guards and elapsed interview budgets."""

import hashlib
import json
from pathlib import Path

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from core.session.state import LifecycleStatus
from tests.integration.test_production_hardening import _create, _headers, _llm, _settings


def require_confirmation(workspace):
    brief = workspace.read_json("context/interview_brief.json")
    brief["status"] = "needs_confirmation"
    workspace.write_artifact("context/interview_brief.json", artifact_type="interview_brief", payload=brief)
    return brief


@pytest.mark.asyncio
async def test_controller_requires_confirmation_and_records_current_digest(tmp_path: Path):
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True), llm_client=_llm(), settings=settings)
    sid, _ = await _create(app, settings)
    workspace = app.state.workspace_mgr.get_or_create_workspace(sid)
    brief = require_confirmation(workspace)
    controller = app.state.controller
    blocked = await controller.start_interview(sid)
    assert blocked.error_code == "PLAN_CONFIRMATION_REQUIRED"
    assert (await app.state.state_mgr.aget(sid)).status == LifecycleStatus.READY
    controller.clock = lambda: 1000.0
    accepted = await controller.start_interview(sid, confirm_limitations=True)
    assert accepted.status == "completed"
    state = await app.state.state_mgr.aget(sid)
    digest = hashlib.sha256(json.dumps(brief, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert state.metadata["plan_confirmation"]["brief_sha256"] == digest
    assert state.metadata["interview_started_at"] == 1000.0
    assert state.metadata["interview_deadline_at"] == 1600.0
    controller.clock = lambda: 1200.0
    assert (await controller.start_interview(sid)).status == "replayed"
    assert (await app.state.state_mgr.aget(sid)).metadata["interview_started_at"] == 1000.0
    await controller.drain()


def test_rest_and_websocket_enforce_same_confirmation(tmp_path: Path):
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True), llm_client=_llm(), settings=settings)
    with TestClient(app) as client:
        created = client.post("/api/v1/sessions", json={"candidate_name": "Tester", "target_role": "Senior Backend Engineer", "duration_minutes": 10}, headers=_headers(settings)).json()
        sid = created["session_id"]
        require_confirmation(app.state.workspace_mgr.get_or_create_workspace(sid))
        assert client.post(f"/api/v1/sessions/{sid}/start", headers=_headers(settings)).status_code == 409
        with client.websocket_connect(f"/ws/sessions/{sid}/stream", headers={"Authorization": f"Bearer {created['session_token']}"}) as ws:
            ws.send_json({"type": "start_interview"})
            frame = ws.receive_json()
            while frame["type"] != "error":
                frame = ws.receive_json()
            assert frame["payload"]["code"] == "PLAN_CONFIRMATION_REQUIRED"
            ws.send_json({"type": "start_interview", "confirm_limitations": "false"})
            frame = ws.receive_json()
            while frame["type"] != "error":
                frame = ws.receive_json()
            assert frame["payload"]["code"] == "INVALID_MESSAGE"
            ws.send_json({"type": "start_interview", "confirm_limitations": True})
            while ws.receive_json()["type"] != "turn_complete":
                pass


@pytest.mark.asyncio
async def test_elapsed_budget_reserves_qa_but_scores_accepted_technical_answer(tmp_path: Path):
    settings = _settings()
    app = create_app(workspace_base=str(tmp_path), redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True), llm_client=_llm(), settings=settings)
    sid, _ = await _create(app, settings)
    controller = app.state.controller
    controller.clock = lambda: 1000.0
    assert (await controller.start_interview(sid)).ok
    runner = await controller.runners.get(sid)
    stages = [
        {"stage_id": "technical", "target_turns": 6, "criteria": ["technical_skill"]},
        {"stage_id": "design", "target_turns": 4},
        {"stage_id": "candidate_questions", "stage_type": "candidate_questions", "planned_minutes": 2, "target_turns": 1,
         "question_candidates": [{"question_id": "candidate_questions_invite", "text": "What questions do you have for us?"}]},
    ]
    runner.stage_director.stages = stages
    runner.stage_director._stages_by_id = {s["stage_id"]: s for s in stages}
    # Loading artifacts must not replace this controlled plan.
    runner.workspace.write_artifact("context/stage_plan.json", artifact_type="stage_plan", payload={"stages": stages})
    state = await app.state.state_mgr.aget(sid)
    await app.state.state_mgr.atransition(state, expected_version=state.state_version, current_stage_id="technical")
    runner.stage_director.clock = lambda: 1490.0
    result = await controller.submit_answer(sid, "deadline-answer", "I use transactional idempotency keys.")
    assert result.ok
    state = await app.state.state_mgr.aget(sid)
    assert state.current_stage_id == "candidate_questions"
    candidate = runner.workspace.read_json("turns/turn_1_candidate.json")
    interviewer = runner.workspace.read_json("turns/turn_1_interviewer.json")
    assert candidate["stage_id"] == "technical"
    assert interviewer["question_id"] == "candidate_questions_invite"
    assert interviewer["stage_id"] == "candidate_questions"
    runner.stage_director.clock = lambda: 2000.0
    result = await controller.submit_answer(sid, "qa-answer", "No additional questions, thank you.")
    assert result.ok
    state = await app.state.state_mgr.aget(sid)
    assert state.status != LifecycleStatus.STAGE_ACTIVE
    await controller.drain()
