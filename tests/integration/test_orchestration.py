"""Integration tests for the control plane: brief, speaker context, orchestrator cycles, gates, summary."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from api.auth import create_service_token
from api.main import create_app
from config.settings import Settings, get_settings
from core.orchestration.brief import BRIEF_PATH, InterviewBriefWriter
from core.orchestration.coverage import CoverageTracker
from core.orchestration.directive import Directive, plan_fallback
from core.orchestration.notebook import NOTEBOOK_PATH
from core.runtime.providers.mock import MockLLMClient
from core.runtime.types import LoopEventType
from tests.fakes import ScriptedInterviewLLM

ANSWERS = [
    "I lead the payments platform and own the ledger service end to end.",
    "We guarantee exactly once payments with idempotency keys stored in the same transaction.",
    "To avoid stampedes we use request coalescing plus jittered expirations on hot keys.",
    "I would shard by merchant id and add read replicas behind a regional cache tier.",
    "What does the on call rotation look like for this team?",
]


def _settings(**overrides: Any) -> Settings:
    base = {"RETRIEVAL_BACKEND": "mock", "ASSESSOR_POOL_SIZE": 0, "ANSWER_BURST": 100.0, "ORCHESTRATOR_JOIN_MS": 300}
    base.update(overrides)
    return get_settings().model_copy(update=base)


async def _app_with_session(tmp_path: Path, llm: Any, settings: Settings) -> Tuple[Any, str]:
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=llm,
        settings=settings,
    )
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        res = await client.post(
            "/api/v1/sessions",
            json={"candidate_name": "Orchestrated", "target_company": "Acme Systems", "target_role": "Senior Backend Engineer"},
            headers={"Authorization": f"Bearer {token}"},
        )
    return app, res.json()["session_id"]


@pytest.mark.asyncio
async def test_brief_is_written_after_profiling_and_status_exposed(tmp_path: Path):
    app, session_id = await _app_with_session(tmp_path, ScriptedInterviewLLM(), _settings())
    state = await app.state.state_mgr.aget(session_id)
    assert state.metadata["brief_status"] == "ok"
    ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    brief = ws.read_text(BRIEF_PATH)
    for section in ("## Interview frame", "## Stages", "## CV points to probe", "## Risks"):
        assert section in brief
    assert "level 3 looks like" in brief


@pytest.mark.asyncio
async def test_brief_falls_back_and_never_leaks_expected_points(tmp_path: Path):
    app, session_id = await _app_with_session(tmp_path, ScriptedInterviewLLM(), _settings(BRIEF_ENABLED=False))
    ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    knowledge = ws.read_json("context/retrieved_knowledge.json")
    point = next(p for qs in knowledge["question_bank"].values() for q in qs for p in q["expected_points"] if len(p) > 12)

    def leaking(prompt: str) -> Dict[str, Any]:
        return {"frame": f"Make sure they mention {point}.", "persona_notes": "x", "cv_probes": [], "risks": []}

    writer = InterviewBriefWriter(ScriptedInterviewLLM(overrides={"interview_brief": leaking}))
    assert await writer.write(ws) == "fallback"
    assert point.lower() not in ws.read_text(BRIEF_PATH).lower()
    assert await InterviewBriefWriter(MockLLMClient(canned_tokens=["no json"])).write(ws) == "fallback"


@pytest.mark.asyncio
async def test_orchestrated_turns_use_directives_with_stable_speaker_prefix(tmp_path: Path):
    captured: List[List[Dict[str, Any]]] = []

    class Spy(ScriptedInterviewLLM):
        async def stream_chat(self, messages, tools=None, **kwargs):  # type: ignore[override]
            if not any("Respond with JSON only for `" in str(m.get("content", "")) for m in messages if m.get("role") == "user"):
                captured.append({"messages": messages, "tools": tools})  # speaker calls only
            async for chunk in super().stream_chat(messages, tools=tools, **kwargs):
                yield chunk

    app, session_id = await _app_with_session(tmp_path, Spy(), _settings())
    controller = app.state.controller
    assert (await controller.start_interview(session_id)).status == "completed"
    for i, answer in enumerate(ANSWERS[:3], start=1):
        result = await controller.submit_answer(session_id, f"m{i}", answer)
        assert result.status == "completed"
        await controller.drain()

    ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    opening = ws.read_json("turns/turn_0_interviewer.json")
    assert opening["directive_source"] == "fallback" and opening["question_id"]
    later = [ws.read_json(f"turns/turn_{i}_interviewer.json") for i in (2, 3)]
    assert all(t["directive_source"] == "orchestrator" for t in later)
    assert all(t["playbook_ref"] for t in later)

    # Speaker: no tools, directive present, stable system prefix within a stage
    speaker_calls = captured
    assert all(call["tools"] is None for call in speaker_calls)
    assert any("# Directive for this turn" in m["content"] for m in speaker_calls[-1]["messages"] if m["role"] == "system")
    technical_prefixes = {
        call["messages"][0]["content"] for call in speaker_calls[2:4]
    }
    assert len(technical_prefixes) == 1  # turns 2 and 3 are both in the technical stage

    # Orchestrator artifacts and events
    assert ws.exists("orchestration/directive_4.json")
    notebook = ws.read_text(NOTEBOOK_PATH)
    for heading in ("## 3. Coverage", "## 5. Observations", "## 7. Evidence log"):
        assert heading in notebook
    events = await app.state.event_store.get_events_async(session_id)
    types = {e.type for e in events}
    assert LoopEventType.DIRECTIVE_READY in types and LoopEventType.DELEGATION_COMPLETED in types
    directive_events = [e for e in events if e.type == LoopEventType.DIRECTIVE_READY]
    assert all("question_text" not in e.payload for e in directive_events)  # never leak the next question

    blackboard = await app.state.blackboard.snapshot(session_id)
    assert "coverage" in blackboard and blackboard["last_cycle"]["source"] == "orchestrator"
    assert app.state.metrics.counter_value("directive_ready_total", {"source": "orchestrator", "on_time": "true"}) >= 1


@pytest.mark.asyncio
async def test_orchestrator_falls_back_when_decision_invalid(tmp_path: Path):
    llm = ScriptedInterviewLLM(overrides={"orchestrator_decision": lambda prompt: {"nonsense": True}})
    app, session_id = await _app_with_session(tmp_path, llm, _settings())
    await app.state.controller.start_interview(session_id)
    await app.state.controller.submit_answer(session_id, "m1", ANSWERS[0])
    await app.state.controller.drain()
    ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    directive = ws.read_json("orchestration/directive_2.json")
    assert directive["generated_by"] == "fallback" and directive["degraded"] is True


@pytest.mark.asyncio
async def test_coverage_gate_blocks_premature_transition(tmp_path: Path):
    def eager(prompt: str) -> Dict[str, Any]:
        return {
            "directive": {"mode": "transition", "question_text": "Let us move to system design.", "reason": "bored"},
            "complete_stage": True,
        }

    app, session_id = await _app_with_session(
        tmp_path, ScriptedInterviewLLM(overrides={"orchestrator_decision": eager}), _settings()
    )
    controller = app.state.controller
    await controller.start_interview(session_id)
    await controller.submit_answer(session_id, "m1", ANSWERS[0])  # warmup (1 turn) -> technical
    await controller.drain()
    ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    gated = ws.read_json("orchestration/directive_2.json")
    assert gated["mode"] == "next_question"
    assert gated["reason"].startswith("coverage gate")


def test_budget_fallback_transitions_and_wraps_up():
    stages = [
        {"stage_id": "a", "target_turns": 2, "question_candidates": [{"question_id": "qa", "text": "A?"}]},
        {"stage_id": "b", "target_turns": 1, "question_candidates": [{"question_id": "qb", "text": "B?"}]},
    ]
    assert plan_fallback(stages, "a", 0, set()).mode == "next_question"
    transition = plan_fallback(stages, "a", 1, set())
    assert transition.mode == "transition" and transition.question_id == "qb"
    wrap = plan_fallback(stages, "b", 0, set())
    assert wrap.mode == "wrap_up" and "Do not ask another question" in wrap.speaker_instruction()


@pytest.mark.asyncio
async def test_rolling_summary_replaces_old_turns_in_speaker_context(tmp_path: Path):
    settings = _settings(SPEAKER_RECENT_MESSAGES=2)
    app, session_id = await _app_with_session(tmp_path, ScriptedInterviewLLM(), settings)
    controller = app.state.controller
    await controller.start_interview(session_id)
    for i, answer in enumerate(ANSWERS[:3], start=1):
        await controller.submit_answer(session_id, f"m{i}", answer)
        await controller.drain()
    ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    summary = ws.read_json("turns/rolling_summary.json")
    assert summary["last_turn_included"] == 3

    runner = await app.state.runners.get(session_id)
    state = await app.state.state_mgr.aget(session_id)
    messages, provenance = runner.build_messages(state, 4, "next answer")
    assert provenance["summary_included"] is True
    joined = " ".join(str(m["content"]) for m in messages)
    assert "Earlier in this interview" in joined
    assert ANSWERS[0] not in joined  # covered by the summary, not repeated raw


def test_coverage_tracker_reads_v2_and_legacy():
    rubric = {"competencies": [
        {"competency_id": "a", "name": "A", "weight": 0.5, "must_have": True, "pass_level": 3},
        {"competency_id": "b", "name": "B", "weight": 0.5, "must_have": False, "pass_level": 3},
    ]}
    tracker = CoverageTracker(rubric, min_confidence=0.6)
    coverage = tracker.build([
        {"turn_id": "turn_001", "criteria": [
            {"competency_id": "a", "status": "scored", "level": 3, "confidence": 0.4,
             "evidence": [{"quote": "x y z", "verified": True}]},
            {"competency_id": "b", "status": "insufficient_evidence"},
        ]},
        {"turn_id": "turn_002", "rubric_target": ["a"], "content_score": 10},
    ])
    assert coverage["a"].confidence == 0.9 and coverage["a"].level == pytest.approx((3 * 0.4 + 4 * 0.5) / 0.9, 0.01)
    summary = tracker.summary(coverage)
    assert summary["must_have_uncovered"] == [] and summary["uncovered"] == ["b"]
