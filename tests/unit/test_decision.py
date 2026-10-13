"""Decision engine, narrative verification and report/review API."""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from api.auth import create_service_token
from api.main import create_app
from config.settings import get_settings
from core.agents.narrative import NarrativeWriter
from core.assessor.decision import FAIL, NEEDS_REVIEW, PASS, DecisionConfig, compute_decision, decision_json
from core.runtime.providers.mock import MockLLMClient
from tests.fakes import ScriptedInterviewLLM

RUBRIC = {
    "rubric_version": "r1",
    "pass_threshold": 3.0,
    "competencies": [
        {"competency_id": "design", "name": "Design", "weight": 0.5, "must_have": True, "pass_level": 3},
        {"competency_id": "data", "name": "Data", "weight": 0.3, "must_have": False, "pass_level": 3},
        {"competency_id": "comms", "name": "Comms", "weight": 0.2, "must_have": False, "pass_level": 3},
    ],
}


def _eval(turn: str, levels: Dict[str, Any], confidence: float = 0.8) -> Dict[str, Any]:
    criteria = []
    for cid, level in levels.items():
        if level is None:
            criteria.append({"competency_id": cid, "status": "insufficient_evidence"})
        else:
            criteria.append({
                "competency_id": cid, "status": "scored", "level": level, "confidence": confidence,
                "evidence": [{"quote": f"quote for {cid} in {turn}", "verified": True}],
            })
    return {"turn_id": turn, "criteria": criteria, "content_score": 7, "communication_score": 7, "evaluated_at": time.time()}


@pytest.mark.parametrize(
    "evaluations, verdict, reason_fragment",
    [
        ([_eval("t1", {"design": 4, "data": 4, "comms": 4})], PASS, "≥ threshold"),
        ([_eval("t1", {"design": 2, "data": 4, "comms": 4})], FAIL, "must-have below"),
        ([_eval("t1", {"design": None, "data": 4, "comms": 4})], NEEDS_REVIEW, "insufficient evidence for must-have"),
        ([_eval("t1", {"design": 3, "data": 3, "comms": 3})], NEEDS_REVIEW, "within ±0.25"),
        ([_eval("t1", {"design": 3, "data": 2, "comms": 1})], FAIL, "< threshold"),
        ([], NEEDS_REVIEW, "insufficient evidence for must-have"),
    ],
)
def test_decision_rules(evaluations: List[Dict[str, Any]], verdict: str, reason_fragment: str):
    decision = compute_decision(RUBRIC, evaluations, DecisionConfig())
    assert decision["verdict"] == verdict
    assert any(reason_fragment in r for r in decision["verdict_reasons"])
    assert decision["requires_human_review"] is (verdict == NEEDS_REVIEW)


def test_low_confidence_evidence_is_not_coverage():
    decision = compute_decision(RUBRIC, [_eval("t1", {"design": 4, "data": 4, "comms": 4}, confidence=0.3)], DecisionConfig())
    assert decision["verdict"] == NEEDS_REVIEW
    assert set(decision["uncovered_criteria"]) == {"design", "data", "comms"}


def test_repeated_unverified_scores_do_not_become_coverage():
    rubric = {
        "pass_threshold": 3.0,
        "competencies": [
            {"competency_id": "design", "weight": 1.0, "must_have": False, "pass_level": 3},
        ],
    }
    evaluations = [
        {
            "turn_id": f"t{i}",
            "criteria": [{
                "competency_id": "design",
                "status": "scored",
                "level": 4,
                "confidence": 0.45,
                "evidence": [],
            }],
        }
        for i in (1, 2)
    ]

    decision = compute_decision(rubric, evaluations, DecisionConfig())

    assert decision["verdict"] == NEEDS_REVIEW
    assert decision["uncovered_criteria"] == ["design"]


def test_decision_is_byte_identical_for_same_inputs_in_any_order():
    evaluations = [_eval(f"turn_{i:03d}", {"design": 3 + i % 2, "data": 3, "comms": 4}) for i in range(1, 6)]
    versions = {"code": "abc", "rubric_version": "r1"}
    first = decision_json(compute_decision(RUBRIC, evaluations, DecisionConfig(), versions))
    shuffled = list(evaluations)
    random.Random(7).shuffle(shuffled)
    for e in shuffled:
        e["evaluated_at"] = time.time() + 100  # timestamps must not matter
    second = decision_json(compute_decision(RUBRIC, shuffled, DecisionConfig(), dict(reversed(list(versions.items())))))
    assert first == second
    changed = [dict(e) for e in evaluations]
    changed[0] = _eval("turn_001", {"design": 1, "data": 3, "comms": 4})
    assert decision_json(compute_decision(RUBRIC, changed, DecisionConfig(), versions)) != first


def test_hiring_assist_requires_review_when_injection_flagged():
    flagged = _eval("t1", {"design": 4, "data": 4, "comms": 4})
    flagged["flags"] = {"injection_suspected": True}
    practice = compute_decision(RUBRIC, [flagged], DecisionConfig(product_mode="practice"))
    hiring = compute_decision(RUBRIC, [flagged], DecisionConfig(product_mode="hiring_assist"))
    assert practice["verdict"] == hiring["verdict"] == PASS
    assert practice["requires_human_review"] is False and hiring["requires_human_review"] is True
    assert hiring["flags"]["injection_suspected_turns"] == ["t1"]


@pytest.mark.asyncio
async def test_narrative_removes_unverifiable_citations_and_falls_back():
    decision = compute_decision(RUBRIC, [_eval("t1", {"design": 4, "data": 4, "comms": 4})], DecisionConfig())

    def narrative(prompt: str) -> Dict[str, Any]:
        return {
            "summary": "Strong design reasoning overall.",
            "criterion_findings": [{"competency_id": "design", "finding": "Clear trade-offs.",
                                    "quotes": ["we shard by merchant id", "I invented this quote entirely"]}],
            "strengths": ["Trade-offs"], "improvements": ["Numbers"], "study_plan": ["Capacity math"],
        }

    writer = NarrativeWriter(ScriptedInterviewLLM(overrides={"report_narrative": narrative}))
    result = await writer.write(decision, [], ["Honestly we shard by merchant id and replicate reads."])
    assert result["source"] == "model" and result["citations_removed"] == 1
    assert result["criterion_findings"][0]["quotes"] == ["we shard by merchant id"]

    fallback = await NarrativeWriter(MockLLMClient(canned_tokens=["no"])).write(decision, [], [])
    assert fallback["source"] == "template" and "PASS" in fallback["summary"]


def _run_interview(tmp_path: Path, product_mode: str, level: int) -> Any:
    settings = get_settings().model_copy(update={
        "RETRIEVAL_BACKEND": "mock",
        "ASSESSOR_BLOCK_MS": 50, "ASSESSOR_RECOVERY_INTERVAL_SECONDS": 0.2, "REPORT_SWEEP_INTERVAL_SECONDS": 0.2,
        "ANSWER_BURST": 100.0, "ORCHESTRATOR_JOIN_MS": 300, "PRODUCT_MODE": product_mode,
    })
    app = create_app(
        workspace_base=str(tmp_path / "ws"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=ScriptedInterviewLLM(level_for=lambda answer, cid: level),
        settings=settings,
    )
    headers = {"Authorization": f"Bearer {create_service_token(issuer='backend', audience='ai-runtime', secret=settings.JWT_SECRET)}"}
    return app, settings, headers


@pytest.mark.parametrize("product_mode, level, verdict", [("practice", 4, PASS), ("hiring_assist", 1, FAIL)])
def test_report_api_exposes_verdict_decision_and_review(tmp_path: Path, product_mode: str, level: int, verdict: str):
    app, settings, headers = _run_interview(tmp_path, product_mode, level)
    with TestClient(app) as client:
        created = client.post("/api/v1/sessions", json={
            "candidate_name": "R", "target_company": "Acme Systems", "target_role": "Senior Backend Engineer",
        }, headers=headers).json()
        sid = created["session_id"]
        with client.websocket_connect(f"/ws/sessions/{sid}/stream", headers={"Authorization": f"Bearer {created['session_token']}"}) as ws:
            ws.send_json({"type": "start_interview"})
            while ws.receive_json()["type"] != "turn_complete":
                pass
            for i in range(1, 6):
                ws.send_json({"type": "candidate_answer", "message_id": f"m{i}",
                              "candidate_answer": f"Answer number {i}: we partition by tenant, cache hot keys and measure p99 latency."})
                frame = ws.receive_json()
                while frame["type"] not in ("turn_complete", "error"):
                    frame = ws.receive_json()
                assert frame["type"] == "turn_complete"
                if frame["payload"]["stage_id"] is None:
                    break
        deadline = time.time() + 15
        while time.time() < deadline:
            res = client.get(f"/api/v1/sessions/{sid}/report", headers=headers)
            if res.status_code == 200:
                break
            time.sleep(0.1)
        body = res.json()
        assert res.status_code == 200, body
        assert body["verdict"] == verdict and body["decision"]["verdict"] == verdict
        assert "Verdict" in body["report_markdown"]
        if product_mode == "hiring_assist":
            assert "support, not replace" in body["final_assessment"]["disclaimer"]
        else:
            assert "coaching feedback" in body["final_assessment"]["disclaimer"]

        ws_dir = app.state.workspace_mgr.get_or_create_workspace(sid)
        raw = ws_dir.read_text("reports/decision.json")
        assert json.loads(raw)["input_digest"] and raw == json.dumps(json.loads(raw), sort_keys=True, indent=2, ensure_ascii=False) + "\n"

        review = client.post(f"/api/v1/sessions/{sid}/review", json={"reviewer_id": "hr-1", "final_verdict": "PASS"}, headers=headers)
        assert review.status_code == 200 and review.json()["decision"]["review"]["computed_verdict"] == verdict
        after = client.get(f"/api/v1/sessions/{sid}/report", headers=headers).json()
        assert after["verdict"] == "PASS" and after["decision"]["verdict"] == verdict
        assert ws_dir.read_text("reports/decision.json") == raw  # computed decision stays immutable
