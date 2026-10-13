"""Simulated candidates through the real app, plus quality scoring (gates G4 directive ratio, G6, G7)."""

import json
from pathlib import Path
from typing import Any

import fakeredis.aioredis
import pytest

from api.auth import create_service_token
from api.main import create_app
from config.settings import get_settings
from core.simulation import CandidatePersona, SimulationRunner, cohen_kappa, mean_absolute_error, score
from core.simulation.quality import main as quality_main
from tests.fakes import ScriptedInterviewLLM

PERSONAS = [
    CandidatePersona(name="strong", default_level=4),
    CandidatePersona(name="weak_design", default_level=4, levels={"be_system_design": 2}),
    CandidatePersona(name="borderline", default_level=3),
    CandidatePersona(name="injector", default_level=4, inject_at_turn=2),
    CandidatePersona(name="wanderer", default_level=3, off_topic_at_turn=3),
]


def _runner(tmp_path: Path) -> SimulationRunner:
    settings = get_settings().model_copy(update={
        "RETRIEVAL_BACKEND": "mock",
        "ANSWER_BURST": 100.0,
        "ORCHESTRATOR_JOIN_MS": 1500,
        "ASSESSOR_BLOCK_MS": 20,
        "ASSESSOR_RECOVERY_INTERVAL_SECONDS": 0.1,
        "REPORT_SWEEP_INTERVAL_SECONDS": 0.1,
    })

    def factory(persona: CandidatePersona) -> Any:
        return create_app(
            workspace_base=str(tmp_path / persona.name),
            redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
            llm_client=ScriptedInterviewLLM(level_for=persona.level_for),
            settings=settings,
        )

    token = create_service_token(issuer="sim", audience="ai-runtime", secret=settings.JWT_SECRET)
    return SimulationRunner(factory, token)


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", PERSONAS, ids=[p.name for p in PERSONAS])
async def test_simulated_candidate_reaches_expected_verdict(tmp_path: Path, persona: CandidatePersona):
    result = await _runner(tmp_path).run(persona)
    assert result.status == "REPORT_READY", result
    assert result.verdict == result.expected_verdict
    assert result.directive_on_time_ratio is not None and result.directive_on_time_ratio >= 0.95
    if persona.name == "strong":
        assert result.verdict == "PASS"
    if persona.name == "weak_design":
        assert result.verdict == "FAIL" and "be_system_design" in result.decision["must_have_failed"]
    if persona.name == "injector":
        assert result.decision["flags"]["injection_suspected_turns"] == ["turn_002"]


@pytest.mark.asyncio
async def test_quality_scoring_over_simulated_sessions(tmp_path: Path):
    runner = _runner(tmp_path)
    golden, decisions = [], {}
    for persona in PERSONAS[:3]:
        result = await runner.run(persona)
        decisions[result.session_id] = result.decision
        golden.append({
            "session_id": result.session_id,
            "gold_verdict": result.expected_verdict,
            "gold_levels": {c["competency_id"]: persona.level_for("", c["competency_id"])
                            for c in result.decision["criteria"] if c["covered"]},
        })
    report = score(golden, decisions)
    assert report.sessions == 3 and report.kappa == 1.0 and report.criterion_mae == 0.0
    assert report.gates["kappa"] and report.gates["criterion_mae"]

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text("\n".join(json.dumps(g) for g in golden), encoding="utf-8")
    assert quality_main(["--golden", str(golden_path), "--decisions-dir", str(tmp_path / "strong")]) in (0, 1)


def test_quality_metric_primitives():
    assert cohen_kappa(["PASS", "FAIL", "PASS"], ["PASS", "FAIL", "PASS"]) == 1.0
    assert cohen_kappa(["PASS", "PASS", "FAIL", "FAIL"], ["PASS", "FAIL", "PASS", "FAIL"]) == 0.0
    assert mean_absolute_error([(3, 4), (2, 2), (None, 1)]) == 0.5
    report = score(
        [{"session_id": "s", "gold_verdict": "PASS", "gold_levels": {"a": 4}}],
        {"s": {"verdict": "FAIL", "criteria": [{"competency_id": "a", "level": 2}], "must_have_uncovered": ["b"]}},
        evaluations={"s": [{"criteria": [{"evidence": [{"quote": "x"}]}], "flags": {"unverified_quotes_dropped": 1}}]},
    )
    assert report.criterion_mae == 2.0 and report.must_have_coverage == 0.0
    assert report.invented_evidence_rate == 0.5 and not report.passed
