"""Interview-quality rules: evidence relevance, borderline must-haves, stage extension, quotes, hedging."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from core.agents.quotes import verify_quote
from core.assessor.decision import FAIL, NEEDS_REVIEW, DecisionConfig, compute_decision
from core.interview.director import StageDirector
from core.orchestration.coverage import CoverageTracker
from core.orchestration.orchestrator import InterviewOrchestrator
from core.runtime.client import ProviderCapabilities
from core.runtime.priority import HedgedLLMClient
from core.runtime.types import IterationChunk
from core.session.state import LifecycleStatus, SessionStateManager

RUBRIC = {
    "pass_threshold": 3.0,
    "competencies": [
        {"competency_id": "design", "name": "System Design", "weight": 0.6, "must_have": True, "pass_level": 3},
        {"competency_id": "comms", "name": "Comms", "weight": 0.4, "must_have": False, "pass_level": 3},
    ],
}


def _criterion(cid: str, level: int, confidence: float, primary: Optional[bool] = None) -> Dict[str, Any]:
    item = {
        "competency_id": cid,
        "status": "scored",
        "level": level,
        "confidence": confidence,
        "evidence": [{"quote": f"verified evidence for {cid}", "verified": True}],
    }
    if primary is not None:
        item["primary"] = primary
    return item


# ---------------------------------------------------------------- evidence relevance


def test_incidental_evidence_weighs_half():
    evaluations = [
        {"turn_id": "t1", "criteria": [_criterion("design", 3, 0.8, primary=True)]},
        {"turn_id": "t2", "criteria": [_criterion("design", 2, 0.8, primary=False)]},
    ]
    item = CoverageTracker(RUBRIC).build(evaluations)["design"]
    # (3*0.8 + 2*0.4) / 1.2
    assert item.level == pytest.approx(2.67, abs=0.01)
    assert item.confidence == pytest.approx(1.2)


def test_evidence_without_primary_flag_counts_fully():
    evaluations = [
        {"turn_id": "t1", "criteria": [_criterion("design", 3, 0.8)]},
        {"turn_id": "t2", "criteria": [_criterion("design", 2, 0.8)]},
    ]
    assert CoverageTracker(RUBRIC).build(evaluations)["design"].level == 2.5


# ---------------------------------------------------------------- decision


def test_must_have_just_below_pass_level_needs_review():
    evaluations = [{"turn_id": "t1", "criteria": [_criterion("design", 3, 0.8), _criterion("comms", 4, 0.8)]},
                   {"turn_id": "t2", "criteria": [_criterion("design", 2, 0.2)]}]
    decision = compute_decision(RUBRIC, evaluations, DecisionConfig())
    assert decision["verdict"] == NEEDS_REVIEW
    assert decision["must_have_borderline"] == ["design"]
    assert any("of pass level" in r for r in decision["verdict_reasons"])


def test_must_have_clearly_below_pass_level_fails():
    evaluations = [{"turn_id": "t1", "criteria": [_criterion("design", 2, 0.8), _criterion("comms", 4, 0.8)]}]
    decision = compute_decision(RUBRIC, evaluations, DecisionConfig())
    assert decision["verdict"] == FAIL
    assert decision["must_have_failed"] == ["design"]


# ---------------------------------------------------------------- stage extension


@pytest.mark.asyncio
async def test_director_extends_stage_only_when_asked_and_within_limit():
    state_mgr = SessionStateManager()
    director = StageDirector(stages=[{"stage_id": "s1", "target_turns": 2}, {"stage_id": "s2"}],
                             state_mgr=state_mgr, max_extra_turns=1)
    state = state_mgr.create_session("sess_ext", initial_status=LifecycleStatus.STAGE_ACTIVE)
    state = await director.record_turn(state)
    state = await director.record_turn(state, extend_stage=True)  # budget reached, extended
    assert state.is_stage_complete is False
    state = await director.record_turn(state, extend_stage=True)  # hard limit (2 + 1) reached
    assert state.is_stage_complete is True


def _orchestrator() -> InterviewOrchestrator:
    return InterviewOrchestrator(
        llm=None, state_store=None, workspace_resolver=None, delegation=None, blackboard=None, max_extra_turns=1
    )


STAGE = {"stage_id": "s1", "target_turns": 2, "criteria": ["design", "comms"]}
NEXT_STAGE = {"stage_id": "s2", "target_turns": 2, "criteria": ["comms"]}
POOL = [
    {"question_id": "q_design_2", "competency_id": "design", "text": "How would you shard it?", "source": "current_stage"},
    {"question_id": "q_next", "competency_id": "comms", "text": "Tell me about a conflict.", "source": "next_stage"},
]


def _decision(mode: str, target: Optional[str] = None) -> Dict[str, Any]:
    return {"directive": {"mode": mode, "question_text": "Tell me about a conflict.", "reason": "r",
                          "target_criterion": target}, "complete_stage": mode == "transition"}


def test_gate_extends_stage_for_must_have_below_bar():
    coverage = {"must_have_uncovered": [], "must_have_below_bar": ["design"]}
    directive = _orchestrator()._gate(_decision("transition", "comms"), STAGE, NEXT_STAGE, True, coverage,
                                      RUBRIC, POOL, turns_used=1)
    assert directive.extend_stage is True
    assert directive.mode == "next_question"
    assert directive.target_criterion == "design"
    assert directive.question_id == "q_design_2"


def test_gate_transitions_when_extension_exhausted():
    coverage = {"must_have_uncovered": ["design"], "must_have_below_bar": []}
    directive = _orchestrator()._gate(_decision("transition", "comms"), STAGE, NEXT_STAGE, True, coverage,
                                      RUBRIC, POOL, turns_used=2)
    assert directive.extend_stage is False
    assert directive.mode == "transition"


def test_gate_transitions_when_must_haves_are_met():
    coverage = {"must_have_uncovered": [], "must_have_below_bar": []}
    directive = _orchestrator()._gate(_decision("next_question", "comms"), STAGE, NEXT_STAGE, True, coverage,
                                      RUBRIC, POOL, turns_used=1)
    assert directive.extend_stage is False
    assert directive.mode == "transition"


# ---------------------------------------------------------------- quotes


def test_quote_verification_ignores_spoken_fillers():
    source = "we needed to make sure we weren't, like, losing any events, and ordering was, you know, important"
    assert verify_quote("we needed to make sure we weren't losing any events", source)
    assert verify_quote("ordering was important", source)
    assert not verify_quote("ordering was not important", source)


@pytest.mark.parametrize(
    "quote, ok",
    [
        ("I owned the design and the migration", True),
        ("I owned the design ... cut over without downtime", True),
        ("I owned the design … cut over without downtime", True),
        ("cut over without downtime ... I owned the design", False),  # out of order
        ("I owned ... downtime", False),  # one-word segment cannot be stitched
        ("I designed the whole platform", False),
    ],
)
def test_quote_verification_with_ellipsis(quote: str, ok: bool):
    source = "So I owned the design and the migration, and I wrote the bridge so we could cut over without downtime."
    assert verify_quote(quote, source) is ok


# ---------------------------------------------------------------- hedging


class _LatencyLLM:
    """Each call's first chunk arrives after the next delay in `delays`."""

    def __init__(self, delays: List[float]) -> None:
        self.delays = list(delays)
        self.calls = 0
        self.closed = 0
        self.capabilities = ProviderCapabilities(streaming=True, usage_events=True, cancellation=True)

    async def stream_chat(self, messages: Any, tools: Any = None, **kwargs: Any) -> AsyncIterator[IterationChunk]:
        index = self.calls
        self.calls += 1
        try:
            await asyncio.sleep(self.delays[index])
            yield IterationChunk(token=f"call{index}-a")
            yield IterationChunk(token=f"call{index}-b")
        finally:
            self.closed += 1


async def _collect(client: HedgedLLMClient) -> List[str]:
    return [c.token async for c in client.stream_chat(messages=[{"role": "user", "content": "hi"}])]


@pytest.mark.asyncio
async def test_hedge_not_started_when_primary_is_fast():
    inner = _LatencyLLM([0.0, 0.0])
    assert await _collect(HedgedLLMClient(inner, hedge_after_seconds=0.2)) == ["call0-a", "call0-b"]
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_hedge_wins_when_primary_stalls_and_loser_is_closed():
    inner = _LatencyLLM([5.0, 0.0])
    chunks = await asyncio.wait_for(_collect(HedgedLLMClient(inner, hedge_after_seconds=0.05)), timeout=2)
    assert chunks == ["call1-a", "call1-b"]
    assert inner.calls == 2
    assert inner.closed == 2  # stalled primary cancelled and closed, winner finished
