"""Criterion coverage: how much verified evidence the interview has gathered per competency."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.workspace.session import SessionWorkspace

# Weight of evidence for a criterion the question did not probe. Such evidence is real but thinner
# (the candidate was not asked to show depth there), so it must not dominate directly probed evidence.
INCIDENTAL_EVIDENCE_WEIGHT = 0.5


def legacy_level(content_score: Any) -> Optional[int]:
    """Map a legacy 1-10 content score onto the 1-4 rubric scale."""
    try:
        score = float(content_score)
    except (TypeError, ValueError):
        return None
    score = max(1.0, min(10.0, score))
    return int(round(1 + (score - 1) / 9 * 3))


@dataclass
class EvidenceItem:
    """One scored observation for a criterion."""

    turn_id: str
    level: int
    confidence: float
    quotes: List[str] = field(default_factory=list)
    rationale: str = ""
    primary: bool = True

    @property
    def weight(self) -> float:
        """Aggregation weight: incidental evidence (not what the question probed) counts less."""
        return self.confidence if self.primary else self.confidence * INCIDENTAL_EVIDENCE_WEIGHT


@dataclass
class CriterionCoverage:
    """Aggregated evidence for one competency."""

    competency_id: str
    name: str
    weight: float
    must_have: bool
    pass_level: int
    evidence: List[EvidenceItem] = field(default_factory=list)
    insufficient_turns: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Evidence weight: summed (relevance-weighted) confidence."""
        return round(sum(e.weight for e in self.evidence), 3)

    @property
    def level(self) -> Optional[float]:
        total = sum(e.weight for e in self.evidence)
        if total <= 0:
            return None
        return round(sum(e.level * e.weight for e in self.evidence) / total, 2)

    def covered(self, min_confidence: float) -> bool:
        return self.confidence >= min_confidence

    def to_dict(self, min_confidence: float) -> Dict[str, Any]:
        data = asdict(self)
        data.update({"confidence": self.confidence, "level": self.level, "covered": self.covered(min_confidence)})
        return data


class CoverageTracker:
    """Builds per-criterion coverage from evaluation artifacts.

    Reads schema 2.0 (per-criterion levels with verified quotes) and legacy 1.0 artifacts
    (two scores for the turn's rubric_target, mapped to levels with confidence 0.5).
    """

    LEGACY_CONFIDENCE = 0.5

    def __init__(self, rubric: Dict[str, Any], min_confidence: float = 0.6) -> None:
        self.rubric = rubric or {}
        self.min_confidence = min_confidence

    def _empty(self) -> Dict[str, CriterionCoverage]:
        return {
            c["competency_id"]: CriterionCoverage(
                competency_id=c["competency_id"],
                name=c.get("name", c["competency_id"]),
                weight=float(c.get("weight", 0.0)),
                must_have=bool(c.get("must_have", False)),
                pass_level=int(c.get("pass_level", 3)),
            )
            for c in self.rubric.get("competencies", [])
            if c.get("competency_id")
        }

    def build(self, evaluations: List[Dict[str, Any]]) -> Dict[str, CriterionCoverage]:
        coverage = self._empty()
        for evaluation in sorted(evaluations, key=lambda e: str(e.get("turn_id", ""))):
            turn_id = str(evaluation.get("turn_id", ""))
            criteria = evaluation.get("criteria")
            if isinstance(criteria, list):
                for item in criteria:
                    cid = item.get("competency_id")
                    if cid not in coverage:
                        continue
                    if item.get("status") != "scored" or item.get("level") is None:
                        coverage[cid].insufficient_turns.append(turn_id)
                        continue
                    quotes = [e.get("quote", "") for e in item.get("evidence", []) if e.get("verified")]
                    coverage[cid].evidence.append(
                        EvidenceItem(
                            turn_id=turn_id,
                            level=int(item["level"]),
                            confidence=float(item.get("confidence", 0.0)),
                            quotes=quotes,
                            rationale=str(item.get("rationale", ""))[:300],
                            primary=bool(item.get("primary", True)),
                        )
                    )
            else:
                level = legacy_level(evaluation.get("content_score"))
                if level is None:
                    continue
                for cid in evaluation.get("rubric_target") or []:
                    if cid in coverage:
                        coverage[cid].evidence.append(
                            EvidenceItem(turn_id=turn_id, level=level, confidence=self.LEGACY_CONFIDENCE)
                        )
        return coverage

    def summary(self, coverage: Dict[str, CriterionCoverage]) -> Dict[str, Any]:
        """Compact, JSON-serializable view for state metadata, blackboard and prompts."""
        return {
            "min_confidence": self.min_confidence,
            "criteria": {cid: c.to_dict(self.min_confidence) for cid, c in coverage.items()},
            "must_have_uncovered": [
                cid for cid, c in coverage.items() if c.must_have and not c.covered(self.min_confidence)
            ],
            "uncovered": [cid for cid, c in coverage.items() if not c.covered(self.min_confidence)],
            # Covered must-haves still below their pass level: the candidate deserves a direct probe.
            "must_have_below_bar": [
                cid
                for cid, c in coverage.items()
                if c.must_have and c.covered(self.min_confidence) and c.level is not None and c.level < c.pass_level
            ],
        }


def load_evaluations(workspace: SessionWorkspace) -> List[Dict[str, Any]]:
    """Read every evaluation artifact payload of a session."""
    evaluations: List[Dict[str, Any]] = []
    for path in sorted(workspace.list_artifacts("evaluations")):
        if path.name.endswith("_eval.json"):
            rel = path.relative_to(workspace.root_path).as_posix()
            try:
                data = workspace.read_json(rel)
            except Exception:
                continue
            evaluations.append(data.get("payload", data))
    return evaluations
