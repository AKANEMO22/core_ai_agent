"""Quality metrics against a human-graded golden set (release gate G6).

Golden set: JSONL, one object per session:
    {"session_id": "...", "gold_verdict": "PASS|FAIL|NEEDS_REVIEW", "gold_levels": {"<competency_id>": 1-4}}
Predictions: the `decision.json` produced for each session (path or object).

Usage:
    python -m core.simulation.quality --golden golden.jsonl --decisions-dir data/workspaces
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

GATES = {"kappa_min": 0.6, "mae_max": 0.5, "must_have_coverage_min": 0.95, "invented_evidence_max": 0.0}


def cohen_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Cohen's kappa for two categorical raters over the same items."""
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("raters must label the same non-empty set of items")
    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    counts_a, counts_b = Counter(rater_a), Counter(rater_b)
    expected = sum(counts_a[c] * counts_b[c] for c in set(counts_a) | set(counts_b)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def mean_absolute_error(pairs: Iterable[tuple]) -> Optional[float]:
    """MAE over (predicted, gold) pairs, skipping missing predictions."""
    diffs = [abs(float(p) - float(g)) for p, g in pairs if p is not None and g is not None]
    return round(sum(diffs) / len(diffs), 4) if diffs else None


@dataclass
class QualityReport:
    """Aggregated metrics and gate results."""

    sessions: int
    kappa: Optional[float]
    criterion_mae: Optional[float]
    must_have_coverage: Optional[float]
    invented_evidence_rate: Optional[float]
    gates: Dict[str, bool] = field(default_factory=dict)
    verdict_confusion: Dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(self.gates.values())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


def score(
    golden: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
    evaluations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    gates: Optional[Dict[str, float]] = None,
) -> QualityReport:
    """Compare decisions with the golden set.

    `evaluations` (session -> evaluation payloads) enables the invented-evidence rate: the share of
    model quotes that failed verification (`flags.unverified_quotes_dropped`) among all quotes offered.
    """
    gates = {**GATES, **(gates or {})}
    matched = [g for g in golden if g["session_id"] in decisions]
    gold_verdicts = [g["gold_verdict"] for g in matched]
    pred_verdicts = [decisions[g["session_id"]]["verdict"] for g in matched]
    kappa = cohen_kappa(gold_verdicts, pred_verdicts) if matched else None

    pairs = []
    for g in matched:
        predicted = {c["competency_id"]: c.get("level") for c in decisions[g["session_id"]].get("criteria", [])}
        pairs += [(predicted.get(cid), level) for cid, level in (g.get("gold_levels") or {}).items()]
    mae = mean_absolute_error(pairs)

    coverage_flags = [not decisions[g["session_id"]].get("must_have_uncovered") for g in matched]
    coverage = round(sum(coverage_flags) / len(coverage_flags), 4) if coverage_flags else None

    invented_rate = None
    if evaluations is not None:
        offered = dropped = 0
        for session_evals in evaluations.values():
            for evaluation in session_evals:
                kept = sum(len(c.get("evidence", [])) for c in evaluation.get("criteria", []) or [])
                lost = int((evaluation.get("flags") or {}).get("unverified_quotes_dropped", 0))
                offered += kept + lost
                dropped += lost
        invented_rate = round(dropped / offered, 4) if offered else 0.0

    confusion = Counter(f"{g}->{p}" for g, p in zip(gold_verdicts, pred_verdicts))
    results = {
        "kappa": kappa is not None and kappa >= gates["kappa_min"],
        "criterion_mae": mae is not None and mae <= gates["mae_max"],
        "must_have_coverage": coverage is not None and coverage >= gates["must_have_coverage_min"],
    }
    if invented_rate is not None:
        results["invented_evidence"] = invented_rate <= gates["invented_evidence_max"]
    return QualityReport(
        sessions=len(matched),
        kappa=kappa,
        criterion_mae=mae,
        must_have_coverage=coverage,
        invented_evidence_rate=invented_rate,
        gates=results,
        verdict_confusion=dict(confusion),
    )


def load_golden(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_decisions(root: Path, session_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Find `reports/decision.json` for each session under a workspaces root."""
    from core.workspace.session import normalize_session_id

    found: Dict[str, Dict[str, Any]] = {}
    for session_id in session_ids:
        path = root / normalize_session_id(session_id) / "reports" / "decision.json"
        if path.exists():
            found[session_id] = json.loads(path.read_text(encoding="utf-8"))
    return found


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score decisions against a golden set (gate G6).")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--decisions-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    golden = load_golden(args.golden)
    report = score(golden, load_decisions(args.decisions_dir, [g["session_id"] for g in golden]))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
