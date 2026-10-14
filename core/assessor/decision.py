"""Deterministic decision engine: verdict computed in code from verified, structured evidence.

Rules (in order):
    1. A must-have criterion covered but below its pass level by more
       than borderline_band                                                 -> FAIL
    2. A must-have criterion without enough evidence (not covered), or
       below its pass level by at most borderline_band                      -> NEEDS_REVIEW
    3. No covered criterion at all                                          -> NEEDS_REVIEW
    4. |weighted_score - pass_threshold| <= borderline_band                 -> NEEDS_REVIEW
    5. weighted_score >= pass_threshold                                     -> PASS, else FAIL

`weighted_score` is the weight-renormalized mean of final levels over covered criteria, where a
criterion's final level is the confidence-weighted mean of its evidence levels.

The output is a pure function of its inputs (no timestamps, sorted keys), so identical evaluations,
rubric and versions always produce a byte-identical `decision.json`.


Tính kết luận bằng quy tắc Python từ rubric và các đánh giá có bằng chứng.
ReportSynthesizer gọi compute_decision: CoverageTracker gộp bằng chứng, sau đó
xét tiêu chí bắt buộc, vùng sát ngưỡng và điểm trung bình có trọng số.
Trả decision cùng digest đầu vào; decision_json chuẩn hóa thứ tự để lưu/tái kiểm tra.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.orchestration.coverage import CoverageTracker

PASS, FAIL, NEEDS_REVIEW = "PASS", "FAIL", "NEEDS_REVIEW"


@dataclass(frozen=True)
class DecisionConfig:
    """Decision parameters."""

    min_confidence: float = 0.6
    borderline_band: float = 0.25
    product_mode: str = "practice"


@lru_cache(maxsize=1)
def code_version() -> str:
    """Build identifier: BUILD_SHA env var, else the git HEAD, else 'unknown'."""
    env = os.getenv("BUILD_SHA")
    if env:
        return env.strip()[:40]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return out.stdout.strip()[:40]
    except Exception:
        pass
    return "unknown"


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)


def compute_decision(
    rubric: Dict[str, Any],
    evaluations: List[Dict[str, Any]],
    config: DecisionConfig,
    versions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the verdict and per-criterion detail. Pure and deterministic."""
    # CoverageTracker gộp mức/confidence từ nhiều lượt; decision chỉ áp dụng luật kết luận.
    tracker = CoverageTracker(rubric, config.min_confidence)
    coverage = tracker.build(evaluations)
    pass_threshold = float(rubric.get("pass_threshold", 3.0))

    criteria: List[Dict[str, Any]] = []
    must_have_failed: List[str] = []
    must_have_uncovered: List[str] = []
    must_have_borderline: List[str] = []
    uncovered: List[str] = []
    weighted_sum = 0.0
    weight_total = 0.0
    # Must-have được xét trước điểm trung bình; điểm cao ở tiêu chí khác không thể bù cho tiêu chí bắt buộc thiếu.
    for cid in sorted(coverage):
        item = coverage[cid]
        covered = item.covered(config.min_confidence)
        level = item.level
        meets_bar = covered and level is not None and level >= item.pass_level
        if not covered:
            uncovered.append(cid)
            if item.must_have:
                must_have_uncovered.append(cid)
        elif item.must_have and not meets_bar:
            if level is not None and item.pass_level - level <= config.borderline_band:
                must_have_borderline.append(cid)
            else:
                must_have_failed.append(cid)
        if covered and level is not None and item.weight > 0:
            weighted_sum += level * item.weight
            weight_total += item.weight
        quotes = [q for e in item.evidence for q in e.quotes][:3]
        criteria.append(
            {
                "competency_id": cid,
                "name": item.name,
                "weight": _round(item.weight),
                "must_have": item.must_have,
                "pass_level": item.pass_level,
                "level": _round(level),
                "confidence": _round(item.confidence),
                "covered": covered,
                "meets_bar": meets_bar,
                "evidence_count": len(item.evidence),
                "evidence_turns": sorted({e.turn_id for e in item.evidence}),
                "quotes": quotes,
            }
        )

    # Mẫu số chỉ gồm trọng số của tiêu chí đủ bằng chứng, không coi thiếu dữ liệu là điểm 0.
    weighted_score = _round(weighted_sum / weight_total) if weight_total > 0 else None
    reasons: List[str] = []
    if must_have_failed:
        verdict = FAIL
        reasons.append("must-have below pass level: " + ", ".join(must_have_failed))
    elif must_have_uncovered:
        verdict = NEEDS_REVIEW
        reasons.append("insufficient evidence for must-have: " + ", ".join(must_have_uncovered))
    elif must_have_borderline:
        verdict = NEEDS_REVIEW
        reasons.append(
            f"must-have within ±{config.borderline_band} of pass level: " + ", ".join(must_have_borderline)
        )
    elif weighted_score is None:
        verdict = NEEDS_REVIEW
        reasons.append("no criterion has enough evidence")
    elif abs(weighted_score - pass_threshold) <= config.borderline_band:
        verdict = NEEDS_REVIEW
        reasons.append(
            f"weighted score {weighted_score} within ±{config.borderline_band} of threshold {pass_threshold}"
        )
    elif weighted_score >= pass_threshold:
        verdict = PASS
        reasons.append(f"weighted score {weighted_score} ≥ threshold {pass_threshold}")
    else:
        verdict = FAIL
        reasons.append(f"weighted score {weighted_score} < threshold {pass_threshold}")

    flagged_injection = sorted(
        str(e.get("turn_id")) for e in evaluations if (e.get("flags") or {}).get("injection_suspected")
    )
    flagged_off_topic = sorted(
        str(e.get("turn_id")) for e in evaluations if (e.get("flags") or {}).get("off_topic")
    )
    requires_review = verdict == NEEDS_REVIEW or (config.product_mode == "hiring_assist" and bool(flagged_injection))

    # Bỏ timestamp xử lý khỏi fingerprint để cùng dữ liệu chấm cho cùng digest.
    canonical_inputs = json.dumps(
        {
            "rubric": rubric,
            "evaluations": sorted(
                [{k: v for k, v in e.items() if k not in ("evaluated_at", "created_at")} for e in evaluations],
                key=lambda e: str(e.get("turn_id")),
            ),
            "config": config.__dict__,
        },
        sort_keys=True,
        default=str,
    )
    return {
        "schema_version": "1.0",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "requires_human_review": requires_review,
        "product_mode": config.product_mode,
        "weighted_score": weighted_score,
        "pass_threshold": pass_threshold,
        "borderline_band": config.borderline_band,
        "min_confidence": config.min_confidence,
        "scale": {"min": 1, "max": 4},
        "criteria": criteria,
        "uncovered_criteria": uncovered,
        "must_have_failed": must_have_failed,
        "must_have_uncovered": must_have_uncovered,
        "must_have_borderline": must_have_borderline,
        "flags": {"injection_suspected_turns": flagged_injection, "off_topic_turns": flagged_off_topic},
        "versions": dict(sorted((versions or {}).items())),
        "input_digest": hashlib.sha256(canonical_inputs.encode("utf-8")).hexdigest(),
    }


def decision_json(decision: Dict[str, Any]) -> str:
    """Canonical serialization used for the decision artifact."""
    return json.dumps(decision, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"
