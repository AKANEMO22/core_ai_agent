"""Evaluate generated responses with independent, human-reviewed claim annotations."""

import numpy as np
from pydantic import BaseModel, Field


class ClaimAnnotation(BaseModel):
    text: str
    supported: bool
    source_locators: list[str] = Field(default_factory=list)


class GroundingSample(BaseModel):
    id: str
    response: str
    retrieved_source_locators: list[str]
    claims: list[ClaimAnnotation]
    human_reviewed: bool = False
    predicted_grade: float | None = Field(default=None, ge=1, le=5)
    reference_grade: float | None = Field(default=None, ge=1, le=5)


def evaluate_grounding(samples: list[GroundingSample]):
    rows = []
    for sample in samples:
        if not sample.human_reviewed:
            raise ValueError(
                "Grounding acceptance requires independent human-reviewed claims"
            )
        if not sample.response.strip() or not sample.claims:
            raise ValueError("Response and claim annotations are required")
        known = set(sample.retrieved_source_locators)
        if any(
            claim.supported
            and (not claim.source_locators or not set(claim.source_locators) <= known)
            for claim in sample.claims
        ):
            raise ValueError("Supported claims must cite retrieved sources")
        rows.append(
            {
                "id": sample.id,
                "faithfulness": sum(c.supported for c in sample.claims)
                / len(sample.claims),
                "citation_validity": sum(
                    set(c.source_locators) <= known for c in sample.claims
                )
                / len(sample.claims),
                "grade_absolute_error": abs(
                    sample.predicted_grade - sample.reference_grade
                )
                if sample.predicted_grade is not None
                and sample.reference_grade is not None
                else None,
            }
        )
    if not rows:
        raise ValueError("No grounding samples")
    errors = [
        r["grade_absolute_error"] for r in rows if r["grade_absolute_error"] is not None
    ]
    return {
        "samples": len(rows),
        "faithfulness": float(np.mean([r["faithfulness"] for r in rows])),
        "citation_validity": float(np.mean([r["citation_validity"] for r in rows])),
        "grade_mae": float(np.mean(errors)) if errors else None,
        "grading_sample_count": len(errors),
        "records": rows,
    }
