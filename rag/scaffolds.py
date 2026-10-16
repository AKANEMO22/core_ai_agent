"""Inspectable, project-authored interview scaffolds; never company evidence.

These templates fill explicit corpus gaps. Their anchors are engineering guidance,
not empirically validated hiring benchmarks. Source questions keep their own level.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

VERSION = "authored-role-level-v1"
LEVEL_EXPECTATIONS = {
    "intern": "Explain fundamentals using a small coursework or personal project; identify where guidance would be needed.",
    "fresher": "Implement and validate a bounded entry-level task; explain personal decisions and mistakes with a concrete example.",
    "junior": "Own a bounded task, compare a practical alternative, debug failures and demonstrate validation.",
    "mid": "Independently own an end-to-end work item, quantify constraints, compare alternatives and validate delivery risks.",
    "senior": "Frame an ambiguous cross-component problem, justify trade-offs with evidence, coordinate dependencies and improve others' decisions.",
}
SCENARIOS = {
    "frontend": "an accessible interface with asynchronous data, slow networks and rendering performance constraints",
    "backend": "an API and persistence workflow with concurrent updates, partial failures and data consistency constraints",
    "data": "a data or model workflow with incomplete inputs, leakage risks, reproducibility and measurable evaluation",
    "qa": "a release with uncertain requirements, regression risks, flaky checks and limited test time",
    "devops": "a deployment and recovery workflow with availability targets, secrets, observability and rollback constraints",
    "mobile": "a mobile user workflow with offline state, lifecycle changes, device constraints and release compatibility",
    "embedded": "a hardware-facing workflow with timing, memory, concurrency and device failure constraints",
    "infrastructure": "a network change with routing, access controls, outage diagnosis and recovery constraints",
    "game": "a playable interaction with frame-time, memory, physics and asset lifecycle constraints",
    "enterprise": "an ERP process change with business rules, access controls, integration and migration constraints",
    "business_analyst": "a business workflow with conflicting stakeholder needs, acceptance criteria, traceability and change impact",
}
# Used for roles whose family is unmapped ("any") instead of failing the whole rubric.
GENERIC_SCENARIO = "a realistic work item with unclear requirements, competing constraints and a need to verify the outcome"


def _content_hash() -> str:
    # Hash the module source (templates, expectations, scenarios) so any wording change yields a new version.
    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


RUBRIC_VERSION = f"{VERSION}:{_content_hash()}"


def behavioral_competency(pillar: dict) -> bool:
    name = str(pillar.get("pillar_en") or pillar.get("pillar", ""))
    return bool(re.search(r"communication|collaborat|teamwork|behavior|motivation|culture|attitude|learning|mindset|soft skill|leadership", name, re.I))


def authored_probe(role_id: str, framework: dict, pillar: dict, family: str, level: str) -> dict:
    """Bind a role's actual criterion to a level-specific scenario and scoring anchors."""
    name = str(pillar.get("pillar_en") or pillar["pillar"])
    criteria = str(pillar.get("criteria") or name)
    scope = LEVEL_EXPECTATIONS[level]
    scenario = SCENARIOS.get(family, GENERIC_SCENARIO)
    role = framework["role_title"]
    if behavioral_competency(pillar):
        text = (
            f"For a {level} {role} role, describe a concrete situation relevant to {name}. "
            f"What was your responsibility, how did you work with others, and what changed? {scope}"
        )
        stage = "behavioral"
    else:
        topics = "; ".join(str(x) for x in framework.get("target_competencies", [])[:3])
        text = (
            f"For a {level} {role} role, consider {scenario}. "
            f"Focus on {name}: {criteria}. "
            f"Use a relevant example involving {topics or role}. "
            f"What would you do, why, and how would you check the result? {scope}"
        )
        stage = "technical"
    return {
        "text": text,
        "stage_type": stage,
        "expected_points": [f"Role-specific evidence for: {criteria}", scope, "Personal responsibility, an explicit constraint and observable validation; acknowledge missing evidence."],
        "follow_ups": ["Which assumption could fail, and how would you check it?", "What would change your decision?"],
        "anchors": {
            "1": "Cannot explain the role-specific task or distinguish personal work from claims.",
            "2": f"Partial evidence for {name}; does not meet this scope: {scope}",
            "3": f"Concrete role-relevant evidence meeting this scope: {scope}",
            "4": f"Meets the scope, compares alternatives and demonstrates the result with evidence: {scope}",
        },
        "provenance": "project_authored",
        "trust": "project_authored_unreviewed",
        "calibration_level": level,
        "calibration_expectations": scope,
        "selected_role": role,
        "source_levels": [],
        "source": f"{VERSION}#{role_id}/{level}/{pillar['competency_id']}",
    }
