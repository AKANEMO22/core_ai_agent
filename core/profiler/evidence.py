"""Deterministic, traceable probes over submitted CV claims and JD requirements.

These records describe claims to examine, not verified facts or scoring criteria.
The original profile and uploaded documents remain the source of record.
"""
import copy
import hashlib
import json
import re
from typing import Any, Iterable


FIELDS = {
    "projects": ("name", "problem", "technologies", "personal_contribution", "decisions", "challenges", "outcomes"),
    "experience": ("role", "company", "period", "responsibilities", "results"),
    "skills": ("name", "category"),
    "career_goals": (),
    "education": ("institution", "field", "period", "achievements"),
    "awards": ("name", "issuer", "date"),
    "certifications": ("name", "issuer", "date"),
}
PRIORITY = ("projects", "experience", "jd_requirements", "skills", "career_goals", "education", "certifications", "awards")
TECHNICAL = {"projects", "skills", "jd_requirements"}


def _identity(section: str, facts: Any) -> str:
    canonical = json.dumps([section, facts], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def build_evidence(profile: dict, job_description: str | None = None) -> list[dict]:
    """Use only substantive allowlisted fields, excluding contact/age/identity data."""
    records: list[dict] = []
    seen: set[str] = set()

    def add(section: str, facts: Any, source_ref: str) -> None:
        if not facts:
            return
        evidence_id = "evidence_" + _identity(section, facts)
        if evidence_id in seen:
            return
        seen.add(evidence_id)
        records.append({
            "evidence_id": evidence_id, "section": section, "facts": copy.deepcopy(facts),
            "source_ref": source_ref,
            "source_status": "submitted_jd_unverified" if section == "jd_requirements" else "candidate_claim_unverified",
        })

    for section, keys in FIELDS.items():
        values = profile.get(section) or []
        if not isinstance(values, list):
            values = [values]
        for index, value in enumerate(values):
            if isinstance(value, dict):
                facts = {key: value[key] for key in keys if value.get(key)}
                source_ref = str(value.get("evidence_ref") or f"candidate_profile.{section}[{index}]")
            elif isinstance(value, str) and value.strip():
                facts, source_ref = value.strip(), f"candidate_profile.{section}[{index}]"
            else:
                continue
            add(section, facts, source_ref)
    if isinstance(job_description, str):
        for index, requirement in enumerate(re.split(r"[\n\r]+|(?<=[.!?])\s+", job_description)):
            requirement = requirement.strip(" \t-*•")
            if requirement:
                add("jd_requirements", requirement, f"submitted_job_description.requirement[{index}]")
    return records


def _probe_text(record: dict) -> str:
    facts = record["facts"]
    label = (facts.get("name") or facts.get("role") or facts.get("institution") or "this claim") if isinstance(facts, dict) else facts
    label = str(label)[:180]
    prompts = {
        "projects": "In your project {label}, what did you personally implement, why did you choose that approach, and how did you measure the outcome?",
        "experience": "In your experience as {label}, can you explain one decision you owned and its measurable result?",
        "skills": "You list {label}; can you explain a concrete use case, the tradeoff you faced, and how you checked your solution?",
        "career_goals": "You mention {label}; how does this role fit that goal and what would you aim to learn or deliver?",
        "education": "From your studies at {label}, which course or assignment best prepared you for this role, and how did you apply it?",
        "awards": "For {label}, what was your own contribution and how was the result evaluated?",
        "certifications": "How have you applied knowledge from {label} to a practical problem beyond the certification exam?",
        "jd_requirements": "The submitted JD mentions {label}; what relevant example can you explain, or how would you approach it if it is new to you?",
    }
    return prompts[record["section"]].format(label=label)


def attach_evidence_probes(stage_plan: Any, profile: dict, job_description: str | None = None) -> Any:
    """Attach after rubric grounding; select a bounded, section-diverse set.

    Mapping to an assessor competency is intentionally left empty: a CV claim alone
    cannot establish that it tests a specific rubric criterion. Existing grounded
    questions and stage criteria are retained for calibrated competency assessment.
    """
    if not isinstance(stage_plan, dict) or not isinstance(stage_plan.get("stages"), list):
        return stage_plan
    result = copy.deepcopy(stage_plan)
    evidence = build_evidence(profile, job_description)
    stages = [s for s in result["stages"] if isinstance(s, dict) and s.get("stage_id") not in {"opening", "candidate_questions", "closing"}]
    capacities = {s["stage_id"]: max(1, int(s.get("target_turns") or s.get("turns") or 1)) for s in stages}
    minutes = max(1, int(result.get("planned_minutes") or 30))
    limit = min(16, max(1, minutes // 3), sum(capacities.values()))
    queues = {section: [e for e in evidence if e["section"] == section] for section in PRIORITY}
    ordered: list[dict] = []
    while any(queues.values()):
        for section in PRIORITY:
            if queues[section]:
                ordered.append(queues[section].pop(0))
    selected: list[dict] = []
    by_stage: dict[str, list[dict]] = {s["stage_id"]: [] for s in stages}
    for record in ordered:
        if len(selected) >= limit:
            break
        preferred = ("technical", "system_design", "warmup") if record["section"] in TECHNICAL else ("behavioral", "warmup")
        target = next((s for kind in preferred for s in stages if s.get("stage_type") == kind and capacities[s["stage_id"]] > 0), None)
        if target is None:
            continue
        stage_id = target["stage_id"]
        probe = {
            "question_id": "cv_probe_" + _identity(record["section"], record["facts"]),
            "text": _probe_text(record), "stage_id": stage_id, "stage_type": target.get("stage_type"),
            "evidence_ids": [record["evidence_id"]], "source_refs": [record["source_ref"]],
            "source_status": record["source_status"], "provenance": "project_authored_candidate_evidence_probe",
            "competency_id": None, "follow_ups": [],
        }
        selected.append(probe)
        by_stage[stage_id].append(probe)
        capacities[stage_id] -= 1
    for stage in stages:
        generic = [q for q in stage.get("question_candidates", []) if not q.get("evidence_ids")]
        stage["question_candidates"] = by_stage[stage["stage_id"]] + generic
    selected_ids = {eid for probe in selected for eid in probe["evidence_ids"]}
    result["evidence_plan"] = {
        "schema_version": 1, "evidence": evidence, "selected_probes": selected,
        "omitted_evidence_ids": [e["evidence_id"] for e in evidence if e["evidence_id"] not in selected_ids],
        "selection_basis": "Section-diverse submitted claims; bounded by stage turns and one probe per three planned minutes. Omitted or unasked claims are not assessed.",
    }
    return result


def evidence_coverage(stage_plan: dict, asked_ids: Iterable[str]) -> dict:
    """Count only persisted question IDs, never guessed semantic/text coverage."""
    plan = stage_plan.get("evidence_plan") or {}
    probes = plan.get("selected_probes") or []
    asked = set(asked_ids)
    asked_probes = [p for p in probes if p["question_id"] in asked]
    unasked = [p for p in probes if p["question_id"] not in asked]
    asked_evidence = list(dict.fromkeys(eid for p in asked_probes for eid in p["evidence_ids"]))
    return {
        "selected_probe_ids": [p["question_id"] for p in probes],
        "asked_probe_ids": [p["question_id"] for p in asked_probes],
        "unasked_probe_ids": [p["question_id"] for p in unasked],
        "asked_evidence_ids": asked_evidence,
        "unasked_evidence_ids": list(dict.fromkeys(eid for p in unasked for eid in p["evidence_ids"] if eid not in asked_evidence)),
        "omitted_evidence_ids": list(plan.get("omitted_evidence_ids") or []),
        "coverage_basis": "Persisted interviewer question IDs; asking a probe does not verify the claim or prove answer quality.",
    }
