"""Traceable interview probes use submitted facts, without claiming verification."""
from copy import deepcopy

from core.profiler.agent import _shape_and_ground_stage_plan


PROFILE = {
    "skills": [{"name": "Python", "evidence_ref": "cv:skills"}],
    "projects": [{"name": "Ledger", "technologies": ["Redis"], "personal_contribution": "Built cache",
                  "decisions": ["TTL 30s"], "challenges": ["stale reads"], "outcomes": ["latency -40%"], "evidence_ref": "cv:project1"}],
    "experience": [{"role": "Developer", "company": "Acme", "results": ["reduced errors"], "evidence_ref": "cv:work"}],
    "career_goals": ["Build reliable services"],
    "education": [{"institution": "University A", "field": "CS"}],
    "awards": [{"name": "Hackathon winner"}],
    "certifications": [{"name": "AWS Developer"}],
}


def plan(profile=None, minutes=60):
    return _shape_and_ground_stage_plan({}, {
        "duration": {"planned_minutes": minutes}, "interview_round": {"value": "combined"},
        "seniority": {"value": "junior"},
    }, profile or PROFILE, {}, knowledge_ok=False, job_description="Must know SQL. Experience with API design.")


def test_probes_survive_shaping_with_standard_sections_and_original_facts():
    result = plan()
    assert "evidence_plan" in result
    ep = result["evidence_plan"]
    assert {e["section"] for e in ep["evidence"]} == {
        "skills", "projects", "experience", "career_goals", "education", "awards", "certifications", "jd_requirements",
    }
    project = next(e for e in ep["evidence"] if e["section"] == "projects")
    assert project["facts"]["technologies"] == ["Redis"]
    assert project["source_ref"] == "cv:project1"
    assert project["source_status"] == "candidate_claim_unverified"
    candidates = [q for s in result["stages"][1:-1] for q in s["question_candidates"]]
    assert {p["question_id"] for p in ep["selected_probes"]} <= {q["question_id"] for q in candidates}
    assert result["stages"][0]["criteria"] == []
    assert result["stages"][-1]["stage_id"] == "candidate_questions"


def test_stable_ids_reviewed_corrections_and_honest_budget_coverage():
    from core.profiler.evidence import evidence_coverage

    original = deepcopy(PROFILE)
    short = plan(minutes=10)
    assert PROFILE == original
    ep = short["evidence_plan"]
    assert 1 <= len(ep["selected_probes"]) <= 3
    assert ep["omitted_evidence_ids"]
    assert ep == plan(minutes=10)["evidence_plan"]
    corrected = deepcopy(PROFILE)
    corrected["projects"][0]["technologies"] = ["PostgreSQL"]
    before = next(e for e in ep["evidence"] if e["section"] == "projects")
    after = next(e for e in plan(corrected)["evidence_plan"]["evidence"] if e["section"] == "projects")
    assert before["evidence_id"] != after["evidence_id"]
    first = ep["selected_probes"][0]
    coverage = evidence_coverage(short, {first["question_id"], "generic-question"})
    assert coverage["asked_probe_ids"] == [first["question_id"]]
    assert set(coverage["unasked_probe_ids"]) == {p["question_id"] for p in ep["selected_probes"][1:]}
    assert coverage["omitted_evidence_ids"] == ep["omitted_evidence_ids"]


def test_speaker_gets_selected_fact_details_and_unverified_status(tmp_path):
    from core.interview.speaker import SpeakerContextBuilder
    from core.workspace.session import SessionWorkspace

    ws = SessionWorkspace("evidence", tmp_path)
    ws.initialize()
    result = plan()
    ws.write_artifact("context/stage_plan.json", artifact_type="stage_plan", payload=result)
    ws.write_artifact("context/candidate_profile.json", artifact_type="candidate_profile", payload=PROFILE)
    stage = next(s for s in result["stages"] if any(q.get("evidence_ids") for q in s.get("question_candidates", [])))
    prefix, _ = SpeakerContextBuilder(ws).stable_prefix(stage)
    assert "candidate_claim_unverified" in prefix
    assert "TTL 30s" in prefix
    assert "latency -40%" in prefix
