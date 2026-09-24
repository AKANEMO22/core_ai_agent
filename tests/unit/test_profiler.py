"""Unit tests for ProfilerAgent pre-flight reconnaissance and artifact generation."""

from pathlib import Path
import pytest
from core.profiler.agent import ProfilerAgent
from core.runtime.providers.mock import MockLLMClient
from core.workspace.session import SessionWorkspace


@pytest.mark.asyncio
async def test_profiler_generates_required_artifacts(tmp_path):
    """Verify profiler ingests form input and generates 4 required artifacts in context/."""
    ws = SessionWorkspace(session_id="sess_p1", root_path=tmp_path / "sess_p1")
    ws.initialize()

    # Write a mock form input
    form_data = {
        "candidate_name": "Nguyen Van A",
        "experience_years": 5,
        "target_company": "TechCorp",
        "target_role": "Senior Backend Engineer",
        "skills": ["Python", "Distributed Systems", "Redis"],
    }
    ws.write_artifact("inputs/form_input.json", artifact_type="input", payload=form_data)

    mock_client = MockLLMClient(
        canned_json_responses={
            "company_intel": {
                "company": "TechCorp",
                "culture": "High Bar",
                "tech_stack": ["Python", "Kafka"],
            },
            "candidate_profile": {
                "strengths": ["Python"],
                "gaps": ["Kubernetes"],
                "probes": ["Concurrency"],
            },
            "interview_spec": {
                "interviewer_persona": "Tech Lead",
                "tone": "rigorous_challenger",
                "knowledge_scope": ["Concurrency"],
            },
            "stage_plan": {
                "stages": [{"stage_id": "stage_1", "name": "Warmup", "turns": 2}]
            },
        }
    )
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)

    assert res["status"] == "SUCCESS"
    assert (ws.root_path / "context" / "company_intel.json").exists()
    assert (ws.root_path / "context" / "candidate_profile.json").exists()
    assert (ws.root_path / "context" / "interview_spec.json").exists()
    assert (ws.root_path / "context" / "stage_plan.json").exists()

    # Check contents of artifacts
    intel = ws.read_json("context/company_intel.json")
    assert intel["company"] == "TechCorp"
    assert intel["artifact_type"] == "company_intel"
    assert "checksum_sha256" in intel

    profile = ws.read_json("context/candidate_profile.json")
    assert profile["strengths"] == ["Python"]
    assert profile["artifact_type"] == "candidate_profile"

    spec = ws.read_json("context/interview_spec.json")
    assert spec["interviewer_persona"] == "Tech Lead"
    assert spec["artifact_type"] == "interview_spec"

    plan = ws.read_json("context/stage_plan.json")
    assert len(plan["stages"]) == 1
    assert plan["artifact_type"] == "stage_plan"


@pytest.mark.asyncio
async def test_profiler_with_cv_text_input(tmp_path):
    """Verify profiler ingests plain text CV file inputs/cv_input.txt."""
    ws = SessionWorkspace(session_id="sess_p2", root_path=tmp_path / "sess_p2")
    ws.initialize()

    cv_text = (
        "Name: Jane Doe\n"
        "Experience: 7 years\n"
        "Target Company: CloudSystems\n"
        "Role: Staff Engineer\n"
        "Skills: Go, Kubernetes, Raft, Distributed Databases\n"
    )
    ws.write_text("inputs/cv_input.txt", cv_text)

    mock_client = MockLLMClient(
        canned_json_responses={
            "company_intel": {"company": "CloudSystems", "culture": "Distributed Systems First"},
            "candidate_profile": {"strengths": ["Raft", "Distributed Databases"], "gaps": []},
            "interview_spec": {"interviewer_persona": "Principal Engineer", "tone": "collaborative_inquisitive"},
            "stage_plan": {"stages": [{"stage_id": "s1", "name": "System Architecture", "turns": 4}]},
        }
    )
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)

    assert res["status"] == "SUCCESS"
    assert (ws.root_path / "context" / "company_intel.json").exists()
    assert (ws.root_path / "context" / "candidate_profile.json").exists()
    assert (ws.root_path / "context" / "interview_spec.json").exists()
    assert (ws.root_path / "context" / "stage_plan.json").exists()


@pytest.mark.asyncio
async def test_profiler_missing_inputs(tmp_path):
    """Verify profiler fails gracefully when no input files exist in inputs/."""
    ws = SessionWorkspace(session_id="sess_p3", root_path=tmp_path / "sess_p3")
    ws.initialize()

    mock_client = MockLLMClient(canned_tokens=["{}"])
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)

    assert res["status"] == "FAILED"
    assert "no candidate input" in res["error"].lower()


@pytest.mark.asyncio
async def test_profiler_scrubs_secrets_from_errors(tmp_path):
    """Verify profiler error handling sanitizes API keys and confidential candidate data."""
    ws = SessionWorkspace(session_id="sess_p4", root_path=tmp_path / "sess_p4")
    ws.initialize()

    # Write form input with candidate sensitive info
    form_data = {
        "candidate_name": "Classified Candidate",
        "secret_note": "CONFIDENTIAL_SALARY_DATA",
    }
    ws.write_artifact("inputs/form_input.json", artifact_type="input", payload=form_data)

    # Provider that raises an error containing an API key
    mock_client = MockLLMClient(
        raise_error=RuntimeError("Provider connection failed with key sk-1234567890abcdef1234567890")
    )
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)

    assert res["status"] == "FAILED"
    assert "sk-" not in res["error"]
    assert "[REDACTED_API_KEY]" in res["error"]
    assert "CONFIDENTIAL_SALARY_DATA" not in res["error"]


@pytest.mark.asyncio
async def test_profiler_single_bundle_generation(tmp_path):
    """Verify profiler can process an all-in-one bundle containing all 4 blueprints."""
    ws = SessionWorkspace(session_id="sess_p5", root_path=tmp_path / "sess_p5")
    ws.initialize()

    ws.write_artifact(
        "inputs/form_input.json",
        artifact_type="input",
        payload={"candidate_name": "Test Candidate", "target_company": "StartupX"},
    )

    all_in_one_bundle = {
        "company_intel": {"company": "StartupX", "culture": "Move fast"},
        "candidate_profile": {"strengths": ["Fast learner"]},
        "interview_spec": {"interviewer_persona": "Founder", "tone": "direct"},
        "stage_plan": {"stages": [{"stage_id": "s1", "name": "Culture", "turns": 2}]},
    }
    mock_client = MockLLMClient(canned_tokens=[json_str := str(all_in_one_bundle).replace("'", '"')])
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)

    assert res["status"] == "SUCCESS"
    assert (ws.root_path / "context" / "company_intel.json").exists()
    assert (ws.root_path / "context" / "candidate_profile.json").exists()
    assert (ws.root_path / "context" / "interview_spec.json").exists()
    assert (ws.root_path / "context" / "stage_plan.json").exists()
