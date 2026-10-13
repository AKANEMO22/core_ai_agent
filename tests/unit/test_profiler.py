"""Unit tests for ProfilerAgent pre-flight reconnaissance and artifact generation."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from core.profiler.agent import ProfilerAgent, redact_sensitive_candidate_data
from core.profiler.tools import MarketIntelTool, SafeWebSearchTool
from core.retrieval import InterviewKnowledgeService, MockRetrievalClient
from core.runtime.providers.mock import MockLLMClient
from core.workspace.session import SessionWorkspace


def _blueprint_llm() -> MockLLMClient:
    """Mock LLM returning a stage plan without criteria, as an ungrounded model would."""
    return MockLLMClient(
        canned_json_responses={
            "company_intel": {"company": "Acme Systems", "culture": "Rigor"},
            "candidate_profile": {"strengths": ["Python"], "gaps": ["Kafka"]},
            "interview_spec": {"interviewer_persona": "Tech Lead", "tone": "direct"},
            "stage_plan": {
                "stages": [
                    {"stage_id": "warmup", "name": "Warmup", "turns": 1},
                    {"stage_id": "technical", "name": "Technical Depth", "turns": 3},
                    {"stage_id": "design", "name": "System Design", "turns": 2},
                    {"stage_id": "close", "name": "Q&A and Close", "turns": 1},
                ]
            },
        }
    )


def _write_backend_form(ws: SessionWorkspace) -> None:
    ws.write_artifact(
        "inputs/form_input.json",
        artifact_type="input",
        payload={
            "candidate_name": "Nguyen Van A",
            "experience_years": 6,
            "target_company": "Acme Systems",
            "target_role": "Senior Backend Engineer",
            "skills": ["Python", "Redis"],
        },
    )


@pytest.mark.asyncio
async def test_profiler_grounds_stage_plan_with_retrieved_knowledge(tmp_path):
    """Verify profiler retrieves rubric/questions, grounds the stage plan and writes knowledge artifacts."""
    ws = SessionWorkspace(session_id="sess_rag1", root_path=tmp_path / "sess_rag1")
    ws.initialize()
    _write_backend_form(ws)

    retrieval = MockRetrievalClient()
    profiler = ProfilerAgent(
        llm_client=_blueprint_llm(),
        tools=[],
        knowledge_service=InterviewKnowledgeService(client=retrieval),
    )
    res = await profiler.run_profiling(ws)

    assert res["status"] == "SUCCESS"
    assert res["retrieval_status"] == "ok"
    assert {"retrieved_knowledge", "rubric"} <= set(res["artifacts"])

    knowledge = ws.read_json("context/retrieved_knowledge.json")
    assert knowledge["status"] == "ok"
    assert knowledge["retrieval_backend"] == "mock"

    rubric = ws.read_json("context/rubric.json")
    assert rubric["rubric_version"] == "backend_senior_v1"
    rubric_ids = {c["competency_id"] for c in rubric["competencies"]}

    plan = ws.read_json("context/stage_plan.json")
    assert [s["stage_id"] for s in plan["stages"]] == ["warmup", "technical", "design", "close"]
    assert {cid for s in plan["stages"] for cid in s["criteria"]} == rubric_ids
    assert all(s["question_candidates"] for s in plan["stages"])

    # Retrieval queries never carry candidate identity
    assert retrieval.queries
    assert all("Nguyen" not in q.text for q in retrieval.queries)


@pytest.mark.asyncio
async def test_profiler_degrades_when_retrieval_fails(tmp_path):
    """Verify retrieval failure does not fail profiling and is recorded as degraded."""
    ws = SessionWorkspace(session_id="sess_rag2", root_path=tmp_path / "sess_rag2")
    ws.initialize()
    _write_backend_form(ws)

    profiler = ProfilerAgent(
        llm_client=_blueprint_llm(),
        tools=[],
        knowledge_service=InterviewKnowledgeService(
            client=MockRetrievalClient(fail_with=RuntimeError("index offline"))
        ),
    )
    res = await profiler.run_profiling(ws)

    assert res["status"] == "SUCCESS"
    assert res["retrieval_status"] == "degraded"
    knowledge = ws.read_json("context/retrieved_knowledge.json")
    assert knowledge["status"] == "degraded"
    assert "index offline" in knowledge["error"]
    assert not ws.exists("context/rubric.json")

    plan = ws.read_json("context/stage_plan.json")
    assert all("criteria" not in s for s in plan["stages"])


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
async def test_profiler_recursive_candidate_redaction(tmp_path):
    """Verify recursive redaction scrub nested candidate identifiers, lists, emails, and phones."""
    ws = SessionWorkspace(session_id="sess_p4_b", root_path=tmp_path / "sess_p4_b")
    ws.initialize()

    form_data = {
        "candidate_name": "Jane TopSecret",
        "nested": {
            "skills": ["CustomSecretAlgorithm", "ProprietaryLedger"],
            "contact": {
                "email": "jane.secret@privatecorp.internal",
                "phone": "+1-555-019-9988",
            },
        },
    }
    ws.write_artifact("inputs/form_input.json", artifact_type="input", payload=form_data)

    mock_client = MockLLMClient(
        raise_error=RuntimeError(
            "Call failed for candidate Jane TopSecret with skill CustomSecretAlgorithm, email jane.secret@privatecorp.internal"
        )
    )
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)

    assert res["status"] == "FAILED"
    assert "Jane TopSecret" not in res["error"]
    assert "CustomSecretAlgorithm" not in res["error"]
    assert "jane.secret@privatecorp.internal" not in res["error"]


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


@pytest.mark.asyncio
async def test_safe_web_search_enforces_timeout_and_max_bytes():
    """Verify SafeWebSearchTool enforces timeout and max_bytes when fetching a target URL."""
    tool = SafeWebSearchTool(timeout=3.0, max_bytes=50)
    assert tool.timeout == 3.0
    assert tool.max_bytes == 50

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"A" * 200  # 200 bytes, exceeding max_bytes=50

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("core.profiler.tools.validate_outbound_url", return_value=True):
            res = await tool.execute({"query": "recon", "target_url": "https://example.com/api"})
            assert res["status"] == "success"
            assert res["truncated"] is True
            assert res["total_bytes"] == 200
            assert len(res["content"]) <= 50


@pytest.mark.asyncio
async def test_gather_recon_coordinates_tools(tmp_path):
    """Verify ProfilerAgent._gather_recon coordinates both MarketIntelTool and SafeWebSearchTool."""
    ws = SessionWorkspace(session_id="sess_p6", root_path=tmp_path / "sess_p6")
    ws.initialize()

    mock_client = MockLLMClient(canned_tokens=["{}"])
    profiler = ProfilerAgent(llm_client=mock_client)

    recon = await profiler._gather_recon("Google", "Senior SRE", workspace=ws)
    assert recon is not None
    assert "market_intel" in recon
    assert recon["market_intel"]["company"] == "Google"
    assert "web_recon" in recon
    assert len(recon["web_recon"]) > 0
