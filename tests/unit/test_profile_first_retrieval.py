"""Retrieval uses the extracted candidate profile; the report follows the chosen interview language."""

from typing import Any, Dict, List

import pytest

from core.agents.narrative import NarrativeWriter
from core.profiler.agent import ProfilerAgent, merge_profile_into_form
from core.runtime.providers.mock import MockLLMClient
from core.report_language import localize_reason
from core.workspace.session import SessionWorkspace


class RecordingKnowledge:
    backend_name = "recording"

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def build_pack(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        raise RuntimeError("no corpus in this test")


def _cv_only_workspace(tmp_path) -> SessionWorkspace:
    ws = SessionWorkspace(session_id="sess_cv", root_path=tmp_path / "sess_cv")
    ws.initialize()
    # Shape FE-test sends in CV mode: role/company left empty.
    ws.write_artifact(
        "inputs/form_input.json",
        artifact_type="input",
        payload={"candidate_name": "", "target_role": "", "target_company": "", "skills": [], "interview_language": "vi"},
    )
    ws.write_text("inputs/cv_input.txt", "Tran B\nJunior Frontend Developer at Shop Co (2023-2025)\nReact, TypeScript")
    return ws


@pytest.mark.asyncio
async def test_cv_only_session_retrieves_with_profile_role_not_senior_default(tmp_path):
    ws = _cv_only_workspace(tmp_path)
    knowledge = RecordingKnowledge()
    llm = MockLLMClient(
        canned_json_responses={
            "candidate_profile": {
                "experience": [{"role": "Junior Frontend Developer", "company": "Shop Co"}],
                "skills": [{"name": "React"}, {"name": "TypeScript"}],
                "experience_years": 2,
            },
            "company_intel": {},
            "interview_spec": {},
            "stage_plan": {"stages": [{"stage_id": "s1", "name": "Intro", "turns": 1}]},
        }
    )
    result = await ProfilerAgent(llm_client=llm, tools=[], knowledge_service=knowledge).run_profiling(ws)

    assert result["status"] == "SUCCESS"
    assert len(knowledge.calls) == 1
    call = knowledge.calls[0]
    assert call["target_role"] == "Junior Frontend Developer"
    assert "senior" not in call["target_role"].casefold()
    assert call["skills"] == ["React", "TypeScript"]
    assert call["experience_years"] == 2
    brief = ws.read_json("context/interview_brief.json")["payload"]
    assert brief["seniority"]["value"] == "junior"


def test_form_values_win_over_profile():
    merged = merge_profile_into_form(
        {"target_role": "Backend Engineer", "skills": ["Go"], "experience_years": 4},
        {"target_role": "Designer", "skills": [{"name": "Go"}, {"name": "Redis"}], "experience_years": 1},
    )
    assert merged["target_role"] == "Backend Engineer"
    assert merged["experience_years"] == 4
    assert merged["skills"] == ["Go", "Redis"]


@pytest.mark.asyncio
async def test_template_narrative_follows_report_language():
    decision = {
        "verdict": "FAIL",
        "verdict_reasons": ["weighted score 2.1 < threshold 3.0"],
        "criteria": [{"competency_id": "c1", "covered": False}],
    }
    narrative = await NarrativeWriter(None).write(decision, [], [], language="vi")
    assert narrative["summary"].startswith("Kết quả: FAIL.")
    assert "Điểm có trọng số 2.1 < ngưỡng 3.0" in narrative["summary"]
    assert narrative["criterion_findings"][0]["finding"].startswith("Chưa thu thập đủ bằng chứng")
    assert localize_reason("weighted score 2.1 < threshold 3.0", "en") == "weighted score 2.1 < threshold 3.0"
