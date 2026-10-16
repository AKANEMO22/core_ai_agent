"""Final reports distinguish planned CV/JD probes from actually asked probes."""

import json

import fakeredis.aioredis
import pytest

from core.assessor.synthesizer import ReportSynthesizer
from core.workspace.session import SessionWorkspace


@pytest.mark.asyncio
async def test_report_persists_asked_and_unasked_cv_jd_probes(tmp_path):
    workspace = SessionWorkspace("evidence_report", tmp_path / "evidence_report")
    workspace.initialize()
    workspace.write_artifact(
        "context/stage_plan.json", artifact_type="stage_plan", payload={
            "stages": [],
            "evidence_plan": {
                "schema_version": 1,
                "evidence": [
                    {"evidence_id": "cv_project", "section": "projects", "source_status": "candidate_claim_unverified"},
                    {"evidence_id": "jd_sql", "section": "job_description", "source_status": "submitted_jd_unverified"},
                    {"evidence_id": "cv_award", "section": "awards", "source_status": "candidate_claim_unverified"},
                ],
                "selected_probes": [
                    {"question_id": "probe_project", "evidence_ids": ["cv_project"]},
                    {"question_id": "probe_jd", "evidence_ids": ["jd_sql"]},
                ],
                "omitted_evidence_ids": ["cv_award"],
            },
        },
    )
    # A generated directive alone is not evidence that its question was asked.
    workspace.write_text("turns/turn_001_interviewer.json", json.dumps({
        "content": "Explain your project contribution", "question_id": "probe_project",
    }))
    workspace.write_text("orchestration/directive_2.json", json.dumps({"question_id": "probe_jd"}))
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        report = await ReportSynthesizer(redis, lambda _: workspace).synthesize(
            "evidence_report", expected_assessments=0,
        )
        coverage = report["cv_jd_evidence_coverage"]
        assert coverage["asked_probe_ids"] == ["probe_project"]
        assert coverage["unasked_probe_ids"] == ["probe_jd"]
        assert coverage["omitted_evidence_ids"] == ["cv_award"]
        assert coverage["asked_evidence_ids"] == ["cv_project"]
        assert workspace.read_json("reports/cv_jd_evidence_coverage.json")["payload"] == coverage
        markdown = workspace.read_text("reports/interview_report.md")
        assert "Độ phủ câu hỏi kiểm chứng CV/JD" in markdown  # default interview language: vi
        assert "probe_jd" in markdown
        assert "xác minh độc lập" in markdown
    finally:
        await redis.aclose()
