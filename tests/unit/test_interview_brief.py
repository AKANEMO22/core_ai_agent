from core.profiler.brief import resolve_interview_brief, shape_stage_plan


def test_generic_framework_does_not_claim_company_interview_round():
    brief = resolve_interview_brief(
        {},
        {
            "company_context": [{
                "topic": "generic_interview_framework",
                "content": "Technical interviews often include system design.",
                "is_generic": True,
                "source": "framework://generic",
            }],
            "rubric": {"competencies": []},
        },
        {"web_recon": [], "web_recon_status": "available"},
        target_company="Acme Systems",
        target_role="Senior Backend Engineer",
    )

    assert brief["interview_round"]["value"] == "combined"
    assert brief["interview_round"]["source"] == "default_full_practice"


def test_shape_stage_plan_has_one_final_candidate_questions_and_close_stage():
    brief = {
        "interview_round": {"value": "combined"},
        "seniority": {"value": "senior"},
        "duration": {
            "planned_minutes": 30,
            "range_minutes": [30, 45],
            "source": "fallback_30_45_minutes",
        },
    }
    proposed_plan = {
        "stages": [
            {"stage_id": "technical", "name": "Technical depth", "stage_type": "technical", "target_turns": 3},
            {"stage_id": "wrapup", "name": "Q&A and Close", "stage_type": "wrapup", "target_turns": 1},
        ]
    }

    result = shape_stage_plan(proposed_plan, brief, {"candidate_name": "Test Candidate"})
    wrapups = [stage for stage in result["stages"] if stage["stage_type"] == "wrapup"]

    assert result["stages"][0]["stage_id"] == "opening"
    assert result["stages"][0]["question_candidates"][0]["question_id"] == "opening_introduction"
    assert result["stages"][-1]["stage_id"] == "candidate_questions"
    assert [stage["stage_id"] for stage in wrapups] == ["candidate_questions"]
    assert result["estimated_turns"] == 15
