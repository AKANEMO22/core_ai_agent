import pytest

from core.profiler.brief import _extract_minutes, resolve_interview_brief, shape_stage_plan


def source(text, *, index=1, title="Microsoft Software Engineer Intern interview", tier="official_verified"):
    return {"source_id": f"s{index}", "title": title, "snippet": text,
            "url": f"https://source{index}.example/interview", "trust_tier": tier,
            "query": "Microsoft Software Engineer Intern interview duration"}


def brief(sources, round="technical", knowledge=None):
    return resolve_interview_brief(
        {"seniority": "intern", "interview_round": round}, knowledge or {"rubric": {"competencies": []}},
        {"web_recon": sources, "web_recon_status": "available"},
        target_company="Microsoft", target_role="Software Engineer Intern")


@pytest.mark.parametrize("title", ["Acme cafeteria meal planning", "Google Data Analyst Intern interview", "Microsoft Data Analyst Intern interview"])
def test_own_query_cannot_make_wrong_company_or_role_relevant(title):
    result = brief([source("Lunch break lasts 60 minutes", index=i, title=title, tier="unreviewed") for i in (1, 2)])
    assert result["duration"]["source"] == "fallback_30_45_minutes"
    assert result["duration"]["evidence_ids"] == []


def test_round_specific_duration_from_mixed_source():
    sources = [source("The coding interview lasts 90 minutes. The recruiter HR phone screen lasts 20 minutes.")]
    assert brief(sources, "hr")["duration"]["planned_minutes"] == 20
    assert brief(sources)["duration"]["planned_minutes"] == 90
    assert brief(sources, "combined")["duration"]["source"] == "fallback_30_45_minutes"


@pytest.mark.parametrize("text, expected", [("1-2 hours", (60, 120)), ("1–1.5 hours", (60, 90)), ("30 to 45 minutes", (30, 45)), ("1 đến 2 giờ", (60, 120))])
def test_duration_ranges(text, expected):
    assert _extract_minutes(text) == expected


def test_unrelated_number_in_relevant_source_cannot_set_duration():
    result = brief([source("Our lunch break lasts 60 minutes. The coding interview covers Python.")])
    assert result["duration"]["evidence_ids"] == []


def test_duration_claim_for_other_level_is_not_intern_timing():
    result = brief([source("Senior technical interview lasts 90 minutes. Intern interview covers Python.")])
    assert result["duration"]["evidence_ids"] == []


def test_same_sentence_durations_still_match_their_round():
    sources = [source("Coding interview lasts 90 minutes and recruiter HR screen lasts 20 minutes.")]
    assert brief(sources, "hr")["duration"]["planned_minutes"] == 20
    assert brief(sources)["duration"]["planned_minutes"] == 90


def test_conflicting_verified_claims_do_not_select_arbitrary_first():
    result = brief([source("Technical interview lasts 30 minutes", index=1), source("Technical interview lasts 90 minutes", index=2)])
    assert result["duration"]["source"] == "fallback_30_45_minutes"
    assert any("conflict" in limitation.casefold() for limitation in result["limitations"])


def test_unreviewed_consensus_cannot_cherry_pick_around_conflicting_source():
    result = brief([source(f"Technical interview lasts {minutes} minutes", index=i, tier="unreviewed") for i, minutes in enumerate((30, 30, 90), 1)])
    assert result["duration"]["source"] == "fallback_30_45_minutes"


def test_framework_duration_must_match_selected_round():
    knowledge = {"rubric": {"competencies": []}, "company_context": [{"source": "framework://process", "is_generic": True, "content": "Technical interview lasts 90 minutes. HR screen lasts 20 minutes."}]}
    result = brief([], "hr", knowledge)
    assert result["duration"]["planned_minutes"] == 20
    assert result["duration"]["source"] == "framework"


@pytest.mark.parametrize("minutes", [10, 11, 30, 38, 45, 90, 120])
def test_stage_minutes_include_intro_and_qa_in_exact_total(minutes):
    result = shape_stage_plan({}, {"interview_round": {"value": "combined"}, "seniority": {"value": "intern"}, "duration": {"planned_minutes": minutes}})
    assert sum(stage["time_budget_minutes"] for stage in result["stages"]) == minutes
    assert all(stage["time_budget_minutes"] > 0 for stage in result["stages"])


@pytest.mark.parametrize("minutes", [1, 2, 3, 4])
def test_very_short_plans_do_not_crash_and_keep_positive_budgets(minutes):
    result = shape_stage_plan({}, {"interview_round": {"value": "combined"}, "seniority": {"value": "intern"}, "duration": {"planned_minutes": minutes}})
    assert sum(stage["time_budget_minutes"] for stage in result["stages"]) == max(3, minutes)
    assert all(stage["time_budget_minutes"] > 0 for stage in result["stages"])


def test_clause_naming_selected_and_other_level_is_still_used():
    result = brief([source("Intern or senior technical interview lasts 60 minutes.")])
    assert result["duration"]["planned_minutes"] == 60
