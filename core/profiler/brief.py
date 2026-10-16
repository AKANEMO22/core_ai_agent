"""Deterministic interview brief and round-safe stage-plan shaping.

The brief keeps user choices separate from inferred values and stores the evidence
used to make every inference. Web search snippets are treated as unreviewed leads;
one such result is never presented as an official company policy.
"""

from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import urlparse


SUPPORTED_ROUNDS = {"hr", "technical", "hiring_manager", "combined"}
SUPPORTED_LEVELS = {"intern", "fresher", "junior", "mid", "senior"}
_LEVEL_TERMS = (
    ("intern", re.compile(r"\b(intern|internship|thực tập sinh)\b", re.I)),
    ("fresher", re.compile(r"\b(fresher|graduate|new grad|entry[- ]level)\b", re.I)),
    ("junior", re.compile(r"\b(junior|jr\.?|associate)\b", re.I)),
    ("mid", re.compile(r"\b(mid[- ]level|middle|intermediate)\b", re.I)),
    ("senior", re.compile(r"\b(senior|sr\.?|lead|staff|principal|architect)\b", re.I)),
)


def _round(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {"hr_interview": "hr", "manager": "hiring_manager", "all": "combined"}
    clean = aliases.get(clean, clean)
    return clean if clean in SUPPORTED_ROUNDS else None


def _extract_minutes(text: str) -> tuple[int, int] | None:
    value = text.casefold().replace("–", "-").replace("—", "-").replace(",", ".")
    ranged = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:-|to|đến)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|giờ|tiếng|minutes?|mins?|phút)\b", value)
    if ranged:
        multiplier = 60 if re.fullmatch(r"hours?|hrs?|giờ|tiếng", ranged.group(3)) else 1
        low, high = sorted(round(float(ranged.group(i)) * multiplier) for i in (1, 2))
        return (low, high) if 10 <= low <= high <= 120 else None
    hour = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|giờ|tiếng)\b", value)
    if hour:
        minutes = round(float(hour.group(1)) * 60)
        return (minutes, minutes) if 10 <= minutes <= 120 else None
    if re.search(r"\bone\s+hour\b", value):
        return (60, 60)
    ranged = re.search(r"\b(\d{1,3})\s*(?:-|to|đến)\s*(\d{1,3})\s*(?:minutes?|mins?|phút)\b", value)
    if ranged:
        low, high = sorted((int(ranged.group(1)), int(ranged.group(2))))
        return (low, high) if 10 <= low <= high <= 120 else None
    single = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?|phút)\b", value)
    if single:
        minutes = int(single.group(1))
        return (minutes, minutes) if 10 <= minutes <= 120 else None
    return None


def _sources(recon_data: dict[str, Any], knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in recon_data.get("web_recon") or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or source.get("source") or "").strip()
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            continue
        key = url or source_id
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "source_id": source_id,
            "title": str(source.get("title") or "")[:400],
            "url": url,
            "snippet": str(source.get("snippet") or "")[:1200],
            "content_excerpt": str(source.get("fetched_text") or "")[:2000],
            "fetch_status": str(source.get("fetch_status") or "not_attempted"),
            "fetched_content_hash": source.get("fetched_content_hash"),
            "source_type": str(source.get("source_type") or "search_result"),
            "trust_tier": str(source.get("trust_tier") or "unreviewed"),
            "verification_status": str(source.get("verification_status") or "unreviewed"),
            "query": str(source.get("query") or ""),
            "published_at": source.get("published_at"),
            "source_age": source.get("source_age"),
            "position": source.get("position"),
            "collected_at": source.get("collected_at"),
            "content_hash": source.get("content_hash"),
        })
    for item in knowledge.get("company_context") or []:
        if not isinstance(item, dict):
            continue
        source_ref = str(item.get("source") or "")
        source_id = str(item.get("source_id") or source_ref)
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        found.append({
            "source_id": source_id,
            "title": str(item.get("topic") or "Interview framework"),
            "url": source_ref,
            "snippet": str(item.get("content") or "")[:1200],
            "source_type": "framework" if item.get("is_generic") else "retrieved_company_context",
            "trust_tier": "framework" if item.get("is_generic") else "unreviewed",
            "verification_status": "framework" if item.get("is_generic") else "unreviewed",
            "query": "",
            "published_at": None,
            "collected_at": None,
            "content_hash": None,
        })
    return found


def _source_matches(source: dict[str, Any], company: str, role: str, level: str | None) -> bool:
    # Search queries describe what we wanted to find, never what a page establishes.
    if source.get("source_type") == "framework":
        return True
    text = " ".join(str(source.get(key) or "") for key in ("title", "snippet", "content_excerpt")).casefold()
    company_clean = company.strip().casefold()
    if company_clean and not re.search(r"(?<!\w)" + re.escape(company_clean) + r"(?!\w)", text):
        return False
    role_text = role.casefold()
    for _, pattern in _LEVEL_TERMS:
        role_text = pattern.sub(" ", role_text)
    # Shared words such as 'engineer' or the requested level cannot make a data
    # analyst page relevant to a software engineer vacancy.
    generic = {"engineer", "engineering", "developer", "development", "specialist"}
    role_words = [word for word in re.findall(r"[a-z0-9+#]+", role_text) if len(word) > 2]
    specific = [word for word in role_words if word not in generic]
    required = specific or role_words
    if required and not all(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) for word in required):
        return False
    level_pattern = dict(_LEVEL_TERMS).get(level or "")
    if level_pattern and not level_pattern.search(text) and any(pattern.search(text) for key, pattern in _LEVEL_TERMS if key != level):
        return False
    return True


def _duration_for_round(source: dict[str, Any], selected_round: str, level: str | None = None) -> tuple[int, int] | None:
    """Only use durations whose local claim concerns the selected interview round and level."""
    level_pattern = dict(_LEVEL_TERMS).get(level or "")
    title = str(source.get("title") or "")
    text = "\n".join(str(source.get(key) or "") for key in ("content_excerpt", "snippet", "title"))
    claims: set[tuple[int, int]] = set()
    for clause in re.split(r"(?<!\d)[.!?;\n]+|[.!?;\n]+(?!\d)|\s+(?:and|while|whereas|còn)\s+", text):
        duration = _extract_minutes(clause)
        if not duration:
            continue
        if re.search(r"\b(lunch|break|meal|commute|coding challenge deadline|nghỉ trưa|ăn trưa)\b", clause, re.I):
            continue
        # A clause about another level (e.g. "Senior interview lasts 90 minutes") is not
        # timing for the selected level unless it also names the selected level.
        if level_pattern and not level_pattern.search(clause) and any(
            pattern.search(clause) for key, pattern in _LEVEL_TERMS if key != level
        ):
            continue
        signals = _round_signal(clause)
        context = clause if signals else title + " " + clause
        if not re.search(r"\b(interview|screen|round|coding|phỏng vấn|vòng)\b", context, re.I):
            continue
        signals = signals or _round_signal(title)
        if selected_round == "combined":
            # A duration for one round does not describe the whole interview.
            if signals and not re.search(r"\b(combined|overall|total|entire|tổng|toàn bộ)\b", clause, re.I):
                continue
        elif signals != {selected_round}:
            continue
        claims.add(duration)
    return next(iter(claims)) if len(claims) == 1 else None


def _consistent_durations(entries: list[tuple]) -> bool:
    return bool(entries) and all(
        max(abs(left[2][0] - right[2][0]), abs(left[2][1] - right[2][1])) <= 15
        for left in entries for right in entries
    )


def _round_signal(text: str) -> set[str]:
    signals = set()
    if re.search(r"\b(hr|human resources|recruiter|recruitment|talent acquisition|nhân sự|vòng nhân sự)\b", text, re.I):
        signals.add("hr")
    if re.search(r"\b(technical|coding|programming|system design|technical screen|kỹ thuật|lập trình|vòng code)\b", text, re.I):
        signals.add("technical")
    if re.search(r"\b(hiring manager|engineering manager|team lead|manager interview|quản lý tuyển dụng|trưởng nhóm)\b", text, re.I):
        signals.add("hiring_manager")
    return signals


def _infer_level(form_data: dict[str, Any], role: str) -> tuple[str | None, str, float]:
    requested = str(form_data.get("seniority") or "").casefold().strip()
    if requested in SUPPORTED_LEVELS:
        return requested, "user_selected", 1.0
    for level, pattern in _LEVEL_TERMS:
        if pattern.search(role):
            if level == "senior" and re.search(r"\b(staff|principal|architect)\b", role, re.I):
                return None, "outside_supported_scope", 0.0
            return level, "role_title", 0.85
    years = form_data.get("experience_years")
    try:
        years = float(years) if years is not None else None
    except (TypeError, ValueError):
        years = None
    if years is None:
        return None, "needs_confirmation", 0.0
    if years < 1:
        return "fresher", "experience_years_heuristic", 0.35
    if years < 3:
        return "junior", "experience_years_heuristic", 0.4
    if years < 7:
        return "mid", "experience_years_heuristic", 0.4
    return "senior", "experience_years_heuristic", 0.35


def resolve_interview_brief(
    form_data: dict[str, Any] | None,
    knowledge: dict[str, Any] | None,
    recon_data: dict[str, Any] | None,
    *,
    target_company: str,
    target_role: str,
    corpus_version: str | None = None,
) -> dict[str, Any]:
    """Resolve round, level, duration, and a provenance snapshot without inventing evidence."""
    form = form_data or {}
    pack = knowledge or {}
    recon = recon_data or {}
    sources = _sources(recon, pack)
    level, level_source, level_confidence = _infer_level(form, target_role)
    limitations: list[str] = []
    if level is None:
        limitations.append("Seniority is outside the supported IT Intern–Senior scope or needs confirmation.")
    if level_source == "experience_years_heuristic":
        limitations.append("Seniority inferred from years only; titles and actual ownership may differ.")

    requested_round = _round(form.get("interview_round"))
    matching_sources = [s for s in sources if _source_matches(s, target_company, target_role, level)]
    # Generic interview frameworks describe common competencies, not how this company
    # structures its interview rounds. Use only company-specific sources for that inference.
    round_sources = [s for s in matching_sources if s.get("source_type") != "framework"]
    source_round_signals = set()
    round_evidence_ids = []
    for source in round_sources:
        signals = _round_signal(" ".join((source.get("title", ""), source.get("snippet", ""), source.get("content_excerpt", ""))))
        source_round_signals.update(signals)
        if signals:
            round_evidence_ids.append(source["source_id"])
    if requested_round:
        round_value, round_source, round_confidence = requested_round, "user_selected", 1.0
    elif len(source_round_signals) > 1:
        round_value, round_source, round_confidence = "combined", "matching_search_sources_unreviewed", 0.35
    elif source_round_signals:
        round_value = next(iter(source_round_signals))
        round_source, round_confidence = "matching_search_source_unreviewed", 0.3
    else:
        round_value, round_source, round_confidence = "combined", "default_full_practice", 0.25
        limitations.append("No matching round evidence was found; combined practice is a low-confidence proposal.")

    requested_minutes = form.get("duration_minutes")
    try:
        requested_minutes = int(requested_minutes) if requested_minutes is not None else None
    except (TypeError, ValueError):
        requested_minutes = None
    duration_candidates = []
    if not requested_minutes:
        for source in matching_sources:
            source_text = " ".join((source.get("content_excerpt", ""), source.get("snippet", ""), source.get("title", "")))
            if not source_text.strip():
                continue
            duration = _duration_for_round(source, round_value, level)
            if duration:
                url = str(source.get("url") or "")
                host = urlparse(url).netloc.casefold()
                tier = str(source.get("trust_tier") or "unreviewed")
                duration_candidates.append((tier, host, duration, source))
        # Search result snippets are unreviewed. Require independent hosts to agree before using them.
        trusted = [entry for entry in duration_candidates if entry[0] in {"official_verified", "firsthand_reviewed"}]
        duration_range = [30, 45]
        duration_source = "fallback_30_45_minutes"
        duration_evidence = []
        duration_confidence = 0.2
        if trusted and _consistent_durations(trusted):
            tier_priority = {"official_verified": 0, "firsthand_reviewed": 1}
            trusted.sort(key=lambda item: (tier_priority.get(item[0], 9), item[1], item[2]))
            selected = trusted[0]
            duration_range = list(selected[2])
            duration_source = selected[0]
            duration_evidence = [selected[3]["source_id"]]
            duration_confidence = 0.85 if selected[0] == "official_verified" else 0.7
        elif trusted:
            limitations.append("Conflicting reviewed duration claims require confirmation; a generic practice duration is proposed.")
        else:
            framework_candidates = [
                entry for entry in duration_candidates if entry[3].get("source_type") == "framework"
            ]
            unreviewed = [entry for entry in duration_candidates if entry[3].get("source_type") != "framework" and entry[1]]
            # All matching claims must agree; never cherry-pick two convenient hosts.
            if len({entry[1] for entry in unreviewed}) >= 2 and _consistent_durations(unreviewed):
                by_host = {entry[1]: entry for entry in unreviewed}
                agreeing = list(by_host.values())
                duration_range = [round(sum(entry[2][i] for entry in agreeing) / len(agreeing)) for i in (0, 1)]
                duration_source = "independent_unreviewed_search_results"
                duration_evidence = [entry[3]["source_id"] for entry in agreeing]
                duration_confidence = 0.4
            elif framework_candidates and _consistent_durations(framework_candidates):
                selected = framework_candidates[0]
                duration_range = list(selected[2])
                duration_source = "framework"
                duration_evidence = [selected[3]["source_id"]]
                duration_confidence = 0.55
                limitations.append("Duration uses a generic interview framework, not verified company policy.")
            if unreviewed and not _consistent_durations(unreviewed):
                limitations.append("Conflicting unreviewed duration claims were not used to infer company timing.")
            elif unreviewed and duration_source != "independent_unreviewed_search_results":
                limitations.append("A single unreviewed duration result was not treated as company policy.")
    else:
        duration_range = [requested_minutes, requested_minutes]
        duration_source = "user_selected"
        duration_evidence = []
        duration_confidence = 1.0
    if requested_minutes:
        planned_minutes = requested_minutes
    else:
        planned_minutes = round(sum(duration_range) / 2)

    if recon.get("web_recon_status") != "available":
        limitations.append("Company web research is unavailable or returned no source results.")
    if not pack.get("rubric"):
        limitations.append("No retrieved role-specific rubric is available; the plan will be explicitly marked as generic/degraded.")

    return {
        "schema_version": "1.0",
        "status": "needs_confirmation" if level is None or not pack.get("rubric") else "ready",
        "target": {"company": target_company, "role": target_role},
        "seniority": {"value": level, "source": level_source, "confidence": level_confidence},
        "interview_round": {
            "value": round_value,
            "source": round_source,
            "confidence": round_confidence,
            "evidence_ids": [] if requested_round else round_evidence_ids,
        },
        "duration": {
            "planned_minutes": planned_minutes,
            "range_minutes": duration_range,
            "source": duration_source,
            "confidence": duration_confidence,
            "evidence_ids": duration_evidence,
        },
        "source_snapshot": {
            "status": recon.get("web_recon_status", "unavailable"),
            "corpus_version": corpus_version,
            "sources": sources,
        },
        "limitations": list(dict.fromkeys(limitations)),
    }


def shape_stage_plan(stage_plan: Any, brief: dict[str, Any], candidate_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Enforce opening/closing and round-appropriate coverage around an LLM's stage proposal."""
    source = stage_plan if isinstance(stage_plan, dict) else {}
    raw_stages = source.get("stages") if isinstance(source.get("stages"), list) else []
    round_value = brief.get("interview_round", {}).get("value", "combined")
    level = brief.get("seniority", {}).get("value")
    minutes = int(brief.get("duration", {}).get("planned_minutes") or 40)
    # Opening, at least one body stage and Q&A each need one whole minute.
    minutes = max(3, minutes)
    keep_types = {
        "hr": {"warmup", "behavioral", "wrapup"},
        "technical": {"warmup", "technical", "system_design", "behavioral", "wrapup"},
        "hiring_manager": {"warmup", "behavioral", "system_design", "wrapup"},
        "combined": {"warmup", "technical", "system_design", "behavioral", "wrapup"},
    }.get(round_value, {"warmup", "technical", "behavioral", "wrapup"})
    allow_design = level in {"mid", "senior"} and round_value in {"technical", "hiring_manager", "combined"}
    body: list[dict[str, Any]] = []
    for item in raw_stages:
        if not isinstance(item, dict):
            continue
        stage = dict(item)
        name = str(stage.get("name") or stage.get("stage_id") or "Interview")
        text = " ".join((name, " ".join(str(x) for x in stage.get("goals", []) if x)))
        kind = str(stage.get("stage_type") or "").casefold()
        if kind not in {"warmup", "technical", "system_design", "behavioral", "wrapup"}:
            if re.search(r"candidate questions|wrap|closing|close", text, re.I):
                kind = "wrapup"
            elif re.search(r"behavior|motivation|conflict|teamwork|culture|leadership", text, re.I):
                kind = "behavioral"
            elif re.search(r"system design|architecture|scalab", text, re.I):
                kind = "system_design"
            elif re.search(r"intro|warm|cv|background|project|experience", text, re.I):
                kind = "warmup"
            else:
                kind = "technical"
        # The core appends one source-aware candidate Q&A and closing stage below. Keep
        # model-proposed wrapups out of the body so the plan cannot close before Q&A or
        # ask the candidate to close twice.
        if kind == "wrapup" or kind not in keep_types or (kind == "system_design" and not allow_design):
            continue
        stage["stage_type"] = kind
        stage["name"] = name
        stage["goals"] = list(stage.get("goals") or [])
        stage["criteria"] = list(stage.get("criteria") or [])
        stage["target_turns"] = max(1, int(stage.get("target_turns") or stage.get("turns") or 2))
        body.append(stage)

    has_kind = {stage["stage_type"] for stage in body}
    if "warmup" not in has_kind:
        body.insert(0, {
            "stage_id": "experience", "name": "Background and project experience", "stage_type": "warmup",
            "target_turns": 2, "goals": ["Ask the candidate to introduce relevant experience and explain one real project."], "criteria": [],
        })
    if round_value in {"hr", "combined"} and "behavioral" not in has_kind:
        body.append({
            "stage_id": "motivation_behavioral", "name": "Motivation and behavioral examples", "stage_type": "behavioral",
            "target_turns": 2, "goals": ["Explore motivation and one specific example using situation, action, and result."], "criteria": [],
        })
    if round_value in {"technical", "combined"} and "technical" not in has_kind:
        body.append({
            "stage_id": "technical_depth", "name": "Role-specific technical depth", "stage_type": "technical",
            "target_turns": 3, "goals": ["Probe evidence in the candidate's projects, technologies, and target-role requirements."], "criteria": [],
        })
    if round_value == "hiring_manager" and "behavioral" not in has_kind:
        body.append({
            "stage_id": "ownership", "name": "Ownership and collaboration", "stage_type": "behavioral",
            "target_turns": 2, "goals": ["Explore ownership, tradeoffs, collaboration, and impact appropriate to the role level."], "criteria": [],
        })

    candidate_name = str((candidate_profile or {}).get("candidate_name") or "the candidate")
    opening = {
        "stage_id": "opening", "name": "Opening and introduction", "stage_type": "warmup", "target_turns": 1,
        "goals": [f"Greet {candidate_name} by name when known, introduce the interview round and planned duration, then invite a short introduction. Do not ask age or unrelated personal details."],
        "question_candidates": [{
            "question_id": "opening_introduction",
            "text": "To start, could you introduce yourself and summarize the experience most relevant to this role?",
        }],
        "criteria": [],
    }
    candidate_questions = {
        "stage_id": "candidate_questions", "name": "Candidate questions", "stage_type": "wrapup", "target_turns": 1,
        "goals": [
            "Invite the candidate to ask about the role, team, or process. On the final response, answer their question only from available source evidence; if the answer is not verified, say so clearly. Thank the candidate and close without inventing next steps or asking for unrelated personal information."
        ],
        "question_candidates": [{
            "question_id": "candidate_questions_invite",
            "text": "Before we wrap up, what questions do you have about the role, team, or interview process?",
        }],
        "criteria": [],
    }
    body = [stage for stage in body if stage.get("stage_id") not in {"opening", "candidate_questions", "closing"}]
    stages = [opening, *body, candidate_questions]
    total_turns = max(6, min(60, round(minutes / 2)))
    main_stages = stages[1:-1]
    main_budget = max(1, total_turns - 2)
    weights = [max(1, int(stage.get("target_turns") or 1)) for stage in main_stages]
    weight_total = sum(weights) or 1
    allocations = [max(1, int(main_budget * weight / weight_total)) for weight in weights]
    while sum(allocations) > main_budget and len(allocations) > 1:
        largest = max(range(len(allocations)), key=allocations.__getitem__)
        if allocations[largest] <= 1:
            break
        allocations[largest] -= 1
    while sum(allocations) < main_budget and allocations:
        largest = max(range(len(weights)), key=weights.__getitem__)
        allocations[largest] += 1
    for stage, allocation in zip(main_stages, allocations):
        stage["target_turns"] = allocation
        stage["turns"] = allocation
    used: set[str] = set()
    for index, stage in enumerate(stages, start=1):
        stage_id = str(stage.get("stage_id") or f"stage_{index}")
        if stage_id in used:
            stage_id = f"{stage_id}_{index}"
        used.add(stage_id)
        stage["stage_id"] = stage_id
        stage["turns"] = int(stage.get("target_turns") or 1)

    reserve = min(max(2, round(minutes * 0.15)), max(2, minutes - 2))
    opening_minutes = max(1, round(minutes * 0.05))
    main_minutes = minutes - reserve - opening_minutes
    if main_minutes < 1:
        # Very short plans: shrink the Q&A reserve so at least one body minute remains.
        reserve = max(1, minutes - opening_minutes - 1)
        main_minutes = minutes - reserve - opening_minutes
    main = stages[1:-1]
    # Keep one minute per stage even for an overlong model proposal. Merge excess
    # stages' goals into the final body stage rather than inventing extra time.
    if len(main) > main_minutes:
        retained = main[:main_minutes]
        for extra in main[main_minutes:]:
            retained[-1]["goals"].extend(extra.get("goals") or [])
        stages = [opening, *retained, candidate_questions]
        main = retained
    weight_total = sum(max(1, int(stage["target_turns"])) for stage in main) or 1
    for stage in stages:
        if stage["stage_id"] == "opening":
            stage["time_budget_minutes"] = opening_minutes
        elif stage["stage_id"] == "candidate_questions":
            stage["time_budget_minutes"] = reserve
    remaining = main_minutes - len(main)
    shares = [remaining * max(1, int(stage["target_turns"])) / weight_total for stage in main]
    budgets = [1 + int(share) for share in shares]
    for index in sorted(range(len(main)), key=lambda i: shares[i] % 1, reverse=True)[:main_minutes - sum(budgets)]:
        budgets[index] += 1
    for stage, budget in zip(main, budgets):
        stage["time_budget_minutes"] = budget
    shaped = dict(source)
    shaped.update({
        "stages": stages,
        "interview_round": round_value,
        "planned_minutes": minutes,
        "duration_range_minutes": brief.get("duration", {}).get("range_minutes"),
        "duration_source": brief.get("duration", {}).get("source"),
        "estimated_turns": sum(int(stage["target_turns"]) for stage in stages),
        "time_allocation_basis": "stage budgets sum to the planned duration, including introduction and reserved candidate Q&A; turn estimates assume two minutes per answer",
    })
    return shaped
