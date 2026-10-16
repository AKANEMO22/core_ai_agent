"""Pre-flight Reconnaissance Profiler Agent coordinating inputs, research tools, and blueprint synthesis.

Pipeline chuẩn bị phiên trước khi speaker bắt đầu: run_profiling đọc form/CV
trong workspace, tạo candidate_profile trước, dùng role/skills/kinh nghiệm từ
profile để lấy recon và knowledge (RAG), rồi tạo company_intel, interview_spec
và stage_plan (hoặc nhận trọn bộ từ LLM).
Knowledge thành công sẽ dùng để neo tiêu chí/câu hỏi và ghi thêm rubric;
retrieval lỗi được ghi degraded để tách khỏi kết quả profiling. Đầu vào ứng viên
được bọc untrusted trong prompt và thông báo lỗi được che dữ liệu cá nhân.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.language import interview_language_instruction, workspace_interview_language
from core.profiler.brief import resolve_interview_brief, shape_stage_plan
from core.profiler.documents import DocumentExtractionError, extract_document_text
from core.profiler.evidence import attach_evidence_probes
from core.profiler.prompts import (
    PROFILER_SYSTEM_PROMPT,
    build_candidate_profile_prompt,
    build_company_intel_prompt,
    build_interview_spec_prompt,
    build_stage_plan_prompt,
    sanitize_prompt_field,
    wrap_untrusted_candidate_data,
)
from core.profiler.tools import SafeWebSearchTool
from core.retrieval.knowledge import InterviewKnowledgeService, ground_stage_plan
from core.runtime.agent import sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.runtime.structured import (  # noqa: F401  (extract_json_payload re-exported)
    StructuredOutputError,
    extract_json_payload,
    run_structured,
)
from core.runtime.tooling import RuntimeTool
from core.workspace.session import SessionWorkspace

logger = logging.getLogger(__name__)


def redact_sensitive_candidate_data(
    error_text: str,
    form_data: Optional[Dict[str, Any]] = None,
    cv_text: Optional[str] = None,
) -> str:
    """Recursively scrub candidate personal data, CV text, and identifiers from error messages.

    Args:
        error_text: Raw error message string.
        form_data: Optional structured candidate form data dictionary.
        cv_text: Optional plain text resume or CV content.

    Returns:
        str: Fully scrubbed error string safe for public logs and API responses.
    """
    clean = sanitize_secrets(error_text)

    # 1. Redact emails and phone numbers via regex
    clean = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "[REDACTED_EMAIL]",
        clean,
    )
    clean = re.sub(r"\+?\d[\d -]{8,}\d", "[REDACTED_PHONE]", clean)

    # 2. Redact full CV text and significant lines
    if cv_text:
        if cv_text in clean:
            clean = clean.replace(cv_text, "[CANDIDATE_CV_REDACTED]")
        for line in cv_text.splitlines():
            line_str = line.strip()
            if len(line_str) >= 5 and line_str in clean:
                clean = clean.replace(line_str, "[REDACTED]")

    # 3. Recursively extract all strings from form_data (including nested dicts/lists)
    def extract_strings(obj: Any) -> List[str]:
        res: List[str] = []
        if isinstance(obj, str):
            s = obj.strip()
            if len(s) >= 3:
                res.append(s)
        elif isinstance(obj, dict):
            for v in obj.values():
                res.extend(extract_strings(v))
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                res.extend(extract_strings(item))
        return res

    if form_data:
        for s in sorted(extract_strings(form_data), key=len, reverse=True):
            if s in clean:
                clean = clean.replace(s, "[REDACTED]")

    return clean


def _shape_and_ground_stage_plan(
    stage_plan: Any,
    brief: Dict[str, Any],
    candidate_profile: Dict[str, Any],
    knowledge: Dict[str, Any],
    *,
    knowledge_ok: bool,
    job_description: Optional[str] = None,
) -> Any:
    """Ground assessable stages after shaping while keeping opening and Q&A unscored.

    CV/JD evidence probes are attached after grounding on both the grounded and the
    degraded (knowledge unavailable) path.
    """
    shaped = shape_stage_plan(stage_plan, brief, candidate_profile)
    profile = candidate_profile if isinstance(candidate_profile, dict) else {}
    if not knowledge_ok or not isinstance(shaped, dict) or not isinstance(shaped.get("stages"), list):
        return attach_evidence_probes(shaped, profile, job_description)
    stages = shaped["stages"]
    if len(stages) < 2:
        return attach_evidence_probes(shaped, profile, job_description)
    body = ground_stage_plan({"stages": stages[1:-1]}, knowledge)
    if not isinstance(body, dict) or not isinstance(body.get("stages"), list):
        return attach_evidence_probes(shaped, profile, job_description)
    grounded = {
        **shaped,
        "stages": [stages[0], *body["stages"], stages[-1]],
        "rubric_version": body.get("rubric_version", knowledge.get("rubric", {}).get("rubric_version")),
    }
    return attach_evidence_probes(grounded, profile, job_description)


def _profile_target_role(candidate_profile: Dict[str, Any]) -> Optional[str]:
    """Role the CV supports: a stated objective first, else the most recent position title."""
    stated = str(candidate_profile.get("target_role") or "").strip()
    if stated:
        return stated
    experience = candidate_profile.get("experience")
    if isinstance(experience, list):
        for item in experience:
            role = str(item.get("role") or "").strip() if isinstance(item, dict) else ""
            if role:
                return role
    return None


def merge_profile_into_form(
    form_data: Optional[Dict[str, Any]], candidate_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill retrieval inputs the form left empty from the extracted candidate profile.

    Explicit form values always win; the profile only supplies what the candidate did not type.
    """
    merged = dict(form_data or {})
    if merged.get("experience_years") is None and candidate_profile.get("experience_years") is not None:
        merged["experience_years"] = candidate_profile["experience_years"]
    if not str(merged.get("target_role") or "").strip():
        role = _profile_target_role(candidate_profile)
        if role:
            merged["target_role"] = role
    if not str(merged.get("target_company") or "").strip() and candidate_profile.get("target_company"):
        merged["target_company"] = str(candidate_profile["target_company"])
    existing_skills = merged.get("skills") if isinstance(merged.get("skills"), list) else []
    profile_skills = candidate_profile.get("skills") if isinstance(candidate_profile.get("skills"), list) else []
    reviewed_skills = [
        str(item.get("name") or "").strip() if isinstance(item, dict) else str(item).strip()
        for item in profile_skills
    ]
    merged["skills"] = list(dict.fromkeys(
        [str(item).strip() for item in existing_skills if str(item).strip()]
        + [item for item in reviewed_skills if item]
    ))
    return merged


def _normalize_company_intel(
    value: Any, target_company: str, recon_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Keep company claims only when the model linked them to retrieved source IDs."""
    payload = value if isinstance(value, dict) else {}
    sources = [
        source for source in (recon_data or {}).get("web_recon", [])
        if isinstance(source, dict) and source.get("source_id")
    ]
    source_by_id = {str(source["source_id"]): source for source in sources}
    requested_ids = payload.get("source_ids") if isinstance(payload.get("source_ids"), list) else []
    valid_ids = list(dict.fromkeys(str(source_id) for source_id in requested_ids if str(source_id) in source_by_id))
    limitations = list(payload.get("limitations") or []) if isinstance(payload.get("limitations"), list) else []
    if not valid_ids:
        return {
            "company": target_company,
            "culture": None,
            "tech_stack": [],
            "hiring_bar": None,
            "interview_focus": None,
            "source_ids": [],
            "evidence_status": "unavailable",
            "limitations": list(dict.fromkeys(limitations + [
                "No reviewed source evidence supports company-specific claims; use only the generic role framework."
            ])),
        }
    return {
        "company": target_company,
        "culture": payload.get("culture"),
        "tech_stack": payload.get("tech_stack") if isinstance(payload.get("tech_stack"), list) else [],
        "hiring_bar": payload.get("hiring_bar"),
        "interview_focus": payload.get("interview_focus"),
        "source_ids": valid_ids,
        "evidence_status": "unreviewed_search_results",
        "limitations": list(dict.fromkeys(limitations + [
            "Search result snippets are unreviewed and may be stale or copied; verify company-specific claims before treating them as policy."
        ])),
    }


class ProfilerAgent:
    """Pre-flight intelligence agent synthesizing interview blueprints and stage progression plans.

    Reads untrusted candidate inputs (form or CV), executes safe reconnaissance tools,
    enforces defensive boundary isolation, and atomically creates 4 foundational artifacts:
    1. context/company_intel.json
    2. context/candidate_profile.json
    3. context/interview_spec.json
    4. context/stage_plan.json

    When a knowledge service is configured, it also retrieves the rubric, question bank and
    company knowledge, grounds the stage plan in them, and writes:
    5. context/retrieved_knowledge.json (status "ok" or "degraded")
    6. context/rubric.json (only when retrieval succeeded)

    Attributes:
        llm_client: Runtime LLM provider adapter.
        tools: Registered research and intelligence tools.
        knowledge_service: Optional retrieval-backed knowledge service.
    """

    def __init__(
        self,
        llm_client: RuntimeLLMClient,
        tools: Optional[List[RuntimeTool]] = None,
        knowledge_service: Optional[InterviewKnowledgeService] = None,
    ) -> None:
        self.llm_client = llm_client
        self.tools = tools if tools is not None else [SafeWebSearchTool()]
        self.tools_map: Dict[str, RuntimeTool] = {t.name: t for t in self.tools}
        self.knowledge_service = knowledge_service

    async def _query_model(
        self,
        user_prompt: str,
        system_prompt: str = PROFILER_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Request a JSON object (provider JSON mode when available) with one repair round."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return await run_structured(self.llm_client, messages, {"type": "object"}, max_repairs=1)
        except StructuredOutputError as e:
            raise ValueError("Could not parse valid JSON object from model output") from e

    async def _gather_recon(
        self,
        company_name: str,
        target_role: Optional[str] = None,
        workspace: Optional[SessionWorkspace] = None,
    ) -> Optional[Dict[str, Any]]:
        """Coordinate market intel and safe web search tools for company reconnaissance."""
        recon_data: Dict[str, Any] = {}

        if "market_intel" in self.tools_map:
            try:
                intel_res = await self.tools_map["market_intel"].execute(
                    {"company_name": company_name, "target_role": target_role},
                    workspace=workspace,
                )
                if isinstance(intel_res, dict) and intel_res.get("status") != "blocked":
                    recon_data["market_intel"] = intel_res
            except Exception as e:
                logger.warning("Market intel tool execution error (skipped): %s", sanitize_secrets(str(e)))

        if "safe_web_search" in self.tools_map:
            search_queries = [
                f'"{company_name}" {target_role or "software engineer"} official careers interview process rounds',
                f'"{company_name}" {target_role or "software engineer"} interview experience technical questions',
                f'"{company_name}" recruiter hiring manager interview duration {target_role or "software engineer"}',
            ]
            results = await asyncio.gather(
                *[
                    self.tools_map["safe_web_search"].execute(
                        {"query": query, "max_results": 4}, workspace=workspace
                    )
                    for query in search_queries
                ],
                return_exceptions=True,
            )
            web_sources: List[Dict[str, Any]] = []
            statuses = []
            seen_urls = set()
            for result in results:
                if isinstance(result, Exception):
                    statuses.append("error")
                    logger.warning(
                        "Safe web search tool execution error (skipped): %s",
                        sanitize_secrets(str(result)),
                    )
                    continue
                if not isinstance(result, dict):
                    statuses.append("error")
                    continue
                statuses.append(str(result.get("status", "unknown")))
                for source in result.get("results") or []:
                    if not isinstance(source, dict):
                        continue
                    url = str(source.get("url") or source.get("source") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    web_sources.append(source)
            recon_data["web_recon"] = web_sources
            if web_sources:
                recon_data["web_recon_status"] = "available"
            elif "error" in statuses:
                recon_data["web_recon_status"] = "error"
            elif "success" in statuses:
                recon_data["web_recon_status"] = "empty"
            else:
                recon_data["web_recon_status"] = "unavailable"
            # Read a bounded number of public result pages so the interview brief can use
            # actual excerpts rather than search-result summaries alone. Every page remains
            # unreviewed evidence and is stored as a short excerpt, never as a full-site copy.
            fetch_slots = asyncio.Semaphore(4)

            async def fetch_source(source: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
                async with fetch_slots:
                    fetched = await self.tools_map["safe_web_search"].execute(
                        {"query": source.get("query") or "company interview source", "target_url": source.get("url")},
                        workspace=workspace,
                    )
                return source, fetched if isinstance(fetched, dict) else {"status": "error"}

            fetch_candidates = []
            for query in search_queries:
                fetch_candidates.extend([source for source in web_sources if source.get("query") == query][:2])
            page_results = await asyncio.gather(
                *(fetch_source(source) for source in fetch_candidates[:6]),
                return_exceptions=True,
            )
            for result in page_results:
                if isinstance(result, Exception):
                    continue
                source, fetched = result
                if fetched.get("status") == "success" and fetched.get("content"):
                    source["fetched_text"] = str(fetched["content"])[:2_000]
                    source["fetched_content_hash"] = fetched.get("content_hash")
                    source["fetch_status"] = "excerpt_saved"
                    source["fetched_at"] = fetched.get("fetched_at")
                else:
                    source["fetch_status"] = str(fetched.get("reason") or fetched.get("status") or "unavailable")

        return recon_data if recon_data else None

    async def _retrieve_knowledge(
        self,
        target_company: str,
        target_role: str,
        form_data: Optional[Dict[str, Any]],
        cv_text: Optional[str],
    ) -> Dict[str, Any]:
        """Retrieve the interview knowledge pack, degrading to an error record on failure.

        Only role, skills, experience, company and job description reach the retrieval backend.

        Returns:
            Dict[str, Any]: The pack with status "ok", or {"status": "degraded", "error": ...}.
        """
        if self.knowledge_service is None:
            return {"status": "disabled"}
        skills: List[str] = []
        experience_years: Optional[int] = None
        job_description: Optional[str] = None
        seniority: Optional[str] = None

        if form_data:
            raw_skills = form_data.get("skills")
            if isinstance(raw_skills, list):
                skills = [sanitize_prompt_field(str(s), max_length=60) for s in raw_skills if s]
            try:
                raw_years = form_data.get("experience_years")
                experience_years = float(raw_years) if raw_years is not None else None
            except (TypeError, ValueError):
                experience_years = None
            raw_jd = form_data.get("job_description")
            job_description = str(raw_jd) if raw_jd else None
            raw_seniority = form_data.get("seniority")
            seniority = str(raw_seniority) if raw_seniority else None

        if not skills and cv_text:
            skills_match = re.search(r"Skills:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
            if skills_match:
                skills = [
                    sanitize_prompt_field(s, max_length=60)
                    for s in skills_match.group(1).split(",")
                ]
        skills = [s for s in skills if s]

        try:
            pack = await self.knowledge_service.build_pack(
                target_company=target_company,
                target_role=target_role,
                skills=skills,
                experience_years=experience_years,
                job_description=job_description,
                seniority=seniority,
            )
            return {"status": "ok", **pack}
        except Exception as e:
            clean_error = redact_sensitive_candidate_data(
                str(e), form_data=form_data, cv_text=cv_text
            )
            logger.warning(
                "Knowledge retrieval failed; continuing without grounding: %s", clean_error
            )
            return {
                "status": "degraded",
                "retrieval_backend": self.knowledge_service.backend_name,
                "error": clean_error,
            }

    def _load_inputs(self, workspace: SessionWorkspace) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Inspect workspace inputs directory and load form input or CV documents.

        Returns:
            tuple[Optional[dict], Optional[str]]: (form_data, cv_text)
        """
        form_data: Optional[Dict[str, Any]] = None
        cv_text: Optional[str] = None

        # 1. Check for form_input.json
        form_path = workspace.inputs_dir / "form_input.json"
        if form_path.exists():
            try:
                raw_form = workspace.read_json("inputs/form_input.json")
                if isinstance(raw_form, dict):
                    # Unwrap standard artifact envelope if wrapped
                    form_data = (
                        raw_form["payload"]
                        if "payload" in raw_form and isinstance(raw_form["payload"], dict)
                        else raw_form
                    )
            except Exception as e:
                logger.warning("Failed to parse inputs/form_input.json: %s", sanitize_secrets(str(e)))

        # 2. Check for cv_input.* files
        cv_candidates = list(workspace.inputs_dir.glob("cv_input.*"))
        if cv_candidates:
            cv_file = cv_candidates[0]
            ext = cv_file.suffix.lower()
            try:
                if ext in (".txt", ".md"):
                    cv_text = cv_file.read_text(encoding="utf-8")
                elif ext == ".json":
                    raw_cv = workspace.read_json(f"inputs/{cv_file.name}")
                    cv_text = json.dumps(
                        raw_cv.get("payload", raw_cv), ensure_ascii=False, indent=2
                    )
                elif ext in (".pdf", ".docx"):
                    try:
                        _extension, cv_text = extract_document_text(cv_file.name, cv_file.read_bytes())
                    except DocumentExtractionError as e:
                        logger.warning("CV document has no usable text: %s", sanitize_secrets(str(e)))
                else:
                    cv_text = cv_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning("Failed to read CV input file: %s", sanitize_secrets(str(e)))

        return form_data, cv_text

    async def rebuild_reviewed_plan(
        self,
        workspace: SessionWorkspace,
        form_data: Dict[str, Any],
        candidate_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Re-retrieve and regenerate the plan after candidate-reviewed inputs change."""
        form_data = merge_profile_into_form(form_data, candidate_profile)

        def payload(path: str) -> Dict[str, Any]:
            if not workspace.exists(path):
                return {}
            raw = workspace.read_json(path)
            return raw.get("payload", raw) if isinstance(raw, dict) and isinstance(raw.get("payload", raw), dict) else {}

        recon_data = payload("context/company_recon.json")
        recon_data.pop("knowledge_base", None)
        _stored_form, cv_text = await asyncio.to_thread(self._load_inputs, workspace)
        knowledge = await self._retrieve_knowledge(
            target_company=str(form_data.get("target_company") or "Target Company"),
            target_role=str(form_data.get("target_role") or "Software Engineer"),
            form_data=form_data,
            cv_text=cv_text,
        )
        if knowledge.get("status") == "ok" and knowledge.get("company_context"):
            recon_data["knowledge_base"] = [
                {
                    "topic": item.get("topic"),
                    "content": item.get("content"),
                    "confidence": item.get("confidence"),
                    "source": item.get("source"),
                    "source_id": item.get("source_id") or item.get("source"),
                    "is_generic": item.get("is_generic", True),
                }
                for item in knowledge["company_context"]
            ]
        rubric = knowledge.get("rubric") if isinstance(knowledge.get("rubric"), dict) else {}
        brief = resolve_interview_brief(
            form_data,
            knowledge,
            recon_data,
            target_company=str(form_data.get("target_company") or "Target Company"),
            target_role=str(form_data.get("target_role") or "Software Engineer"),
            corpus_version=rubric.get("rubric_version"),
        )
        interview_language = str(form_data.get("interview_language") or "vi")
        profiling_system_prompt = (
            PROFILER_SYSTEM_PROMPT + "\n\n" + interview_language_instruction(interview_language)
            + "\nWrite generated natural-language blueprint descriptions, stage goals and "
            + "question text in the interview language; preserve all schema keys and IDs."
        )
        spec_prompt = build_interview_spec_prompt(
            company_intel=payload("context/company_intel.json"),
            candidate_profile=candidate_profile,
            target_role=str(form_data.get("target_role") or "Software Engineer"),
            interview_brief=brief,
        )
        spec_response = await self._query_model(spec_prompt, system_prompt=profiling_system_prompt)
        interview_spec = spec_response.get("interview_spec", spec_response)
        if not isinstance(interview_spec, dict):
            interview_spec = {}
        interview_spec = {
            **interview_spec,
            "interview_language": interview_language,
            "interview_round": brief["interview_round"]["value"],
            "interview_round_provenance": brief["interview_round"],
            "planned_minutes": brief["duration"]["planned_minutes"],
            "duration_range_minutes": brief["duration"]["range_minutes"],
            "duration_provenance": brief["duration"],
            "seniority": brief["seniority"],
        }
        minutes = int(brief["duration"]["planned_minutes"])
        prompt = build_stage_plan_prompt(
            interview_spec=interview_spec,
            candidate_profile=candidate_profile,
            total_turns=max(6, round(minutes / 2)),
            competencies=rubric.get("competencies") if knowledge.get("status") == "ok" else None,
            interview_brief=brief,
            job_description=form_data.get("job_description"),
        )
        response = await self._query_model(prompt, system_prompt=profiling_system_prompt)
        stage_plan = response.get("stage_plan", response)
        stage_plan = _shape_and_ground_stage_plan(
            stage_plan,
            brief,
            candidate_profile,
            knowledge,
            knowledge_ok=knowledge.get("status") == "ok",
            job_description=form_data.get("job_description"),
        )
        return {
            "knowledge": knowledge,
            "brief": brief,
            "interview_spec": interview_spec,
            "stage_plan": stage_plan,
            "company_recon": recon_data,
        }

    async def run_profiling(self, workspace: SessionWorkspace) -> Dict[str, Any]:
        """Execute pre-flight reconnaissance profiling pipeline for the active session workspace.

        Args:
            workspace: The isolated SessionWorkspace holding candidate inputs and target context/.

        Returns:
            Dict[str, Any]: Pipeline outcome status and dictionary of written artifact paths.
        """
        form_data: Optional[Dict[str, Any]] = None
        cv_text: Optional[str] = None

        try:
            form_data, cv_text = self._load_inputs(workspace)
            interview_language = workspace_interview_language(workspace)
            profiling_system_prompt = (
                PROFILER_SYSTEM_PROMPT + "\n\n" + interview_language_instruction(interview_language)
                + "\nWrite generated natural-language blueprint descriptions, stage goals and "
                "question text in the interview language; preserve all schema keys and IDs."
            )

            if not form_data and not cv_text:
                return {
                    "status": "FAILED",
                    "error": "No candidate input found in inputs/ directory (expected form_input.json or cv_input.*)",
                    "session_id": workspace.session_id,
                }

            # Step 1: Extract candidate_profile first; role, skills and experience for retrieval come from it.
            profile_prompt = build_candidate_profile_prompt(form_data=form_data, cv_text=cv_text)
            first_response = await self._query_model(profile_prompt, system_prompt=profiling_system_prompt)
            required_keys = ("company_intel", "candidate_profile", "interview_spec", "stage_plan")
            # Có đủ bốn artifact thì dùng bundle; nếu thiếu, các bước sau phải chờ dữ liệu bước trước.
            bundle = first_response if all(k in first_response for k in required_keys) else None
            candidate_profile = bundle["candidate_profile"] if bundle else first_response.get("candidate_profile", first_response)
            if not isinstance(candidate_profile, dict):
                candidate_profile = {}

            # Form values win; empty fields (CV-only sessions) are filled from the extracted profile.
            form_data = merge_profile_into_form(form_data, candidate_profile)
            raw_company = form_data.get("target_company") or form_data.get("company")
            raw_role = form_data.get("target_role") or form_data.get("role")
            if cv_text and not raw_company:
                comp_match = re.search(r"Target Company:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
                raw_company = comp_match.group(1).strip() if comp_match else None
            if cv_text and not raw_role:
                role_match = re.search(r"Role:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
                raw_role = role_match.group(1).strip() if role_match else None

            target_company = sanitize_prompt_field(raw_company, default="Target Company")
            target_role = sanitize_prompt_field(raw_role, default="Software Engineer")

            # Assemble untrusted candidate context upfront
            candidate_data_parts: List[str] = []
            if form_data:
                candidate_data_parts.append(f"Form Input:\n{json.dumps(form_data, indent=2)}")
            if cv_text:
                candidate_data_parts.append(f"CV/Resume Text:\n{cv_text}")
            combined_candidate = (
                "\n\n".join(candidate_data_parts)
                if candidate_data_parts
                else "No candidate data provided."
            )
            wrapped_candidate = wrap_untrusted_candidate_data(combined_candidate)

            # Step 2: Recon và RAG chạy song song, nhưng chỉ sau khi đã có candidate_profile.
            recon_coro = self._gather_recon(target_company, target_role, workspace=workspace)
            knowledge: Optional[Dict[str, Any]] = None
            if self.knowledge_service is not None:
                recon_data, knowledge = await asyncio.gather(
                    recon_coro,
                    self._retrieve_knowledge(target_company, target_role, form_data, cv_text),
                )
            else:
                recon_data = await recon_coro
            knowledge_ok = knowledge is not None and knowledge.get("status") == "ok"
            recon_data = dict(recon_data or {})

            if knowledge_ok and knowledge.get("company_context"):
                recon_data["knowledge_base"] = [
                    {
                        "topic": c.get("topic"),
                        "content": c.get("content"),
                        "confidence": c.get("confidence"),
                        "source": c.get("source"),
                        "source_id": c.get("source_id") or c.get("source"),
                        "is_generic": c.get("is_generic", True),
                    }
                    for c in knowledge["company_context"]
                ]

            corpus_version = (
                (knowledge or {}).get("rubric", {}).get("rubric_version")
                if isinstance((knowledge or {}).get("rubric"), dict) else None
            )
            interview_brief = resolve_interview_brief(
                form_data,
                knowledge,
                recon_data,
                target_company=target_company,
                target_role=target_role,
                corpus_version=corpus_version,
            )

            if bundle:
                company_intel = _normalize_company_intel(bundle["company_intel"], target_company, recon_data)
                interview_spec = bundle["interview_spec"]
                stage_plan = bundle["stage_plan"]
            else:
                # Step 3: Generate only source-linked company context; unsupported facts are removed below.
                company_prompt = build_company_intel_prompt(
                    company_name=target_company,
                    target_role=target_role,
                    recon_data=recon_data,
                    candidate_context=wrapped_candidate,
                    interview_brief=interview_brief,
                )
                company_res = await self._query_model(company_prompt, system_prompt=profiling_system_prompt)
                company_intel = _normalize_company_intel(
                    company_res.get("company_intel", company_res), target_company, recon_data
                )

                # Step 4: Generate interview spec
                spec_prompt = build_interview_spec_prompt(
                    company_intel=company_intel,
                    candidate_profile=candidate_profile,
                    target_role=target_role,
                    interview_brief=interview_brief,
                )
                spec_res = await self._query_model(spec_prompt, system_prompt=profiling_system_prompt)
                interview_spec = spec_res.get("interview_spec", spec_res)
                interview_spec = {**interview_spec, "interview_language": interview_language}

                # Step 5: Generate stage plan
                plan_prompt = build_stage_plan_prompt(
                    interview_spec=interview_spec,
                    candidate_profile=candidate_profile,
                    total_turns=max(6, round(interview_brief["duration"]["planned_minutes"] / 2)),
                    competencies=knowledge["rubric"]["competencies"] if knowledge_ok else None,
                    interview_brief=interview_brief,
                    job_description=(form_data or {}).get("job_description"),
                )
                plan_res = await self._query_model(plan_prompt, system_prompt=profiling_system_prompt)
                stage_plan = plan_res.get("stage_plan", plan_res)

            if not isinstance(candidate_profile, dict):
                candidate_profile = {}
            if form_data and form_data.get("candidate_name"):
                candidate_profile["candidate_name"] = str(form_data["candidate_name"])
            if not isinstance(interview_spec, dict):
                interview_spec = {}
            # User language, round and duration preferences cannot be overwritten by an LLM bundle.
            interview_spec = {
                **interview_spec,
                "interview_language": interview_language,
                "interview_round": interview_brief["interview_round"]["value"],
                "interview_round_provenance": interview_brief["interview_round"],
                "planned_minutes": interview_brief["duration"]["planned_minutes"],
                "duration_range_minutes": interview_brief["duration"]["range_minutes"],
                "duration_provenance": interview_brief["duration"],
                "seniority": interview_brief["seniority"],
            }
            # Shape first, then ground every assessable stage including deterministic fallbacks.
            stage_plan = _shape_and_ground_stage_plan(
                stage_plan,
                interview_brief,
                candidate_profile,
                knowledge,
                knowledge_ok=knowledge_ok,
                job_description=(form_data or {}).get("job_description"),
            )

            recon_data.setdefault("web_recon", [])
            recon_data.setdefault("web_recon_status", "unavailable")
            recon_data["prepared_at"] = datetime.now(timezone.utc).isoformat()

            # Atomically write all 4 artifacts through SessionWorkspace
            # Mỗi artifact được ghi riêng; chuỗi bốn lần ghi không phải một transaction toàn bộ pipeline.
            p1 = workspace.write_artifact(
                "context/company_intel.json",
                artifact_type="company_intel",
                payload=company_intel,
            )
            p2 = workspace.write_artifact(
                "context/candidate_profile.json",
                artifact_type="candidate_profile",
                payload=candidate_profile,
            )
            p3 = workspace.write_artifact(
                "context/interview_spec.json",
                artifact_type="interview_spec",
                payload=interview_spec,
            )
            p4 = workspace.write_artifact(
                "context/stage_plan.json",
                artifact_type="stage_plan",
                payload=stage_plan,
            )
            p5 = workspace.write_artifact(
                "context/interview_brief.json",
                artifact_type="interview_brief",
                payload=interview_brief,
            )
            p6 = workspace.write_artifact(
                "context/company_recon.json",
                artifact_type="company_recon",
                payload=recon_data,
            )

            artifacts = {
                "company_intel": str(p1),
                "candidate_profile": str(p2),
                "interview_spec": str(p3),
                "stage_plan": str(p4),
                "interview_brief": str(p5),
                "company_recon": str(p6),
            }
            if knowledge is not None:
                artifacts["retrieved_knowledge"] = str(
                    workspace.write_artifact(
                        "context/retrieved_knowledge.json",
                        artifact_type="retrieved_knowledge",
                        payload=knowledge,
                    )
                )
            if knowledge_ok:
                artifacts["rubric"] = str(
                    workspace.write_artifact(
                        "context/rubric.json",
                        artifact_type="rubric",
                        payload=knowledge["rubric"],
                    )
                )

            return {
                "status": "SUCCESS",
                "session_id": workspace.session_id,
                "retrieval_status": knowledge["status"] if knowledge is not None else "disabled",
                "artifacts": artifacts,
            }

        except Exception as e:
            # Strictly scrub credentials, API keys, and candidate personal data from error output
            clean_error = redact_sensitive_candidate_data(
                str(e),
                form_data=form_data,
                cv_text=cv_text,
            )
            logger.error("ProfilerAgent execution failed: %s", clean_error)
            return {
                "status": "FAILED",
                "error": clean_error,
                "session_id": workspace.session_id,
            }
