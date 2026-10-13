"""Pre-flight Reconnaissance Profiler Agent coordinating inputs, research tools, and blueprint synthesis."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.profiler.prompts import (
    PROFILER_SYSTEM_PROMPT,
    build_candidate_profile_prompt,
    build_company_intel_prompt,
    build_interview_spec_prompt,
    build_stage_plan_prompt,
    sanitize_prompt_field,
    wrap_untrusted_candidate_data,
)
from core.profiler.tools import MarketIntelTool, SafeWebSearchTool
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
        self.tools = tools if tools is not None else [SafeWebSearchTool(), MarketIntelTool()]
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
            try:
                search_query = f"{company_name} engineering culture tech stack"
                search_res = await self.tools_map["safe_web_search"].execute(
                    {"query": search_query, "max_results": 3},
                    workspace=workspace,
                )
                if isinstance(search_res, dict) and search_res.get("status") != "blocked":
                    recon_data["web_recon"] = search_res.get("results", [])
            except Exception as e:
                logger.warning("Safe web search tool execution error (skipped): %s", sanitize_secrets(str(e)))

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

        if form_data:
            raw_skills = form_data.get("skills")
            if isinstance(raw_skills, list):
                skills = [sanitize_prompt_field(str(s), max_length=60) for s in raw_skills if s]
            try:
                raw_years = form_data.get("experience_years")
                experience_years = int(raw_years) if raw_years is not None else None
            except (TypeError, ValueError):
                experience_years = None
            raw_jd = form_data.get("job_description")
            job_description = str(raw_jd) if raw_jd else None

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
                elif ext == ".pdf":
                    try:
                        import pypdf

                        reader = pypdf.PdfReader(str(cv_file))
                        pages = [p.extract_text() or "" for p in reader.pages]
                        cv_text = "\n".join(pages)
                    except Exception:
                        cv_text = cv_file.read_text(encoding="utf-8", errors="ignore")
                else:
                    cv_text = cv_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning("Failed to read CV input file: %s", sanitize_secrets(str(e)))

        return form_data, cv_text

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

            if not form_data and not cv_text:
                return {
                    "status": "FAILED",
                    "error": "No candidate input found in inputs/ directory (expected form_input.json or cv_input.*)",
                    "session_id": workspace.session_id,
                }

            # Extract target company and role from inputs with defensive sanitization
            raw_company = "TechCorp"
            raw_role = "Senior Software Engineer"

            if form_data:
                raw_company = form_data.get("target_company") or form_data.get("company", raw_company)
                raw_role = form_data.get("target_role") or form_data.get("role", raw_role)
            elif cv_text:
                comp_match = re.search(r"Target Company:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
                if comp_match:
                    raw_company = comp_match.group(1).strip()
                role_match = re.search(r"Role:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
                if role_match:
                    raw_role = role_match.group(1).strip()

            target_company = sanitize_prompt_field(raw_company, default="Target Company")
            target_role = sanitize_prompt_field(raw_role, default="Senior Software Engineer")

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

            # Recon và framework retrieval chạy song song; đầu vào ứng viên được đánh dấu untrusted trước khi vào prompt.
            # Execute safe reconnaissance and knowledge retrieval concurrently
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

            if knowledge_ok and knowledge.get("company_context"):
                recon_data = dict(recon_data or {})
                recon_data["knowledge_base"] = [
                    {"topic": c["topic"], "content": c["content"], "confidence": c["confidence"]}
                    for c in knowledge["company_context"]
                ]

            # Step 1: Generate company intel (including candidate context so single-bundle works without hallucination)
            company_prompt = build_company_intel_prompt(
                company_name=target_company,
                target_role=target_role,
                recon_data=recon_data,
                candidate_context=wrapped_candidate,
            )
            first_response = await self._query_model(company_prompt)

            # Check if all-in-one bundle returned
            required_keys = ("company_intel", "candidate_profile", "interview_spec", "stage_plan")
            if all(k in first_response for k in required_keys):
                company_intel = first_response["company_intel"]
                candidate_profile = first_response["candidate_profile"]
                interview_spec = first_response["interview_spec"]
                stage_plan = first_response["stage_plan"]
            else:
                company_intel = first_response.get("company_intel", first_response)

                # Step 2: Generate candidate profile
                candidate_prompt = build_candidate_profile_prompt(
                    form_data=form_data,
                    cv_text=cv_text,
                    company_intel=company_intel,
                )
                cand_res = await self._query_model(candidate_prompt)
                candidate_profile = cand_res.get("candidate_profile", cand_res)

                # Step 3: Generate interview spec
                spec_prompt = build_interview_spec_prompt(
                    company_intel=company_intel,
                    candidate_profile=candidate_profile,
                    target_role=target_role,
                )
                spec_res = await self._query_model(spec_prompt)
                interview_spec = spec_res.get("interview_spec", spec_res)

                # Step 4: Generate stage plan
                plan_prompt = build_stage_plan_prompt(
                    interview_spec=interview_spec,
                    candidate_profile=candidate_profile,
                    competencies=knowledge["rubric"]["competencies"] if knowledge_ok else None,
                )
                plan_res = await self._query_model(plan_prompt)
                stage_plan = plan_res.get("stage_plan", plan_res)

            if knowledge_ok:
                stage_plan = ground_stage_plan(stage_plan, knowledge)

            # Atomically write all 4 artifacts through SessionWorkspace
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

            artifacts = {
                "company_intel": str(p1),
                "candidate_profile": str(p2),
                "interview_spec": str(p3),
                "stage_plan": str(p4),
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
