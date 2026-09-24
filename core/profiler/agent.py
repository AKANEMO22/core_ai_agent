"""Pre-flight Reconnaissance Profiler Agent coordinating inputs, research tools, and blueprint synthesis."""

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
)
from core.profiler.tools import MarketIntelTool, SafeWebSearchTool
from core.runtime.agent import sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.runtime.tooling import RuntimeTool
from core.workspace.session import SessionWorkspace

logger = logging.getLogger(__name__)


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Robustly parse JSON payload from LLM response text.

    Handles clean JSON, markdown fenced blocks, and extracts outer JSON objects.

    Args:
        text: Raw response string from LLM provider.

    Returns:
        Dict[str, Any]: Extracted and parsed dictionary.

    Raises:
        ValueError: If valid JSON dictionary cannot be parsed from text.
    """
    clean_text = text.strip()

    # 1. Direct JSON parse
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Markdown fenced code block ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if match:
        block = match.group(1).strip()
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Outer curly braces extraction
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        slice_text = clean_text[start : end + 1]
        try:
            parsed = json.loads(slice_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    sample = clean_text[:80] + "..." if len(clean_text) > 80 else clean_text
    raise ValueError(f"Could not parse valid JSON object from model output: '{sample}'")


class ProfilerAgent:
    """Pre-flight intelligence agent synthesizing interview blueprints and stage progression plans.

    Reads untrusted candidate inputs (form or CV), executes safe reconnaissance tools,
    enforces defensive boundary isolation, and atomically creates 4 foundational artifacts:
    1. context/company_intel.json
    2. context/candidate_profile.json
    3. context/interview_spec.json
    4. context/stage_plan.json

    Attributes:
        llm_client: Runtime LLM provider adapter.
        tools: Registered research and intelligence tools.
    """

    def __init__(
        self,
        llm_client: RuntimeLLMClient,
        tools: Optional[List[RuntimeTool]] = None,
    ) -> None:
        self.llm_client = llm_client
        self.tools = tools or [SafeWebSearchTool(), MarketIntelTool()]
        self.tools_map: Dict[str, RuntimeTool] = {t.name: t for t in self.tools}

    async def _query_model(
        self,
        user_prompt: str,
        system_prompt: str = PROFILER_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Stream completion from LLM client and parse structured JSON dictionary."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokens: List[str] = []
        async for chunk in self.llm_client.stream_chat(messages=messages):
            if chunk.token:
                tokens.append(chunk.token)

        full_output = "".join(tokens)
        return extract_json_payload(full_output)

    async def _gather_recon(
        self,
        company_name: str,
        target_role: Optional[str] = None,
        workspace: Optional[SessionWorkspace] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run safe market intel recon tool if available."""
        if "market_intel" in self.tools_map:
            tool = self.tools_map["market_intel"]
            try:
                res = await tool.execute(
                    {"company_name": company_name, "target_role": target_role},
                    workspace=workspace,
                )
                if isinstance(res, dict) and res.get("status") != "blocked":
                    return res
            except Exception as e:
                logger.warning("Market intel tool execution error (skipped): %s", sanitize_secrets(str(e)))
        return None

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
        try:
            form_data, cv_text = self._load_inputs(workspace)

            if not form_data and not cv_text:
                return {
                    "status": "FAILED",
                    "error": "No candidate input found in inputs/ directory (expected form_input.json or cv_input.*)",
                    "session_id": workspace.session_id,
                }

            # Extract target company and role from inputs
            target_company = "TechCorp"
            target_role = "Senior Software Engineer"

            if form_data:
                target_company = form_data.get("target_company") or form_data.get("company", target_company)
                target_role = form_data.get("target_role") or form_data.get("role", target_role)
            elif cv_text:
                # Basic heuristic extraction if form not provided
                comp_match = re.search(r"Target Company:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
                if comp_match:
                    target_company = comp_match.group(1).strip()
                role_match = re.search(r"Role:\s*([^\n\r]+)", cv_text, re.IGNORECASE)
                if role_match:
                    target_role = role_match.group(1).strip()

            # Execute safe reconnaissance
            recon_data = await self._gather_recon(target_company, target_role, workspace=workspace)

            # Step 1: Generate or check for bundle in first prompt
            company_prompt = build_company_intel_prompt(
                company_name=target_company,
                target_role=target_role,
                recon_data=recon_data,
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
                )
                plan_res = await self._query_model(plan_prompt)
                stage_plan = plan_res.get("stage_plan", plan_res)

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

            return {
                "status": "SUCCESS",
                "session_id": workspace.session_id,
                "artifacts": {
                    "company_intel": str(p1),
                    "candidate_profile": str(p2),
                    "interview_spec": str(p3),
                    "stage_plan": str(p4),
                },
            }

        except Exception as e:
            # Strictly scrub credentials, API keys, and candidate personal data from error output
            clean_error = sanitize_secrets(str(e))
            # Also ensure no raw CV input text is leaked in error
            if cv_text and cv_text in clean_error:
                clean_error = clean_error.replace(cv_text, "[CANDIDATE_CV_REDACTED]")
            if form_data:
                for val in form_data.values():
                    if isinstance(val, str) and val in clean_error:
                        clean_error = clean_error.replace(val, "[REDACTED]")

            logger.error("ProfilerAgent execution failed: %s", clean_error)
            return {
                "status": "FAILED",
                "error": clean_error,
                "session_id": workspace.session_id,
            }
