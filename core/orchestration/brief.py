"""Interview brief: written once after profiling, read-only during the interview.

The model writes only narrative fields (frame, persona notes, CV probes, risks). Stage structure,
criteria with anchors and question candidates are rendered by code from retrieved artifacts, so
the brief cannot invent criteria or leak evaluator-only expected answers.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from core.runtime.client import RuntimeLLMClient
from core.runtime.sanitize import sanitize_secrets
from core.runtime.skills import SkillRegistry
from core.runtime.structured import StructuredOutputError, run_structured
from core.runtime.tokens import TokenCounter
from core.workspace.session import SessionWorkspace

BRIEF_PATH = "context/interview_brief.md"
MAX_BRIEF_TOKENS = 4000

BRIEF_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame": {"type": "string", "minLength": 10, "maxLength": 800},
        "persona_notes": {"type": "string", "maxLength": 500},
        "cv_probes": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 8},
        "risks": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 6},
    },
    "required": ["frame", "persona_notes", "cv_probes", "risks"],
    "additionalProperties": False,
}

REQUIRED_SECTIONS = ("## Interview frame", "## Interviewer persona", "## Stages", "## CV points to probe", "## Risks")

logger = logging.getLogger(__name__)


def _payload(workspace: SessionWorkspace, rel: str) -> Dict[str, Any]:
    if not workspace.exists(rel):
        return {}
    data = workspace.read_json(rel)
    return data.get("payload", data) if isinstance(data, dict) else {}


def _expected_points(knowledge: Dict[str, Any]) -> List[str]:
    points: List[str] = []
    for questions in (knowledge.get("question_bank") or {}).values():
        for question in questions:
            points.extend(str(p) for p in question.get("expected_points", []) if len(str(p)) >= 12)
    return points


class InterviewBriefWriter:
    """Generates and validates `context/interview_brief.md`."""

    def __init__(
        self,
        llm: RuntimeLLMClient,
        skills: Optional[SkillRegistry] = None,
        deadline_seconds: float = 45.0,
        counter: Optional[TokenCounter] = None,
    ) -> None:
        self.llm = llm
        self.skills = skills
        self.deadline_seconds = deadline_seconds
        self.counter = counter or TokenCounter()

    def _render(self, narrative: Dict[str, Any], workspace: SessionWorkspace) -> str:
        spec = _payload(workspace, "context/interview_spec.json")
        rubric = _payload(workspace, "context/rubric.json")
        plan = _payload(workspace, "context/stage_plan.json")
        competencies = {c["competency_id"]: c for c in rubric.get("competencies", []) if c.get("competency_id")}

        lines = ["# Interview brief", ""]
        lines += ["## Interview frame", "", narrative.get("frame", "").strip(), ""]
        if rubric:
            lines.append(
                f"Role family: {rubric.get('role_family', '-')}; level: {rubric.get('level', '-')}; "
                f"rubric {rubric.get('rubric_version', '-')}."
            )
            lines.append("")
        lines += [
            "## Interviewer persona",
            "",
            f"- Persona: {spec.get('interviewer_persona', 'Interviewer')}; tone: {spec.get('tone', 'professional')}",
            f"- {narrative.get('persona_notes', '').strip()}",
            "",
            "## Stages",
            "",
        ]
        for stage in plan.get("stages", []) or []:
            if not isinstance(stage, dict):
                continue
            target = stage.get("target_turns") or stage.get("turns") or "-"
            lines.append(f"### {stage.get('name', stage.get('stage_id'))} ({stage.get('stage_type', '-')}, {target} turns)")
            goals = stage.get("goals") or []
            if goals:
                lines.append("- Goals: " + "; ".join(str(g) for g in goals))
            for cid in stage.get("criteria", []) or []:
                comp = competencies.get(cid)
                if comp is None:
                    continue
                anchors = comp.get("anchors", {})
                lines.append(
                    f"- {comp.get('name', cid)} (weight {comp.get('weight')}, "
                    f"{'must-have' if comp.get('must_have') else 'optional'}): "
                    f"level 3 looks like: {anchors.get('3', '-')}"
                )
            for question in stage.get("question_candidates", []) or []:
                follow = "; ".join(question.get("follow_ups", [])[:2])
                lines.append(f"  - Q: {question.get('text')}" + (f" (follow-ups: {follow})" if follow else ""))
            lines.append("")
        lines += ["## CV points to probe", ""]
        lines += [f"- {p}" for p in narrative.get("cv_probes", [])] or ["- none identified"]
        lines += ["", "## Risks", ""]
        lines += [f"- {r}" for r in narrative.get("risks", [])] or ["- none identified"]
        lines.append("")
        return "\n".join(lines)

    def validate(self, markdown: str, knowledge: Dict[str, Any], narrative: Optional[Dict[str, Any]] = None) -> List[str]:
        """Return validation errors (empty when the brief is acceptable).

        The leak check covers the model-written narrative; code-rendered rubric anchors may
        legitimately share wording with expected points.
        """
        errors = [f"missing section {s}" for s in REQUIRED_SECTIONS if s not in markdown]
        if self.counter.count(markdown) > MAX_BRIEF_TOKENS:
            errors.append(f"brief exceeds {MAX_BRIEF_TOKENS} tokens")
        model_text = json.dumps(narrative, ensure_ascii=False) if narrative is not None else markdown
        lowered = model_text.lower()
        leaked = [p for p in _expected_points(knowledge) if p.lower() in lowered]
        if leaked:
            errors.append(f"brief leaks {len(leaked)} evaluator-only expected point(s)")
        return errors

    def _fallback_narrative(self, workspace: SessionWorkspace) -> Dict[str, Any]:
        profile = _payload(workspace, "context/candidate_profile.json")
        intel = _payload(workspace, "context/company_intel.json")
        probes = [str(c) for c in (profile.get("claims") or profile.get("probes") or [])][:8]
        gaps = [f"Verify depth on: {g}" for g in (profile.get("gaps") or [])][:4]
        return {
            "frame": (
                f"Mock interview for {intel.get('company', 'the target company')}. Gather evidence for each "
                "stage competency with one question at a time and short follow-ups."
            ),
            "persona_notes": "Professional, neutral and concise; the candidate should do most of the talking.",
            "cv_probes": (probes + gaps)[:8],
            "risks": ["Do not reveal scoring or expected answers.", "Avoid stacking multiple questions in one turn."],
        }

    async def write(self, workspace: SessionWorkspace) -> str:
        """Generate, validate and persist the brief. Returns 'ok' or 'fallback'."""
        knowledge = _payload(workspace, "context/retrieved_knowledge.json")
        template = self.skills.get_template("interview_brief") if self.skills else None
        instructions = template.content if template else "Write the interview brief. Respond with JSON only for `interview_brief`."
        context = {
            "company_intel": _payload(workspace, "context/company_intel.json"),
            "candidate_profile": _payload(workspace, "context/candidate_profile.json"),
            "interview_spec": _payload(workspace, "context/interview_spec.json"),
            "stage_plan": [
                {k: s.get(k) for k in ("stage_id", "name", "stage_type", "goals", "criteria_labels")}
                for s in (_payload(workspace, "context/stage_plan.json").get("stages") or [])
                if isinstance(s, dict)
            ],
        }
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": "<untrusted_context>\n"
                + json.dumps(context, ensure_ascii=False, default=str)[:12000]
                + "\n</untrusted_context>\nRespond with JSON only for `interview_brief`.",
            },
        ]
        status = "ok"
        try:
            narrative = await run_structured(
                self.llm, messages, BRIEF_SCHEMA, max_repairs=1, deadline_seconds=self.deadline_seconds, temperature=0.2
            )
        except StructuredOutputError as e:
            logger.warning("Brief generation fell back to template: %s", sanitize_secrets(str(e)))
            narrative, status = self._fallback_narrative(workspace), "fallback"

        markdown = self._render(narrative, workspace)
        if self.validate(markdown, knowledge, narrative):
            narrative, status = self._fallback_narrative(workspace), "fallback"
            markdown = self._render(narrative, workspace)
        await asyncio.to_thread(workspace.write_text, BRIEF_PATH, markdown)
        return status
