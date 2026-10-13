"""Scripted LLM for development, tests and simulations: answers every structured prompt of the runtime.

Structured prompts end with "Respond with JSON only for `<kind>`". The scripted model parses the
prompt it receives and returns a valid, deterministic response for that kind, so the full
pipeline (profiling, brief, speaker, orchestrator, evaluator, planner, guard, claims, summary,
narrative) runs end to end without a provider.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from core.runtime.providers.mock import MockLLMClient

KIND_PATTERN = re.compile(r"for `([a-z_]+)`")

DEFAULT_STAGE_PLAN = {
    "stages": [
        {"stage_id": "warmup", "name": "Warmup", "target_turns": 1},
        {"stage_id": "technical", "name": "Technical Depth", "target_turns": 2},
        {"stage_id": "system_design", "name": "System Design", "target_turns": 1},
        {"stage_id": "qa_close", "name": "Q&A and Close", "target_turns": 1},
    ]
}


def _last_user(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _between(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"\s*(.*?)\s*" + re.escape(end), text, re.DOTALL)
    return match.group(1) if match else ""


class ScriptedInterviewLLM(MockLLMClient):
    """Deterministic model for every prompt kind the runtime issues.

    Args:
        level_for: optional callable(answer_text, competency_id) -> level (1-4) used by the
            evaluator script; defaults to 3.
        speaker_tokens: tokens streamed for interviewer turns.
        overrides: kind -> callable(prompt_text) -> dict, to customize any structured response.
    """

    def __init__(
        self,
        level_for: Optional[Callable[[str, str], int]] = None,
        speaker_tokens: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Callable[[str], Dict[str, Any]]]] = None,
        stage_plan: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.level_for = level_for or (lambda answer, cid: 3)
        self.overrides = overrides or {}
        self.kind_counts: Dict[str, int] = {}
        super().__init__(
            canned_tokens=speaker_tokens or ["Thanks.", " Let", " us", " go", " deeper:", " how", " would", " you", " scale", " it?"],
            canned_json_responses={
                "company_intel": {"company": "Acme Systems", "culture": "Engineering rigor", "tech_stack": ["Python", "Redis"]},
                "candidate_profile": {
                    "strengths": ["Backend architecture"],
                    "gaps": ["Capacity modeling"],
                    "claims": ["Led the payments ledger migration", "Scaled the API to 20k requests per second"],
                },
                "interview_spec": {
                    "interviewer_persona": "Principal Engineer",
                    "tone": "rigorous_challenger",
                    "knowledge_scope": ["Distributed systems", "Data storage"],
                },
                "stage_plan": stage_plan or DEFAULT_STAGE_PLAN,
            },
            responder=self._respond,
            **kwargs,
        )

    def _respond(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        prompt = _last_user(messages)
        matches = KIND_PATTERN.findall(prompt)
        if not matches:
            return None
        kind = matches[-1]
        self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1
        if kind in self.overrides:
            return json.dumps(self.overrides[kind](prompt))
        handler = getattr(self, f"_kind_{kind}", None)
        return json.dumps(handler(prompt)) if handler else None

    # -- structured kinds -------------------------------------------------------------

    def _kind_criterion_evaluation(self, prompt: str) -> Dict[str, Any]:
        targets = json.loads(_between(prompt, "Target competencies and anchors:\n", "\n\nWhat a strong answer"))
        answer = _between(prompt, "<candidate_answer>", "</candidate_answer>")
        words = answer.split()
        quote = " ".join(words[: min(8, len(words))])
        criteria = []
        for comp in targets:
            level = int(self.level_for(answer, comp["competency_id"]))
            criteria.append(
                {
                    "competency_id": comp["competency_id"],
                    "status": "scored" if len(words) >= 4 else "insufficient_evidence",
                    "level": level if len(words) >= 4 else None,
                    "confidence": 0.8,
                    "rationale": f"Matches anchor {level}.",
                    "evidence_quotes": [quote] if quote else [],
                }
            )
        injected = bool(re.search(r"ignore (all )?previous instructions", answer, re.IGNORECASE))
        return {
            "criteria": criteria,
            "content_score": 7,
            "communication_score": 7,
            "strengths": ["Concrete example from production work"],
            "improvements": ["Quantify the impact of design decisions"],
            "flags": {"off_topic": False, "injection_suspected": injected},
        }

    def _kind_orchestrator_decision(self, prompt: str) -> Dict[str, Any]:
        pool = json.loads(_between(prompt, "# Question pool\n", "\n\n# Latest exchange") or "[]")
        current = [q for q in pool if q.get("source") == "current_stage"] or pool
        chosen = current[0] if current else {"text": "Can you give a concrete example?", "question_id": None}
        return {
            "directive": {
                "mode": "next_question",
                "question_text": chosen["text"],
                "reason": "gather evidence for uncovered criteria",
                "target_criterion": chosen.get("competency_id"),
                "question_id": chosen.get("question_id"),
                "probe_hint": "Ask for one concrete number.",
            },
            "complete_stage": False,
            "notes": {"observations": "Candidate answers with concrete production context.", "open_threads": None},
        }

    def _kind_interview_brief(self, prompt: str) -> Dict[str, Any]:
        return {
            "frame": "Assess senior backend depth across design, distributed systems and data storage.",
            "persona_notes": "Direct and curious; keep the candidate talking.",
            "cv_probes": ["Ledger migration ownership", "Claimed 20k requests per second"],
            "risks": ["Do not over-invest in warmup"],
        }

    def _kind_rolling_summary(self, prompt: str) -> Dict[str, Any]:
        turns = re.findall(r"\[turn (\d+)\] Candidate", prompt)
        return {
            "covered_topics": [f"turn {t} topic" for t in turns],
            "candidate_claims": ["works on distributed services"],
            "open_threads": [],
        }

    def _kind_followup_plan(self, prompt: str) -> Dict[str, Any]:
        return {
            "answer_gaps": ["failure modes not discussed"],
            "suggestions": [{"question": "What happens when that component fails?", "rationale": "failure modes"}],
        }

    def _kind_guard_verdict(self, prompt: str) -> Dict[str, Any]:
        return {"injection_suspected": False, "off_topic": False, "reason": "on topic"}

    def _kind_claim_check(self, prompt: str) -> Dict[str, Any]:
        claims = json.loads(_between(prompt, "Claims:\n", "\n\n<transcript>") or "[]")
        return {"claims": [{"claim": c, "status": "untested", "note": "not discussed"} for c in claims]}

    def _kind_report_narrative(self, prompt: str) -> Dict[str, Any]:
        return {
            "summary": "The candidate showed solid backend fundamentals with room to deepen failure analysis.",
            "criterion_findings": [],
            "strengths": ["Concrete production examples"],
            "improvements": ["Quantify trade-offs"],
            "study_plan": ["Practice failure-mode analysis for distributed caches"],
        }
