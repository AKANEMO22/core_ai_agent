"""Directives: the orchestrator's instruction to the speaker for the next turn."""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.workspace.session import SessionWorkspace

DIRECTIVE_MODES = ("follow_up", "next_question", "transition", "wrap_up", "redirect")


@dataclass
class Directive:
    """What the speaker should do when the candidate's next answer arrives.

    Attributes:
        mode: follow_up | next_question | transition | wrap_up | redirect.
        question_text: The question to ask (or the opening of the next stage on transition).
        reason: Why the orchestrator chose this (audit only; never spoken).
        target_criterion: Competency id the question gathers evidence for.
        question_id: Question bank id when the question came from retrieval.
        probe_hint: Conditional follow-up: if the answer is shallow on this, probe it first.
        generated_by: "orchestrator" | "fallback".
        degraded: True when produced by a fallback path.
        extend_stage: Keep the stage open past its turn budget (bounded) because a must-have
            criterion still needs evidence; set only by the orchestrator's coverage gate.
    """

    mode: str
    question_text: str
    reason: str = ""
    target_criterion: Optional[str] = None
    question_id: Optional[str] = None
    probe_hint: Optional[str] = None
    generated_by: str = "orchestrator"
    degraded: bool = False
    extend_stage: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Directive":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        values.setdefault("question_text", "")
        values.setdefault("mode", "next_question")
        return cls(**values)

    def speaker_instruction(self) -> str:
        """Render the directive as a short instruction block for the speaker prompt."""
        lines = ["# Directive for this turn (internal; never read it aloud)"]
        if self.mode == "follow_up":
            lines.append(f"- Ask this follow-up: {self.question_text}")
        elif self.mode == "transition":
            lines.append("- Briefly close the current topic, then move to the next stage.")
            lines.append(f"- Ask: {self.question_text}")
        elif self.mode == "wrap_up":
            lines.append("- This is the final turn. Briefly acknowledge the answer, thank the candidate,")
            lines.append("  and say feedback will follow. Do not ask another question.")
            lines.append(f"- Closing line to adapt: {self.question_text}")
            return "\n".join(lines)
        elif self.mode == "redirect":
            lines.append("- Politely steer the conversation back to the interview.")
            lines.append(f"- Then ask: {self.question_text}")
        else:
            lines.append(f"- Next question: {self.question_text}")
        if self.probe_hint:
            lines.append(
                f"- Before that, if the candidate's latest answer was shallow on this point, "
                f"ask one short follow-up instead: {self.probe_hint}"
            )
        lines.append("- One question only. React to what the candidate just said in at most one short sentence.")
        return "\n".join(lines)


class DirectiveStore:
    """Persists directives as `orchestration/directive_{turn}.json` (consumed by that turn)."""

    @staticmethod
    def path(turn_number: int) -> str:
        return f"orchestration/directive_{turn_number}.json"

    def write(self, workspace: SessionWorkspace, turn_number: int, directive: Directive) -> str:
        rel = self.path(turn_number)
        workspace.write_artifact(rel, artifact_type="directive", payload=directive.to_dict())
        return rel

    def read(self, workspace: SessionWorkspace, turn_number: int) -> Optional[Directive]:
        rel = self.path(turn_number)
        if not workspace.exists(rel):
            return None
        data = workspace.read_json(rel)
        return Directive.from_dict(data.get("payload", data))


def asked_question_ids(workspace: SessionWorkspace) -> Set[str]:
    """Question bank ids already asked, from interviewer turn artifacts."""
    asked: Set[str] = set()
    for path in workspace.list_artifacts("turns"):
        if path.name.endswith("_interviewer.json"):
            rel = path.relative_to(workspace.root_path).as_posix()
            try:
                question_id = workspace.read_json(rel).get("question_id")
            except Exception:
                continue
            if question_id:
                asked.add(str(question_id))
    return asked


def stage_position(
    stages: List[Dict[str, Any]], current_stage_id: Optional[str]
) -> tuple:
    """(index, stage, next_stage) for the current stage id."""
    index = 0
    for i, stage in enumerate(stages):
        if stage.get("stage_id") == current_stage_id:
            index = i
            break
    stage = stages[index] if stages else {}
    next_stage = stages[index + 1] if index + 1 < len(stages) else None
    return index, stage, next_stage


def is_final_turn_of_stage(stage: Dict[str, Any], turns_used: int) -> bool:
    """True when the next candidate turn is the last one the stage budget allows."""
    target = stage.get("target_turns") or stage.get("turns") or 3
    try:
        return turns_used + 1 >= int(target)
    except (TypeError, ValueError):
        return False


def can_extend_stage(stage: Dict[str, Any], turns_used: int, max_extra_turns: int) -> bool:
    """True when the next turn may still belong to this stage as an extension past its budget."""
    target = stage.get("target_turns") or stage.get("turns") or 3
    extra = stage.get("max_extra_turns", max_extra_turns)
    try:
        return turns_used + 1 < int(target) + max(0, int(extra))
    except (TypeError, ValueError):
        return False


def plan_fallback(
    stages: List[Dict[str, Any]],
    current_stage_id: Optional[str],
    stage_turns_used: int,
    asked_ids: Set[str],
    reason: str = "no orchestrator directive available",
) -> Directive:
    """Budget-aware deterministic directive.

    On the last turn of a stage the response must already move on: `transition` with the next
    stage's first unasked question, or `wrap_up` on the final stage. Otherwise the next unasked
    key question of the current stage.
    """
    _, stage, next_stage = stage_position(stages, current_stage_id)
    if is_final_turn_of_stage(stage, stage_turns_used):
        if next_stage is not None:
            opening = fallback_directive(next_stage, asked_ids, reason=reason)
            opening.mode = "transition"
            return opening
        return Directive(
            mode="wrap_up",
            question_text="Thank you for your time today. That concludes the interview; you will receive feedback soon.",
            reason=reason,
            generated_by="fallback",
            degraded=True,
        )
    return fallback_directive(stage, asked_ids, reason=reason)


def fallback_directive(
    stage: Dict[str, Any],
    asked_ids: Set[str],
    next_stage: Optional[Dict[str, Any]] = None,
    reason: str = "no orchestrator directive available",
) -> Directive:
    """Deterministic directive: next unasked key question of the stage, else a generic probe."""
    candidates: List[Dict[str, Any]] = [
        q for q in (stage.get("question_candidates") or []) if isinstance(q, dict) and q.get("text")
    ]
    for question in candidates:
        if question.get("question_id") not in asked_ids:
            follow_ups = question.get("follow_ups") or []
            return Directive(
                mode="next_question",
                question_text=str(question["text"]),
                question_id=question.get("question_id"),
                target_criterion=question.get("competency_id"),
                probe_hint=str(follow_ups[0]) if follow_ups else None,
                reason=reason,
                generated_by="fallback",
                degraded=True,
            )
    goals = stage.get("goals") or []
    goal = goals[0] if isinstance(goals, list) and goals else stage.get("name", "the current topic")
    return Directive(
        mode="next_question",
        question_text=f"Can you walk me through a concrete example related to {goal}?",
        reason=reason,
        generated_by="fallback",
        degraded=True,
    )
