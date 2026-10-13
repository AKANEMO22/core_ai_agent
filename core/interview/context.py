"""Context assembler enforcing strict token budgets and deterministic truncation priority."""

import math
from typing import Any, Dict, List, Optional


class ContextAssembler:
    """Assembles prompt contexts for interviewer subagents under a strict token ceiling.
    
    Truncation Priority (when estimated tokens exceed max_tokens):
    1. Drop old raw dialogue turns from oldest to newest, preserving the latest turn / current Q&A.
    2. Drop low-confidence research and auxiliary company intelligence.
    3. Drop non-critical profile details (claims, secondary skills, metadata, hobbies).
    
    Inviolable Core:
    Persona, tone, stage goals (with retrieved competencies and up to 3 key questions),
    and current Q&A are strictly preserved and never dropped.
    """

    def __init__(self, max_tokens: int = 1500) -> None:
        """Initialize the context assembler.

        Args:
            max_tokens: Maximum allowable token count for the assembled prompt (default 1500).
        """
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: Optional[str]) -> int:
        """Estimate the token count of a given text.

        Uses the standard heuristic of ~4 characters per token for English text.

        Args:
            text: Input string or None.

        Returns:
            int: Estimated token count (0 if empty/None).
        """
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4.0))

    def assemble_prompt(
        self,
        spec: Dict[str, Any],
        stage: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        recent_turns: Optional[List[Dict[str, Any]]] = None,
        company_intel: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Assemble an interviewer prompt strictly within max_tokens.

        Args:
            spec: Interview specification containing persona, tone, knowledge_scope.
            stage: Stage definition containing stage_id, name, goals.
            profile: Optional candidate profile data (strengths, claims, gaps, etc.).
            recent_turns: Optional list of past dialogue turns (role, content).
            company_intel: Optional company research data.
            **kwargs: Extra metadata or stage completion signals.

        Returns:
            str: Deterministically assembled, truncated prompt string under token budget.
        """
        turns = list(recent_turns or [])
        profile_data = dict(profile or {})
        intel_data = dict(company_intel or {})

        # Core non-droppable elements
        persona = spec.get("interviewer_persona", "Technical Interviewer")
        tone = spec.get("tone", "professional")
        knowledge_scope = spec.get("knowledge_scope", [])
        if isinstance(knowledge_scope, list):
            scope_str = ", ".join(str(s) for s in knowledge_scope)
        else:
            scope_str = str(knowledge_scope)

        stage_name = stage.get("name", stage.get("stage_id", "Current Stage"))
        stage_id = stage.get("stage_id", "")
        goals = stage.get("goals", [])
        if isinstance(goals, list):
            goals_str = "; ".join(str(g) for g in goals)
        else:
            goals_str = str(goals)

        # Retrieval-grounded stage fields (optional): competencies to evidence and key questions
        criteria_labels = stage.get("criteria_labels")
        criteria_str = (
            ", ".join(str(c) for c in criteria_labels) if isinstance(criteria_labels, list) else ""
        )
        raw_questions = stage.get("question_candidates")
        key_questions = [
            str(q.get("text", ""))[:240]
            for q in (raw_questions if isinstance(raw_questions, list) else [])
            if isinstance(q, dict) and q.get("text")
        ][:3]

        # Separate inviolable current Q&A from older turns
        # If 2 or more turns, last 2 represent current Q&A (Interviewer Q + Candidate A)
        # If 1 turn, it represents current Q&A
        min_turns_to_keep = min(2, len(turns))
        old_turns = turns[:-min_turns_to_keep] if min_turns_to_keep > 0 else []
        current_qa_turns = turns[-min_turns_to_keep:] if min_turns_to_keep > 0 else []

        # Levels:
        # turns_count: number of old turns to include (from len(old_turns) down to 0)
        # intel_level: 2 (full), 1 (high confidence only), 0 (none)
        # profile_level: 2 (full), 1 (core strengths/probes only), 0 (top strength only), -1 (none)
        turns_to_keep = list(old_turns)
        intel_level = 2 if intel_data else 0
        profile_level = 2 if profile_data else -1

        def render_prompt(
            active_old_turns: List[Dict[str, Any]],
            i_level: int,
            p_level: int,
            qa_turns: Optional[List[Dict[str, Any]]] = None,
        ) -> str:
            sections = []

            # 1. Inviolable Core: Role, Tone, Scope
            sections.append(
                f"# Interviewer Persona & Guidelines\n"
                f"- Role: {persona}\n"
                f"- Tone: {tone}\n"
                f"- Knowledge Scope: {scope_str}"
            )

            # 2. Inviolable Core: Stage Goals, Competencies and Key Questions
            stage_lines = [
                f"# Current Stage: {stage_name} ({stage_id})",
                f"- Stage Goals: {goals_str}",
            ]
            if criteria_str:
                stage_lines.append(f"- Competencies to Evidence: {criteria_str}")
            if key_questions:
                stage_lines.append(
                    "- Key Questions (ask one at a time, adapt to the conversation, "
                    "follow up when an answer is shallow):"
                )
                stage_lines.extend(f"  {idx}. {q}" for idx, q in enumerate(key_questions, 1))
            sections.append("\n".join(stage_lines))

            # 3. Candidate Profile (Droppable Priority 3)
            if p_level >= 2:
                prof_lines = ["# Candidate Profile"]
                for k, v in profile_data.items():
                    if isinstance(v, list):
                        items = ", ".join(str(item) for item in v)
                        prof_lines.append(f"- {k.replace('_', ' ').capitalize()}: {items}")
                    elif isinstance(v, dict):
                        items = ", ".join(f"{dk}: {dv}" for dk, dv in v.items())
                        prof_lines.append(f"- {k.replace('_', ' ').capitalize()}: {items}")
                    else:
                        prof_lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
                sections.append("\n".join(prof_lines))
            elif p_level == 1:
                prof_lines = ["# Candidate Profile"]
                if "strengths" in profile_data:
                    strengths = ", ".join(str(s) for s in profile_data["strengths"])
                    prof_lines.append(f"- Strengths: {strengths}")
                if "probes" in profile_data:
                    probes = ", ".join(str(p) for p in profile_data["probes"])
                    prof_lines.append(f"- Focus Areas: {probes}")
                sections.append("\n".join(prof_lines))
            elif p_level == 0:
                strengths = profile_data.get("strengths", [])
                top_strength = strengths[0] if strengths else ""
                if top_strength:
                    sections.append(f"# Candidate Profile\n- Core Competency: {top_strength}")

            # 4. Research & Company Intel (Droppable Priority 2)
            if i_level >= 2:
                intel_lines = ["# Company & Role Context"]
                for k, v in intel_data.items():
                    if isinstance(v, list):
                        items = ", ".join(str(item) for item in v)
                        intel_lines.append(f"- {k.replace('_', ' ').capitalize()}: {items}")
                    else:
                        intel_lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
                sections.append("\n".join(intel_lines))
            elif i_level == 1:
                # Drop low-confidence, auxiliary, or rumor fields
                intel_lines = ["# Company & Role Context"]
                for k, v in intel_data.items():
                    if k.lower() in ("low_confidence_notes", "auxiliary_research", "rumors", "notes"):
                        continue
                    if isinstance(v, list):
                        items = ", ".join(str(item) for item in v)
                        intel_lines.append(f"- {k.replace('_', ' ').capitalize()}: {items}")
                    else:
                        intel_lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
                sections.append("\n".join(intel_lines))

            # 5. Conversation History (Droppable Priority 1: old turns) + Inviolable Current Q&A
            qa_list = qa_turns if qa_turns is not None else current_qa_turns
            all_turns = active_old_turns + qa_list
            if all_turns:
                turn_lines = ["# Conversation History"]
                for t in all_turns:
                    role_tag = "Interviewer" if t.get("role") == "interviewer" else "Candidate"
                    turn_lines.append(f"[{role_tag}]: {t.get('content', '')}")
                sections.append("\n".join(turn_lines))

            # 6. Operational Instructions
            sections.append(
                "# Directives\n"
                "- Conduct the dialogue naturally following the specified persona and tone.\n"
                "- Ensure every question actively probes the current stage goals."
            )

            return "\n\n".join(sections)

        # Initial render
        prompt = render_prompt(turns_to_keep, intel_level, profile_level)
        if self.estimate_tokens(prompt) <= self.max_tokens:
            return prompt

        # --- Truncation Priority 1: Drop old raw turns from oldest to newest ---
        while turns_to_keep:
            turns_to_keep.pop(0)  # drop oldest turn
            prompt = render_prompt(turns_to_keep, intel_level, profile_level)
            if self.estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        # --- Truncation Priority 2: Drop low-confidence research -> drop all research ---
        if intel_level > 1:
            intel_level = 1  # drop low-confidence and auxiliary notes
            prompt = render_prompt(turns_to_keep, intel_level, profile_level)
            if self.estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        if intel_level > 0:
            intel_level = 0  # drop research completely
            prompt = render_prompt(turns_to_keep, intel_level, profile_level)
            if self.estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        # --- Truncation Priority 3: Drop non-critical profile detail -> top strength -> none ---
        if profile_level > 1:
            profile_level = 1  # drop claims, hobbies, metadata, keep strengths/probes
            prompt = render_prompt(turns_to_keep, intel_level, profile_level)
            if self.estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        if profile_level > 0:
            profile_level = 0  # top strength only
            prompt = render_prompt(turns_to_keep, intel_level, profile_level)
            if self.estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        if profile_level >= 0:
            profile_level = -1  # drop profile completely
            prompt = render_prompt(turns_to_keep, intel_level, profile_level)
            if self.estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        # --- Truncation Priority 4: Truncate verbose candidate content in current Q&A ---
        active_qa_turns = [dict(t) for t in current_qa_turns]
        while self.estimate_tokens(prompt) > self.max_tokens:
            excess_tokens = self.estimate_tokens(prompt) - self.max_tokens
            excess_chars = excess_tokens * 4
            candidate_turns = [
                (idx, t)
                for idx, t in enumerate(active_qa_turns)
                if t.get("role") == "candidate" and len(t.get("content", "")) > 60
            ]
            if not candidate_turns:
                candidate_turns = [
                    (idx, t)
                    for idx, t in enumerate(active_qa_turns)
                    if len(t.get("content", "")) > 60
                ]
                if not candidate_turns:
                    break

            idx, longest_turn = max(
                candidate_turns, key=lambda pair: len(pair[1].get("content", ""))
            )
            curr_content = longest_turn.get("content", "")
            reduction = max(100, excess_chars + 20)
            new_len = max(40, len(curr_content) - reduction)
            active_qa_turns[idx]["content"] = (
                curr_content[:new_len].rstrip() + "... [truncated]"
            )
            prompt = render_prompt(
                turns_to_keep, intel_level, profile_level, active_qa_turns
            )
            if new_len <= 40:
                break

        return prompt
