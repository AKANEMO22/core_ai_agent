"""Turn-level rubric evaluation subagent with non-scoring descriptive delivery observations.

Evaluator tương thích cũ: chấm nội dung và cách trình bày trên thang 1–10.
TurnEvaluator nhận câu hỏi, câu trả lời và mục tiêu rubric, gom token LLM rồi
đọc JSON, giới hạn điểm và trả phản hồi. CriterionEvaluator dùng nó khi thiếu rubric.
VAD chỉ tạo quan sát mô tả, không tham gia tính điểm năng lực.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from core.runtime.client import RuntimeLLMClient

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are an expert interview assessment subagent.
Your role is to rigorously evaluate a candidate's answer for an interview turn against the specified rubric targets.

Evaluation Dimensions:
1. Content Score (1-10): Technical accuracy, depth, edge-case consideration, and problem-solving reasoning.
2. Communication Structure Score (1-10): Structural clarity, answer organization, relevance, and precision.
3. Strengths: Distinct demonstrable positive points in the candidate's answer.
4. Improvements: Concrete, constructive recommendations for technical depth or clearer delivery.

Output Requirements:
- Respond ONLY with a valid JSON object matching the requested schema.
- Do NOT include markdown code fences or conversational text.
"""


class TurnEvaluator:
    """Evaluates candidate interview turns against rubrics and descriptive audio timing."""

    def __init__(
        self,
        llm_client: RuntimeLLMClient,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize turn evaluator with an LLM provider client.

        Args:
            llm_client: Runtime LLM provider client.
            system_prompt: Optional custom system instructions.
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt or EVALUATOR_SYSTEM_PROMPT

    def format_delivery_observations(
        self, vad_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format non-scoring descriptive delivery observations from VAD metrics.

        CRITICAL POLICY: VAD timing metrics must never be used to infer candidate
        competence, personality, disability, truthfulness, or hiring suitability.
        Observations are strictly descriptive of timing dynamics.

        Args:
            vad_metrics: Raw audio/VAD metrics dictionary.

        Returns:
            Dict[str, Any]: Descriptive observation payload.
        """
        metrics = vad_metrics or {}
        speech_ms = metrics.get("speech_duration_ms", 0)
        pauses = metrics.get("pause_count", 0)
        silence_ms = metrics.get("total_silence_ms", 0)
        wpm = metrics.get("words_per_minute", 0)

        speech_sec = speech_ms / 1000.0 if speech_ms else 0.0

        if speech_ms > 0:
            obs_desc = (
                f"Candidate spoke for {speech_sec:.1f} seconds with {pauses} pause(s) "
                f"({silence_ms}ms total silence) at an estimated pace of {wpm} words per minute. "
                "Delivery observations are strictly descriptive of audio timing dynamics and "
                "are not scored or used to infer candidate competence, personality, or suitability."
            )
        else:
            obs_desc = (
                "No verbal audio metrics recorded for this turn. "
                "Delivery observations are strictly descriptive and non-evaluative."
            )

        return {
            "speech_duration_ms": speech_ms,
            "pause_count": pauses,
            "total_silence_ms": silence_ms,
            "words_per_minute": wpm,
            "observations": obs_desc,
        }

    async def evaluate_turn(
        self,
        question: str,
        candidate_text: str,
        rubric_target: Optional[List[str]] = None,
        rubric_version: str = "rubric_v1",
        vad_metrics: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Evaluate a single interview turn using the LLM provider (legacy two-score mode).

        Args:
            question: Interviewer prompt / question.
            candidate_text: Candidate verbal or text response.
            rubric_target: Competencies or topics evaluated (e.g. ['Redis', 'PubSub']).
            rubric_version: Rubric specification version.
            vad_metrics: Voice activity and timing metrics.

        Returns:
            Dict[str, Any]: Structured evaluation scores, feedback, and descriptive observations.
        """
        # Prompt chấm chỉ chứa câu hỏi/câu trả lời và target; dữ liệu VAD được mô tả ở bước sau.
        targets = rubric_target or []
        user_prompt = (
            f"Generate evaluation for `assessment` in JSON format with schema:\n"
            f'{{"content_score": int, "communication_score": int, "strengths": [str], "improvements": [str]}}\n\n'
            f"Rubric Version: {rubric_version}\n"
            f"Target Competencies: {', '.join(targets)}\n"
            f"Interviewer Question: {question}\n"
            f"Candidate Answer: {candidate_text}\n"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Stream and accumulate response
        # Client trả token theo stream; phải gom đủ response trước khi giải mã JSON.
        full_text = ""
        async for chunk in self.llm_client.stream_chat(messages=messages):
            if chunk.token:
                full_text += chunk.token

        # Parse JSON output
        parsed: Dict[str, Any] = {}
        cleaned_text = full_text.strip()
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Fallback if wrapped in an assessment envelope
            logger.warning("Failed direct JSON decode of LLM response; applying fallback parsing.")
            import re
            match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    pass

        # Extract dimensions with robust defaults and score clamping (1-10)
        # Nhánh legacy dùng giá trị mặc định khi thiếu trường, khác nhánh structured có retry.
        content_score = int(parsed.get("content_score", 5))
        content_score = max(1, min(10, content_score))

        comm_score = int(parsed.get("communication_score", 5))
        comm_score = max(1, min(10, comm_score))

        strengths = list(parsed.get("strengths", []))
        improvements = list(parsed.get("improvements", []))

        delivery_observations = self.format_delivery_observations(vad_metrics)

        return {
            "content_score": content_score,
            "communication_score": comm_score,
            "strengths": strengths,
            "improvements": improvements,
            "delivery_observations": delivery_observations,
            "rubric_version": rubric_version,
            "rubric_target": targets,
        }
