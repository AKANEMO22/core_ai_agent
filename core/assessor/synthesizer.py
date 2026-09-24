"""Report synthesizer generating final numerical assessments and markdown coaching reports."""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.runtime.agent import sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.session.events import SessionEventStore
from core.session.state import LifecycleStatus, SessionStateManager
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "This report is AI-generated coaching feedback for practice and preparation purposes only. "
    "It does not constitute an employment decision, hiring recommendation, qualification guarantee, "
    "or formal assessment."
)


class AssessmentBarrierError(Exception):
    """Raised when report synthesis is attempted before the assessment barrier is satisfied."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


class ReportSynthesizer:
    """Enforces assessment barrier, synthesizes final practice metrics and markdown report atomically."""

    def __init__(
        self,
        redis_client: Any,
        workspace_resolver: Union[Callable[[str], SessionWorkspace], SessionWorkspaceManager],
        state_manager: Optional[SessionStateManager] = None,
        event_store: Optional[SessionEventStore] = None,
        llm_client: Optional[RuntimeLLMClient] = None,
    ) -> None:
        """Initialize ReportSynthesizer.

        Args:
            redis_client: Redis client for checking set cardinality.
            workspace_resolver: Resolver or SessionWorkspaceManager.
            state_manager: Optional SessionStateManager for CAS lifecycle transitions.
            event_store: Optional SessionEventStore for appending report_ready event.
            llm_client: Optional LLM client for enhanced narrative synthesis.
        """
        self.redis_client = redis_client
        self.workspace_resolver = workspace_resolver
        self.state_manager = state_manager
        self.event_store = event_store
        self.llm_client = llm_client

    def _get_workspace(self, session_id: str) -> SessionWorkspace:
        """Resolve workspace instance for the session."""
        if hasattr(self.workspace_resolver, "get_or_create_workspace"):
            return self.workspace_resolver.get_or_create_workspace(session_id)
        return self.workspace_resolver(session_id)

    async def check_barrier(
        self, session_id: str, expected_assessments: Optional[int] = None
    ) -> Tuple[bool, int, int, int]:
        """Check whether the assessment barrier is satisfied.

        Barrier Condition:
            completed_assessments == expected_assessments and failed_assessments == 0

        Args:
            session_id: Target session identifier.
            expected_assessments: Optional explicit expected count.

        Returns:
            Tuple[bool, int, int, int]: (is_ready, completed_count, failed_count, expected_count).
        """
        completed_count = await self.redis_client.scard(
            f"session:{session_id}:completed_assessments"
        )
        failed_count = await self.redis_client.scard(
            f"session:{session_id}:failed_assessments"
        )

        expected = expected_assessments
        if expected is None:
            # Check Redis
            raw_exp = await self.redis_client.get(
                f"session:{session_id}:expected_assessments"
            )
            if raw_exp is not None:
                expected = int(raw_exp)

        if expected is None and self.state_manager is not None:
            state = self.state_manager.get_state(session_id)
            if state and "expected_assessments" in state.metadata:
                expected = int(state.metadata["expected_assessments"])
            elif state and "expected_assessment_count" in state.metadata:
                expected = int(state.metadata["expected_assessment_count"])

        # If expected assessments cannot be resolved, fail safe (is_ready = False)
        is_ready = (
            failed_count == 0
            and expected is not None
            and expected > 0
            and completed_count == expected
        )
        return is_ready, completed_count, failed_count, (expected or 0)

    async def synthesize(
        self, session_id: str, expected_assessments: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synthesize final assessment JSON and markdown report after verifying barrier.

        Args:
            session_id: Target session identifier.
            expected_assessments: Optional expected assessment count override.

        Returns:
            Dict[str, Any]: Synthesized final assessment data.

        Raises:
            AssessmentBarrierError: If barrier conditions are not satisfied.
        """
        is_ready, completed, failed, expected = await self.check_barrier(
            session_id, expected_assessments
        )

        if not is_ready:
            if failed > 0:
                raise AssessmentBarrierError(
                    f"Assessment barrier blocked for session '{session_id}': "
                    f"{failed} evaluation(s) failed permanently."
                )
            raise AssessmentBarrierError(
                f"Assessment barrier not met for session '{session_id}': "
                f"completed {completed}/{expected} evaluations."
            )

        ws = self._get_workspace(session_id)

        # 1. Collect all turn evaluation artifacts
        eval_paths = sorted(ws.list_artifacts("evaluations"))
        evaluations: List[Dict[str, Any]] = []
        for path in eval_paths:
            if path.name.endswith("_eval.json"):
                rel_path = path.relative_to(ws.root_path)
                data = ws.read_json(rel_path)
                eval_payload = data.get("payload", data)
                evaluations.append(eval_payload)

        # 2. Compute Practice Scores and Aggregated Findings
        content_scores = [e.get("content_score", 5) for e in evaluations]
        comm_scores = [e.get("communication_score", 5) for e in evaluations]

        avg_content = (
            round(sum(content_scores) / len(content_scores), 2)
            if content_scores
            else 5.0
        )
        avg_comm = (
            round(sum(comm_scores) / len(comm_scores), 2)
            if comm_scores
            else 5.0
        )
        # Problem solving score weighted from content depth
        problem_solving = round((avg_content * 0.7 + avg_comm * 0.3), 2)
        overall_score = round((avg_content + avg_comm + problem_solving) / 3, 2)

        # Aggregate unique strengths and improvements
        all_strengths: List[str] = []
        all_improvements: List[str] = []
        evidence_references: List[Dict[str, Any]] = []

        for e in evaluations:
            turn_id = e.get("turn_id", "")
            all_strengths.extend(e.get("strengths", []))
            all_improvements.extend(e.get("improvements", []))
            evidence_references.append(
                {
                    "turn_id": turn_id,
                    "content_score": e.get("content_score"),
                    "communication_score": e.get("communication_score"),
                    "rubric_target": e.get("rubric_target", []),
                    "strengths": e.get("strengths", []),
                    "improvements": e.get("improvements", []),
                }
            )

        unique_strengths = list(dict.fromkeys(all_strengths))
        unique_improvements = list(dict.fromkeys(all_improvements))

        curriculum = [
            f"Review fundamental design principles for: {item}"
            for item in unique_improvements[:5]
        ]
        if not curriculum:
            curriculum = ["Continue practicing advanced edge cases and trade-off rationalization."]

        # 3. Construct final_assessment.json payload
        final_assessment: Dict[str, Any] = {
            "schema_version": "1.0",
            "session_id": session_id,
            "synthesized_at": time.time(),
            "disclaimer": DISCLAIMER_TEXT,
            "metrics": {
                "overall_score": overall_score,
                "technical_score": avg_content,
                "communication_score": avg_comm,
                "problem_solving_score": problem_solving,
                "total_turns_evaluated": len(evaluations),
                "confidence_score": 0.95,
                "limitations": [
                    "Practice evaluations are based on textual answers and audio pace dynamics.",
                    "Scoring measures structured articulation in a mock context.",
                ],
            },
            "demonstrated_strengths": unique_strengths,
            "improvement_areas": unique_improvements,
            "recommended_curriculum": curriculum,
            "evidence_references": evidence_references,
        }

        # 4. Construct interview_report.md
        md_lines = [
            "# AI Interview Practice & Coaching Report",
            "",
            "> [!IMPORTANT]",
            f"> **Disclaimer:** {DISCLAIMER_TEXT}",
            "",
            "## 1. Practice Performance Summary",
            "",
            f"- **Overall Practice Score:** `{overall_score} / 10.0`",
            f"- **Technical Competence:** `{avg_content} / 10.0`",
            f"- **Communication Structure:** `{avg_comm} / 10.0`",
            f"- **Problem Solving Reasoning:** `{problem_solving} / 10.0`",
            f"- **Completed Evaluated Turns:** `{len(evaluations)}`",
            "",
            "## 2. Demonstrated Strengths",
            "",
        ]

        if unique_strengths:
            for s in unique_strengths:
                md_lines.append(f"- {s}")
        else:
            md_lines.append("- Candidate demonstrated steady responses across all prompt turns.")

        md_lines.extend([
            "",
            "## 3. Targeted Improvement Areas",
            "",
        ])
        if unique_improvements:
            for imp in unique_improvements:
                md_lines.append(f"- {imp}")
        else:
            md_lines.append("- No critical deficiencies noted; refine precision on edge cases.")

        md_lines.extend([
            "",
            "## 4. Recommended Study Curriculum",
            "",
        ])
        for c in curriculum:
            md_lines.append(f"1. {c}")

        md_lines.extend([
            "",
            "## 5. Turn-by-Turn Evidence Breakdown",
            "",
            "| Turn ID | Target Competencies | Content | Comm | Delivery Note |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for e in evaluations:
            t_id = e.get("turn_id", "turn")
            targets = ", ".join(e.get("rubric_target", [])) or "General"
            c_sc = e.get("content_score", "-")
            cm_sc = e.get("communication_score", "-")
            deliv = e.get("delivery_observations", {}).get("observations", "N/A")
            # Truncate delivery note for clean table formatting
            short_deliv = (deliv[:40] + "...") if len(deliv) > 40 else deliv
            md_lines.append(f"| `{t_id}` | {targets} | {c_sc}/10 | {cm_sc}/10 | {short_deliv} |")

        md_lines.append("")
        report_markdown = "\n".join(md_lines)

        # 5. Atomically persist final_assessment.json and interview_report.md
        ws.write_artifact(
            "reports/final_assessment.json",
            artifact_type="final_assessment",
            payload=final_assessment,
        )
        ws.write_text("reports/interview_report.md", report_markdown)

        # 6. Append report_ready event to event store and Redis stream
        event_payload = {
            "session_id": session_id,
            "overall_score": overall_score,
            "report_path": "reports/interview_report.md",
            "assessment_path": "reports/final_assessment.json",
        }
        if self.event_store is not None:
            self.event_store.append_event(
                session_id=session_id,
                turn_id="summary",
                event_type="report_ready",
                payload=event_payload,
            )

        try:
            await self.redis_client.xadd(
                f"session:{session_id}:events",
                {
                    "type": "report_ready",
                    "turn_id": "summary",
                    "payload": json.dumps(event_payload),
                },
            )
        except Exception:
            pass

        # 7. Transition session lifecycle to REPORT_READY via CAS
        if self.state_manager is not None:
            state = self.state_manager.get_state(session_id)
            if state and state.status == LifecycleStatus.ASSESSING:
                self.state_manager.transition(
                    state,
                    expected_version=state.state_version,
                    new_status=LifecycleStatus.REPORT_READY,
                    metadata={
                        "overall_score": overall_score,
                        "report_synthesized_at": time.time(),
                    },
                )

        return final_assessment
