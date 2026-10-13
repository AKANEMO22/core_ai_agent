"""Report synthesizer: assessment barrier, deterministic decision, verified narrative, report artifacts."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.agents.narrative import NarrativeWriter
from core.assessor.decision import (
    NEEDS_REVIEW,
    DecisionConfig,
    code_version,
    compute_decision,
    decision_json,
)
from core.assessor.queue import (
    completed_assessments_key,
    expected_assessments_key,
    failed_assessments_key,
)
from core.orchestration.coverage import load_evaluations
from core.runtime.agent import sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.runtime.types import LoopEvent, LoopEventType
from core.session.events import SessionEventStore
from core.session.state import LifecycleStatus, SessionStateStore
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "This report is AI-generated coaching feedback for practice and preparation purposes only. "
    "It does not constitute an employment decision, hiring recommendation, qualification guarantee, "
    "or formal assessment."
)
HIRING_ASSIST_DISCLAIMER = (
    "This report is an AI-assisted assessment intended to support, not replace, a human hiring "
    "decision. A qualified reviewer must confirm the outcome before it is used; it is not an "
    "employment decision on its own."
)
DECISION_PATH = "reports/decision.json"
REVIEW_PATH = "reports/review.json"


class AssessmentBarrierError(Exception):
    """Raised when report synthesis is attempted before the assessment barrier is satisfied."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


def _payload(workspace: SessionWorkspace, rel: str) -> Dict[str, Any]:
    if not workspace.exists(rel):
        return {}
    data = workspace.read_json(rel)
    return data.get("payload", data) if isinstance(data, dict) else {}


class ReportSynthesizer:
    """Builds `decision.json`, `final_assessment.json` and `interview_report.md` once the barrier holds."""

    def __init__(
        self,
        redis_client: Any,
        workspace_resolver: Union[Callable[[str], SessionWorkspace], SessionWorkspaceManager],
        state_manager: Optional[SessionStateStore] = None,
        event_store: Optional[SessionEventStore] = None,
        llm_client: Optional[RuntimeLLMClient] = None,
        connection_manager: Optional[Any] = None,
        decision_config: Optional[DecisionConfig] = None,
        narrative_writer: Optional[NarrativeWriter] = None,
        extra_versions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ReportSynthesizer.

        Args:
            redis_client: Redis client for barrier counters.
            workspace_resolver: Resolver or SessionWorkspaceManager.
            state_manager: Optional session state store for the REPORT_READY transition.
            event_store: Optional event store (used when no connection manager is configured).
            llm_client: Optional LLM client for the narrative.
            connection_manager: Optional publisher for live events.
            decision_config: Decision thresholds and product mode.
            narrative_writer: Narrative generator (defaults to one using `llm_client`).
            extra_versions: Version fields recorded in every decision (e.g. prompt versions).
        """
        self.redis_client = redis_client
        self.workspace_resolver = workspace_resolver
        self.state_manager = state_manager
        self.event_store = event_store
        self.llm_client = llm_client
        self.connection_manager = connection_manager
        self.decision_config = decision_config or DecisionConfig()
        self.narrative_writer = narrative_writer or NarrativeWriter(llm_client)
        self.extra_versions = dict(extra_versions or {})

    def _get_workspace(self, session_id: str) -> SessionWorkspace:
        if hasattr(self.workspace_resolver, "get_or_create_workspace"):
            return self.workspace_resolver.get_or_create_workspace(session_id)
        return self.workspace_resolver(session_id)

    async def check_barrier(
        self, session_id: str, expected_assessments: Optional[int] = None
    ) -> Tuple[bool, int, int, int]:
        """Barrier: completed >= expected and failed == 0, with expected known (0 allowed).

        Returns:
            Tuple[bool, int, int, int]: (is_ready, completed, failed, expected).
        """
        completed_count = await self.redis_client.scard(completed_assessments_key(session_id))
        failed_count = await self.redis_client.scard(failed_assessments_key(session_id))

        expected = expected_assessments
        if expected is None:
            raw_exp = await self.redis_client.get(expected_assessments_key(session_id))
            if raw_exp is not None:
                expected = int(raw_exp)
        if expected is None and self.state_manager is not None:
            state = await self.state_manager.aget(session_id)
            if state and "expected_assessments" in state.metadata:
                expected = int(state.metadata["expected_assessments"])
            elif state and "expected_assessment_count" in state.metadata:
                expected = int(state.metadata["expected_assessment_count"])

        is_ready = failed_count == 0 and expected is not None and expected >= 0 and completed_count >= expected
        return is_ready, completed_count, failed_count, (expected or 0)

    def _versions(self, workspace: SessionWorkspace, rubric: Dict[str, Any], evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        playbooks = set()
        for path in workspace.list_artifacts("turns"):
            if path.name.endswith("_interviewer.json"):
                ref = workspace.read_json(path.relative_to(workspace.root_path).as_posix()).get("playbook_ref")
                if ref:
                    playbooks.add(ref)
        knowledge = _payload(workspace, "context/retrieved_knowledge.json")
        return {
            "code": code_version(),
            "rubric_version": rubric.get("rubric_version"),
            "retrieval_backend": knowledge.get("retrieval_backend"),
            "evaluator_models": sorted({str((e.get("model") or {}).get("id")) for e in evaluations if e.get("model")}),
            "evaluator_prompts": sorted(
                {str((e.get("model") or {}).get("prompt_version")) for e in evaluations if e.get("model")}
            ),
            "playbooks": sorted(playbooks),
            **self.extra_versions,
        }

    @staticmethod
    def _legacy_rubric(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Equal-weight rubric over the legacy rubric targets (retrieval degraded)."""
        targets = sorted({t for e in evaluations for t in (e.get("rubric_target") or [])})
        weight = round(1.0 / len(targets), 4) if targets else 0.0
        return {
            "rubric_version": "legacy",
            "pass_threshold": 3.0,
            "competencies": [
                {"competency_id": t, "name": t, "weight": weight, "must_have": False, "pass_level": 3} for t in targets
            ],
        }

    async def synthesize(self, session_id: str, expected_assessments: Optional[int] = None) -> Dict[str, Any]:
        """Synthesize decision, assessment and report after verifying the barrier.

        Raises:
            AssessmentBarrierError: If barrier conditions are not satisfied.
        """
        # Chờ đủ kết quả assessment trước khi tạo verdict để báo cáo không phản ánh trạng thái xử lý dở dang.
        is_ready, completed, failed, expected = await self.check_barrier(session_id, expected_assessments)
        if not is_ready:
            if failed > 0:
                raise AssessmentBarrierError(
                    f"Assessment barrier blocked for session '{session_id}': {failed} evaluation(s) failed permanently."
                )
            raise AssessmentBarrierError(
                f"Assessment barrier not met for session '{session_id}': completed {completed}/{expected} evaluations."
            )

        ws = self._get_workspace(session_id)
        evaluations = load_evaluations(ws)
        rubric = _payload(ws, "context/rubric.json")
        rubric_source = "retrieved"
        if not rubric.get("competencies"):
            rubric, rubric_source = self._legacy_rubric(evaluations), "legacy"

        decision = compute_decision(rubric, evaluations, self.decision_config, self._versions(ws, rubric, evaluations))
        if rubric_source == "legacy" and decision["verdict"] != NEEDS_REVIEW:
            decision["verdict"] = NEEDS_REVIEW
            decision["requires_human_review"] = True
            decision["verdict_reasons"].append("rubric unavailable (retrieval degraded): human review required")

        candidate_texts = []
        for path in sorted(ws.list_artifacts("turns")):
            if path.name.endswith("_candidate.json"):
                candidate_texts.append(str(ws.read_json(path.relative_to(ws.root_path).as_posix()).get("content", "")))
        notebook = ws.read_text("context/interview_notebook.md") if ws.exists("context/interview_notebook.md") else ""
        claims = _payload(ws, "orchestration/claims.json")
        narrative = await self.narrative_writer.write(decision, evaluations, candidate_texts, notebook, claims)

        mode = self.decision_config.product_mode
        disclaimer = HIRING_ASSIST_DISCLAIMER if mode == "hiring_assist" else DISCLAIMER_TEXT
        content_scores = [e["content_score"] for e in evaluations if isinstance(e.get("content_score"), (int, float))]
        comm_scores = [e["communication_score"] for e in evaluations if isinstance(e.get("communication_score"), (int, float))]
        covered = [c for c in decision["criteria"] if c["covered"]]
        confidence = round(len(covered) / len(decision["criteria"]), 3) if decision["criteria"] else None

        final_assessment: Dict[str, Any] = {
            "schema_version": "2.0",
            "session_id": session_id,
            "synthesized_at": time.time(),
            "disclaimer": disclaimer,
            "product_mode": mode,
            "verdict": decision["verdict"],
            "requires_human_review": decision["requires_human_review"],
            "metrics": {
                "weighted_score": decision["weighted_score"],
                "pass_threshold": decision["pass_threshold"],
                "overall_score": decision["weighted_score"],
                "technical_score": round(sum(content_scores) / len(content_scores), 2) if content_scores else None,
                "communication_score": round(sum(comm_scores) / len(comm_scores), 2) if comm_scores else None,
                "total_turns_evaluated": len(evaluations),
                "criteria_coverage": confidence,
                "confidence_score": confidence,
                "limitations": [
                    "Scores reflect evidence in this mock interview's transcript only.",
                    "Transcripts come from speech recognition and may contain errors.",
                    "Delivery metrics (pace, pauses) are descriptive and never scored.",
                ],
            },
            "criteria": decision["criteria"],
            "uncovered_criteria": decision["uncovered_criteria"],
            "narrative": narrative,
            "claims": claims.get("claims", []),
            "demonstrated_strengths": narrative.get("strengths", []),
            "improvement_areas": narrative.get("improvements", []),
            "recommended_curriculum": narrative.get("study_plan", []),
            "evidence_references": [
                {
                    "turn_id": e.get("turn_id"),
                    "rubric_target": e.get("rubric_target", []),
                    "content_score": e.get("content_score"),
                    "communication_score": e.get("communication_score"),
                }
                for e in evaluations
            ],
        }
        report_markdown = self._render_markdown(decision, narrative, final_assessment, rubric, evaluations, claims)

        await asyncio.to_thread(ws.write_text, DECISION_PATH, decision_json(decision))
        await asyncio.to_thread(
            ws.write_artifact, "reports/final_assessment.json", artifact_type="final_assessment", payload=final_assessment
        )
        await asyncio.to_thread(ws.write_text, "reports/interview_report.md", report_markdown)

        event_payload = {
            "session_id": session_id,
            "verdict": decision["verdict"],
            "requires_human_review": decision["requires_human_review"],
            "weighted_score": decision["weighted_score"],
            "overall_score": decision["weighted_score"],
            "report_path": "reports/interview_report.md",
            "assessment_path": "reports/final_assessment.json",
        }
        if self.state_manager is not None:
            state = await self.state_manager.aget(session_id)
            if state and state.status == LifecycleStatus.ASSESSING:
                await self.state_manager.atransition(
                    state,
                    expected_version=state.state_version,
                    new_status=LifecycleStatus.REPORT_READY,
                    metadata={
                        "verdict": decision["verdict"],
                        "weighted_score": decision["weighted_score"],
                        "overall_score": decision["weighted_score"],
                        "report_synthesized_at": time.time(),
                    },
                )

        await self._publish(session_id, LoopEventType.DECISION_READY, {
            "verdict": decision["verdict"], "requires_human_review": decision["requires_human_review"],
        })
        await self._publish(session_id, LoopEventType.REPORT_READY, event_payload)
        return final_assessment

    def _render_markdown(
        self,
        decision: Dict[str, Any],
        narrative: Dict[str, Any],
        final_assessment: Dict[str, Any],
        rubric: Dict[str, Any],
        evaluations: List[Dict[str, Any]],
        claims: Dict[str, Any],
    ) -> str:
        anchors = {c["competency_id"]: c.get("anchors", {}) for c in rubric.get("competencies", []) if c.get("competency_id")}
        verdict_label = {"PASS": "Ready (pass)", "FAIL": "Not yet ready (fail)", "NEEDS_REVIEW": "Needs human review"}
        lines = [
            "# AI Interview Practice & Coaching Report",
            "",
            "> [!IMPORTANT]",
            f"> **Disclaimer:** {final_assessment['disclaimer']}",
            "",
            "## 1. Outcome",
            "",
            f"- **Verdict:** {verdict_label.get(decision['verdict'], decision['verdict'])}",
            f"- **Weighted score:** {decision['weighted_score'] if decision['weighted_score'] is not None else 'n/a'} "
            f"/ 4 (pass threshold {decision['pass_threshold']}, borderline ±{decision['borderline_band']})",
        ]
        lines += [f"- {reason}" for reason in decision["verdict_reasons"]]
        if decision["requires_human_review"]:
            lines.append("- A human reviewer must confirm this outcome.")
        lines += ["", "## 2. Summary", "", narrative.get("summary", ""), ""]

        lines += [
            "## 3. Criteria",
            "",
            "| Criterion | Weight | Must-have | Level (1-4) | Evidence weight (sum of confidence) | Meets bar |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for c in decision["criteria"]:
            level = "insufficient evidence" if not c["covered"] else f"{c['level']}"
            lines.append(
                f"| {c['name']} | {c['weight']} | {'yes' if c['must_have'] else 'no'} | {level} | "
                f"{c['confidence']} | {'yes' if c['meets_bar'] else 'no'} |"
            )
        lines.append("")

        findings = {f.get("competency_id"): f for f in narrative.get("criterion_findings", [])}
        lines += ["## 4. Findings by criterion", ""]
        for c in decision["criteria"]:
            lines.append(f"### {c['name']}")
            finding = findings.get(c["competency_id"], {})
            lines.append(finding.get("finding") or ("Not enough evidence was gathered." if not c["covered"] else ""))
            if c["covered"] and c["level"] is not None:
                anchor = anchors.get(c["competency_id"], {}).get(str(int(round(c["level"]))))
                if anchor:
                    lines.append(f"- Closest level description: {anchor}")
            for quote in (finding.get("quotes") or [])[:2]:
                lines.append(f"> \"{quote}\"")
            lines.append("")

        lines += ["## 5. Strengths", ""] + [f"- {s}" for s in narrative.get("strengths", [])] + [""]
        lines += ["## 6. Areas to improve", ""] + [f"- {s}" for s in narrative.get("improvements", [])] + [""]
        lines += ["## 7. Study plan", ""] + [f"{i}. {s}" for i, s in enumerate(narrative.get("study_plan", []), 1)] + [""]

        claim_items = claims.get("claims", [])
        if claim_items:
            lines += ["## 8. CV claims checked", "", "| Claim | Status |", "| :--- | :--- |"]
            lines += [f"| {c.get('claim')} | {c.get('status')} |" for c in claim_items]
            lines.append("")
        if decision["uncovered_criteria"]:
            lines += ["## 9. Not covered in this interview", ""]
            lines += [f"- {cid}" for cid in decision["uncovered_criteria"]]
            lines.append("")

        lines += ["## 10. Delivery observations (not scored)", ""]
        for e in evaluations:
            note = (e.get("delivery_observations") or {}).get("observations")
            if note:
                lines.append(f"- {e.get('turn_id')}: {note}")
        lines += ["", "## 11. Limitations", ""] + [f"- {l}" for l in final_assessment["metrics"]["limitations"]] + [""]
        return "\n".join(lines)

    async def _publish(self, session_id: str, event_type: LoopEventType, payload: Dict[str, Any]) -> None:
        try:
            event = LoopEvent(
                schema_version="1.0",
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                turn_id="summary",
                sequence=0,
                timestamp=time.time(),
                type=event_type,
                payload=payload,
            )
            if self.connection_manager is not None:
                recorded = await self.connection_manager.record_event_async(session_id, event)
                await self.connection_manager.send_event(session_id, recorded)
            elif self.event_store is not None:
                await self.event_store.append_async(session_id, event)
        except Exception as e:
            logger.warning("Failed to publish %s: %s", event_type.value, sanitize_secrets(str(e)))


class ReviewService:
    """Records a human reviewer's final verdict next to the (immutable) computed decision."""

    def __init__(self, workspace_resolver: Any) -> None:
        self.workspace_resolver = workspace_resolver

    async def record_review(
        self, session_id: str, reviewer_id: str, final_verdict: str, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Persist `reports/review.json` and return the decision with the review attached."""
        ws = (
            self.workspace_resolver.get_or_create_workspace(session_id)
            if hasattr(self.workspace_resolver, "get_or_create_workspace")
            else self.workspace_resolver(session_id)
        )
        decision = json.loads(ws.read_text(DECISION_PATH)) if ws.exists(DECISION_PATH) else {}
        review = {
            "reviewer_id": reviewer_id,
            "final_verdict": final_verdict,
            "notes": notes,
            "computed_verdict": decision.get("verdict"),
            "decision_digest": decision.get("input_digest"),
            "reviewed_at": time.time(),
        }
        await asyncio.to_thread(ws.write_artifact, REVIEW_PATH, artifact_type="human_review", payload=review)
        return {**decision, "review": review}
