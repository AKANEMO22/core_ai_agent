"""Interactive interviewer runner: turn execution, streaming generation, persistence and assessment."""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from core.interview.context import ContextAssembler
from core.interview.director import StageDirector, StageSessionConflictError
from core.interview.queue import AssessmentJob, AssessmentQueue
from core.interview.speaker import SpeakerContextBuilder
from core.orchestration.directive import (
    Directive,
    DirectiveStore,
    asked_question_ids,
    fallback_directive,
    plan_fallback,
)
from core.runtime.agent import RuntimeAgent, sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType
from core.session.state import LifecycleStatus, SessionState, SessionStateStore
from core.workspace.session import SessionWorkspace

logger = logging.getLogger(__name__)

OPENING_INSTRUCTION = (
    "Begin the interview now. Greet the candidate in one short sentence, explain the format "
    "of this stage in one sentence, then ask the first question. Ask exactly one question."
)

TURN_FAILED_CODE = "TURN_FAILED"

DialogueCompletedCallback = Callable[[str], Awaitable[None]]
TurnCompletedCallback = Callable[[str, int, SessionState], Awaitable[None]]


def turn_id_for(turn_number: int) -> str:
    """Canonical turn identifier (turn 0 is the interviewer's opening)."""
    return f"turn_{turn_number:03d}"


class InteractiveSubagentRunner:
    """Executes interviewer turns for one session.

    Turn contract:
        - The candidate artifact is persisted concurrently with generation.
        - The assessment job is enqueued only after the interviewer turn succeeded or was
          interrupted, and before the turn counter advances, so a crash in between is
          recovered by an idempotent retry of the same `message_id`.
        - A failed generation (provider error, deadline without output) does not advance
          the turn; an `error` event with `code=TURN_FAILED` and `retryable=True` is emitted.
        - A cancelled or deadline-cut generation with partial output is persisted with
          `interrupted=True` and completes the turn.

    Attributes:
        workspace: Isolated session workspace for artifact storage.
        runtime_agent: Streaming agent executing LLM interactions and tool calls.
        context_assembler: Context assembler enforcing the prompt token ceiling.
        stage_director: Director tracking stages, turn counters, and CAS transitions.
        assessment_queue: Optional async queue for turn evaluation jobs.
        state_mgr: Session state store.
        turn_deadline_seconds: Hard deadline for one interviewer generation.
    """

    def __init__(
        self,
        workspace: SessionWorkspace,
        runtime_agent: RuntimeAgent,
        context_assembler: ContextAssembler,
        stage_director: StageDirector,
        assessment_queue: Optional[AssessmentQueue] = None,
        state_mgr: Optional[SessionStateStore] = None,
        turn_deadline_seconds: Optional[float] = None,
        on_dialogue_completed: Optional[DialogueCompletedCallback] = None,
        on_turn_completed: Optional[TurnCompletedCallback] = None,
        context_builder: Optional[SpeakerContextBuilder] = None,
        directive_store: Optional[DirectiveStore] = None,
    ) -> None:
        self.workspace = workspace
        self.runtime_agent = runtime_agent
        self.context_assembler = context_assembler
        self.stage_director = stage_director
        self.assessment_queue = assessment_queue
        self.state_mgr = state_mgr or stage_director.state_mgr
        self.turn_deadline_seconds = turn_deadline_seconds
        self.on_dialogue_completed = on_dialogue_completed
        self.on_turn_completed = on_turn_completed
        self.context_builder = context_builder
        self.directive_store = directive_store or DirectiveStore()
        self._cached_spec: Optional[Dict[str, Any]] = None
        self._cached_profile: Optional[Dict[str, Any]] = None
        self._cached_intel: Optional[Dict[str, Any]] = None
        self._cached_rubric: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def _read_json_safe(self, rel_path: str, default: Any = None) -> Any:
        """Safely read a JSON artifact from workspace if it exists."""
        try:
            if self.workspace.exists(rel_path):
                return self.workspace.read_json(rel_path)
        except Exception as e:
            logger.warning(
                "Error reading artifact '%s': %s", rel_path, sanitize_secrets(str(e))
            )
        return default if default is not None else {}

    def _get_previous_question(self, turn_number: int) -> str:
        """Retrieve the interviewer text that preceded the candidate's answer for this turn."""
        if turn_number >= 1:
            prev_file = f"turns/turn_{turn_number - 1}_interviewer.json"
            if self.workspace.exists(prev_file):
                content = self._read_json_safe(prev_file, {}).get("content")
                if content:
                    return str(content)
        return "Initial stage introduction"

    def _load_recent_turns(self, up_to_turn: int) -> List[Dict[str, Any]]:
        """Load dialogue turns 0..up_to_turn in chronological order."""
        turns: List[Dict[str, Any]] = []
        for i in range(0, up_to_turn + 1):
            cand_path = f"turns/turn_{i}_candidate.json"
            if self.workspace.exists(cand_path):
                content = self._read_json_safe(cand_path, {}).get("content", "")
                if content:
                    turns.append({"role": "candidate", "content": content})

            int_path = f"turns/turn_{i}_interviewer.json"
            if self.workspace.exists(int_path):
                content = self._read_json_safe(int_path, {}).get("content", "")
                if content:
                    turns.append({"role": "interviewer", "content": content})
        return turns

    def _load_blueprints(self) -> None:
        if self._cached_spec is None:
            self._cached_spec = self._read_json_safe(
                "context/interview_spec.json",
                default={"interviewer_persona": "Interviewer", "tone": "professional"},
            )
        if self._cached_profile is None:
            self._cached_profile = self._read_json_safe("context/candidate_profile.json", default={})
        if self._cached_intel is None:
            self._cached_intel = self._read_json_safe("context/company_intel.json", default={})
        if self._cached_rubric is None:
            self._cached_rubric = self._read_json_safe("context/rubric.json", default={})

    def build_system_prompt(self, state: SessionState, up_to_turn: int) -> str:
        """Assemble the interviewer system prompt for the active stage."""
        self._load_blueprints()
        return self.context_assembler.assemble_prompt(
            spec=self._cached_spec or {},
            stage=self.stage_director.get_current_stage(state) or {},
            profile=self._cached_profile,
            company_intel=self._cached_intel,
            recent_turns=self._load_recent_turns(up_to_turn=up_to_turn),
        )

    def resolve_directive(self, state: SessionState, turn_number: int) -> Directive:
        """Directive written by the orchestrator for this turn, or the deterministic fallback."""
        directive = None
        try:
            directive = self.directive_store.read(self.workspace, turn_number)
        except Exception as e:
            logger.warning("Directive unreadable; using fallback: %s", sanitize_secrets(str(e)))
        if directive is not None and directive.question_text:
            return directive
        stage = self.stage_director.get_current_stage(state) or {}
        asked = asked_question_ids(self.workspace)
        if turn_number == 0:
            return fallback_directive(stage, asked)
        used = int((state.metadata.get("stage_turns") or {}).get(state.current_stage_id or "", 0))
        return plan_fallback(self.stage_director.stages, state.current_stage_id, used, asked)

    def build_messages(
        self, state: SessionState, turn_number: int, final_user_content: str
    ) -> tuple:
        """Return (messages, provenance) for a turn using the speaker builder when configured."""
        if self.context_builder is None:
            return (
                [
                    {"role": "system", "content": self.build_system_prompt(state, up_to_turn=turn_number - 1)},
                    {"role": "user", "content": final_user_content},
                ],
                {},
            )
        directive = self.resolve_directive(state, turn_number)
        stage = self.stage_director.get_current_stage(state) or {}
        context = self.context_builder.build(stage, turn_number - 1, directive, final_user_content)
        return context.messages, {
            "directive": directive,
            "playbook_ref": context.playbook_ref,
            "raw_turns_included": context.raw_turns_included,
            "summary_included": context.summary_included,
        }

    @staticmethod
    def _provenance_fields(provenance: Dict[str, Any]) -> Dict[str, Any]:
        directive: Optional[Directive] = provenance.get("directive")
        if directive is None:
            return {}
        return {
            "question_id": directive.question_id,
            "target_criterion": directive.target_criterion,
            "directive_mode": directive.mode,
            "directive_source": directive.generated_by,
            "playbook_ref": provenance.get("playbook_ref"),
        }

    @staticmethod
    def _event(session_id: str, turn_id: str, event_type: LoopEventType, payload: Dict[str, Any]) -> LoopEvent:
        return LoopEvent(
            schema_version="1.0",
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=turn_id,
            sequence=0,
            timestamp=time.time(),
            type=event_type,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Generation core
    # ------------------------------------------------------------------

    async def _generate(
        self,
        session_id: str,
        turn_id: str,
        messages: List[Dict[str, Any]],
        cancellation_token: Optional[asyncio.Event],
        outcome: Dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """Stream the interviewer generation, recording the outcome in `outcome`.

        outcome keys: text, tool_calls, usage, status ("ok" | "interrupted" | "failed"),
        interrupt_reason, error_code.
        """
        token = cancellation_token or asyncio.Event()
        deadline_hit = {"value": False}
        watchdog: Optional[asyncio.Task[None]] = None
        if self.turn_deadline_seconds:
            async def _watchdog() -> None:
                await asyncio.sleep(self.turn_deadline_seconds)
                deadline_hit["value"] = True
                token.set()

            watchdog = asyncio.create_task(_watchdog())

        tokens: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        outcome.update({"status": "ok", "usage": None, "error_code": None, "interrupt_reason": None})
        try:
            async for event in self.runtime_agent.run_loop_streaming(
                session_id=session_id,
                turn_id=turn_id,
                messages=messages,
                cancellation_token=token,
                **kwargs,
            ):
                if event.type == LoopEventType.RESPONSE_TOKEN:
                    tokens.append(event.payload.get("token", ""))
                elif event.type == LoopEventType.TOOL_CALL_START:
                    tool_calls.append(event.payload)
                elif event.type == LoopEventType.COMPLETE:
                    outcome["usage"] = event.payload.get("usage")
                    continue
                elif event.type == LoopEventType.ERROR:
                    code = event.payload.get("code")
                    if code in ("CANCELLED", "TURN_DEADLINE"):
                        outcome["status"] = "interrupted"
                        outcome["interrupt_reason"] = (
                            "deadline" if deadline_hit["value"] or code == "TURN_DEADLINE" else "cancelled"
                        )
                        continue
                    outcome["status"] = "failed"
                    outcome["error_code"] = code
                yield event
        finally:
            if watchdog is not None:
                watchdog.cancel()

        outcome["text"] = "".join(tokens)
        outcome["tool_calls"] = tool_calls
        if outcome["status"] == "ok" and not outcome["text"].strip():
            # An empty interviewer turn must never be recorded as spoken.
            outcome["status"] = "failed"
            outcome["error_code"] = "EMPTY_COMPLETION"
        if outcome["status"] == "interrupted" and not outcome["text"]:
            outcome["status"] = "failed"
            outcome["error_code"] = "TURN_DEADLINE" if deadline_hit["value"] else "CANCELLED"

    # ------------------------------------------------------------------
    # Opening (turn 0)
    # ------------------------------------------------------------------

    async def run_opening(
        self,
        session_id: str,
        cancellation_token: Optional[asyncio.Event] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """Stream the interviewer's opening question (turn 0) for a STAGE_ACTIVE session.

        The session must be STAGE_ACTIVE with no completed candidate turns. On failure the
        session stays STAGE_ACTIVE at turn 0 and the opening can be retried.
        """
        state = await self.state_mgr.aget(session_id)
        if state is None:
            raise StageSessionConflictError(f"Session '{session_id}' not found")
        if state.status != LifecycleStatus.STAGE_ACTIVE or state.current_turn != 0:
            raise StageSessionConflictError(
                f"Opening requires STAGE_ACTIVE at turn 0 (status {state.status.value}, "
                f"turn {state.current_turn})"
            )

        turn_id = turn_id_for(0)
        yield self._event(
            session_id,
            turn_id,
            LoopEventType.INTERVIEW_STARTED,
            {"stage_id": state.current_stage_id, "turn_number": 0},
        )

        messages, provenance = self.build_messages(state, 0, OPENING_INSTRUCTION)
        outcome: Dict[str, Any] = {}
        async for event in self._generate(session_id, turn_id, messages, cancellation_token, outcome, **kwargs):
            yield event

        if outcome["status"] == "failed":
            yield self._event(
                session_id,
                turn_id,
                LoopEventType.ERROR,
                {"code": TURN_FAILED_CODE, "retryable": True, "cause": outcome.get("error_code")},
            )
            return

        await asyncio.to_thread(
            self.workspace.write_artifact,
            "turns/turn_0_interviewer.json",
            artifact_type="turn_interviewer",
            payload={
                "turn_id": turn_id,
                "turn_number": 0,
                "role": "interviewer",
                "session_id": session_id,
                "stage_id": state.current_stage_id,
                "content": outcome["text"],
                "interrupted": outcome["status"] == "interrupted",
                "timestamp": time.time(),
                **self._provenance_fields(provenance),
            },
        )
        yield self._event(
            session_id,
            turn_id,
            LoopEventType.TURN_COMPLETE,
            {
                "turn_number": 0,
                "turn_id": turn_id,
                "stage_id": state.current_stage_id,
                "interviewer_response": outcome["text"],
                "interrupted": outcome["status"] == "interrupted",
                "usage": outcome.get("usage"),
            },
        )

    # ------------------------------------------------------------------
    # Candidate turn
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        session_id: str,
        candidate_answer: str,
        vad_metrics: Optional[Dict[str, Any]] = None,
        expected_turn: Optional[int] = None,
        cancellation_token: Optional[asyncio.Event] = None,
        rubric_version: Optional[str] = None,
        rubric_target: Optional[List[str]] = None,
        message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """Execute a candidate turn and stream the interviewer's response.

        Args:
            session_id: Session identifier.
            candidate_answer: Candidate text answer transcript.
            vad_metrics: Optional voice activity detection and speech delivery metrics.
            expected_turn: Optional expected 1-indexed turn number for stale/order validation.
            cancellation_token: Optional asyncio event signaling external cancellation.
            rubric_version: Rubric version for assessment job evaluation. Defaults to the
                version in context/rubric.json, then "rubric_v1".
            rubric_target: Specific competencies/topics to score. Defaults to the active
                stage's retrieved `criteria`.
            message_id: Client message identifier recorded on the candidate artifact and job.
            **kwargs: Extra parameters passed to the runtime agent LLM provider.

        Yields:
            LoopEvent: Streaming lifecycle events for the turn.

        Raises:
            StageSessionConflictError: If session is not in STAGE_ACTIVE or turn is stale.
        """
        state = await self.state_mgr.aget(session_id)
        if state is None:
            raise StageSessionConflictError(
                f"Session '{session_id}' not found in state manager registry"
            )
        if state.status != LifecycleStatus.STAGE_ACTIVE:
            raise StageSessionConflictError(
                f"Session '{session_id}' is not in STAGE_ACTIVE status "
                f"(current status: {state.status.value})"
            )

        turn_number = state.current_turn + 1
        if expected_turn is not None and expected_turn != turn_number:
            raise StageSessionConflictError(
                f"Out-of-order turn submission for session '{session_id}': "
                f"expected turn {turn_number}, but received {expected_turn}"
            )
        turn_id = turn_id_for(turn_number)
        self._load_blueprints()
        curr_stage = self.stage_director.get_current_stage(state) or {}

        candidate_payload = {
            "turn_id": turn_id,
            "turn_number": turn_number,
            "role": "candidate",
            "session_id": session_id,
            "stage_id": state.current_stage_id,
            "message_id": message_id,
            "content": candidate_answer,
            "vad_metrics": dict(vad_metrics or {}),
            "timestamp": time.time(),
        }
        # Ghi transcript độc lập với lượt gọi model để câu trả lời vẫn được giữ nếu generation lỗi hoặc bị ngắt.
        persist_task = asyncio.create_task(
            asyncio.to_thread(
                self.workspace.write_artifact,
                f"turns/turn_{turn_number}_candidate.json",
                artifact_type="turn_candidate",
                payload=candidate_payload,
            )
        )

        messages, provenance = self.build_messages(state, turn_number, candidate_answer)
        directive: Optional[Directive] = provenance.get("directive")
        outcome: Dict[str, Any] = {}
        try:
            async for event in self._generate(
                session_id, turn_id, messages, cancellation_token, outcome, **kwargs
            ):
                yield event
        finally:
            await persist_task

        if outcome["status"] == "failed":
            yield self._event(
                session_id,
                turn_id,
                LoopEventType.ERROR,
                {"code": TURN_FAILED_CODE, "retryable": True, "cause": outcome.get("error_code")},
            )
            return

        interrupted = outcome["status"] == "interrupted"
        if interrupted:
            yield self._event(
                session_id,
                turn_id,
                LoopEventType.GENERATION_CANCELLED,
                {"reason": outcome.get("interrupt_reason"), "partial_length": len(outcome["text"])},
            )

        await asyncio.to_thread(
            self.workspace.write_artifact,
            f"turns/turn_{turn_number}_interviewer.json",
            artifact_type="turn_interviewer",
            payload={
                "turn_id": turn_id,
                "turn_number": turn_number,
                "role": "interviewer",
                "session_id": session_id,
                "stage_id": state.current_stage_id,
                "content": outcome["text"],
                "interrupted": interrupted,
                "timestamp": time.time(),
                **self._provenance_fields(provenance),
            },
        )

        # Enqueue assessment before advancing the turn so a crash in between is retried
        # idempotently (same message_id -> same job_id and input).
        if self.assessment_queue is not None:
            stage_criteria = curr_stage.get("criteria")
            if rubric_target is not None:
                resolved_target = list(rubric_target)
            elif isinstance(stage_criteria, list):
                resolved_target = [str(c) for c in stage_criteria]
            else:
                resolved_target = []
            resolved_version = (
                rubric_version
                or (self._cached_rubric or {}).get("rubric_version")
                or "rubric_v1"
            )
            job = AssessmentJob(
                schema_version="1.0",
                job_id=f"{session_id}:{turn_id}:{resolved_version}",
                session_id=session_id,
                turn_id=turn_id,
                stage_id=state.current_stage_id or "",
                question=self._get_previous_question(turn_number),
                candidate_text=candidate_answer,
                vad_metrics=dict(vad_metrics or {}),
                rubric_version=resolved_version,
                rubric_target=resolved_target,
                created_at=time.time(),
            )
            job_dict = job.to_dict()
            job_dict["message_id"] = message_id
            await self.assessment_queue.enqueue(job_dict)

        prev_stage_id = state.current_stage_id
        # A transition directive means this response already moved on to the next stage.
        transition_requested = directive is not None and directive.mode == "transition"
        # An extension directive asked another question of this stage, so the stage stays open.
        extend_stage = (
            directive is not None and directive.extend_stage and directive.mode in ("follow_up", "next_question")
        )
        state = await self.stage_director.record_turn(
            state, complete_stage=transition_requested, tool_calls=outcome["tool_calls"], extend_stage=extend_stage
        )
        if state.is_stage_complete:
            state = await self.stage_director.advance_stage(state)
            yield self._event(
                session_id,
                turn_id,
                LoopEventType.STAGE_TRANSITION,
                {
                    "from_stage": prev_stage_id,
                    "to_stage": state.current_stage_id,
                    "is_curriculum_complete": state.status == LifecycleStatus.DIALOGUE_COMPLETED,
                    "new_status": state.status.value,
                },
            )

        yield self._event(
            session_id,
            turn_id,
            LoopEventType.TURN_COMPLETE,
            {
                "turn_number": turn_number,
                "turn_id": turn_id,
                "stage_id": state.current_stage_id,
                "interviewer_response": outcome["text"],
                "interrupted": interrupted,
                "usage": outcome.get("usage"),
            },
        )

        if self.on_turn_completed is not None:
            try:
                await self.on_turn_completed(session_id, turn_number, state)
            except Exception as e:
                logger.warning("on_turn_completed callback failed: %s", sanitize_secrets(str(e)))
        if state.status == LifecycleStatus.DIALOGUE_COMPLETED and self.on_dialogue_completed is not None:
            try:
                await self.on_dialogue_completed(session_id)
            except Exception as e:
                logger.warning("on_dialogue_completed callback failed: %s", sanitize_secrets(str(e)))
