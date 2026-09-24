"""Interactive subagent runner coordinating turns, streaming generation, persistence, and assessment."""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from core.interview.context import ContextAssembler
from core.interview.director import StageDirector, StageSessionConflictError
from core.interview.queue import AssessmentJob, AssessmentQueue
from core.runtime.agent import RuntimeAgent, sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType
from core.session.state import LifecycleStatus, SessionState, SessionStateManager
from core.workspace.session import SessionWorkspace

logger = logging.getLogger(__name__)


class InteractiveSubagentRunner:
    """Coordinates turn-by-turn conversational flow between candidate and interviewer subagent.

    Attributes:
        workspace: Isolated session workspace for artifact storage.
        runtime_agent: Streaming agent executing LLM interactions and tool calls.
        context_assembler: Context assembler ensuring prompt <= 1500 tokens.
        stage_director: Director tracking stages, turn counters, and CAS transitions.
        assessment_queue: Optional async queue for turn evaluation jobs.
        state_mgr: Reference to the session state manager.
    """

    def __init__(
        self,
        workspace: SessionWorkspace,
        runtime_agent: RuntimeAgent,
        context_assembler: ContextAssembler,
        stage_director: StageDirector,
        assessment_queue: Optional[AssessmentQueue] = None,
        state_mgr: Optional[SessionStateManager] = None,
    ) -> None:
        self.workspace = workspace
        self.runtime_agent = runtime_agent
        self.context_assembler = context_assembler
        self.stage_director = stage_director
        self.assessment_queue = assessment_queue
        self.state_mgr = state_mgr or stage_director.state_mgr
        self._cached_spec: Optional[Dict[str, Any]] = None
        self._cached_profile: Optional[Dict[str, Any]] = None
        self._cached_intel: Optional[Dict[str, Any]] = None

    def _read_json_safe(self, rel_path: str, default: Any = None) -> Any:
        """Safely read a JSON artifact from workspace if it exists."""
        try:
            if self.workspace.exists(rel_path):
                return self.workspace.read_json(rel_path)
        except Exception as e:
            logger.warning(
                f"Error reading artifact '{rel_path}': {sanitize_secrets(str(e))}"
            )
        return default if default is not None else {}

    def _get_previous_question(self, turn_number: int) -> str:
        """Retrieve question asked by interviewer in the previous turn."""
        if turn_number > 1:
            prev_file = f"turns/turn_{turn_number - 1}_interviewer.json"
            if self.workspace.exists(prev_file):
                data = self._read_json_safe(prev_file, {})
                content = data.get("content")
                if content:
                    return str(content)

        return "Initial stage introduction"

    def _load_recent_turns(self, up_to_turn: int) -> List[Dict[str, Any]]:
        """Load conversation turns up to the current turn number in chronological sequence."""
        turns: List[Dict[str, Any]] = []
        for i in range(1, up_to_turn + 1):
            cand_path = f"turns/turn_{i}_candidate.json"
            if self.workspace.exists(cand_path):
                c_data = self._read_json_safe(cand_path, {})
                content = c_data.get("content", "")
                if content:
                    turns.append({"role": "candidate", "content": content})

            int_path = f"turns/turn_{i}_interviewer.json"
            if self.workspace.exists(int_path):
                i_data = self._read_json_safe(int_path, {})
                content = i_data.get("content", "")
                if content:
                    turns.append({"role": "interviewer", "content": content})

        return turns

    async def run_turn(
        self,
        session_id: str,
        candidate_answer: str,
        vad_metrics: Optional[Dict[str, Any]] = None,
        expected_turn: Optional[int] = None,
        cancellation_token: Optional[asyncio.Event] = None,
        rubric_version: str = "rubric_v1",
        rubric_target: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """Execute a candidate turn and stream the resulting interviewer subagent dialogue.

        Args:
            session_id: Session identifier.
            candidate_answer: Candidate text answer transcript.
            vad_metrics: Optional voice activity detection and speech delivery metrics.
            expected_turn: Optional expected 1-indexed turn number for stale/order validation.
            cancellation_token: Optional asyncio event signaling external cancellation.
            rubric_version: Rubric version for assessment job evaluation.
            rubric_target: Specific competencies/topics to score.
            **kwargs: Extra parameters passed to the runtime agent LLM provider.

        Yields:
            LoopEvent: Streaming lifecycle events for the turn.

        Raises:
            StageSessionConflictError: If session is not in STAGE_ACTIVE or turn is stale.
        """
        # 1. State Validation
        state = self.state_mgr.get_state(session_id)
        if state is None:
            raise StageSessionConflictError(
                f"Session '{session_id}' not found in state manager registry"
            )

        if state.status != LifecycleStatus.STAGE_ACTIVE:
            raise StageSessionConflictError(
                f"Session '{session_id}' is not in STAGE_ACTIVE status "
                f"(current status: {state.status.value})"
            )

        expected_next = state.current_turn + 1
        if expected_turn is not None and expected_turn != expected_next:
            raise StageSessionConflictError(
                f"Out-of-order turn submission for session '{session_id}': "
                f"expected turn {expected_next}, but received {expected_turn}"
            )

        turn_number = expected_next
        turn_id = f"turn_{turn_number:03d}"

        # 2. Atomically persist candidate turn artifact
        candidate_payload = {
            "turn_id": turn_id,
            "turn_number": turn_number,
            "role": "candidate",
            "session_id": session_id,
            "stage_id": state.current_stage_id,
            "content": candidate_answer,
            "vad_metrics": dict(vad_metrics or {}),
            "timestamp": time.time(),
        }
        # 2. Asynchronously persist candidate turn artifact and enqueue background evaluation job
        enqueue_job = None
        if self.assessment_queue is not None:
            prev_question = self._get_previous_question(turn_number)
            enqueue_job = AssessmentJob(
                schema_version="1.0",
                job_id=f"{session_id}:{turn_id}:{rubric_version}",
                session_id=session_id,
                turn_id=turn_id,
                stage_id=state.current_stage_id or "",
                question=prev_question,
                candidate_text=candidate_answer,
                vad_metrics=dict(vad_metrics or {}),
                rubric_version=rubric_version,
                rubric_target=list(rubric_target or []),
                created_at=time.time(),
            )

        async def _persist_candidate_and_enqueue() -> None:
            await asyncio.to_thread(
                self.workspace.write_artifact,
                f"turns/turn_{turn_number}_candidate.json",
                artifact_type="turn_candidate",
                payload=candidate_payload,
            )
            if enqueue_job is not None and self.assessment_queue is not None:
                await self.assessment_queue.enqueue(enqueue_job)

        bg_task = asyncio.create_task(_persist_candidate_and_enqueue())

        # 4. Assemble context prompt within token ceiling (using cached blueprints)
        if self._cached_spec is None:
            self._cached_spec = self._read_json_safe(
                "context/interview_spec.json",
                default={"interviewer_persona": "Interviewer", "tone": "professional"},
            )
        if self._cached_profile is None:
            self._cached_profile = self._read_json_safe("context/candidate_profile.json", default={})
        if self._cached_intel is None:
            self._cached_intel = self._read_json_safe("context/company_intel.json", default={})

        spec = self._cached_spec
        profile = self._cached_profile
        company_intel = self._cached_intel
        curr_stage = self.stage_director.get_current_stage(state) or {}
        recent_turns = self._load_recent_turns(up_to_turn=turn_number - 1)

        system_prompt = self.context_assembler.assemble_prompt(
            spec=spec,
            stage=curr_stage,
            profile=profile,
            company_intel=company_intel,
            recent_turns=recent_turns,
        )

        # 5. Stream through RuntimeAgent
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidate_answer},
        ]

        interviewer_tokens: List[str] = []
        tool_calls_emitted: List[Dict[str, Any]] = []
        complete_payload: Dict[str, Any] = {}

        try:
            async for event in self.runtime_agent.run_loop_streaming(
                session_id=session_id,
                turn_id=turn_id,
                messages=messages,
                cancellation_token=cancellation_token,
                **kwargs,
            ):
                if event.type == LoopEventType.RESPONSE_TOKEN:
                    interviewer_tokens.append(event.payload.get("token", ""))
                elif event.type == LoopEventType.TOOL_CALL_START:
                    tool_calls_emitted.append(event.payload)
                elif event.type == LoopEventType.COMPLETE:
                    complete_payload = event.payload
                    continue

                yield event
        finally:
            # Guarantee candidate persistence and assessment enqueue task resolution
            if not bg_task.done():
                await bg_task
            elif bg_task.exception() is not None:
                # Re-raise or retrieve background exception if task failed
                _ = bg_task.exception()

        # 6. Atomically persist interviewer turn artifact
        interviewer_text = "".join(interviewer_tokens)
        interviewer_payload = {
            "turn_id": turn_id,
            "turn_number": turn_number,
            "role": "interviewer",
            "session_id": session_id,
            "stage_id": state.current_stage_id,
            "content": interviewer_text,
            "timestamp": time.time(),
        }
        await asyncio.to_thread(
            self.workspace.write_artifact,
            f"turns/turn_{turn_number}_interviewer.json",
            artifact_type="turn_interviewer",
            payload=interviewer_payload,
        )

        # 7. Record turn progression once with tool_calls and advance stage if completed
        prev_stage_id = state.current_stage_id
        state = self.stage_director.record_turn(state, tool_calls=tool_calls_emitted)
        if state.is_stage_complete:
            state = self.stage_director.advance_stage(state)
            yield LoopEvent(
                schema_version="1.0",
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                turn_id=turn_id,
                sequence=0,
                timestamp=time.time(),
                type=LoopEventType.STAGE_TRANSITION,
                payload={
                    "from_stage": prev_stage_id,
                    "to_stage": state.current_stage_id,
                    "is_curriculum_complete": state.status == LifecycleStatus.DIALOGUE_COMPLETED,
                    "new_status": state.status.value,
                },
            )

        yield LoopEvent(
            schema_version="1.0",
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=turn_id,
            sequence=0,
            timestamp=time.time(),
            type=LoopEventType.TURN_COMPLETE,
            payload={
                "turn_number": turn_number,
                "turn_id": turn_id,
                "stage_id": state.current_stage_id,
                "interviewer_response": interviewer_text,
                "usage": complete_payload.get("usage"),
            },
        )
