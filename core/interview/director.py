"""Stage director managing stage index, turn counters, completion triggers, and CAS progression."""

from typing import Any, Dict, List, Optional

from core.runtime.agent import sanitize_secrets
from core.session.state import (
    InvalidStateTransitionError,
    LifecycleStatus,
    SessionState,
    SessionStateManager,
)


class StageSessionConflictError(ValueError):
    """Raised when an operation is attempted on a session in an incompatible lifecycle status."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


class StageDirector:
    """Orchestrates stage lifecycles, turn limits, and state transitions for interview sessions.

    Attributes:
        stages: Ordered list of stage definitions.
        state_mgr: Reference to the session state manager for CAS mutations.
    """

    def __init__(
        self,
        stages: List[Dict[str, Any]],
        state_mgr: SessionStateManager,
    ) -> None:
        """Initialize the stage director.

        Args:
            stages: Ordered list of stage configuration dictionaries.
            state_mgr: The SessionStateManager controlling FSM and CAS validation.
        """
        self.stages = stages
        self.state_mgr = state_mgr
        self._stages_by_id: Dict[str, Dict[str, Any]] = {
            s["stage_id"]: s for s in stages if "stage_id" in s
        }

    def _find_stage_index(self, stage_id: Optional[str]) -> int:
        """Locate the 0-based index of a stage ID in the stage list."""
        if not stage_id:
            return 0
        for idx, stage in enumerate(self.stages):
            if stage.get("stage_id") == stage_id:
                return idx
        return 0

    def get_current_stage(self, state: SessionState) -> Optional[Dict[str, Any]]:
        """Retrieve the stage configuration dict for the session's active stage."""
        stage_id = state.current_stage_id or (
            self.stages[0]["stage_id"] if self.stages else None
        )
        if not stage_id:
            return None
        return self._stages_by_id.get(stage_id)

    def get_stage_by_id(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """Look up a stage configuration dictionary by its unique stage_id."""
        return self._stages_by_id.get(stage_id)

    def get_stage_index(self, state: SessionState) -> int:
        """Return the 0-based stage index for the session's active stage."""
        return self._find_stage_index(state.current_stage_id)

    def is_last_stage(self, state: SessionState) -> bool:
        """Check if the active stage is the final stage in the interview curriculum."""
        if not self.stages:
            return True
        curr_idx = self._find_stage_index(state.current_stage_id)
        return curr_idx >= len(self.stages) - 1

    def record_turn(
        self,
        state: SessionState,
        complete_stage: bool = False,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> SessionState:
        """Record an executed dialogue turn, advancing counters and checking budget or triggers.

        Args:
            state: The current SessionState snapshot.
            complete_stage: Explicit boolean flag requesting immediate stage completion.
            tool_calls: Optional list of tool calls emitted during turn for trigger checks.

        Returns:
            SessionState: Updated session state snapshot reflecting new turn and completion flag.

        Raises:
            StageSessionConflictError: If session is not currently in STAGE_ACTIVE status.
        """
        if state.status != LifecycleStatus.STAGE_ACTIVE:
            raise StageSessionConflictError(
                f"Cannot record turn for session '{state.session_id}': status is {state.status.value}, "
                f"expected {LifecycleStatus.STAGE_ACTIVE.value}"
            )

        # 1. Determine active stage identifier
        curr_stage_id = state.current_stage_id or (
            self.stages[0]["stage_id"] if self.stages else "stage_1"
        )
        stage_cfg = self._stages_by_id.get(curr_stage_id, {})
        target_turns = stage_cfg.get("target_turns") or stage_cfg.get("turns", 3)

        # 2. Update stage and overall turn counters
        metadata = dict(state.metadata)
        stage_turns_dict = dict(metadata.get("stage_turns", {}))
        current_stage_turns = stage_turns_dict.get(curr_stage_id, 0) + 1
        stage_turns_dict[curr_stage_id] = current_stage_turns
        metadata["stage_turns"] = stage_turns_dict

        overall_turn = state.current_turn + 1

        # 3. Check completion conditions
        tool_triggered = False
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") or (
                    tc.get("function", {}).get("name")
                    if isinstance(tc.get("function"), dict)
                    else ""
                )
                if name == "complete_stage":
                    tool_triggered = True
                    break

        is_complete = bool(
            complete_stage or tool_triggered or (current_stage_turns >= target_turns)
        )

        # 4. Atomically transition state via SessionStateManager CAS
        if state.session_id not in self.state_mgr._states:
            self.state_mgr._states[state.session_id] = state

        registered = self.state_mgr.get_state(state.session_id)
        if registered is not None and state.state_version != registered.state_version:
            raise InvalidStateTransitionError(
                f"CAS mismatch for session '{state.session_id}': expected version "
                f"{registered.state_version}, but provided state has version {state.state_version}"
            )

        return self.state_mgr.transition(
            state=state,
            expected_version=state.state_version,
            current_stage_id=curr_stage_id,
            is_stage_complete=is_complete,
            current_turn=overall_turn,
            metadata=metadata,
        )

    def advance_stage(self, state: SessionState) -> SessionState:
        """Advance session to the subsequent stage or mark dialogue completed if at end.

        Args:
            state: The current SessionState snapshot with is_stage_complete == True.

        Returns:
            SessionState: Updated session state snapshot at the next stage or DIALOGUE_COMPLETED.

        Raises:
            StageSessionConflictError: If session is not currently in STAGE_ACTIVE status.
            InvalidStateTransitionError: If CAS expected_version does not match current state.
        """
        if state.status != LifecycleStatus.STAGE_ACTIVE:
            raise StageSessionConflictError(
                f"Cannot advance stage for session '{state.session_id}': status is {state.status.value}, "
                f"expected {LifecycleStatus.STAGE_ACTIVE.value}"
            )

        registered = self.state_mgr.get_state(state.session_id)
        if registered is not None and state.state_version != registered.state_version:
            raise InvalidStateTransitionError(
                f"CAS mismatch for session '{state.session_id}': expected version "
                f"{registered.state_version}, but provided state has version {state.state_version}"
            )

        curr_idx = self._find_stage_index(state.current_stage_id)
        next_idx = curr_idx + 1

        if next_idx < len(self.stages):
            next_stage = self.stages[next_idx]
            next_stage_id = next_stage["stage_id"]
            return self.state_mgr.transition(
                state=state,
                expected_version=state.state_version,
                current_stage_id=next_stage_id,
                is_stage_complete=False,
            )
        else:
            # Reached curriculum completion -> transition to DIALOGUE_COMPLETED
            return self.state_mgr.transition(
                state=state,
                expected_version=state.state_version,
                new_status=LifecycleStatus.DIALOGUE_COMPLETED,
                is_stage_complete=True,
            )
