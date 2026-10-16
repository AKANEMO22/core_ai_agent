"""Điều khiển tiến độ stage từ curriculum và snapshot trạng thái phiên.

Runner gọi record_turn sau khi lưu câu trả lời/gửi job chấm; hàm tăng bộ đếm,
xét directive, tool complete_stage và ngân sách lượt. Khi stage hoàn tất, runner
gọi advance_stage để chọn stage kế tiếp hoặc chuyển DIALOGUE_COMPLETED.
Các hàm trả snapshot từ CAS của state store; chúng không sinh câu hỏi bằng LLM
và không tự viết report. max_extra_turns chặn việc gia hạn stage vô hạn.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from core.runtime.agent import sanitize_secrets
from core.session.state import (
    LifecycleStatus,
    SessionState,
    SessionStateStore,
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
        state_mgr: SessionStateStore,
        max_extra_turns: int = 1,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the stage director.

        Args:
            stages: Ordered list of stage configuration dictionaries.
            state_mgr: Session state store controlling FSM and CAS validation.
            max_extra_turns: Turns a stage may run past its budget when the orchestrator extends it
                (a must-have criterion still lacks evidence). A stage's own `max_extra_turns` wins.
        """
        self.stages = stages
        self.state_mgr = state_mgr
        self.max_extra_turns = max(0, int(max_extra_turns))
        self.clock = clock
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

    def elapsed_budget_target(self, state: SessionState) -> Optional[Dict[str, Any]]:
        """Reserve the final Q&A at an answer boundary, never interrupt candidate input.

        A time budget is an upper planning bound. Turn budgets can finish a quick
        interview earlier; there is no minimum wait or attempt to fill wall time.
        """
        deadline = state.metadata.get("interview_deadline_at")
        if not isinstance(deadline, (float, int)):
            return None
        current_index = self.get_stage_index(state)
        for index, stage in enumerate(self.stages):
            if stage.get("stage_type") == "candidate_questions" or stage.get("stage_id") == "candidate_questions":
                if index <= current_index:
                    return None
                reserve = stage.get("planned_minutes", stage.get("duration_minutes", 2))
                try:
                    reserve_seconds = max(30.0, min(600.0, float(reserve) * 60))
                except (TypeError, ValueError):
                    reserve_seconds = 120.0
                if self.clock() >= deadline - reserve_seconds:
                    return stage
        return None

    async def advance_to_budget_target(self, state: SessionState, stage: Dict[str, Any]) -> SessionState:
        """Commit the Q&A invitation after recording/scoring the accepted answer."""
        metadata = dict(state.metadata)
        metadata["elapsed_budget_transition"] = {"from_stage": state.current_stage_id, "to_stage": stage["stage_id"], "at": self.clock()}
        return await self.state_mgr.atransition(
            state, expected_version=state.state_version, current_stage_id=stage["stage_id"],
            is_stage_complete=False, metadata=metadata,
        )

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

    async def record_turn(
        self,
        state: SessionState,
        complete_stage: bool = False,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        extend_stage: bool = False,
    ) -> SessionState:
        """Record an executed dialogue turn, advancing counters and checking budget or triggers.

        Args:
            state: The current SessionState snapshot.
            complete_stage: Explicit boolean flag requesting immediate stage completion.
            tool_calls: Optional list of tool calls emitted during turn for trigger checks.
            extend_stage: The turn's directive keeps the stage open past its budget (bounded by
                max_extra_turns), because the response asked another question of this stage.

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
        # current_turn đếm toàn phiên; stage_turns đếm riêng stage để kiểm tra ngân sách từng đoạn.
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
                name = (
                    tc.get("tool_name")
                    or tc.get("name")
                    or (
                        tc.get("function", {}).get("name")
                        if isinstance(tc.get("function"), dict)
                        else ""
                    )
                )
                if name == "complete_stage":
                    tool_triggered = True
                    break

        extra = stage_cfg.get("max_extra_turns", self.max_extra_turns)
        hard_limit = int(target_turns) + max(0, int(extra))
        # Có thể gia hạn stage khi interviewer hỏi tiếp, nhưng hard_limit vẫn chặn vòng hỏi kéo dài vô hạn.
        budget_reached = current_stage_turns >= hard_limit or (
            current_stage_turns >= target_turns and not extend_stage
        )
        is_complete = bool(complete_stage or tool_triggered or budget_reached)

        # 4. Atomically transition state via the store's CAS
        return await self.state_mgr.atransition(
            state=state,
            expected_version=state.state_version,
            current_stage_id=curr_stage_id,
            is_stage_complete=is_complete,
            current_turn=overall_turn,
            metadata=metadata,
        )

    async def advance_stage(self, state: SessionState) -> SessionState:
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

        curr_idx = self._find_stage_index(state.current_stage_id)
        next_idx = curr_idx + 1

        # Sang stage kế tiếp chỉ đổi stage_id; status vẫn STAGE_ACTIVE ở nhánh này.
        if next_idx < len(self.stages):
            next_stage = self.stages[next_idx]
            next_stage_id = next_stage["stage_id"]
            return await self.state_mgr.atransition(
                state=state,
                expected_version=state.state_version,
                current_stage_id=next_stage_id,
                is_stage_complete=False,
            )
        else:
            # Reached curriculum completion -> transition to DIALOGUE_COMPLETED
            return await self.state_mgr.atransition(
                state=state,
                expected_version=state.state_version,
                new_status=LifecycleStatus.DIALOGUE_COMPLETED,
                is_stage_complete=True,
            )

    async def complete_current_stage(self, state: SessionState) -> SessionState:
        """Mark active stage complete via CAS transition without incrementing turn counters.

        Args:
            state: The current SessionState snapshot.

        Returns:
            SessionState: Updated session state snapshot with is_stage_complete == True.

        Raises:
            StageSessionConflictError: If session is not currently in STAGE_ACTIVE status.
            InvalidStateTransitionError: If CAS expected_version does not match registered state.
        """
        if state.status != LifecycleStatus.STAGE_ACTIVE:
            raise StageSessionConflictError(
                f"Cannot complete stage for session '{state.session_id}': status is {state.status.value}, "
                f"expected {LifecycleStatus.STAGE_ACTIVE.value}"
            )

        return await self.state_mgr.atransition(
            state=state,
            expected_version=state.state_version,
            is_stage_complete=True,
        )
