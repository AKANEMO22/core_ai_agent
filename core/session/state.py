"""Session state management and FSM lifecycle transitions with compare-and-set (CAS) validation."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set

from core.runtime.agent import sanitize_secrets


class LifecycleStatus(str, Enum):
    """Enumeration of valid lifecycle statuses for an interview session FSM."""

    INITIALIZED = "INITIALIZED"
    PROFILING = "PROFILING"
    READY = "READY"
    STAGE_ACTIVE = "STAGE_ACTIVE"
    STAGE_TRANSITIONING = "STAGE_TRANSITIONING"
    DIALOGUE_COMPLETED = "DIALOGUE_COMPLETED"
    ASSESSING = "ASSESSING"
    REPORT_READY = "REPORT_READY"
    ASSESSMENT_FAILED = "ASSESSMENT_FAILED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid lifecycle transition or CAS version mismatch occurs."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


# Terminal states from which no further progression is allowed
TERMINAL_STATUSES: Set[LifecycleStatus] = {
    LifecycleStatus.REPORT_READY,
    LifecycleStatus.FAILED,
    LifecycleStatus.CANCELLED,
    LifecycleStatus.EXPIRED,
}

# Strict Finite State Machine (FSM) allowed transition graph
VALID_TRANSITIONS: Dict[LifecycleStatus, Set[LifecycleStatus]] = {
    LifecycleStatus.INITIALIZED: {
        LifecycleStatus.PROFILING,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
    },
    LifecycleStatus.PROFILING: {
        LifecycleStatus.READY,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
    },
    LifecycleStatus.READY: {
        LifecycleStatus.STAGE_ACTIVE,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.EXPIRED,
    },
    LifecycleStatus.STAGE_ACTIVE: {
        LifecycleStatus.STAGE_TRANSITIONING,
        LifecycleStatus.DIALOGUE_COMPLETED,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.EXPIRED,
    },
    LifecycleStatus.STAGE_TRANSITIONING: {
        LifecycleStatus.STAGE_ACTIVE,
        LifecycleStatus.DIALOGUE_COMPLETED,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.EXPIRED,
    },
    LifecycleStatus.DIALOGUE_COMPLETED: {
        LifecycleStatus.ASSESSING,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
    },
    LifecycleStatus.ASSESSING: {
        LifecycleStatus.REPORT_READY,
        LifecycleStatus.ASSESSMENT_FAILED,
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
    },
    LifecycleStatus.ASSESSMENT_FAILED: {
        LifecycleStatus.ASSESSING,  # Retry capability
        LifecycleStatus.FAILED,
        LifecycleStatus.CANCELLED,
    },
    LifecycleStatus.REPORT_READY: set(),
    LifecycleStatus.FAILED: set(),
    LifecycleStatus.CANCELLED: set(),
    LifecycleStatus.EXPIRED: set(),
}


@dataclass
class SessionState:
    """Immutable data snapshot representing the state of an interview session at a specific version.

    Attributes:
        session_id: Unique session identifier.
        status: Current FSM lifecycle status.
        state_version: Monotonically increasing revision number for compare-and-set (CAS).
        current_stage_id: Identifier of the currently executing interview stage (if active).
        is_stage_complete: Flag indicating whether the active stage has fulfilled its criteria.
        current_turn: 1-indexed turn counter within the session.
        metadata: Arbitrary extensible metadata associated with the session.
        created_at: Unix epoch timestamp when the session was created.
        updated_at: Unix epoch timestamp when this state version was recorded.
    """

    session_id: str
    status: LifecycleStatus = LifecycleStatus.INITIALIZED
    state_version: int = 1
    current_stage_id: Optional[str] = None
    is_stage_complete: bool = False
    current_turn: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a JSON-serializable dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "state_version": self.state_version,
            "current_stage_id": self.current_stage_id,
            "is_stage_complete": self.is_stage_complete,
            "current_turn": self.current_turn,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Reconstruct a SessionState instance from a dictionary."""
        status_raw = data.get("status", LifecycleStatus.INITIALIZED)
        status = (
            LifecycleStatus(status_raw)
            if isinstance(status_raw, str)
            else status_raw
        )
        return cls(
            session_id=data["session_id"],
            status=status,
            state_version=data.get("state_version", 1),
            current_stage_id=data.get("current_stage_id"),
            is_stage_complete=data.get("is_stage_complete", False),
            current_turn=data.get("current_turn", 0),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class SessionStateManager:
    """Manages session lifecycle states with optimistic concurrency via Compare-And-Set (CAS)."""

    def __init__(self) -> None:
        self._states: Dict[str, SessionState] = {}

    def create_session(
        self,
        session_id: str,
        initial_status: LifecycleStatus = LifecycleStatus.INITIALIZED,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SessionState:
        """Create and register a new session state with initial version 1."""
        now = time.time()
        state = SessionState(
            session_id=session_id,
            status=initial_status,
            state_version=1,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
            **kwargs,
        )
        self._states[session_id] = state
        return state

    def get_state(self, session_id: str) -> Optional[SessionState]:
        """Retrieve the latest registered state for a session, or None if unknown."""
        return self._states.get(session_id)

    def transition(
        self,
        state: SessionState,
        expected_version: int,
        new_status: Optional[LifecycleStatus] = None,
        **updates: Any,
    ) -> SessionState:
        """Atomically transition state validating CAS version and FSM status transitions.

        Args:
            state: The current SessionState snapshot being transitioned.
            expected_version: The expected version number (must match state.state_version).
            new_status: Optional target LifecycleStatus. If provided, verified against FSM.
            **updates: Additional field updates (e.g. current_stage_id, is_stage_complete).

        Returns:
            SessionState: Newly constructed immutable state with state_version incremented.

        Raises:
            InvalidStateTransitionError: If version check fails or transition is disallowed.
        """
        # 1. CAS Check: Verify expected version matches current version
        if state.state_version != expected_version:
            raise InvalidStateTransitionError(
                f"CAS mismatch for session '{state.session_id}': expected version "
                f"{expected_version}, but current version is {state.state_version}"
            )

        # 2. FSM Transition Check: Validate lifecycle status change
        target_status = state.status
        if new_status is not None and new_status != state.status:
            # Check terminal state lock
            if state.status in TERMINAL_STATUSES:
                raise InvalidStateTransitionError(
                    f"Cannot transition from terminal state {state.status.value} "
                    f"for session '{state.session_id}'"
                )

            # Check allowed transition paths
            allowed = VALID_TRANSITIONS.get(state.status, set())
            if new_status not in allowed:
                raise InvalidStateTransitionError(
                    f"Invalid transition from {state.status.value} to {new_status.value} "
                    f"for session '{state.session_id}'. Allowed transitions: "
                    f"{[s.value for s in allowed]}"
                )
            target_status = new_status

        # 3. Construct incremented state version
        new_version = state.state_version + 1
        now = time.time()

        # Build new field values
        new_stage_id = updates.get("current_stage_id", state.current_stage_id)
        new_is_complete = updates.get("is_stage_complete", state.is_stage_complete)
        new_turn = updates.get("current_turn", state.current_turn)
        
        merged_metadata = dict(state.metadata)
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            merged_metadata.update(updates["metadata"])

        updated_state = SessionState(
            session_id=state.session_id,
            status=target_status,
            state_version=new_version,
            current_stage_id=new_stage_id,
            is_stage_complete=new_is_complete,
            current_turn=new_turn,
            metadata=merged_metadata,
            created_at=state.created_at,
            updated_at=now,
        )

        # Update in-memory registry if this session is tracked
        if state.session_id in self._states:
            self._states[state.session_id] = updated_state

        return updated_state

    def transition_session(
        self,
        session_id: str,
        expected_version: int,
        new_status: Optional[LifecycleStatus] = None,
        **updates: Any,
    ) -> SessionState:
        """Helper to transition a registered session directly by ID."""
        current = self.get_state(session_id)
        if current is None:
            raise InvalidStateTransitionError(
                f"Session '{session_id}' not found in state manager registry"
            )
        return self.transition(current, expected_version, new_status, **updates)
