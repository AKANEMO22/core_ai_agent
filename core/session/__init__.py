"""Session management package for state FSM lifecycle, CAS transitions, event store, and idempotency."""

from core.session.events import (
    EventNotFoundError,
    EventReplayer,
    SequenceOrderingError,
    SessionEventError,
    SessionEventStore,
)
from core.session.idempotency import (
    CachedResult,
    IdempotencyConflictError,
    IdempotencyError,
    IdempotencyManager,
    TokenBucket,
    compute_payload_hash,
)
from core.session.state import (
    InvalidStateTransitionError,
    LifecycleStatus,
    SessionState,
    SessionStateManager,
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
)

__all__ = [
    # State & FSM
    "LifecycleStatus",
    "SessionState",
    "SessionStateManager",
    "InvalidStateTransitionError",
    "TERMINAL_STATUSES",
    "VALID_TRANSITIONS",
    # Events & Replay
    "SessionEventStore",
    "EventReplayer",
    "SessionEventError",
    "EventNotFoundError",
    "SequenceOrderingError",
    # Idempotency & Rate Limiting
    "IdempotencyManager",
    "IdempotencyError",
    "IdempotencyConflictError",
    "CachedResult",
    "TokenBucket",
    "compute_payload_hash",
]
