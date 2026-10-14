"""Cổng import cho các thành phần quản lý phiên: trạng thái, nhật ký sự kiện và chống xử lý lặp.

API/controller và các tác vụ nền dùng những lớp này để đọc trạng thái, cập nhật qua CAS,
lưu sự kiện rồi phát lại khi client kết nối lại. File này chỉ gom API công khai;
không khởi tạo Redis hay tự chạy vòng đời phiên khi import.
"""

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
