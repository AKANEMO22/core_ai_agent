"""Data models and event definitions for agent streaming runtime execution loop."""

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class LoopEventType(str, Enum):
    """Enumeration of event types emitted by the streaming agent runtime loop."""

    PROGRESS = "progress"
    REASONING_SUMMARY = "reasoning_summary"
    RESPONSE_TOKEN = "response_token"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    COMPLETE = "complete"
    ERROR = "error"
    TURN_COMPLETE = "turn_complete"
    STAGE_TRANSITION = "stage_transition"
    ASSESSMENT_UPDATE = "assessment_update"
    ASSESSMENT_COMPLETED = "assessment_completed"
    REPORT_READY = "report_ready"
    SESSION_FAILED = "session_failed"
    SESSION_CANCELLED = "session_cancelled"
    INTERVIEW_STARTED = "interview_started"
    BUSY = "busy"
    RESYNC_REQUIRED = "resync_required"
    GENERATION_CANCELLED = "generation_cancelled"
    DIRECTIVE_READY = "directive_ready"
    DELEGATION_STARTED = "delegation_started"
    DELEGATION_COMPLETED = "delegation_completed"
    COMPACTION = "compaction"
    DECISION_READY = "decision_ready"


class LoopStatus(str, Enum):
    """Execution status states of the agent runtime loop."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IterationChunk(BaseModel):
    """Represents an incremental streaming output chunk emitted by an LLM provider.

    Attributes:
        token: Incremental assistant response text fragment.
        reasoning_summary: Chain-of-thought or reasoning preview string.
        tool_call_chunks: Partial tool call chunks (name, index, partial arguments).
        tool_calls: Assembled complete tool call specifications.
        finish_reason: Generation termination indicator ('stop', 'tool_calls', 'length').
        usage: Token consumption metrics if provided by the model stream.
    """

    token: Optional[str] = None
    reasoning_summary: Optional[str] = None
    tool_call_chunks: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LoopEvent(BaseModel):
    """Standardized event envelope emitted during agent loop streaming execution.

    Adheres strictly to the global system specification requiring:
    schema_version, event_id, session_id, turn_id, sequence, timestamp, type, payload.

    Attributes:
        schema_version: Specification contract version (default '1.0').
        event_id: Unique event identifier prefixed with 'evt_'.
        session_id: Target session identifier.
        turn_id: Current turn identifier within the interview session.
        sequence: Monotonically increasing sequence number within the turn.
        timestamp: Unix epoch timestamp in seconds.
        type: Event classifier from LoopEventType.
        payload: Event-specific structured data dictionary.

    Note:
        Instances may hold an internal `_cached_json` string attribute dynamically
        assigned by SessionEventStore.append_async to avoid redundant Pydantic
        re-serialization during high-throughput WebSocket broadcast.
    """

    schema_version: str = "1.0"
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:16]}")
    session_id: str
    turn_id: str
    sequence: int
    timestamp: Union[float, str] = Field(default_factory=time.time)
    type: LoopEventType
    payload: Dict[str, Any] = Field(default_factory=dict)
