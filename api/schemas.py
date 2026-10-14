"""Pydantic REST request/response schemas and WebSocket protocol event envelopes."""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from core.runtime.types import LoopEvent, LoopEventType
from core.session.state import LifecycleStatus


# ============================================================================
# REST Request / Response Schemas
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Payload for initializing a new candidate interview session."""

    model_config = ConfigDict(extra="ignore")

    candidate_name: str = Field(..., description="Full name of the interview candidate")
    experience_years: Optional[float] = Field(default=None, description="Years of professional experience")
    target_company: str = Field(default="Target Company", description="Target employer for interview")
    target_role: str = Field(default="Senior Software Engineer", description="Target job role")
    skills: List[str] = Field(default_factory=list, description="Primary technical and domain skills")
    session_id: Optional[str] = Field(default=None, description="Optional custom session identifier")
    cv_text: Optional[str] = Field(default=None, max_length=180_000, description="Optional resume or CV plain text content")
    job_description: Optional[str] = Field(default=None, max_length=180_000, description="Optional target job description")
    interview_format: Optional[str] = Field(default=None, description="Optional interview format guideline")
    seniority: Optional[Literal["intern", "fresher", "junior", "mid", "senior"]] = Field(
        default=None, description="Optional IT seniority; omitted values are inferred and surfaced with provenance"
    )
    interview_round: Optional[Literal["hr", "technical", "hiring_manager", "combined"]] = Field(
        default=None, description="Optional interview round; omitted values are inferred and surfaced with provenance"
    )
    duration_minutes: Optional[int] = Field(
        default=None, ge=10, le=120,
        description="Optional requested interview duration; omitted values are inferred from matching evidence",
    )
    interview_language: Literal["vi", "en"] = Field(
        default="vi", description="Interview conversation language; independent of the interface locale"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom session metadata")


class PlanPreferenceRequest(BaseModel):
    """Optional edits to an inferred interview brief before the first interview turn."""

    model_config = ConfigDict(extra="forbid")

    seniority: Optional[Literal["intern", "fresher", "junior", "mid", "senior"]] = None
    interview_round: Optional[Literal["hr", "technical", "hiring_manager", "combined"]] = None
    duration_minutes: Optional[int] = Field(default=None, ge=10, le=120)


class ProfileCorrectionRequest(BaseModel):
    """Candidate-reviewed extraction; kept separate from the original CV file/text."""

    model_config = ConfigDict(extra="forbid")
    candidate_profile: Dict[str, Any]


class CreateSessionResponse(BaseModel):
    """Response returned upon successful creation and acceptance of an interview session (HTTP 202)."""

    session_id: str
    status: LifecycleStatus = LifecycleStatus.PROFILING
    state_version: int = 2
    session_token: Optional[str] = None
    websocket_stream_url: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class SessionStatusResponse(BaseModel):
    """Current FSM lifecycle and progression snapshot for an active or completed session."""

    session_id: str
    status: LifecycleStatus
    state_version: int
    current_stage_id: Optional[str] = None
    is_stage_complete: bool = False
    current_turn: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None
    retrieval_status: Optional[str] = None
    brief_status: Optional[str] = None
    coverage: Optional[Dict[str, Any]] = None
    created_at: float
    updated_at: float


class StartInterviewResponse(BaseModel):
    """Result of starting the interview over REST (backend-driven flows)."""

    session_id: str
    status: str
    turn_id: Optional[str] = None
    interviewer_response: Optional[str] = None
    error_code: Optional[str] = None


class RequeueAssessmentsResponse(BaseModel):
    """Result of an operator requeue of dead-lettered assessments."""

    session_id: str
    requeued_jobs: int
    status: LifecycleStatus


class ReviewRequest(BaseModel):
    """Human reviewer decision for a report that requires review."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(..., min_length=1, max_length=128)
    final_verdict: str = Field(..., pattern="^(PASS|FAIL)$")
    notes: Optional[str] = Field(default=None, max_length=4000)


class SessionReportResponse(BaseModel):
    """Final assessment evaluation and structured interview report (HTTP 200) or pending status (HTTP 202)."""

    session_id: str
    status: LifecycleStatus
    report_markdown: Optional[str] = None
    final_assessment: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    message: Optional[str] = None


class CancelSessionResponse(BaseModel):
    """Confirmation payload for session cancellation."""

    session_id: str
    status: LifecycleStatus = LifecycleStatus.CANCELLED
    message: str = "Session cancelled successfully"


class DeleteSessionResponse(BaseModel):
    """Confirmation payload for session workspace retention deletion."""

    session_id: str
    status: str = "DELETED"
    message: str = "Session and associated workspace artifacts deleted successfully"


class HealthResponse(BaseModel):
    """Response model for health probes /health/live and /health/ready."""

    status: str
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# WebSocket Protocol Envelopes & Incoming Messages
# ============================================================================

class ProtocolEnvelope(BaseModel):
    """Standardized WebSocket and streaming protocol envelope.

    All protocol envelopes MUST include:
    schema_version, event_id, session_id, turn_id, sequence, timestamp, type, payload.
    """

    schema_version: str = "1.0"
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:16]}")
    session_id: str
    turn_id: str
    sequence: int
    timestamp: Union[float, str] = Field(default_factory=time.time)
    type: Union[str, LoopEventType]
    payload: Dict[str, Any] = Field(default_factory=dict)

    def to_loop_event(self) -> LoopEvent:
        """Convert to internal LoopEvent."""
        typed_type = (
            self.type
            if isinstance(self.type, LoopEventType)
            else LoopEventType(self.type)
        )
        return LoopEvent(
            schema_version=self.schema_version,
            event_id=self.event_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            sequence=self.sequence,
            timestamp=self.timestamp,
            type=typed_type,
            payload=self.payload,
        )

    @classmethod
    def from_loop_event(cls, event: LoopEvent) -> "ProtocolEnvelope":
        """Instantiate ProtocolEnvelope from LoopEvent."""
        return cls(
            schema_version=event.schema_version,
            event_id=event.event_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            sequence=event.sequence,
            timestamp=event.timestamp,
            type=event.type,
            payload=event.payload,
        )


class CandidateAnswerMessage(BaseModel):
    """Incoming WebSocket message frame submitted by candidate client."""

    model_config = ConfigDict(extra="ignore")

    type: str = "candidate_answer"
    message_id: Optional[str] = None
    candidate_answer: Optional[str] = None
    answer: Optional[str] = None
    vad_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    expected_turn: Optional[int] = None
    last_sequence: Optional[int] = None
    last_event_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def get_answer_text(self) -> str:
        """Extract candidate answer string from direct field or nested payload."""
        if self.candidate_answer is not None:
            return self.candidate_answer
        if self.answer is not None:
            return self.answer
        if self.payload and isinstance(self.payload, dict):
            return str(
                self.payload.get("candidate_answer")
                or self.payload.get("answer")
                or ""
            )
        return ""

    def get_message_id(self) -> Optional[str]:
        """Extract the client message UUID used for exactly-once turn processing."""
        if self.message_id:
            return str(self.message_id)
        if self.payload and isinstance(self.payload, dict) and self.payload.get("message_id"):
            return str(self.payload["message_id"])
        return None

    def get_vad_metrics(self) -> Dict[str, Any]:
        """Extract VAD delivery metrics dictionary."""
        if self.vad_metrics:
            return dict(self.vad_metrics)
        if self.payload and isinstance(self.payload, dict):
            metrics = self.payload.get("vad_metrics")
            if isinstance(metrics, dict):
                return dict(metrics)
        return {}

    def get_expected_turn(self) -> Optional[int]:
        """Extract expected turn number if specified."""
        if self.expected_turn is not None:
            return self.expected_turn
        if self.payload and isinstance(self.payload, dict):
            return self.payload.get("expected_turn")
        return None
