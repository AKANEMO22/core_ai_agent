"""Hợp đồng job và hàng đợi tách chấm điểm khỏi luồng phát câu trả lời.

Runner tạo AssessmentJob từ câu hỏi trước đó, câu trả lời ứng viên, VAD và rubric,
rồi enqueue; bên đánh giá nền dequeue và chấm theo payload này. job_id nối
session/turn/rubric để bên xử lý có khóa nhận diện ổn định. Bản InMemoryAssessmentQueue
chỉ giữ dữ liệu RAM và lịch sử enqueue; nó không tự deduplicate hay bảo đảm phục hồi
sau restart, vì vậy không nên suy ra độ bền Redis từ hợp đồng này.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union


@dataclass
class AssessmentJob:
    """Standardized payload for an asynchronous turn assessment job.

    Attributes:
        schema_version: Event schema version identifier (default "1.0").
        job_id: Unique idempotency key (format: {session_id}:{turn_id}:{rubric_version}).
        session_id: Session identifier.
        turn_id: Turn identifier (e.g. "turn_001").
        stage_id: Active stage identifier during this turn.
        question: Question asked by interviewer preceding this candidate answer.
        candidate_text: Candidate's verbal or written answer.
        vad_metrics: Audio/VAD metrics (speech_duration_ms, pause_count, etc.).
        rubric_version: Evaluation rubric schema version.
        rubric_target: List of target skills, topics, or competencies evaluated.
        created_at: Unix epoch timestamp when the job was generated.
    """

    schema_version: str = "1.0"
    job_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    stage_id: str = ""
    question: str = ""
    candidate_text: str = ""
    vad_metrics: Dict[str, Any] = field(default_factory=dict)
    rubric_version: str = "rubric_v1"
    rubric_target: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "stage_id": self.stage_id,
            "question": self.question,
            "candidate_text": self.candidate_text,
            "vad_metrics": dict(self.vad_metrics),
            "rubric_version": self.rubric_version,
            "rubric_target": list(self.rubric_target),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessmentJob":
        """Reconstruct an AssessmentJob instance from a dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            job_id=data.get("job_id", ""),
            session_id=data.get("session_id", ""),
            turn_id=data.get("turn_id", ""),
            stage_id=data.get("stage_id", ""),
            question=data.get("question", ""),
            candidate_text=data.get("candidate_text", ""),
            vad_metrics=dict(data.get("vad_metrics", {})),
            rubric_version=data.get("rubric_version", "rubric_v1"),
            rubric_target=list(data.get("rubric_target", [])),
            created_at=data.get("created_at", time.time()),
        )


class AssessmentQueue(Protocol):
    """Protocol defining the interface for assessment job producers and consumers."""

    async def enqueue(self, job: Union[Dict[str, Any], AssessmentJob]) -> str:
        """Enqueue an assessment job and return its assigned job ID."""
        ...

    async def dequeue(self, timeout_seconds: float = 0.0) -> Optional[Dict[str, Any]]:
        """Dequeue the next available job, or return None if queue is empty or timed out."""
        ...

    async def get_queue_size(self) -> int:
        """Return the count of currently pending assessment jobs in the queue."""
        ...


class InMemoryAssessmentQueue:
    """Thread-safe asyncio in-memory queue implementation for development and testing."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._history: List[Dict[str, Any]] = []

    async def enqueue(self, job: Union[Dict[str, Any], AssessmentJob]) -> str:
        """Enqueue a job dictionary or AssessmentJob."""
        job_dict = job.to_dict() if isinstance(job, AssessmentJob) else dict(job)
        if not job_dict.get("job_id"):
            session_id = job_dict.get("session_id", "sess")
            turn_id = job_dict.get("turn_id", "turn")
            rubric_v = job_dict.get("rubric_version", "rubric_v1")
            job_dict["job_id"] = f"{session_id}:{turn_id}:{rubric_v}"

        job_id = job_dict["job_id"]
        # Đây là FIFO RAM: job_id ổn định không đồng nghĩa enqueue tự loại job trùng.
        await self._queue.put(job_dict)
        self._history.append(job_dict)
        return job_id

    async def dequeue(self, timeout_seconds: float = 0.0) -> Optional[Dict[str, Any]]:
        """Dequeue the next job payload or None on timeout."""
        try:
            if timeout_seconds > 0:
                return await asyncio.wait_for(
                    self._queue.get(), timeout=timeout_seconds
                )
            else:
                return self._queue.get_nowait()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    async def get_queue_size(self) -> int:
        """Return number of pending unconsumed items."""
        return self._queue.qsize()

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Return a copy of all jobs enqueued historically."""
        # History chứa cả job đã dequeue; nó không phản ánh số tác vụ đang chờ.
        return list(self._history)
