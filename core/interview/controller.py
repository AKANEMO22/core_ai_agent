"""Điểm điều khiển phiên dùng chung cho adapter REST và WebSocket.

Adapter gọi start_interview hoặc submit_answer với session_id/nội dung client.
Controller kiểm tra hủy, tốc độ và ledger, lấy lease phiên cùng slot generation,
rồi gọi runner và publish các event. Kết quả TurnRequestResult cho adapter biết
lượt hoàn tất, bị ngắt, phát lại, bị từ chối hay cần thử lại. Sau lượt thành công,
hook chấm/tóm tắt được lên lịch nền; xử lý request trùng phải giữ message → turn
và payload hash để tránh gọi LLM lại cho một câu trả lời đã hoàn tất.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.interview.director import StageSessionConflictError
from core.interview.runner import InteractiveSubagentRunner, turn_id_for
from core.runtime.agent import sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType
from core.session.cancellation import CancellationRegistry
from core.session.idempotency import (
    IdempotencyConflictError,
    LedgerEntry,
    MessageLedger,
    RedisLease,
    TokenBucket,
    compute_payload_hash,
)
from core.session.state import LifecycleStatus, SessionState, SessionStateStore

logger = logging.getLogger(__name__)

RunnerFactory = Callable[[str], Awaitable[Optional[InteractiveSubagentRunner]]]
AfterTurnHook = Callable[[str, int], Awaitable[Any]]
SessionHook = Callable[[str], Awaitable[Any]]

# Throttle for the per-token durable stop-fence check in _consume.
FENCE_CHECK_INTERVAL_SECONDS = 0.1


@dataclass
class TurnRequestResult:
    """Outcome of a start or answer request.

    status values:
        completed  - turn finished normally
        interrupted- turn finished with a cancelled/deadline-cut interviewer response
        failed     - generation failed; the same message_id may be retried
        replayed   - duplicate request; `replay_events` holds the recorded events
        rejected   - request refused (see error_code)
        busy       - capacity or rate limit; retry after `retry_after_ms`
    """

    status: str
    turn_id: Optional[str] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    retry_after_ms: Optional[int] = None
    replay_events: List[LoopEvent] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True for statuses that represent a delivered turn."""
        return self.status in ("completed", "interrupted", "replayed")

    def to_event(self, session_id: str) -> Optional[LoopEvent]:
        """Ephemeral (unrecorded) event telling the requester why the request was not executed."""
        if self.status not in ("rejected", "busy", "failed"):
            return None
        event_type = LoopEventType.BUSY if self.status == "busy" else LoopEventType.ERROR
        payload: Dict[str, Any] = {"code": self.error_code, "message": self.message}
        if self.retry_after_ms is not None:
            payload["retry_after_ms"] = self.retry_after_ms
        if self.status == "failed":
            payload["retryable"] = True
        return LoopEvent(
            schema_version="1.0",
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=self.turn_id or "control",
            sequence=0,
            timestamp=time.time(),
            type=event_type,
            payload=payload,
        )


class RunnerRegistry:
    """Bounded LRU cache of per-session runners built by a factory."""

    def __init__(self, factory: RunnerFactory, max_size: int = 500) -> None:
        self.factory = factory
        self.max_size = max_size
        self._runners: "OrderedDict[str, InteractiveSubagentRunner]" = OrderedDict()

    async def get(self, session_id: str) -> Optional[InteractiveSubagentRunner]:
        runner = self._runners.get(session_id)
        if runner is not None:
            self._runners.move_to_end(session_id)
            return runner
        runner = await self.factory(session_id)
        if runner is None:
            return None
        self._runners[session_id] = runner
        while len(self._runners) > self.max_size:
            self._runners.popitem(last=False)
        return runner

    def evict(self, session_id: str) -> None:
        self._runners.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._runners


class InterviewSessionController:
    """Coordinates opening and candidate turns for all sessions of this process."""

    def __init__(
        self,
        state_store: SessionStateStore,
        publisher: Any,
        runners: RunnerRegistry,
        ledger: MessageLedger,
        lease: RedisLease,
        cancellations: CancellationRegistry,
        generation_slots: asyncio.Semaphore,
        answer_limiter: Optional[TokenBucket] = None,
        queue_timeout_ms: int = 1500,
        busy_retry_after_ms: int = 1000,
        turn_lock_ttl_ms: int = 60000,
        metrics: Optional[Any] = None,
        hot_path_gate: Optional[Any] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.state_store = state_store
        self.publisher = publisher
        self.runners = runners
        self.ledger = ledger
        self.lease = lease
        self.cancellations = cancellations
        self.generation_slots = generation_slots
        self.answer_limiter = answer_limiter
        self.queue_timeout_ms = queue_timeout_ms
        self.busy_retry_after_ms = busy_retry_after_ms
        self.turn_lock_ttl_ms = turn_lock_ttl_ms
        self.metrics = metrics
        self.hot_path_gate = hot_path_gate
        self.clock = clock
        self.after_turn_hooks: List[AfterTurnHook] = []
        self.dialogue_completed_hooks: List[SessionHook] = []
        self._background: set = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lock_key(session_id: str) -> str:
        return f"session:{session_id}:turn_lock"

    def _spawn(self, session_id: str, coro: Awaitable[Any], name: str) -> asyncio.Task:
        task = asyncio.create_task(self.cancellations.run(session_id, coro, self.lease.redis), name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        self.cancellations.register_task(session_id, task)

        def _log_failure(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception() is not None:
                logger.error("Background task %s failed: %s", name, sanitize_secrets(str(t.exception())))

        task.add_done_callback(_log_failure)
        return task

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait for scheduled background work (used by tests and graceful shutdown)."""
        if self._background:
            await asyncio.wait(list(self._background), timeout=timeout)

    async def _acquire_slot(self) -> bool:
        try:
            await asyncio.wait_for(self.generation_slots.acquire(), timeout=self.queue_timeout_ms / 1000)
            return True
        except asyncio.TimeoutError:
            if self.metrics is not None:
                self.metrics.inc("busy_total")
            return False

    def _busy(self, code: str, turn_id: Optional[str] = None) -> TurnRequestResult:
        return TurnRequestResult(
            status="busy",
            turn_id=turn_id,
            error_code=code,
            message="Interviewer is at capacity; retry shortly.",
            retry_after_ms=self.busy_retry_after_ms,
        )

    async def _stream(
        self,
        session_id: str,
        events: Any,
        accepted_at: float,
    ) -> Dict[str, Any]:
        """Publish a runner event stream and summarize it."""
        summary: Dict[str, Any] = {"status": "failed", "dialogue_completed": False, "turn_number": None}
        first_token_seen = False
        gate = self.hot_path_gate.hot() if self.hot_path_gate is not None else contextlib.nullcontext()
        async with gate:
            try:
                await self.cancellations.run(session_id, self._consume(session_id, events, accepted_at, summary), self.lease.redis)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                summary["status"] = "interrupted"
        if self.metrics is not None:
            self.metrics.observe("turn_duration_seconds", time.perf_counter() - accepted_at)
        return summary

    async def _consume(self, session_id: str, events: Any, accepted_at: float, summary: Dict[str, Any]) -> None:
        first_token_seen = False
        last_fence_check = float("-inf")
        async for event in events:
            # Tokens check the durable fence at most every FENCE_CHECK_INTERVAL; lifecycle events
            # always check it. CancellationRegistry.run polls the same fence concurrently.
            now = time.monotonic()
            if event.type != LoopEventType.RESPONSE_TOKEN or now - last_fence_check >= FENCE_CHECK_INTERVAL_SECONDS:
                last_fence_check = now
                if await self.lease.redis.exists(f"session:{session_id}:work_stopped"):
                    summary["status"] = "ended"
                    return
            await self.publisher.publish(session_id, event)
            if event.type == LoopEventType.RESPONSE_TOKEN and not first_token_seen:
                first_token_seen = True
                if self.metrics is not None:
                    self.metrics.observe("ttft_seconds", time.perf_counter() - accepted_at)
            elif event.type == LoopEventType.STAGE_TRANSITION and event.payload.get("is_curriculum_complete"):
                summary["dialogue_completed"] = True
            elif event.type == LoopEventType.TURN_COMPLETE:
                summary["status"] = "interrupted" if event.payload.get("interrupted") else "completed"
                summary["turn_number"] = event.payload.get("turn_number")

    @staticmethod
    def _ended(turn_id: Optional[str]) -> TurnRequestResult:
        return TurnRequestResult(
            status="rejected", turn_id=turn_id, error_code="SESSION_ENDED",
            message="The session was ended; the final report is being prepared.",
        )

    async def _replay(self, session_id: str, turn_id: str) -> TurnRequestResult:
        events = await self.publisher.event_store.get_turn_events(session_id, turn_id)
        return TurnRequestResult(status="replayed", turn_id=turn_id, replay_events=events)

    async def _recover_answer(
        self, session_id: str, entry: LedgerEntry, state: SessionState, answer_text: str
    ) -> Optional[TurnRequestResult]:
        """Reconcile an abandoned message while holding the session execution lease."""
        replay = await self._replay(session_id, entry.turn_id)
        if any(event.type == LoopEventType.TURN_COMPLETE for event in replay.replay_events):
            # Turn đã hoàn tất; sửa ledger bị bỏ dở rồi phát lại, không gọi LLM lần nữa.
            await self.ledger.complete(session_id, entry.message_id)
            return replay

        recovery_required = TurnRequestResult(
            status="rejected", turn_id=entry.turn_id, error_code="TURN_RECOVERY_REQUIRED",
            message="The abandoned turn needs reconciliation before it can be retried.",
        )
        try:
            turn_number = int(entry.turn_id.removeprefix("turn_"))
        except ValueError:
            return recovery_required
        if turn_number < 1 or entry.turn_id != turn_id_for(turn_number):
            return recovery_required
        # Chưa chuyển stage sau commit thì chưa có kết quả cuối đủ chắc chắn để replay.
        if (
            state.current_turn == turn_number
            and state.is_stage_complete
            and state.status == LifecycleStatus.STAGE_ACTIVE
        ):
            return recovery_required
        runner = await self.runners.get(session_id)
        if runner is None:
            return TurnRequestResult(
                status="rejected", turn_id=entry.turn_id, error_code="RUNNER_UNAVAILABLE"
            )
        interviewer_path = f"turns/turn_{turn_number}_interviewer.json"
        if state.current_turn < turn_number:
            # Chỉ chạy lại turn chưa commit và chưa lưu câu trả lời; giữ nguyên message → turn.
            if state.current_turn + 1 == turn_number and not runner.workspace.exists(interviewer_path):
                return None
            return recovery_required
        try:
            candidate = await asyncio.to_thread(
                runner.workspace.read_json, f"turns/turn_{turn_number}_candidate.json"
            )
            interviewer = await asyncio.to_thread(runner.workspace.read_json, interviewer_path)
        except (OSError, ValueError):
            return recovery_required
        if not isinstance(candidate, dict) or not isinstance(interviewer, dict):
            return recovery_required
        if (
            candidate.get("message_id") != entry.message_id
            or candidate.get("content") != answer_text
            or candidate.get("turn_id") != entry.turn_id
            or interviewer.get("turn_id") != entry.turn_id
            or candidate.get("session_id") != session_id
            or interviewer.get("session_id") != session_id
            or not isinstance(interviewer.get("content"), str)
            or not interviewer["content"].strip()
        ):
            return recovery_required
        # State đã commit nhưng thiếu event cuối: dựng lại từ artifact đúng message và turn.
        await self.publisher.publish(
            session_id,
            LoopEvent(
                session_id=session_id, turn_id=entry.turn_id, sequence=0,
                type=LoopEventType.TURN_COMPLETE,
                payload={
                    "turn_number": turn_number, "turn_id": entry.turn_id,
                    "stage_id": interviewer.get("stage_id"),
                    "interviewer_response": interviewer["content"],
                    "interrupted": bool(interviewer.get("interrupted")),
                },
            ),
        )
        await self.ledger.complete(session_id, entry.message_id)
        return await self._replay(session_id, entry.turn_id)

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    async def start_interview(self, session_id: str, *, confirm_limitations: bool = False) -> TurnRequestResult:
        """Start the interview: READY → STAGE_ACTIVE and stream the opening question (turn 0).

        Idempotent: once the opening exists, repeated calls replay its events.
        """
        turn_id = turn_id_for(0)
        if self.cancellations.is_session_cancelled(session_id):
            return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="SESSION_CANCELLED")

        # Lease theo phiên tuần tự hóa lượt giữa các process; slot generation giới hạn toàn bộ tải.
        lock_token = await self.lease.acquire(self._lock_key(session_id), self.turn_lock_ttl_ms)
        if lock_token is None:
            return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="TURN_IN_PROGRESS")
        try:
            state = await self.state_store.aget(session_id)
            if state is None:
                return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="SESSION_NOT_FOUND")

            runner = await self.runners.get(session_id)
            if runner is None:
                return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="RUNNER_UNAVAILABLE")

            # The controller owns this guard: REST, WebSocket and internal callers share it.
            brief = {}
            if runner.workspace.exists("context/interview_brief.json"):
                brief = await asyncio.to_thread(runner.workspace.read_json, "context/interview_brief.json")
                brief = brief.get("payload", brief)
            digest = hashlib.sha256(json.dumps(brief, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            accepted_confirmation = state.metadata.get("plan_confirmation") or {}
            if brief.get("status") == "needs_confirmation" and (
                accepted_confirmation.get("brief_sha256") != digest and confirm_limitations is not True
            ):
                return TurnRequestResult(
                    status="rejected", turn_id=turn_id, error_code="PLAN_CONFIRMATION_REQUIRED",
                    message="Review the current plan's evidence limitations and confirm before starting",
                )

            if state.status == LifecycleStatus.STAGE_ACTIVE and (
                state.current_turn > 0 or runner.workspace.exists("turns/turn_0_interviewer.json")
            ):
                return await self._replay(session_id, turn_id)
            if state.status == LifecycleStatus.READY:
                stages = runner.stage_director.stages
                first_stage = stages[0]["stage_id"] if stages else "warmup"
                metadata = dict(state.metadata)
                started_at = self.clock()
                metadata["interview_started_at"] = started_at
                minutes = (brief.get("duration") or {}).get("planned_minutes")
                if isinstance(minutes, (int, float)) and not isinstance(minutes, bool) and 0 < minutes <= 120:
                    metadata["interview_deadline_at"] = started_at + minutes * 60
                if confirm_limitations is True:
                    metadata["plan_confirmation"] = {"brief_sha256": digest, "accepted_at": started_at}
                state = await self.state_store.atransition(
                    state,
                    expected_version=state.state_version,
                    new_status=LifecycleStatus.STAGE_ACTIVE,
                    current_stage_id=first_stage,
                    metadata=metadata,
                )
            elif state.status != LifecycleStatus.STAGE_ACTIVE:
                code = "PROFILING_IN_PROGRESS" if state.status in (
                    LifecycleStatus.INITIALIZED,
                    LifecycleStatus.PROFILING,
                ) else "INVALID_STATE"
                return TurnRequestResult(
                    status="rejected",
                    turn_id=turn_id,
                    error_code=code,
                    message=f"Session status is {state.status.value}",
                )

            accepted_at = time.perf_counter()  # TTFT includes queue wait
            if not await self._acquire_slot():
                return self._busy("GENERATION_BUSY", turn_id)
            token = self.cancellations.token_for(session_id)
            try:
                summary = await self._stream(
                    session_id, runner.run_opening(session_id, cancellation_token=token), accepted_at
                )
            finally:
                self.generation_slots.release()
                self.cancellations.finish_generation(session_id, token)
            if summary["status"] == "ended":
                return self._ended(turn_id)
            if summary["status"] == "failed":
                return TurnRequestResult(status="failed", turn_id=turn_id, error_code="TURN_FAILED")
            return TurnRequestResult(status=summary["status"], turn_id=turn_id)
        except StageSessionConflictError as e:
            return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="INVALID_STATE", message=str(e))
        finally:
            await self.lease.release(self._lock_key(session_id), lock_token)

    # ------------------------------------------------------------------
    # Candidate answers
    # ------------------------------------------------------------------

    async def submit_answer(
        self,
        session_id: str,
        message_id: Optional[str],
        answer_text: str,
        vad_metrics: Optional[Dict[str, Any]] = None,
        expected_turn: Optional[int] = None,
    ) -> TurnRequestResult:
        """Execute or recover a candidate turn identified by an idempotent `message_id`."""
        if not message_id:
            return TurnRequestResult(
                status="rejected", error_code="MISSING_MESSAGE_ID", message="message_id is required"
            )
        if not answer_text or not answer_text.strip():
            return TurnRequestResult(status="rejected", error_code="EMPTY_ANSWER")
        if self.cancellations.is_session_cancelled(session_id):
            return TurnRequestResult(status="rejected", error_code="SESSION_CANCELLED")

        payload_hash = compute_payload_hash({"answer": answer_text})
        try:
            existing = await self.ledger.get(session_id, message_id)
            if existing is not None:
                if existing.payload_hash and existing.payload_hash != payload_hash:
                    return TurnRequestResult(status="rejected", error_code="MESSAGE_ID_CONFLICT")
                if existing.status == "completed":
                    return await self._replay(session_id, existing.turn_id)
        except IdempotencyConflictError:
            return TurnRequestResult(status="rejected", error_code="MESSAGE_ID_CONFLICT")

        if self.answer_limiter is not None and not self.answer_limiter.acquire(session_id):
            return self._busy("RATE_LIMITED")

        lock_token = await self.lease.acquire(self._lock_key(session_id), self.turn_lock_ttl_ms)
        if lock_token is None:
            return self._busy("TURN_IN_PROGRESS")

        registered = False
        turn_id: Optional[str] = None
        try:
            # Lease mới cho phép đối soát ledger cũ; lease còn hiệu lực vẫn chặn retry.
            existing = await self.ledger.get(session_id, message_id)
            if existing is not None:
                if existing.payload_hash and existing.payload_hash != payload_hash:
                    return TurnRequestResult(status="rejected", error_code="MESSAGE_ID_CONFLICT")
                if existing.status == "completed":
                    return await self._replay(session_id, existing.turn_id)
            state = await self.state_store.aget(session_id)
            if state is None:
                return TurnRequestResult(status="rejected", error_code="SESSION_NOT_FOUND")
            if existing is not None:
                turn_id = existing.turn_id
                recovered = await self._recover_answer(session_id, existing, state, answer_text)
                if recovered is not None:
                    return recovered
            if state.status == LifecycleStatus.READY:
                return TurnRequestResult(
                    status="rejected",
                    error_code="INTERVIEW_NOT_STARTED",
                    message="Send start_interview before answering",
                )
            if state.status != LifecycleStatus.STAGE_ACTIVE:
                return TurnRequestResult(
                    status="rejected",
                    error_code="INVALID_STATE",
                    message=f"Session status is {state.status.value}",
                )

            turn_number = state.current_turn + 1
            turn_id = turn_id_for(turn_number)
            if expected_turn is not None and expected_turn != turn_number:
                return TurnRequestResult(
                    status="rejected",
                    turn_id=turn_id,
                    error_code="OUT_OF_ORDER_TURN",
                    message=f"Expected turn {turn_number}, received {expected_turn}",
                )

            try:
                prior = await self.ledger.begin(session_id, message_id, turn_id, payload_hash)
            except IdempotencyConflictError:
                return TurnRequestResult(status="rejected", error_code="MESSAGE_ID_CONFLICT")
            if prior is not None:
                if prior.status == "completed":
                    return await self._replay(session_id, prior.turn_id)
                if prior.turn_id != turn_id:
                    return TurnRequestResult(
                        status="rejected", turn_id=prior.turn_id, error_code="TURN_RECOVERY_REQUIRED"
                    )
            registered = True

            runner = await self.runners.get(session_id)
            if runner is None:
                return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="RUNNER_UNAVAILABLE")

            accepted_at = time.perf_counter()  # TTFT includes queue wait
            if not await self._acquire_slot():
                return self._busy("GENERATION_BUSY", turn_id)

            token = self.cancellations.token_for(session_id)
            try:
                summary = await self._stream(
                    session_id,
                    runner.run_turn(
                        session_id=session_id,
                        candidate_answer=answer_text,
                        vad_metrics=vad_metrics,
                        expected_turn=turn_number,
                        cancellation_token=token,
                        message_id=message_id,
                    ),
                    accepted_at,
                )
            finally:
                self.generation_slots.release()
                self.cancellations.finish_generation(session_id, token)

            if summary["status"] == "ended":
                return self._ended(turn_id)
            if summary["status"] == "failed":
                return TurnRequestResult(status="failed", turn_id=turn_id, error_code="TURN_FAILED")

            # Chỉ đánh dấu hoàn tất sau khi stream đã publish kết quả; retry sau đó đi qua replay.
            await self.ledger.complete(session_id, message_id)
            registered = False
            # Hook chạy nền và được registry theo dõi để hủy phiên có thể dừng công việc liên quan.
            for hook in self.after_turn_hooks:
                self._spawn(session_id, hook(session_id, turn_number), f"after_turn:{turn_id}")
            if summary["dialogue_completed"]:
                self.runners.evict(session_id)
                for session_hook in self.dialogue_completed_hooks:
                    self._spawn(session_id, session_hook(session_id), "dialogue_completed")
            return TurnRequestResult(status=summary["status"], turn_id=turn_id)
        except StageSessionConflictError as e:
            return TurnRequestResult(status="rejected", turn_id=turn_id, error_code="INVALID_STATE", message=str(e))
        finally:
            if registered:
                await self.ledger.discard(session_id, message_id)
            await self.lease.release(self._lock_key(session_id), lock_token)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_generation(self, session_id: str) -> bool:
        """Barge-in: stop the in-flight interviewer generation. Returns True if one was running."""
        return self.cancellations.cancel_generation(session_id, reason="barge_in")

    def cancel_session(self, session_id: str) -> int:
        """Stop the generation and all background work of a session."""
        self.runners.evict(session_id)
        return self.cancellations.cancel_session(session_id)
