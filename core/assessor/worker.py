"""Background assessor worker consuming Redis Streams evaluation tasks.

Xử lý một job chấm điểm từ queue với cơ chế giao ít nhất một lần.
AssessorWorker kiểm tra phiên bị xóa, câu trả lời bị thay thế và artifact đã có,
gọi evaluator rồi ghi artifact trước khi ghi completed/ACK và báo pipeline.
Lỗi được đưa sang lịch retry hoặc dead-letter; pool chịu trách nhiệm đọc job liên tục.
"""

import asyncio
import hashlib
import logging
import random
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from core.assessor.queue import (
    RedisAssessmentQueue,
    completed_assessments_key,
    failed_assessments_key,
)
from core.assessor.subagent import TurnEvaluator
from core.observability import session_ref
from core.runtime.agent import sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.runtime.types import LoopEvent, LoopEventType
from core.session.events import SessionEventStore
from core.session.cancellation import CancellationRegistry
from core.session.state import LifecycleStatus, SessionStateStore
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager

logger = logging.getLogger(__name__)

AssessmentCompletedCallback = Callable[[str], Awaitable[Any]]


class AssessorWorker:
    """Consumes assessment jobs, runs the evaluator, and persists evaluation artifacts.

    Guarantees:
        - At-least-once delivery with idempotency on (job_id, input_sha256, rubric_version).
        - XACK only after the artifact is atomically written and the job_id recorded.
        - Jobs superseded by a newer answer for the same turn (different `message_id` on the
          candidate artifact) are acknowledged without evaluation.
        - Transient failures are retried with exponential backoff and jitter through the
          queue's delayed set; exhausted jobs go to the dead-letter stream.
    """

    def __init__(
        self,
        redis_client: Any,
        queue: RedisAssessmentQueue,
        llm_client: RuntimeLLMClient,
        workspace_resolver: Union[Callable[[str], SessionWorkspace], SessionWorkspaceManager],
        state_manager: Optional[SessionStateStore] = None,
        event_store: Optional[SessionEventStore] = None,
        max_retries: int = 3,
        evaluator: Optional[Any] = None,
        connection_manager: Optional[Any] = None,
        on_assessment_completed: Optional[AssessmentCompletedCallback] = None,
        llm_semaphore: Optional[asyncio.Semaphore] = None,
        retry_base_delay_seconds: float = 1.0,
        lease_refresh_seconds: float = 15.0,
    ) -> None:
        """Initialize AssessorWorker.

        Args:
            redis_client: Redis client instance.
            queue: RedisAssessmentQueue managing stream operations.
            llm_client: Runtime LLM provider client.
            workspace_resolver: Resolver returning SessionWorkspace for a session_id.
            state_manager: Optional session state store for lifecycle transitions.
            event_store: Optional SessionEventStore for appending replayable events.
            max_retries: Maximum execution attempts before dead-letter routing (default 3).
            evaluator: Optional evaluator exposing `evaluate_turn(...)`.
            connection_manager: Optional ConnectionManager for recording and pushing live events.
            on_assessment_completed: Optional callback invoked after each persisted evaluation.
            llm_semaphore: Optional semaphore bounding concurrent evaluator LLM calls.
            retry_base_delay_seconds: Base of the exponential retry backoff.
            lease_refresh_seconds: How often an in-flight job's pending idle time is reset
                (so XAUTOCLAIM recovery does not steal a slow but live evaluation).
        """
        self.redis_client = redis_client
        self.queue = queue
        self.llm_client = llm_client
        self.workspace_resolver = workspace_resolver
        self.state_manager = state_manager
        self.event_store = event_store
        self.max_retries = max_retries
        self.evaluator = evaluator or TurnEvaluator(llm_client=llm_client)
        self.connection_manager = connection_manager
        self.on_assessment_completed = on_assessment_completed
        self.llm_semaphore = llm_semaphore
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.lease_refresh_seconds = lease_refresh_seconds
        self.cancellations = CancellationRegistry()
        self._active_jobs: Dict[str, int] = {}

    def _get_workspace(self, session_id: str) -> SessionWorkspace:
        """Resolve SessionWorkspace instance for the given session ID."""
        if hasattr(self.workspace_resolver, "get_or_create_workspace"):
            return self.workspace_resolver.get_or_create_workspace(session_id)
        return self.workspace_resolver(session_id)

    async def process_next_job(
        self,
        group_name: str = "assessor_workers",
        consumer_name: str = "worker_1",
        block_ms: int = 0,
    ) -> bool:
        """Fetch and process the next assessment job from the stream consumer group.

        Returns:
            bool: True if a job was processed or verified idempotent, False otherwise.
        """
        jobs = await self.queue.read_jobs(
            group_name=group_name,
            consumer_name=consumer_name,
            count=1,
            block_ms=block_ms,
        )
        if not jobs:
            return False
        message_id, job_data = jobs[0]
        return await self._process_job_data(group_name, message_id, job_data, consumer_name=consumer_name)

    async def recover_and_process_idle_jobs(
        self,
        group_name: str = "assessor_workers",
        consumer_name: str = "recovery_worker",
        min_idle_time_ms: int = 0,
        count: int = 10,
    ) -> int:
        """Claim idle unacknowledged jobs from crashed consumers (XAUTOCLAIM) and process them."""
        claimed = await self.queue.claim_idle_jobs(
            group_name=group_name,
            consumer_name=consumer_name,
            min_idle_time_ms=min_idle_time_ms,
            count=count,
        )
        processed_count = 0
        for msg_id, job_data in claimed:
            if await self._process_job_data(group_name, msg_id, job_data, consumer_name=consumer_name):
                processed_count += 1
        return processed_count

    def _is_superseded(self, ws: SessionWorkspace, job_data: Dict[str, Any]) -> bool:
        """True if the turn's candidate artifact now belongs to a different client message."""
        job_message_id = job_data.get("message_id")
        turn_id = job_data.get("turn_id", "")
        if not job_message_id or not turn_id.startswith("turn_"):
            return False
        try:
            turn_number = int(turn_id.split("_", 1)[1])
        except ValueError:
            return False
        rel_path = f"turns/turn_{turn_number}_candidate.json"
        if not ws.exists(rel_path):
            return False
        try:
            artifact_message_id = ws.read_json(rel_path).get("message_id")
        except Exception:
            return False
        return bool(artifact_message_id) and artifact_message_id != job_message_id

    async def _publish(self, session_id: str, turn_id: str, event_type: LoopEventType, payload: Dict[str, Any]) -> None:
        """Record a replayable event and push it to the live connection when present."""
        event = LoopEvent(
            schema_version="1.0",
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=turn_id,
            sequence=0,
            timestamp=time.time(),
            type=event_type,
            payload=payload,
        )
        if self.connection_manager is not None:
            recorded = await self.connection_manager.record_event_async(session_id, event)
            await self.connection_manager.send_event(session_id, recorded)
        elif self.event_store is not None:
            await self.event_store.append_async(session_id, event)

    async def _keep_claim(self, group_name: str, consumer_name: str, message_id: str) -> None:
        """Periodically reset the job's idle time while it is being evaluated."""
        while True:
            await asyncio.sleep(self.lease_refresh_seconds)
            try:
                await self.queue.refresh_claim(group_name, consumer_name, message_id)
            except Exception as e:
                logger.warning("Assessment claim refresh failed: %s", sanitize_secrets(str(e)))

    async def _evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        if self.llm_semaphore is None:
            return await self.evaluator.evaluate_turn(**kwargs)
        async with self.llm_semaphore:
            return await self.evaluator.evaluate_turn(**kwargs)

    async def _discard_if_deleted(self, session_id: str, group_name: str, message_id: str) -> bool:
        """Fence stale deliveries before resolving a workspace or committing any side effect."""
        deleted = await self.queue.is_revoked(session_id)
        if not deleted and self.state_manager is not None:
            deleted = await self.state_manager.aget(session_id) is None
        if deleted:
            # Phiên đã xóa chỉ được ACK/XDEL; tuyệt đối không dựng lại file hoặc event.
            await self.queue.ack_job(group_name, message_id)
            await self.redis_client.xdel(self.queue.stream_key, message_id)
        return deleted

    async def _process_job_data(
        self,
        group_name: str,
        message_id: str,
        job_data: Dict[str, Any],
        consumer_name: Optional[str] = None,
    ) -> bool:
        """Process a single assessment job with idempotency, evaluation, persistence and events."""
        session_id = job_data.get("session_id", "")
        self._active_jobs[session_id] = self._active_jobs.get(session_id, 0) + 1
        try:
            return await self._process_job(group_name, message_id, job_data, consumer_name)
        finally:
            remaining = self._active_jobs.get(session_id, 1) - 1
            if remaining <= 0:
                # No in-flight job remains: drop RAM cancellation state (the durable Redis
                # fence is re-checked by every new job) so the registry does not grow forever.
                self._active_jobs.pop(session_id, None)
                self.cancellations.forget(session_id)
            else:
                self._active_jobs[session_id] = remaining

    async def _process_job(
        self,
        group_name: str,
        message_id: str,
        job_data: Dict[str, Any],
        consumer_name: Optional[str],
    ) -> bool:
        session_id = job_data.get("session_id", "")
        turn_id = job_data.get("turn_id", "")
        job_id = job_data.get("job_id", "")
        stage_id = job_data.get("stage_id", "")
        question = job_data.get("question", "")
        candidate_text = job_data.get("candidate_text", "")
        rubric_version = job_data.get("rubric_version", "rubric_v1")
        rubric_target = job_data.get("rubric_target", [])
        vad_metrics = job_data.get("vad_metrics", {})

        input_sha256 = job_data.get("input_sha256")
        if not input_sha256:
            input_sha256 = hashlib.sha256(f"{question}:{candidate_text}".encode("utf-8")).hexdigest()

        # Giữ lease khi kiểm tra/ghi trạng thái ngắn để phối hợp với thao tác xóa phiên.
        async with self.queue.session_mutation(session_id):
            if await self._discard_if_deleted(session_id, group_name, message_id):
                return True
            ws = self._get_workspace(session_id)
            eval_rel_path = f"evaluations/{turn_id}_eval.json"

            # 1. Superseded answers are acknowledged without evaluation
            if self._is_superseded(ws, job_data):
                # The newer answer shares this job_id, so the enqueued set already counts it once.
                await self.queue.ack_job(group_name, message_id)
                logger.info("Skipped superseded assessment job (session %s, %s)", session_ref(session_id), turn_id)
                return True

            # 2. Idempotency: an existing artifact with the same key is reused
            if ws.exists(eval_rel_path):
                try:
                    existing_doc = ws.read_json(eval_rel_path)
                    existing_payload = existing_doc.get("payload", existing_doc)
                    if (
                        existing_payload.get("job_id") == job_id
                        and existing_payload.get("input_sha256") == input_sha256
                        and existing_payload.get("rubric_version") == rubric_version
                    ):
                        await self.redis_client.sadd(completed_assessments_key(session_id), job_id)
                        await self.queue.ack_job(group_name, message_id)
                        logger.info(
                            "Idempotent assessment hit (session %s, %s); acknowledged without re-evaluation",
                            session_ref(session_id),
                            turn_id,
                        )
                        if self.on_assessment_completed is not None:
                            await self.on_assessment_completed(session_id)
                        return True
                except Exception as e:
                    logger.warning(
                        "Failed checking existing evaluation artifact for idempotency: %s",
                        sanitize_secrets(str(e)),
                    )

        # Gọi LLM ngoài lease: không giữ khóa phiên suốt thời gian mạng/model xử lý.
        # 3. Evaluate
        try:
            keepalive = (
                asyncio.create_task(self._keep_claim(group_name, consumer_name, message_id))
                if consumer_name else None
            )
            try:
                eval_result = await self.cancellations.run(session_id, self._evaluate(
                    question=question,
                    candidate_text=candidate_text,
                    rubric_target=rubric_target,
                    rubric_version=rubric_version,
                    vad_metrics=vad_metrics,
                    session_id=session_id,
                    turn_id=turn_id,
                ), self.redis_client)
            finally:
                if keepalive is not None:
                    keepalive.cancel()
                    await asyncio.gather(keepalive, return_exceptions=True)

            artifact_payload = {
                "schema_version": eval_result.get("schema_version", "1.0"),
                "job_id": job_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "stage_id": stage_id,
                "message_id": job_data.get("message_id"),
                "input_sha256": input_sha256,
                "rubric_version": rubric_version,
                "rubric_target": rubric_target,
                "evaluated_at": time.time(),
            }
            for key, value in eval_result.items():
                if key not in artifact_payload:
                    artifact_payload[key] = value

            # Kiểm tra lại sau LLM vì phiên có thể đã bị xóa trong lúc đang chấm.
            async with self.queue.session_mutation(session_id):
                if await self._discard_if_deleted(session_id, group_name, message_id):
                    return True
                # 4. Atomically write the artifact, then record completion, then ACK
                # Thứ tự artifact -> completed set -> ACK giúp lần giao lại tái dùng kết quả đã ghi.
                write_task = asyncio.create_task(asyncio.to_thread(
                    ws.write_artifact, eval_rel_path, artifact_type="turn_evaluation",
                    payload=artifact_payload, schema_version=str(artifact_payload["schema_version"]),
                ))
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    # Cancel coroutine không dừng thread: giữ lease đến khi ghi file kết thúc.
                    while not write_task.done():
                        try:
                            await asyncio.shield(write_task)
                        except asyncio.CancelledError:
                            continue
                    write_task.result()
                    raise
                await self.redis_client.sadd(completed_assessments_key(session_id), job_id)
                await self.queue.ack_job(group_name, message_id)

                await self._publish(
                    session_id,
                    turn_id,
                    LoopEventType.ASSESSMENT_UPDATE,
                    {
                        "job_id": job_id,
                        "turn_id": turn_id,
                        "stage_id": stage_id,
                        "content_score": artifact_payload.get("content_score"),
                        "communication_score": artifact_payload.get("communication_score"),
                        "status": "COMPLETED",
                    },
                )

            if self.on_assessment_completed is not None:
                await self.on_assessment_completed(session_id)
            return True

        except asyncio.CancelledError:
            # Only a cancellation of this task itself (pool shutdown) propagates. A per-session
            # cancellation must not terminate the shared pool consumer, even if the stop fence
            # has already disappeared again (e.g. the session was deleted meanwhile).
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            if await self.queue.is_revoked(session_id):
                await self.queue.ack_job(group_name, message_id)
                await self.redis_client.xdel(self.queue.stream_key, message_id)
                return True
            async with self.queue.session_mutation(session_id):
                if await self._discard_if_deleted(session_id, group_name, message_id):
                    return True
            # Still live: leave the delivery pending so idle recovery re-evaluates it.
            logger.warning(
                "Assessment evaluation cancelled without a stop fence (session %s, %s); left pending",
                session_ref(session_id),
                turn_id,
            )
            return False
        except Exception as exc:
            async with self.queue.session_mutation(session_id):
                if await self._discard_if_deleted(session_id, group_name, message_id):
                    return True
                clean_error = sanitize_secrets(str(exc))
                logger.error(
                    "Evaluation failed (session %s, %s): %s", session_ref(session_id), turn_id, clean_error
                )
                retry_count = int(job_data.get("retry_count", 0)) + 1
                job_data["retry_count"] = retry_count

                # Đưa job vào delayed set trước ACK; hết budget thì DLQ và failed set chặn report.
                if retry_count < self.max_retries:
                    delay = self.retry_base_delay_seconds * (2 ** (retry_count - 1))
                    delay += random.uniform(0, delay / 2)
                    await self.queue.schedule_retry(job_data, delay)
                    await self.queue.ack_job(group_name, message_id)
                else:
                    await self.queue.send_to_dead_letter(
                        job_data=job_data,
                        error_reason=clean_error,
                        message_id=message_id,
                    )
                    # Record the failure before ACK: a crash between the two must leave the job
                    # redeliverable rather than ACKed but absent from the barrier's failed set.
                    await self.redis_client.sadd(failed_assessments_key(session_id), job_id)
                    await self.queue.ack_job(group_name, message_id)

                    if self.state_manager is not None:
                        try:
                            state = await self.state_manager.aget(session_id)
                            if state and state.status == LifecycleStatus.ASSESSING:
                                await self.state_manager.atransition(
                                    state,
                                    expected_version=state.state_version,
                                    new_status=LifecycleStatus.ASSESSMENT_FAILED,
                                    metadata={
                                        "failure_code": "ASSESSMENT_EXHAUSTED",
                                        "failed_job_id": job_id,
                                    },
                                )
                        except Exception as trans_err:
                            logger.warning(
                                "Failed state transition to ASSESSMENT_FAILED: %s",
                                sanitize_secrets(str(trans_err)),
                            )
                return False
