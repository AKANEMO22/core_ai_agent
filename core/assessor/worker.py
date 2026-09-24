"""Background assessor worker pool consuming Redis Streams evaluation tasks."""

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.assessor.queue import RedisAssessmentQueue
from core.assessor.subagent import TurnEvaluator
from core.runtime.agent import sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.runtime.types import LoopEvent, LoopEventType
from core.session.events import SessionEventStore
from core.session.state import LifecycleStatus, SessionStateManager
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager

logger = logging.getLogger(__name__)


class AssessorWorker:
    """Consumes assessment jobs from Redis Streams, executes rubric evaluations, and persists artifacts.

    Enforces at-least-once delivery with idempotency (job_id + input_sha256 + rubric_version),
    atomic artifact writes, set cardinality tracking, dead-letter routing, and event stream appending.
    """

    def __init__(
        self,
        redis_client: Any,
        queue: RedisAssessmentQueue,
        llm_client: RuntimeLLMClient,
        workspace_resolver: Union[Callable[[str], SessionWorkspace], SessionWorkspaceManager],
        state_manager: Optional[SessionStateManager] = None,
        event_store: Optional[SessionEventStore] = None,
        max_retries: int = 3,
        evaluator: Optional[TurnEvaluator] = None,
        connection_manager: Optional[Any] = None,
    ) -> None:
        """Initialize AssessorWorker.

        Args:
            redis_client: Redis client instance.
            queue: RedisAssessmentQueue managing stream operations.
            llm_client: Runtime LLM provider client.
            workspace_resolver: Resolver returning SessionWorkspace for a session_id.
            state_manager: Optional SessionStateManager for lifecycle transitions.
            event_store: Optional SessionEventStore for appending replayable events.
            max_retries: Maximum execution attempts before dead-letter routing (default 3).
            evaluator: Optional TurnEvaluator instance.
            connection_manager: Optional ConnectionManager for dispatching live events.
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
            bool: True if a job was successfully processed or verified idempotent,
                  False if no jobs were available or a fatal failure occurred.
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
        return await self._process_job_data(group_name, message_id, job_data)

    async def recover_and_process_idle_jobs(
        self,
        group_name: str = "assessor_workers",
        consumer_name: str = "recovery_worker",
        min_idle_time_ms: int = 0,
        count: int = 10,
    ) -> int:
        """Claim idle unacknowledged jobs from other workers using XAUTOCLAIM and process them.

        Args:
            group_name: Consumer group name.
            consumer_name: Worker consumer identifier claiming the jobs.
            min_idle_time_ms: Minimum idle duration in milliseconds.
            count: Maximum number of jobs to reclaim.

        Returns:
            int: Number of idle jobs successfully claimed and processed.
        """
        claimed = await self.queue.claim_idle_jobs(
            group_name=group_name,
            consumer_name=consumer_name,
            min_idle_time_ms=min_idle_time_ms,
            count=count,
        )
        processed_count = 0
        for msg_id, job_data in claimed:
            success = await self._process_job_data(group_name, msg_id, job_data)
            if success:
                processed_count += 1
        return processed_count

    async def _process_job_data(
        self,
        group_name: str,
        message_id: str,
        job_data: Dict[str, Any],
    ) -> bool:
        """Process a single assessment job with idempotency, evaluation, persistence, and event dispatch."""
        session_id = job_data.get("session_id", "")
        turn_id = job_data.get("turn_id", "")
        job_id = job_data.get("job_id", "")
        stage_id = job_data.get("stage_id", "")
        question = job_data.get("question", "")
        candidate_text = job_data.get("candidate_text", "")
        rubric_version = job_data.get("rubric_version", "rubric_v1")
        rubric_target = job_data.get("rubric_target", [])
        vad_metrics = job_data.get("vad_metrics", {})

        # Compute SHA256 input fingerprint
        input_sha256 = job_data.get("input_sha256")
        if not input_sha256:
            raw_input = f"{question}:{candidate_text}".encode("utf-8")
            input_sha256 = hashlib.sha256(raw_input).hexdigest()

        ws = self._get_workspace(session_id)
        eval_rel_path = f"evaluations/{turn_id}_eval.json"

        # 1. Idempotency Check: Existing evaluation artifact matching key
        if ws.exists(eval_rel_path):
            try:
                existing_doc = ws.read_json(eval_rel_path)
                existing_payload = existing_doc.get("payload", existing_doc)
                if (
                    existing_payload.get("job_id") == job_id
                    and existing_payload.get("input_sha256") == input_sha256
                    and existing_payload.get("rubric_version") == rubric_version
                ):
                    # Already evaluated: update Redis set and XACK without re-running LLM
                    await self.redis_client.sadd(
                        f"session:{session_id}:completed_assessments", job_id
                    )
                    await self.queue.ack_job(group_name, message_id)
                    logger.info(
                        f"Idempotent hit for job '{job_id}'; acknowledged without re-evaluation."
                    )
                    return True
            except Exception as e:
                logger.warning(
                    f"Failed checking existing evaluation artifact for idempotency: {e}"
                )

        # 2. Execute Evaluation Subagent with Retry Budget
        try:
            eval_result = await self.evaluator.evaluate_turn(
                question=question,
                candidate_text=candidate_text,
                rubric_target=rubric_target,
                rubric_version=rubric_version,
                vad_metrics=vad_metrics,
            )

            artifact_payload = {
                "schema_version": "1.0",
                "job_id": job_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "stage_id": stage_id,
                "input_sha256": input_sha256,
                "rubric_version": rubric_version,
                "rubric_target": rubric_target,
                "content_score": eval_result["content_score"],
                "communication_score": eval_result["communication_score"],
                "strengths": eval_result["strengths"],
                "improvements": eval_result["improvements"],
                "delivery_observations": eval_result["delivery_observations"],
                "evaluated_at": time.time(),
            }

            # 3. Atomically write evaluation artifact
            ws.write_artifact(
                eval_rel_path,
                artifact_type="turn_evaluation",
                payload=artifact_payload,
            )

            # 4. Atomically record job_id in Redis completed set
            await self.redis_client.sadd(
                f"session:{session_id}:completed_assessments", job_id
            )

            # 5. XACK only after atomic file persistence and Redis set recording
            await self.queue.ack_job(group_name, message_id)

            # 6. Append assessment_completed to replayable event store
            if self.event_store is not None:
                self.event_store.append_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    event_type="assessment_completed",
                    payload=artifact_payload,
                )

            # Broadcast assessment_update to connection manager if configured
            if self.connection_manager is not None:
                update_event = LoopEvent(
                    schema_version="1.0",
                    event_id=f"evt_{uuid.uuid4().hex[:16]}",
                    session_id=session_id,
                    turn_id=turn_id,
                    sequence=0,
                    timestamp=time.time(),
                    type=LoopEventType.ASSESSMENT_UPDATE,
                    payload={
                        "job_id": job_id,
                        "turn_id": turn_id,
                        "stage_id": stage_id,
                        "content_score": artifact_payload["content_score"],
                        "communication_score": artifact_payload["communication_score"],
                        "status": "COMPLETED",
                    },
                )
                if hasattr(self.connection_manager, "record_event_async"):
                    await self.connection_manager.record_event_async(session_id, update_event)
                else:
                    self.connection_manager.record_event(session_id, update_event)
                await self.connection_manager.send_event(session_id, update_event)

            # Synchronize to Redis stream session:{session_id}:events
            try:
                redis_stream_key = f"session:{session_id}:events"
                await self.redis_client.xadd(
                    redis_stream_key,
                    {
                        "type": "assessment_completed",
                        "turn_id": turn_id,
                        "payload": json.dumps(artifact_payload),
                    },
                )
            except Exception:
                pass

            return True

        except Exception as exc:
            logger.error(
                f"Evaluation failed for job '{job_id}': {sanitize_secrets(str(exc))}"
            )
            retry_count = int(job_data.get("retry_count", 0)) + 1
            job_data["retry_count"] = retry_count

            if retry_count < self.max_retries:
                # Re-enqueue with incremented retry count and acknowledge current offset
                await self.queue.enqueue(job_data)
                await self.queue.ack_job(group_name, message_id)
            else:
                # Exhausted retry budget: forward to dead letter queue
                await self.queue.send_to_dead_letter(
                    job_data=job_data,
                    error_reason=sanitize_secrets(str(exc)),
                    message_id=message_id,
                )
                await self.queue.ack_job(group_name, message_id)
                await self.redis_client.sadd(
                    f"session:{session_id}:failed_assessments", job_id
                )

                # Transition session lifecycle to ASSESSMENT_FAILED if state_manager available
                if self.state_manager is not None:
                    try:
                        state = self.state_manager.get_state(session_id)
                        if state and state.status == LifecycleStatus.ASSESSING:
                            self.state_manager.transition(
                                state,
                                expected_version=state.state_version,
                                new_status=LifecycleStatus.ASSESSMENT_FAILED,
                                metadata={
                                    "failure_code": "ASSESSMENT_EXHAUSTED",
                                    "failed_job_id": job_id,
                                    "error": sanitize_secrets(str(exc)),
                                },
                            )
                    except Exception as trans_err:
                        logger.warning(
                            f"Failed state transition to ASSESSMENT_FAILED: {trans_err}"
                        )

            return False
