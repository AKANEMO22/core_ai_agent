"""Redis Streams assessment queue producer, consumer group manager, and dead-letter queue."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from core.interview.queue import AssessmentJob

logger = logging.getLogger(__name__)


class RedisAssessmentQueue:
    """Production-grade Redis Streams assessment queue with consumer groups, ACK, and DLQ.

    Attributes:
        redis_client: Redis or FakeRedis client with async commands support.
        stream_key: Name of the primary assessment tasks stream.
        dead_letter_stream_key: Name of the stream for permanently failed jobs.
    """

    def __init__(
        self,
        redis_client: Any,
        stream_key: str = "interview:tasks:assessment",
        dead_letter_stream_key: str = "interview:tasks:assessment:dead_letter",
    ) -> None:
        """Initialize Redis assessment queue with designated stream keys.

        Args:
            redis_client: Async-compatible Redis client.
            stream_key: Primary Redis stream key for assessment jobs.
            dead_letter_stream_key: Target stream key for dead letter payloads.
        """
        self.redis_client = redis_client
        self.stream_key = stream_key
        self.dead_letter_stream_key = dead_letter_stream_key

    async def setup_consumer_group(
        self, group_name: str = "assessor_workers", start_id: str = "$"
    ) -> None:
        """Create the consumer group on the assessment stream if it does not already exist.

        Args:
            group_name: Consumer group identifier.
            start_id: Initial message ID offset ('$' for new messages, '0' for all).
        """
        try:
            await self.redis_client.xgroup_create(
                self.stream_key,
                group_name,
                id=start_id,
                mkstream=True,
            )
        except Exception as e:
            err_msg = str(e)
            if "BUSYGROUP" not in err_msg and "already exists" not in err_msg.lower():
                logger.warning(f"Unexpected response creating consumer group: {e}")
                raise

    async def enqueue(self, job_data: Union[Dict[str, Any], AssessmentJob]) -> str:
        """Enqueue an assessment job to the Redis stream.

        Computes input_sha256 and idempotency job_id if not explicitly provided.

        Args:
            job_data: AssessmentJob instance or equivalent dictionary.

        Returns:
            str: Unique idempotency job_id.
        """
        job_dict = job_data.to_dict() if hasattr(job_data, "to_dict") else dict(job_data)

        # Standardize schema version
        if "schema_version" not in job_dict:
            job_dict["schema_version"] = "1.0"

        session_id = job_dict.get("session_id", "default_sess")
        turn_id = job_dict.get("turn_id", "turn_001")
        rubric_version = job_dict.get("rubric_version", "rubric_v1")

        if not job_dict.get("job_id"):
            job_dict["job_id"] = f"{session_id}:{turn_id}:{rubric_version}"

        question = job_dict.get("question", "")
        candidate_text = job_dict.get("candidate_text", "")
        if not job_dict.get("input_sha256"):
            content_fingerprint = f"{question}:{candidate_text}".encode("utf-8")
            job_dict["input_sha256"] = hashlib.sha256(content_fingerprint).hexdigest()

        if "created_at" not in job_dict:
            job_dict["created_at"] = time.time()

        if "retry_count" not in job_dict:
            job_dict["retry_count"] = 0

        serialized = json.dumps(job_dict)
        fields = {
            "payload": serialized,
            "job_id": job_dict["job_id"],
            "session_id": session_id,
        }
        await self.redis_client.xadd(self.stream_key, fields)
        return job_dict["job_id"]

    def _parse_message_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize message fields dictionary, unpacking JSON payload if present."""
        if "payload" in fields:
            try:
                return json.loads(fields["payload"])
            except Exception:
                pass
        return dict(fields)

    async def read_jobs(
        self,
        group_name: str,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 0,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Read pending or new jobs for this consumer from the consumer group.

        Args:
            group_name: Consumer group name.
            consumer_name: Worker consumer identifier.
            count: Maximum number of jobs to fetch.
            block_ms: Milliseconds to block waiting for messages (0 for non-blocking).

        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of (message_id, parsed_job_dict).
        """
        streams_arg = {self.stream_key: ">"}
        block_val = block_ms if block_ms > 0 else None
        response = await self.redis_client.xreadgroup(
            group_name,
            consumer_name,
            streams_arg,
            count=count,
            block=block_val,
        )

        results: List[Tuple[str, Dict[str, Any]]] = []
        if not response:
            return results

        for stream_name, messages in response:
            for msg_id, fields in messages:
                job_data = self._parse_message_fields(fields)
                results.append((msg_id, job_data))

        return results

    async def ack_job(self, group_name: str, message_id: str) -> None:
        """Acknowledge successful completion of a job.

        Args:
            group_name: Consumer group name.
            message_id: Redis stream message identifier.
        """
        await self.redis_client.xack(self.stream_key, group_name, message_id)

    async def claim_idle_jobs(
        self,
        group_name: str,
        consumer_name: str,
        min_idle_time_ms: int = 60000,
        count: int = 10,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Reclaim pending jobs exceeding lease timeout via XAUTOCLAIM.

        Args:
            group_name: Consumer group name.
            consumer_name: Reclaiming worker consumer identifier.
            min_idle_time_ms: Minimum idle duration in milliseconds before reclamation.
            count: Maximum jobs to reclaim.

        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of reclaimed (message_id, parsed_job_dict).
        """
        response = await self.redis_client.xautoclaim(
            self.stream_key,
            group_name,
            consumer_name,
            min_idle_time=min_idle_time_ms,
            start_id="0-0",
            count=count,
        )

        results: List[Tuple[str, Dict[str, Any]]] = []
        if not response or len(response) < 2:
            return results

        messages = response[1]
        for msg in messages:
            if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                msg_id, fields = msg[0], msg[1]
                job_data = self._parse_message_fields(fields)
                results.append((msg_id, job_data))

        return results

    async def send_to_dead_letter(
        self,
        job_data: Dict[str, Any],
        error_reason: str,
        message_id: Optional[str] = None,
    ) -> str:
        """Forward permanently failed job to dead letter stream.

        Args:
            job_data: The failing job dictionary.
            error_reason: Explanation or exception trace summary.
            message_id: Optional original Redis stream message ID.

        Returns:
            str: Dead letter stream message identifier.
        """
        dlq_entry = {
            "payload": json.dumps(job_data),
            "job_id": job_data.get("job_id", ""),
            "session_id": job_data.get("session_id", ""),
            "turn_id": job_data.get("turn_id", ""),
            "error_reason": error_reason,
            "failed_at": time.time(),
            "original_message_id": message_id or "",
        }
        res = await self.redis_client.xadd(self.dead_letter_stream_key, dlq_entry)
        return str(res)

    async def get_queue_size(self) -> int:
        """Return count of items in the primary assessment stream."""
        return await self.redis_client.xlen(self.stream_key)

    async def dequeue(self, timeout_seconds: float = 0.0) -> Optional[Dict[str, Any]]:
        """Protocol compatibility method for AssessmentQueue."""
        block_ms = int(timeout_seconds * 1000)
        jobs = await self.read_jobs(
            group_name="default_group",
            consumer_name="default_consumer",
            count=1,
            block_ms=block_ms,
        )
        if jobs:
            return jobs[0][1]
        return None
