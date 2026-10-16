"""Redis Streams assessment queue producer, consumer group manager, and dead-letter queue.

Biên giới Redis của luồng đánh giá: enqueue, đọc consumer group, ACK và retry.
Producer đưa AssessmentJob vào stream; worker/pool đọc lại payload để chấm, còn
các set theo session lưu job_id đã gửi/hoàn tất/lỗi cho barrier tổng hợp báo cáo.
Retry chờ trong sorted set; job hết lượt thử nằm ở dead-letter stream để xử lý lại.
"""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, List, Optional, Tuple, Union

from redis.exceptions import ResponseError, WatchError

from core.interview.queue import AssessmentJob
from core.session.idempotency import RedisLease

logger = logging.getLogger(__name__)


def enqueued_assessments_key(session_id: str) -> str:
    """Set of job_ids enqueued for a session."""
    return f"session:{session_id}:enqueued_assessments"


def completed_assessments_key(session_id: str) -> str:
    """Set of job_ids whose evaluation artifact is durably persisted."""
    return f"session:{session_id}:completed_assessments"


def failed_assessments_key(session_id: str) -> str:
    """Set of job_ids that exhausted their retry budget."""
    return f"session:{session_id}:failed_assessments"


def expected_assessments_key(session_id: str) -> str:
    """Expected assessment count frozen when the dialogue completes."""
    return f"session:{session_id}:expected_assessments"


def session_assessment_keys(session_id: str) -> List[str]:
    """Every Redis key holding assessment bookkeeping for a session (used by deletion)."""
    return [
        enqueued_assessments_key(session_id),
        completed_assessments_key(session_id),
        failed_assessments_key(session_id),
        expected_assessments_key(session_id),
    ]


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
        counter_ttl_seconds: int = 7 * 24 * 3600,
        mutation_lock_ttl_ms: int = 120000,
    ) -> None:
        """Initialize Redis assessment queue with designated stream keys.

        Args:
            redis_client: Async-compatible Redis client.
            stream_key: Primary Redis stream key for assessment jobs.
            dead_letter_stream_key: Target stream key for dead letter payloads.
            counter_ttl_seconds: TTL of per-session assessment bookkeeping sets.
        """
        self.redis_client = redis_client
        self.stream_key = stream_key
        self.dead_letter_stream_key = dead_letter_stream_key
        self.counter_ttl_seconds = counter_ttl_seconds
        self.mutation_lock_ttl_ms = mutation_lock_ttl_ms

    @staticmethod
    def revoked_key(session_id: str) -> str:
        return f"session:{session_id}:revoked"

    async def is_revoked(self, session_id: str) -> bool:
        return bool(await self.redis_client.exists(self.revoked_key(session_id), self.stopped_key(session_id)))

    @staticmethod
    def stopped_key(session_id: str) -> str:
        return f"session:{session_id}:work_stopped"

    @asynccontextmanager
    async def session_mutation(self, session_id: str, purpose: str = "assessment_lock"):
        """Serialize short artifact/event mutations with session deletion across processes."""
        lease = RedisLease(self.redis_client)
        key = f"session:{session_id}:{purpose}"
        token = None
        while token is None:
            token = await lease.acquire(key, self.mutation_lock_ttl_ms)
            if token is None:
                await asyncio.sleep(0.01)
        # Lease dùng chung với xóa phiên; gia hạn để các thao tác I/O chưa xong không mất khóa.
        renewal = asyncio.create_task(self._renew_mutation_lease(key, token))
        try:
            yield
        finally:
            renewal.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await renewal
            finally:
                await lease.release(key, token)

    async def _renew_mutation_lease(self, key: str, token: str) -> None:
        """Keep a mutation lease alive while its owner awaits I/O."""
        while True:
            await asyncio.sleep(self.mutation_lock_ttl_ms / 3000)
            while True:
                async with self.redis_client.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(key)
                        owner = await pipe.get(key)
                        if isinstance(owner, bytes):
                            owner = owner.decode()
                        if owner != token:
                            raise RuntimeError("Session mutation lease ownership lost")
                        # Chỉ chủ sở hữu được gia hạn; không kéo dài lease của tiến trình khác.
                        pipe.multi()
                        pipe.pexpire(key, self.mutation_lock_ttl_ms)
                        await pipe.execute()
                        break
                    except WatchError:
                        continue

    async def _write_if_active(self, session_id: str, configure: Any) -> Any:
        """Commit queue writes only while the session's revocation fence is absent."""
        while True:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self.revoked_key(session_id), self.stopped_key(session_id))
                    if await pipe.exists(self.revoked_key(session_id), self.stopped_key(session_id)):
                        return None
                    # WATCH buộc mọi enqueue/retry/DLQ dừng nếu DELETE dựng hàng rào.
                    pipe.multi()
                    configure(pipe)
                    return await pipe.execute()
                except WatchError:
                    continue

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

    def _prepare_job(self, job_data: Union[Dict[str, Any], AssessmentJob]) -> Dict[str, Any]:
        """Normalize payload defaults, its input hash and idempotency job ID."""
        job_dict = job_data.to_dict() if hasattr(job_data, "to_dict") else dict(job_data)

        # Standardize schema version
        if "schema_version" not in job_dict:
            job_dict["schema_version"] = "1.0"

        # job_id nhận diện lượt/rubric, còn input_sha256 phân biệt nội dung câu trả lời thay đổi.
        session_id = job_dict.setdefault("session_id", "default_sess")
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

        return job_dict

    async def enqueue(self, job_data: Union[Dict[str, Any], AssessmentJob]) -> str:
        """Atomically enqueue and count an active session job; revoked sessions return an empty ID."""
        job_dict = self._prepare_job(job_data)
        session_id = job_dict["session_id"]
        fields = {
            "payload": json.dumps(job_dict),
            "job_id": job_dict["job_id"],
            "session_id": session_id,
        }
        # Register the job in the session's enqueued set in the same transaction as XADD,
        # so the expected-assessment count frozen at dialogue completion is exact.
        def configure(pipe):
            pipe.xadd(self.stream_key, fields)
            pipe.sadd(enqueued_assessments_key(session_id), job_dict["job_id"])
            pipe.expire(enqueued_assessments_key(session_id), self.counter_ttl_seconds)

        result = await self._write_if_active(session_id, configure)
        return job_dict["job_id"] if result is not None else ""

    def _parse_message_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize message fields dictionary, unpacking JSON payload if present."""
        fields = {key.decode() if isinstance(key, bytes) else key: value for key, value in fields.items()}
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
        # Dấu > lấy message mới của group; pending từ consumer chết được lấy bằng claim_idle_jobs.
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
        results: List[Tuple[str, Dict[str, Any]]] = []
        # XAUTOCLAIM scans at most ~count*10 PEL entries per call; follow the returned
        # cursor so idle jobs behind a long run of fresh pending entries are still reclaimed.
        cursor: Any = "0-0"
        while len(results) < count:
            response = await self.redis_client.xautoclaim(
                self.stream_key,
                group_name,
                consumer_name,
                min_idle_time=min_idle_time_ms,
                start_id=cursor,
                count=count - len(results),
            )
            if not response or len(response) < 2:
                break
            for msg in response[1]:
                if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                    msg_id, fields = msg[0], msg[1]
                    job_data = self._parse_message_fields(fields)
                    results.append((msg_id, job_data))
            cursor = response[0]
            if isinstance(cursor, bytes):
                cursor = cursor.decode()
            if not cursor or cursor == "0-0":
                break

        return results

    async def refresh_claim(self, group_name: str, consumer_name: str, message_id: str) -> None:
        """Reset a pending job's idle time so a long evaluation is not reclaimed mid-flight."""
        # XCLAIM to the current owner with min-idle 0 only resets the delivery idle timer.
        await self.redis_client.xclaim(
            self.stream_key, group_name, consumer_name, min_idle_time=0, message_ids=[message_id], justid=True
        )

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
        result = await self._write_if_active(
            job_data.get("session_id", ""), lambda pipe: pipe.xadd(self.dead_letter_stream_key, dlq_entry)
        )
        return str(result[0]) if result is not None else ""

    @property
    def delayed_key(self) -> str:
        """Sorted set of jobs waiting for their retry backoff to elapse."""
        return f"{self.stream_key}:delayed"

    async def schedule_retry(self, job_data: Dict[str, Any], delay_seconds: float) -> None:
        """Park a job until `delay_seconds` from now; `promote_due_retries` re-enqueues it."""
        await self._write_if_active(
            job_data.get("session_id", ""),
            lambda pipe: pipe.zadd(self.delayed_key, {json.dumps(job_data): time.time() + max(0.0, delay_seconds)}),
        )

    async def promote_due_retries(self, limit: int = 100) -> int:
        """Move retry jobs whose backoff elapsed back onto the stream. Safe to run concurrently."""
        # Sorted set dùng thời điểm được phép thử lại làm score; pool gọi hàm này định kỳ.
        due = await self.redis_client.zrangebyscore(
            self.delayed_key, "-inf", time.time(), start=0, num=limit
        )
        promoted = 0
        for raw in due:
            job = self._prepare_job(json.loads(raw))
            session_id = job["session_id"]
            enqueued_key = enqueued_assessments_key(session_id)
            while True:
                async with self.redis_client.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(self.delayed_key, self.stream_key, enqueued_key, self.revoked_key(session_id), self.stopped_key(session_id))
                        if await pipe.zscore(self.delayed_key, raw) is None:
                            break
                        revoked = await pipe.exists(self.revoked_key(session_id), self.stopped_key(session_id))
                        if not revoked:
                            # MULTI không rollback lỗi lệnh: kiểm tra kiểu trước khi xóa retry.
                            for key, expected_type in ((self.stream_key, "stream"), (enqueued_key, "set")):
                                key_type = await pipe.type(key)
                                if isinstance(key_type, bytes):
                                    key_type = key_type.decode()
                                if key_type not in ("none", expected_type):
                                    raise ResponseError("WRONGTYPE retry destination has an incompatible type")
                        pipe.multi()
                        if not revoked:
                            pipe.xadd(self.stream_key, {"payload": json.dumps(job), "job_id": job["job_id"], "session_id": session_id})
                            pipe.sadd(enqueued_key, job["job_id"])
                            pipe.expire(enqueued_key, self.counter_ttl_seconds)
                        # XADD và ZREM cùng EXEC: crash không tạo khoảng trống mất retry.
                        pipe.zrem(self.delayed_key, raw)
                        await pipe.execute()
                        promoted += int(not revoked)
                        break
                    except WatchError:
                        continue
        return promoted

    async def purge_session(self, session_id: str) -> None:
        """Remove one session's retained payloads, including pending stream deliveries."""
        # DELETE phải dựng revoked trước: không producer nào thêm payload sau lượt quét.
        for stream in (self.stream_key, self.dead_letter_stream_key):
            if not await self.redis_client.exists(stream):
                continue
            groups = await self.redis_client.xinfo_groups(stream)
            cursor = "-"
            while True:
                entries = await self.redis_client.xrange(stream, min=cursor, max="+", count=500)
                if not entries:
                    break
                ids = [entry_id for entry_id, fields in entries if self._parse_message_fields(fields).get("session_id") == session_id]
                if ids:
                    async with self.redis_client.pipeline(transaction=True) as pipe:
                        for group in groups:
                            pipe.xack(stream, group.get("name", group.get(b"name")), *ids)
                        pipe.xdel(stream, *ids)
                        await pipe.execute()
                last_id = entries[-1][0]
                cursor = f"({last_id.decode() if isinstance(last_id, bytes) else last_id}"
        while True:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self.delayed_key)
                    cursor = 0
                    matches = []
                    while True:
                        cursor, entries = await pipe.zscan(self.delayed_key, cursor=cursor, count=500)
                        matches.extend(raw for raw, _ in entries if json.loads(raw).get("session_id") == session_id)
                        if cursor == 0:
                            break
                    if not matches:
                        return
                    # Quét xong mới xóa; WATCH tránh bỏ sót khi promoter đổi tập đang quét.
                    pipe.multi()
                    pipe.zrem(self.delayed_key, *matches)
                    await pipe.execute()
                    return
                except WatchError:
                    continue

    async def requeue_dead_letters(self, session_id: str) -> List[str]:
        """Re-enqueue a session's dead-lettered jobs with a fresh retry budget.

        Idempotent: requeued entries are removed from the dead-letter stream and the failed
        set, so a repeated call finds nothing to requeue.

        Returns:
            List[str]: job_ids that were requeued.
        """
        requeued: List[str] = []
        # Khôi phục budget trước khi enqueue lại; bỏ dấu lỗi để barrier có thể tiếp tục chờ kết quả.
        entries = await self.redis_client.xrange(self.dead_letter_stream_key, "-", "+")
        for entry_id, fields in entries:
            job_data = self._parse_message_fields(fields)
            if job_data.get("session_id") != session_id:
                continue
            job_data["retry_count"] = 0
            # A refused enqueue (revoked/stopped session) must keep the DLQ record.
            if not await self.enqueue(job_data):
                continue
            await self.redis_client.xdel(self.dead_letter_stream_key, entry_id)
            await self.redis_client.srem(failed_assessments_key(session_id), job_data.get("job_id", ""))
            requeued.append(job_data.get("job_id", ""))
        return requeued

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
