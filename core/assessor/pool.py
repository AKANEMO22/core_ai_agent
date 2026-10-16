"""Lifecycle-managed pool of assessor consumers with recovery, retry promotion and heartbeats.

Quản lý vòng đời các consumer đánh giá trong tiến trình ứng dụng.
Pool đọc job từ RedisAssessmentQueue và giao AssessorWorker xử lý; vòng bảo trì
đưa retry tới hạn về stream, thu hồi job từ consumer im lặng và gọi sweep báo cáo.
start/stop gắn với vòng đời ứng dụng; heartbeat/status phục vụ kiểm tra readiness.
"""

import asyncio
import logging
import socket
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.assessor.queue import RedisAssessmentQueue
from core.assessor.worker import AssessorWorker
from core.runtime.agent import sanitize_secrets

logger = logging.getLogger(__name__)

HEARTBEAT_KEY_PREFIX = "assessor:heartbeat:"


class AssessorWorkerPool:
    """Runs N stream consumers plus a maintenance loop inside the application process.

    Maintenance loop (every `recovery_interval_seconds`):
        - promotes retry jobs whose backoff elapsed,
        - reclaims jobs idle longer than `lease_ms` (crashed consumers) via XAUTOCLAIM,
        - runs the optional `sweeper` (report pipeline recovery) every `sweep_interval_seconds`.
    Each consumer refreshes a heartbeat key so `/health/ready` can detect a dead pool.
    """

    def __init__(
        self,
        worker: AssessorWorker,
        queue: RedisAssessmentQueue,
        redis_client: Any,
        size: int = 4,
        group_name: str = "assessor_workers",
        block_ms: int = 1000,
        lease_ms: int = 60000,
        recovery_interval_seconds: float = 15.0,
        heartbeat_ttl_seconds: int = 15,
        sweeper: Optional[Callable[[], Awaitable[Any]]] = None,
        sweep_interval_seconds: float = 60.0,
    ) -> None:
        self.worker = worker
        self.queue = queue
        self.redis = redis_client
        self.size = size
        self.group_name = group_name
        self.block_ms = block_ms
        self.lease_ms = lease_ms
        self.recovery_interval_seconds = recovery_interval_seconds
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self.sweeper = sweeper
        self.sweep_interval_seconds = sweep_interval_seconds
        self.instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self._tasks: List[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()
        self._in_flight = 0
        self._processed = 0
        self._last_sweep = 0.0

    def _consumer_name(self, index: int) -> str:
        return f"{self.instance_id}-c{index}"

    async def start(self) -> None:
        """Create the consumer group and start consumer and maintenance tasks."""
        await self.queue.setup_consumer_group(group_name=self.group_name, start_id="0")
        self._stopping.clear()
        for index in range(self.size):
            self._tasks.append(asyncio.create_task(self._consume(index), name=f"assessor-{index}"))
        self._tasks.append(asyncio.create_task(self._maintain(), name="assessor-maintenance"))
        logger.info("Assessor pool started with %d consumers", self.size)

    async def stop(self, grace_seconds: float = 30.0) -> None:
        """Stop reading new jobs, wait for in-flight jobs up to the grace period, then cancel."""
        self._stopping.set()
        if not self._tasks:
            return
        # Cho job đang chạy thời gian kết thúc; sau grace period mới hủy các task còn lại.
        _, pending = await asyncio.wait(self._tasks, timeout=grace_seconds)
        for task in pending:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Assessor pool stopped")

    async def _heartbeat(self, consumer: str) -> None:
        try:
            await self.redis.set(
                f"{HEARTBEAT_KEY_PREFIX}{consumer}", str(time.time()), ex=self.heartbeat_ttl_seconds
            )
        except Exception as e:
            logger.warning("Assessor heartbeat failed: %s", sanitize_secrets(str(e)))

    async def _consume(self, index: int) -> None:
        consumer = self._consumer_name(index)
        while not self._stopping.is_set():
            await self._heartbeat(consumer)
            try:
                jobs = await self.queue.read_jobs(
                    group_name=self.group_name,
                    consumer_name=consumer,
                    count=1,
                    block_ms=self.block_ms,
                )
                if not jobs:
                    # Non-blocking clients (tests) return immediately; avoid a busy loop.
                    await asyncio.sleep(min(0.05, self.block_ms / 1000))
                    continue
                for message_id, job_data in jobs:
                    self._in_flight += 1
                    try:
                        # Worker chịu trách nhiệm ACK/retry và artifact; pool chỉ quản lý vòng đọc cùng bộ đếm.
                        await self.worker._process_job_data(self.group_name, message_id, job_data, consumer_name=consumer)
                        self._processed += 1
                    finally:
                        self._in_flight -= 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Assessor consumer error: %s", sanitize_secrets(str(e)))
                await asyncio.sleep(1.0)

    async def _maintain(self) -> None:
        while not self._stopping.is_set():
            try:
                # Ba bước phục hồi: retry tới hạn, pending của consumer chết, rồi pipeline báo cáo.
                await self.queue.promote_due_retries()
                await self.worker.recover_and_process_idle_jobs(
                    group_name=self.group_name,
                    consumer_name=f"{self.instance_id}-recovery",
                    min_idle_time_ms=self.lease_ms,
                )
                if self.sweeper is not None and time.time() - self._last_sweep >= self.sweep_interval_seconds:
                    self._last_sweep = time.time()
                    await self.sweeper()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Assessor maintenance error: %s", sanitize_secrets(str(e)))
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.recovery_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def is_healthy(self) -> bool:
        """True if at least one consumer heartbeat of this pool is fresh."""
        for index in range(self.size):
            if await self.redis.exists(f"{HEARTBEAT_KEY_PREFIX}{self._consumer_name(index)}"):
                return True
        return False

    def status(self) -> Dict[str, Any]:
        """Snapshot for readiness probes and metrics."""
        return {
            "instance_id": self.instance_id,
            "consumers": self.size,
            "running_tasks": sum(1 for t in self._tasks if not t.done()),
            "in_flight": self._in_flight,
            "processed": self._processed,
        }
