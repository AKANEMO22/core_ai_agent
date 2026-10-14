"""Dialogue completion → assessment barrier → report synthesis pipeline.

Nối sự kiện kết thúc hội thoại với việc chờ đánh giá và tạo báo cáo.
AssessmentPipeline chốt số job cần hoàn tất, chuyển phiên sang ASSESSING và
chỉ gọi ReportSynthesizer khi barrier đủ; callback worker và sweep đều có thể gọi lại.
Lease theo phiên giới hạn các lời gọi tổng hợp đồng thời.
"""

import logging
from typing import Any, Awaitable, Callable, Optional

from core.assessor.queue import enqueued_assessments_key, expected_assessments_key
from core.assessor.synthesizer import ReportSynthesizer
from core.runtime.agent import sanitize_secrets
from core.session.idempotency import RedisLease
from core.session.state import InvalidStateTransitionError, LifecycleStatus, SessionStateStore

logger = logging.getLogger(__name__)

SynthesizeHook = Callable[[str], Awaitable[Any]]


class AssessmentPipeline:
    """Drives a session from DIALOGUE_COMPLETED to REPORT_READY without manual wiring.

    - `on_dialogue_completed`: freezes the expected assessment count from the set of enqueued
      jobs, records it in Redis and in state metadata, and transitions to ASSESSING.
    - `maybe_synthesize`: under a per-session lease, checks the barrier and synthesizes the
      report exactly once; a permanently failed assessment moves the session to
      ASSESSMENT_FAILED instead.
    - `sweep`: recovery for crashes between any of these steps.

    Every method is idempotent and safe to call concurrently from workers, the gateway and
    the periodic sweeper.
    """

    def __init__(
        self,
        redis_client: Any,
        state_store: SessionStateStore,
        synthesizer: ReportSynthesizer,
        counter_ttl_seconds: int = 7 * 24 * 3600,
        synthesis_lock_ttl_seconds: int = 120,
    ) -> None:
        self.redis = redis_client
        self.state_store = state_store
        self.synthesizer = synthesizer
        self.counter_ttl_seconds = counter_ttl_seconds
        self.synthesis_lock_ttl_seconds = synthesis_lock_ttl_seconds
        self._lease = RedisLease(redis_client)

    async def on_dialogue_completed(self, session_id: str) -> None:
        """Freeze the expected count and move DIALOGUE_COMPLETED → ASSESSING, then try synthesis."""
        state = await self.state_store.aget(session_id)
        if state is None:
            return
        if state.status == LifecycleStatus.DIALOGUE_COMPLETED:
            # Đếm job_id duy nhất lúc hội thoại đóng; retry không làm tăng số assessment cần chờ.
            expected = int(await self.redis.scard(enqueued_assessments_key(session_id)))
            await self.redis.set(
                expected_assessments_key(session_id), expected, ex=self.counter_ttl_seconds
            )
            try:
                await self.state_store.atransition(
                    state,
                    expected_version=state.state_version,
                    new_status=LifecycleStatus.ASSESSING,
                    metadata={"expected_assessments": expected},
                )
            except InvalidStateTransitionError:
                # Another caller won the race; the state is now ASSESSING or later.
                pass
        await self.maybe_synthesize(session_id)

    async def maybe_synthesize(self, session_id: str) -> bool:
        """Synthesize the report if the barrier is satisfied. Returns True if a report was written."""
        lock_key = f"session:{session_id}:synth_lock"
        # Worker hoàn tất và sweeper có thể cùng gọi; chỉ người giữ lease đi qua barrier.
        token = await self._lease.acquire(lock_key, self.synthesis_lock_ttl_seconds * 1000)
        if token is None:
            return False
        try:
            state = await self.state_store.aget(session_id)
            if state is None or state.status != LifecycleStatus.ASSESSING:
                return False

            # Có job lỗi vĩnh viễn thì chuyển trạng thái lỗi; chưa đủ kết quả thì đợi callback/sweep sau.
            is_ready, completed, failed, expected = await self.synthesizer.check_barrier(session_id)
            if failed > 0:
                try:
                    await self.state_store.atransition(
                        state,
                        expected_version=state.state_version,
                        new_status=LifecycleStatus.ASSESSMENT_FAILED,
                        metadata={"failure_code": "ASSESSMENT_EXHAUSTED"},
                    )
                except InvalidStateTransitionError:
                    pass
                return False
            if not is_ready:
                return False

            await self.synthesizer.synthesize(session_id)
            return True
        except Exception as e:
            logger.error("Report synthesis failed: %s", sanitize_secrets(str(e)))
            return False
        finally:
            await self._lease.release(lock_key, token)

    async def requeue_failed(self, session_id: str, queue: Any) -> int:
        """Operator action: requeue dead-lettered jobs and move ASSESSMENT_FAILED → ASSESSING."""
        requeued = await queue.requeue_dead_letters(session_id)
        state = await self.state_store.aget(session_id)
        if state is not None and state.status == LifecycleStatus.ASSESSMENT_FAILED:
            await self.state_store.atransition(
                state,
                expected_version=state.state_version,
                new_status=LifecycleStatus.ASSESSING,
                metadata={"failure_code": None, "requeued_jobs": len(requeued)},
            )
        await self.maybe_synthesize(session_id)
        return len(requeued)

    async def sweep(self) -> int:
        """Advance every session stuck in DIALOGUE_COMPLETED or ASSESSING. Returns sessions touched."""
        touched = 0
        for session_id in await self.state_store.alist_by_status(LifecycleStatus.DIALOGUE_COMPLETED):
            await self.on_dialogue_completed(session_id)
            touched += 1
        for session_id in await self.state_store.alist_by_status(LifecycleStatus.ASSESSING):
            await self.maybe_synthesize(session_id)
            touched += 1
        return touched
