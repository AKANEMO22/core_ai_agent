"""Dialogue completion → assessment barrier → report synthesis pipeline.

Nối sự kiện kết thúc hội thoại với việc chờ đánh giá và tạo báo cáo.
AssessmentPipeline chốt số job cần hoàn tất, chuyển phiên sang ASSESSING và
chỉ gọi ReportSynthesizer khi barrier đủ; callback worker và sweep đều có thể gọi lại.
Lease theo phiên giới hạn các lời gọi tổng hợp đồng thời.
"""

import logging
import asyncio
import time
from typing import Any, Awaitable, Callable, Optional

from core.assessor.queue import enqueued_assessments_key, expected_assessments_key
from core.assessor.synthesizer import ReportSynthesizer
from core.runtime.agent import sanitize_secrets
from core.session.idempotency import RedisLease
from core.session.state import (
    TERMINAL_STATUSES,
    InvalidStateTransitionError,
    LifecycleStatus,
    SessionStateStore,
)

logger = logging.getLogger(__name__)

SynthesizeHook = Callable[[str], Awaitable[Any]]
SessionHook = Callable[[str], Awaitable[Any]]

# Bounded CAS retries when concurrent turns keep bumping the version during /end.
END_TRANSITION_ATTEMPTS = 20


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
        max_end_attempts: int = 5,
    ) -> None:
        self.redis = redis_client
        self.state_store = state_store
        self.synthesizer = synthesizer
        self.counter_ttl_seconds = counter_ttl_seconds
        self.synthesis_lock_ttl_seconds = synthesis_lock_ttl_seconds
        self.max_end_attempts = max(1, int(max_end_attempts))
        self._lease = RedisLease(redis_client)
        # Optional hooks wired by the app: stop session work before an early-end report, and
        # release per-process session state once the end reached a final status.
        self.end_prepare: Optional[SessionHook] = None
        self.end_finished: Optional[SessionHook] = None

    async def end_session(self, session_id: str) -> bool:
        """Persist intent before cancellation; retries/sweeps finish an interrupted end."""
        for _ in range(END_TRANSITION_ATTEMPTS):
            state = await self.state_store.aget(session_id)
            if state is None or state.status == LifecycleStatus.REPORT_READY:
                return state is not None
            if state.status == LifecycleStatus.ENDING:
                break
            if state.status in TERMINAL_STATUSES:
                raise InvalidStateTransitionError(f"Cannot end session in status {state.status.value}")
            try:
                await self.state_store.atransition(state, state.state_version, LifecycleStatus.ENDING,
                    metadata={"early_ended": True, "end_requested_at": time.time(), "end_attempts": 0})
                break
            except InvalidStateTransitionError:
                latest = await self.state_store.aget(session_id)
                # Same snapshot means the FSM refused the transition; anything else is a CAS race
                # (e.g. an in-flight turn bumped the version) and is retried against fresh state.
                if (latest is not None and latest.state_version == state.state_version
                        and latest.status == state.status):
                    raise
        else:
            raise InvalidStateTransitionError(f"Concurrent state changes kept ending session '{session_id}' from persisting")
        # The fence must outlive delayed/redelivered work; it expires with the session state.
        await self._set_work_stopped(session_id)
        return await self.finish_ending(session_id)

    async def stop_work(self, session_id: str) -> None:
        """Durable cross-process fence: queued, retried and in-flight work for the session is refused."""
        await self._set_work_stopped(session_id)

    async def release_session(self, session_id: str) -> None:
        """Drop per-process cancellation markers once a session is terminal."""
        await self._end_finished(session_id)

    async def _set_work_stopped(self, session_id: str) -> None:
        await self.redis.set(f"session:{session_id}:work_stopped", "1", ex=self.counter_ttl_seconds)

    async def finish_ending(self, session_id: str) -> bool:
        key = f"session:{session_id}:synth_lock"
        token = await self._lease.acquire(key, self.synthesis_lock_ttl_seconds * 1000)
        if token is None:
            return False
        lease_lost = asyncio.Event()
        work = asyncio.create_task(self._finish_ending_locked(session_id))
        renewal = asyncio.create_task(self._renew_end_lease(key, token, work, lease_lost))
        try:
            return await work
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if lease_lost.is_set() and not (current is not None and current.cancelling()):
                # Another process may now hold the lease; it owns the synthesis from here.
                logger.warning("Early-end synthesis stopped: lease lost for session")
                return False
            raise
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            await self._lease.release(key, token)

    async def _finish_ending_locked(self, session_id: str) -> bool:
        state = await self.state_store.aget(session_id)
        if state is None or state.status != LifecycleStatus.ENDING:
            done = bool(state and state.status == LifecycleStatus.REPORT_READY)
            if state is not None and state.status in (LifecycleStatus.REPORT_READY, LifecycleStatus.ASSESSMENT_FAILED):
                await self._end_finished(session_id)
            return done
        attempts = int(state.metadata.get("end_attempts") or 0)
        if attempts >= self.max_end_attempts:
            # Bounded retries: a report that keeps failing goes to operator review instead of
            # leaving the session in ENDING forever.
            try:
                await self.state_store.atransition(
                    state, expected_version=state.state_version,
                    new_status=LifecycleStatus.ASSESSMENT_FAILED,
                    metadata={"failure_code": "EARLY_END_REPORT_FAILED"},
                )
            except InvalidStateTransitionError:
                return False
            await self._end_finished(session_id)
            return False
        await self.state_store.atransition(state, expected_version=state.state_version,
            metadata={"end_attempts": attempts + 1})
        await self._set_work_stopped(session_id)
        if self.end_prepare is not None:
            await self.end_prepare(session_id)
        await self.synthesizer.synthesize(session_id, early_end=True)
        await self._end_finished(session_id)
        return True

    async def _end_finished(self, session_id: str) -> None:
        if self.end_finished is None:
            return
        try:
            await self.end_finished(session_id)
        except Exception as exc:
            logger.warning("Early-end cleanup failed: %s", sanitize_secrets(str(exc)))

    async def _renew_end_lease(self, key: str, token: str, work: "asyncio.Task[Any]", lost: asyncio.Event) -> None:
        while True:
            await asyncio.sleep(self.synthesis_lock_ttl_seconds / 3)
            try:
                renewed = await self._lease.renew(key, token, self.synthesis_lock_ttl_seconds * 1000)
            except Exception as exc:
                logger.warning("Early-end lease renewal failed: %s", sanitize_secrets(str(exc)))
                renewed = False
            if not renewed:
                lost.set()
                work.cancel()
                return

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
        state = await self.state_store.aget(session_id)
        if (state is not None and state.status == LifecycleStatus.ASSESSMENT_FAILED
                and state.metadata.get("early_ended")):
            # The work fence refuses requeued jobs after an early end; retry the end report instead.
            await self.state_store.atransition(
                state,
                expected_version=state.state_version,
                new_status=LifecycleStatus.ENDING,
                metadata={"failure_code": None, "end_attempts": 0},
            )
            await self.finish_ending(session_id)
            return 0
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
        for session_id in await self.state_store.alist_by_status(LifecycleStatus.ENDING):
            try:
                await self.finish_ending(session_id)
            except Exception as exc:
                logger.error("Early-end report pending: %s", sanitize_secrets(str(exc)))
            touched += 1
        for session_id in await self.state_store.alist_by_status(LifecycleStatus.DIALOGUE_COMPLETED):
            await self.on_dialogue_completed(session_id)
            touched += 1
        for session_id in await self.state_store.alist_by_status(LifecycleStatus.ASSESSING):
            await self.maybe_synthesize(session_id)
            touched += 1
        return touched
