"""Hot-path priority: background LLM calls yield while an interviewer turn is generating.

Every answer triggers the speaker and, at the same moment, background work (evaluation, summary,
orchestration). On a shared provider those calls compete with the speaker exactly when time to
first token matters. Background calls therefore wait (bounded) until no speaker generation is in
flight before starting; calls already streaming are not interrupted.


Luồng sử dụng: Các wrapper provider giúp ưu tiên câu nói trực tiếp của interviewer khi có nhiều agent chạy.
API tạo HotPathGate chung; speaker đánh dấu đang sinh câu, BackgroundLLMClient chờ gate
trong thời gian hữu hạn trước khi gọi model. HedgedLLMClient có thể mở request thứ hai
nếu chunk đầu chậm, chọn stream thắng và đóng stream còn lại; đầu ra vẫn là IterationChunk.
"""

import asyncio
import contextlib
from typing import Any, AsyncIterator, Dict, List, Optional

from core.runtime.types import IterationChunk


class HotPathGate:
    """Counts in-flight speaker generations; background callers wait for zero."""

    def __init__(self) -> None:
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return self._active

    @contextlib.asynccontextmanager
    async def hot(self) -> AsyncIterator[None]:
        """Mark a speaker generation as in flight."""
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    async def wait_idle(self, max_wait_seconds: float) -> bool:
        """Wait until no speaker generation is in flight, at most `max_wait_seconds`. True if idle."""
        if self._active == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=max_wait_seconds)
            return True
        except asyncio.TimeoutError:
            return False


class BackgroundLLMClient:
    """Client wrapper for background agents: yields to the hot path before each call."""

    def __init__(self, inner: Any, gate: HotPathGate, max_wait_seconds: float = 3.0) -> None:
        self.inner = inner
        self.gate = gate
        self.max_wait_seconds = max_wait_seconds
        self.capabilities = inner.capabilities
        self.model = getattr(inner, "model", "")

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        # Hết thời gian chờ vẫn gọi model để tác vụ nền không bị bỏ đói khi speaker chạy liên tục.
        await self.gate.wait_idle(self.max_wait_seconds)
        async for chunk in self.inner.stream_chat(messages=messages, tools=tools, **kwargs):
            yield chunk


async def _discard(task: "asyncio.Future[Any]", stream: Any) -> None:
    """Cancel a losing request and close its stream (stops provider-side generation)."""
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    with contextlib.suppress(BaseException):
        await stream.aclose()


class HedgedLLMClient:
    """Tail-latency hedge for the hot path.

    If the first chunk has not arrived after `hedge_after_seconds`, an identical second request is
    started and whichever produces a first chunk first is streamed; the other is cancelled and its
    stream closed. Shared gateways have long latency tails (queueing behind other calls), and a
    hedge cuts p95/p99 time to first token for at most one extra request on slow turns only.
    """

    def __init__(self, inner: Any, hedge_after_seconds: float, metrics: Optional[Any] = None) -> None:
        self.inner = inner
        self.hedge_after_seconds = hedge_after_seconds
        self.metrics = metrics
        self.capabilities = inner.capabilities
        self.model = getattr(inner, "model", "")

    def _count(self, outcome: str) -> None:
        if self.metrics is not None:
            self.metrics.inc("speaker_hedge_total", 1.0, {"outcome": outcome})

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        if self.hedge_after_seconds <= 0:
            async for chunk in self.inner.stream_chat(messages=messages, tools=tools, **kwargs):
                yield chunk
            return

        def start() -> tuple:
            stream = self.inner.stream_chat(messages=messages, tools=tools, **kwargs)
            return stream, asyncio.ensure_future(stream.__anext__())

        # Chỉ tranh chunk đầu; chọn xong sẽ tiếp tục đúng stream đó thay vì ghép nội dung hai request.
        racers = [start()]
        winner: Optional[tuple] = None
        try:
            done, _ = await asyncio.wait({racers[0][1]}, timeout=self.hedge_after_seconds)
            if not done:
                racers.append(start())
            # The first racer to produce a chunk wins; a racer that fails or ends empty yields to
            # the other. If every racer fails, the last error propagates.
            pending = {task for _, task in racers}
            error: Optional[BaseException] = None
            while pending and winner is None:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for racer in racers:
                    if racer[1] in done and winner is None:
                        exc = racer[1].exception()
                        if exc is None:
                            winner = racer
                        else:
                            error = exc
            if winner is None:
                if error is None or isinstance(error, StopAsyncIteration):
                    return
                raise error
            if len(racers) > 1:
                self._count("hedge_won" if winner is racers[1] else "primary_won")
            for racer in racers:
                if racer is not winner:
                    await _discard(racer[1], racer[0])
            stream, first = winner
            yield first.result()
            async for chunk in stream:
                yield chunk
        finally:
            # Runs on completion, error and consumer cancellation alike; closing is idempotent.
            for stream, task in racers:
                if winner is not None and stream is winner[0]:
                    with contextlib.suppress(BaseException):
                        await stream.aclose()
                else:
                    await _discard(task, stream)
