"""Hook bus for the agent runtime: lifecycle events with per-hook budgets and isolation.

Luồng sử dụng: Bus mở rộng vòng đời: runtime và tầng session phát HookContext, handler nhận dữ liệu sự kiện.
Hook blocking chạy tuần tự và có hạn thời gian; hook nền chạy bằng task riêng.
HookOutcome gom quyết định/lỗi để caller xử lý; lỗi blocking ở BEFORE_TOOL chặn tool,
còn lỗi ở sự kiện khác chỉ được ghi log. drain hỗ trợ chờ task nền khi kết thúc.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from core.runtime.sanitize import sanitize_secrets

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Points in the runtime and session lifecycle where hooks run."""

    TURN_START = "turn_start"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_COMPACTION = "before_compaction"
    AFTER_COMPACTION = "after_compaction"
    TURN_END = "turn_end"
    STAGE_TRANSITION = "stage_transition"
    DIALOGUE_COMPLETED = "dialogue_completed"
    ASSESSMENT_COMPLETED = "assessment_completed"
    REPORT_READY = "report_ready"


@dataclass
class HookContext:
    """Data passed to hooks. `data` is event specific; `extra` may be mutated by blocking hooks."""

    event: HookEvent
    session_id: str = ""
    turn_id: str = ""
    agent: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookDecision:
    """Optional return value of a blocking hook. `deny=True` on BEFORE_TOOL blocks the tool."""

    deny: bool = False
    reason: str = ""


@dataclass
class HookOutcome:
    """Aggregate result of emitting one event."""

    denied: bool = False
    reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


HookFn = Callable[[HookContext], Union[Awaitable[Optional[HookDecision]], Optional[HookDecision]]]


@dataclass
class _Registration:
    name: str
    fn: HookFn
    blocking: bool
    budget_ms: int


class HookBus:
    """Registry and dispatcher of runtime hooks.

    Semantics:
        - Blocking hooks run inline, in registration order, each under its `budget_ms`.
        - Non-blocking hooks run as background tasks and never delay the caller.
        - A failing or timed-out hook is isolated and logged. On BEFORE_TOOL a failure or
          timeout of a blocking hook denies the tool (fail-closed); every other event fails open.
    """

    FAIL_CLOSED_EVENTS = frozenset({HookEvent.BEFORE_TOOL})

    def __init__(self) -> None:
        self._hooks: Dict[HookEvent, List[_Registration]] = {}
        self._background: set = set()

    def register(
        self,
        event: HookEvent,
        fn: HookFn,
        *,
        blocking: bool = False,
        budget_ms: int = 20,
        name: Optional[str] = None,
    ) -> None:
        """Register a hook for an event."""
        self._hooks.setdefault(event, []).append(
            _Registration(name=name or getattr(fn, "__name__", "hook"), fn=fn, blocking=blocking, budget_ms=budget_ms)
        )

    def registered(self, event: HookEvent) -> List[str]:
        """Names of hooks registered for an event (in order)."""
        return [r.name for r in self._hooks.get(event, [])]

    async def _invoke(self, registration: _Registration, ctx: HookContext) -> Optional[HookDecision]:
        result = registration.fn(ctx)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        return result if isinstance(result, HookDecision) else None

    async def emit(self, event: HookEvent, ctx: HookContext) -> HookOutcome:
        """Run every hook registered for `event`."""
        outcome = HookOutcome()
        started = time.perf_counter()
        for registration in self._hooks.get(event, []):
            if not registration.blocking:
                # Task nền không có quyền chặn caller; tập tham chiếu giữ task sống tới khi hoàn tất.
                task = asyncio.create_task(self._run_background(registration, ctx))
                self._background.add(task)
                task.add_done_callback(self._background.discard)
                continue
            try:
                decision = await asyncio.wait_for(
                    self._invoke(registration, ctx), timeout=registration.budget_ms / 1000
                )
                if decision is not None and decision.deny:
                    outcome.denied = True
                    outcome.reasons.append(decision.reason or registration.name)
            except asyncio.TimeoutError:
                outcome.errors.append(f"{registration.name}: timeout")
                # BEFORE_TOOL là chốt thực thi: hook blocking lỗi thì không cho tool chạy.
                if event in self.FAIL_CLOSED_EVENTS:
                    outcome.denied = True
                    outcome.reasons.append(f"{registration.name} timed out")
                logger.warning("Hook %s timed out on %s", registration.name, event.value)
            except Exception as e:
                outcome.errors.append(f"{registration.name}: {sanitize_secrets(str(e))}")
                if event in self.FAIL_CLOSED_EVENTS:
                    outcome.denied = True
                    outcome.reasons.append(f"{registration.name} failed")
                logger.warning("Hook %s failed on %s: %s", registration.name, event.value, sanitize_secrets(str(e)))
        outcome.duration_ms = (time.perf_counter() - started) * 1000
        return outcome

    async def _run_background(self, registration: _Registration, ctx: HookContext) -> None:
        try:
            await self._invoke(registration, ctx)
        except Exception as e:
            logger.warning(
                "Background hook %s failed on %s: %s", registration.name, ctx.event.value, sanitize_secrets(str(e))
            )

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for background hooks (tests and shutdown)."""
        if self._background:
            await asyncio.wait(list(self._background), timeout=timeout)
