"""Delegation runtime: the orchestrator hands typed tasks to subagents, synchronously or not.

Every delegation is bounded (per-agent semaphore, deadline), cancellable with its session,
recorded as `delegation_started` / `delegation_completed` events, and persisted as an artifact.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from core.runtime.sanitize import sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType

logger = logging.getLogger(__name__)

AgentFn = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


class DelegationError(RuntimeError):
    """Raised for unknown agents or forbidden delegation modes."""


@dataclass
class DelegationHandle:
    """A running or finished delegation."""

    delegation_id: str
    agent: str
    session_id: str
    task: "asyncio.Task[Dict[str, Any]]"
    started_at: float = field(default_factory=time.time)

    async def result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for the result; None on timeout, failure or cancellation (never raises)."""
        try:
            return await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None
        except Exception:
            return None

    @property
    def done(self) -> bool:
        return self.task.done()


class DelegationRuntime:
    """Registry and executor of subagents callable by the orchestrator."""

    def __init__(
        self,
        publisher: Optional[Any] = None,
        cancellations: Optional[Any] = None,
        workspace_resolver: Optional[Callable[[str], Any]] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self.publisher = publisher
        self.cancellations = cancellations
        self.workspace_resolver = workspace_resolver
        self.metrics = metrics
        self._agents: Dict[str, AgentFn] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._deadlines: Dict[str, float] = {}

    def register(self, agent: str, fn: AgentFn, *, concurrency: int = 8, default_deadline_ms: int = 8000) -> None:
        """Register a subagent callable `fn(session_id, task) -> result dict`."""
        self._agents[agent] = fn
        self._semaphores[agent] = asyncio.Semaphore(concurrency)
        self._deadlines[agent] = default_deadline_ms / 1000

    def registered(self) -> list:
        return sorted(self._agents)

    async def _publish(self, session_id: str, event_type: LoopEventType, payload: Dict[str, Any], turn_id: str) -> None:
        if self.publisher is None:
            return
        try:
            await self.publisher.publish(
                session_id,
                LoopEvent(
                    schema_version="1.0",
                    event_id=f"evt_{uuid.uuid4().hex[:16]}",
                    session_id=session_id,
                    turn_id=turn_id,
                    sequence=0,
                    timestamp=time.time(),
                    type=event_type,
                    payload=payload,
                ),
            )
        except Exception as e:
            logger.warning("Delegation event publish failed: %s", sanitize_secrets(str(e)))

    async def _run(
        self, delegation_id: str, agent: str, session_id: str, task: Dict[str, Any], deadline: float, turn_id: str
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        status, result, error = "completed", {}, None
        await self._publish(
            session_id, LoopEventType.DELEGATION_STARTED, {"delegation_id": delegation_id, "agent": agent}, turn_id
        )
        try:
            async with self._semaphores[agent]:
                result = await asyncio.wait_for(self._agents[agent](session_id, task), timeout=deadline)
        except asyncio.TimeoutError:
            status, error = "timeout", f"exceeded {deadline}s"
        except asyncio.CancelledError:
            status, error = "cancelled", "cancelled"
            raise
        except Exception as e:
            status, error = "failed", sanitize_secrets(str(e))
        finally:
            duration = time.perf_counter() - started
            if self.metrics is not None and status != "completed":
                self.metrics.inc("degraded_total", 1.0, {"component": f"delegation:{agent}:{status}"})
            await self._publish(
                session_id,
                LoopEventType.DELEGATION_COMPLETED,
                {"delegation_id": delegation_id, "agent": agent, "status": status, "duration_ms": round(duration * 1000)},
                turn_id,
            )
            self._persist(session_id, delegation_id, agent, status, result, error, duration)
        if status != "completed":
            raise RuntimeError(f"delegation {agent} {status}: {error}")
        return result

    def _persist(
        self,
        session_id: str,
        delegation_id: str,
        agent: str,
        status: str,
        result: Dict[str, Any],
        error: Optional[str],
        duration: float,
    ) -> None:
        if self.workspace_resolver is None:
            return
        try:
            workspace = self.workspace_resolver(session_id)
            workspace.write_artifact(
                f"orchestration/delegation_{delegation_id}.json",
                artifact_type="delegation_result",
                payload={
                    "delegation_id": delegation_id,
                    "agent": agent,
                    "status": status,
                    "error": error,
                    "duration_ms": round(duration * 1000),
                    "result": result if status == "completed" else None,
                },
            )
        except Exception as e:
            logger.warning("Delegation artifact write failed: %s", sanitize_secrets(str(e)))

    async def delegate(
        self,
        agent: str,
        session_id: str,
        task: Dict[str, Any],
        *,
        mode: Literal["async", "sync"] = "async",
        deadline_ms: Optional[int] = None,
        turn_id: str = "orchestration",
        caller_profile: Optional[str] = None,
    ) -> DelegationHandle:
        """Start a delegation. `mode="sync"` waits for it; the speaker may never delegate synchronously."""
        if agent not in self._agents:
            raise DelegationError(f"unknown agent '{agent}'")
        if mode == "sync" and caller_profile == "speaker":
            raise DelegationError("the speaker may not delegate synchronously (hot-path latency contract)")
        deadline = (deadline_ms / 1000) if deadline_ms is not None else self._deadlines[agent]
        delegation_id = uuid.uuid4().hex[:12]
        task_handle = asyncio.create_task(
            self._run(delegation_id, agent, session_id, task, deadline, turn_id), name=f"delegate:{agent}"
        )
        if self.cancellations is not None:
            self.cancellations.register_task(session_id, task_handle)
        task_handle.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        handle = DelegationHandle(delegation_id=delegation_id, agent=agent, session_id=session_id, task=task_handle)
        if mode == "sync":
            await handle.result()
        return handle
