"""Trace actual core invocations with durable parentage, including queue boundaries."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import time
from typing import Any, Callable, Optional

from fe_inspect.store import TraceStore, sanitize_payload

_current: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("core_trace", default=None)


class RuntimeTraceRecorder:
    """SQLite is authoritative; context variables carry only the active span identity."""

    def __init__(self, workspace_mgr: Any) -> None:
        self.workspace_mgr = workspace_mgr
        self.store = TraceStore(workspace_mgr.root_base)
        self.provider_metadata = {}

    def read(self, session_id: str) -> list[dict]:
        return self.store.read_spans(session_id)["spans"]

    @staticmethod
    def _arguments(fn: Callable, args: tuple, kwargs: dict) -> dict:
        return dict(inspect.signature(fn).bind_partial(*args, **kwargs).arguments)

    @staticmethod
    def _session(values: dict, owner: Any = None) -> Optional[str]:
        if values.get("session_id"):
            return str(values["session_id"])
        for value in [values.get("workspace"), values.get("state"), getattr(owner, "workspace", None)]:
            if getattr(value, "session_id", None):
                return str(value.session_id)
        job = values.get("job_data") or values.get("job")
        if isinstance(job, dict) and job.get("session_id"):
            return str(job["session_id"])
        context = _current.get()
        return context["session_id"] if context else None

    @staticmethod
    def _input(values: dict) -> dict:
        output = {}
        for key, value in values.items():
            if key in {"cancellation_token", "coro"}:
                continue
            if getattr(value, "session_id", None) and hasattr(value, "root_path"):
                output[key] = {"session_id": value.session_id, "workspace": str(value.root_path)}
            else:
                output[key] = sanitize_payload(value)
        return output

    @staticmethod
    def _result_status(result: Any) -> tuple[str, Optional[str]]:
        status = result if isinstance(result, str) else result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
        status = str(getattr(status, "value", status) or "").lower()
        if status in {"failed", "timeout", "cancelled", "error", "rejected"}:
            error = result.get("error") if isinstance(result, dict) else getattr(result, "error_code", None)
            return status if status not in {"error", "rejected"} else "failed", str(error or status)
        if status in {"interrupted", "fallback", "degraded"}:
            return "degraded", None
        return "completed", None

    def bind_agent(self, owner: Any, agent: str, lifetime_scope: str = "application", session_id: str = None) -> dict:
        instance = self.store.register_instance(agent, type(owner).__name__, lifetime_scope, session_id)
        owner._inspect_instance_id = instance["instance_id"]
        owner._inspect_agent = agent
        owner._inspect_lifetime_scope = lifetime_scope
        return instance

    def wrap(self, owner: Any, method: str, name: str = None, kind: str = "operation", agent: str = None) -> None:
        original = getattr(owner, method)
        if getattr(original, "_core_trace_wrapped", False):
            return
        display_name = name or f"{type(owner).__name__}.{method}"

        async def start(values: dict) -> Optional[dict]:
            session_id = self._session(values, owner)
            if not session_id:
                return None
            current = _current.get()
            job = values.get("job_data") or {}
            parent = job.get("inspect_trace_parent_id") if isinstance(job, dict) else None
            if not parent and current and current["session_id"] == session_id:
                parent = current["span_id"]
            role = values.get("agent") or agent or getattr(owner, "_inspect_agent", None) or (current or {}).get("agent")
            attributes = {"method": method, "class": type(owner).__name__}
            for key in ("turn_id", "turn_number", "through_turn", "delegation_id", "job_id"):
                if key in values:
                    attributes[key] = sanitize_payload(values[key])
                elif isinstance(job, dict) and key in job:
                    attributes[key] = sanitize_payload(job[key])
            if kind == "llm":
                attributes["model"] = values.get("kwargs", {}).get("model") or getattr(owner, "model", None)
                attributes["provider"] = type(owner).__name__
            creation = asyncio.create_task(asyncio.to_thread(
                self.store.start_span, session_id, display_name, kind,
                parent_span_id=parent, agent=role,
                instance_id=getattr(owner, "_inspect_instance_id", None),
                lifetime_scope=getattr(owner, "_inspect_lifetime_scope", None),
                inputs=self._input(values), attributes=attributes,
            ))
            try:
                return await asyncio.shield(creation)
            except asyncio.CancelledError:
                # The filesystem thread may still commit after cancellation. Await its
                # identity and close it instead of leaving a false running span.
                created = await creation
                await asyncio.to_thread(self.store.finish_span, created["span_id"], "cancelled", error="Cancelled while recording invocation start")
                raise

        def context_for(span: dict) -> dict:
            return {"span_id": span["span_id"], "session_id": span["session_id"], "agent": span.get("agent")}

        @functools.wraps(original)
        async def call(*args: Any, **kwargs: Any) -> Any:
            values = self._arguments(original, args, kwargs)
            span = await start(values)
            if span is None:
                return await original(*args, **kwargs)
            token = _current.set(context_for(span))
            try:
                if method == "enqueue" and isinstance(values.get("job_data"), dict):
                    job = dict(values["job_data"], inspect_trace_parent_id=span["span_id"])
                    if "job_data" in kwargs:
                        kwargs = {**kwargs, "job_data": job}
                    else:
                        args = (job, *args[1:])
                result = await original(*args, **kwargs)
                status, error = self._result_status(result)
                if method == "_process_job_data" and result is False:
                    status, error = "degraded", "Assessment did not complete; consult retry/dead-letter events"
                output = result
                if hasattr(result, "delegation_id"):
                    output = {"delegation_id": result.delegation_id, "agent": result.agent, "started_at": result.started_at}
                await asyncio.to_thread(self.store.finish_span, span["span_id"], status, outputs=sanitize_payload(output), error=error)
                return result
            except BaseException as exc:
                status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "timeout" if isinstance(exc, TimeoutError) else "failed"
                await asyncio.to_thread(self.store.finish_span, span["span_id"], status, error=sanitize_payload(str(exc) or type(exc).__name__))
                raise
            finally:
                _current.reset(token)

        @functools.wraps(original)
        async def streaming(*args: Any, **kwargs: Any):
            values = self._arguments(original, args, kwargs)
            span = await start(values)
            iterator = original(*args, **kwargs).__aiter__()
            if span is None:
                async for item in iterator:
                    yield item
                return
            output = {"text": "", "usage": {}, "tool_calls": []}
            status, error = "completed", None
            last_saved = time.monotonic()
            try:
                while True:
                    # Restore context per __anext__: wait_for can consume the iterator
                    # from different tasks. No ContextVar token may span a yield.
                    token = _current.set(context_for(span))
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        _current.reset(token)
                    if getattr(item, "token", None):
                        if len(output["text"]) + len(item.token) > 262144:
                            output["text_truncated"] = True
                            output["text_character_limit"] = 262144
                        output["text"] = (output["text"] + item.token)[:262144]
                    if getattr(item, "usage", None):
                        output["usage"].update(item.usage)
                    if getattr(item, "tool_calls", None):
                        output["tool_calls"].extend(item.tool_calls)
                    event_type = getattr(getattr(item, "type", None), "value", None)
                    if event_type == "error":
                        output["error_event"] = sanitize_payload(item.payload)
                        error = str(item.payload.get("error") or item.payload.get("code") or "Core error")
                        status = "cancelled" if item.payload.get("code") == "CANCELLED" else "failed"
                    elif event_type == "complete":
                        output.update(sanitize_payload(item.payload))
                    if time.monotonic() - last_saved >= .15:
                        await asyncio.to_thread(self.store.update_output, span["span_id"], output)
                        last_saved = time.monotonic()
                    yield item
            except BaseException as exc:
                status = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "timeout" if isinstance(exc, TimeoutError) else "failed"
                error = sanitize_payload(str(exc) or type(exc).__name__)
                raise
            finally:
                token = _current.set(context_for(span))
                try:
                    close = getattr(iterator, "aclose", None)
                    if close:
                        await close()
                finally:
                    _current.reset(token)
                    await asyncio.to_thread(self.store.finish_span, span["span_id"], status, outputs=output, error=error)

        wrapper = streaming if inspect.isasyncgenfunction(original) or method == "stream_chat" else call
        wrapper._core_trace_wrapped = True
        setattr(owner, method, wrapper)

    def traceable(self, *, name: str = None, kind: str = "operation", agent: str = None):
        """Decorator for backend extensions using the same durable trace context."""
        def decorate(fn):
            class Adapter:
                pass
            adapter = Adapter()
            adapter.invoke = fn
            self.wrap(adapter, "invoke", name or fn.__name__, kind, agent)
            return adapter.invoke
        return decorate

    def prepare_task(self, session_id: str, coro: Any, name: str, attributes: dict = None):
        current = _current.get()
        parent = current["span_id"] if current and current["session_id"] == session_id else None
        span = self.store.schedule_span(session_id, name, parent_span_id=parent, agent=(current or {}).get("agent"), attributes=attributes)

        async def execute():
            await asyncio.to_thread(self.store.begin_span, span["span_id"])
            token = _current.set({"session_id": session_id, "span_id": span["span_id"], "agent": span.get("agent")})
            try:
                result = await coro
                status, error = self._result_status(result)
                await asyncio.to_thread(self.store.finish_span, span["span_id"], status, outputs=sanitize_payload(result), error=error)
                return result
            except BaseException as exc:
                await asyncio.to_thread(self.store.finish_span, span["span_id"], "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", error=sanitize_payload(str(exc) or type(exc).__name__))
                raise
            finally:
                _current.reset(token)

        def done(task):
            if task.cancelled():
                self.store.finish_span(span["span_id"], "cancelled", error="Task cancelled")
                if inspect.iscoroutine(coro) and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
                    coro.close()
        return execute(), done

    def spawn_task(self, session_id: str, coro: Any, name: str, attributes: dict = None):
        wrapped, done = self.prepare_task(session_id, coro, name, attributes)
        task = asyncio.create_task(wrapped, name=name)
        task.add_done_callback(done)
        return task

    def instrument_controller(self, controller: Any) -> None:
        self.wrap(controller, "start_interview", "Start interview")
        self.wrap(controller, "submit_answer", "Candidate answer")
        original_spawn = controller._spawn
        def spawn(session_id, coro, name):
            wrapped, done = self.prepare_task(session_id, coro, name)
            task = original_spawn(session_id, wrapped, name)
            task.add_done_callback(done)
            return task
        controller._spawn = spawn

    def instrument_events(self, event_store: Any) -> None:
        original = event_store.append_async
        async def append(session_id, event):
            event = await original(session_id, event)
            await asyncio.to_thread(self.store.save_event, event.model_dump(mode="json"))
            return event
        event_store.append_async = append
        original_sync = event_store.append
        def append_sync(session_id, event):
            event = original_sync(session_id, event)
            self.store.save_event(event.model_dump(mode="json"))
            return event
        event_store.append = append_sync
        original_purge = event_store.purge
        async def purge(session_id):
            await original_purge(session_id)
            await asyncio.to_thread(self.store.purge_session, session_id)
        event_store.purge = purge

    def instrument_state(self, state_store: Any) -> None:
        for method in ("acreate", "atransition"):
            original = getattr(state_store, method)
            def make_wrapper(fn):
                @functools.wraps(fn)
                async def save(*args, **kwargs):
                    state = await fn(*args, **kwargs)
                    snapshot = dict(state.to_dict(), _inspect_provider=self.provider_metadata)
                    await asyncio.to_thread(self.store.save_state, snapshot)
                    return state
                return save
            setattr(state_store, method, make_wrapper(original))

    def instrument_registry(self, registry: Any) -> None:
        original_get, original_evict = registry.get, registry.evict
        async def get(session_id):
            before = dict(registry._runners)
            runner = await original_get(session_id)
            for key, previous in before.items():
                if key not in registry._runners:
                    self.release_runner(previous, "lru_eviction")
            return runner
        def evict(session_id):
            runner = registry._runners.get(session_id)
            original_evict(session_id)
            if runner:
                self.release_runner(runner, "session_runner_released")
        registry.get, registry.evict = get, evict

    def release_runner(self, runner, reason):
        instance_id = getattr(runner.runtime_agent, "_inspect_instance_id", None)
        if instance_id:
            self.store.end_instance(instance_id, reason)

    async def close(self) -> None:
        await asyncio.to_thread(self.store.close_process)
