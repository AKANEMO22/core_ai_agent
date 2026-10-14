"""Regression checks for durable trace parentage across concurrent core task boundaries."""
import asyncio

import pytest

from core.runtime.types import IterationChunk
from core.workspace.session import SessionWorkspaceManager
from fe_inspect.instrumentation import RuntimeTraceRecorder


@pytest.fixture
def recorder(tmp_path):
    return RuntimeTraceRecorder(SessionWorkspaceManager(str(tmp_path)))


@pytest.mark.asyncio
async def test_decorator_parentage_isolated_between_concurrent_calls(recorder):
    @recorder.traceable(name="child")
    async def child(session_id, value):
        await asyncio.sleep(0)
        return value * 2

    @recorder.traceable(name="parent", kind="agent")
    async def parent(session_id, value):
        return await child(session_id, value)

    assert await asyncio.gather(parent("shared", 1), parent("shared", 2)) == [2, 4]
    spans = recorder.read("shared")
    parents = {span["span_id"]: span for span in spans if span["name"] == "parent"}
    children = [span for span in spans if span["name"] == "child"]
    assert len(parents) == len(children) == 2
    for child_span in children:
        parent_span = parents[child_span["parent_span_id"]]
        assert parent_span["inputs"]["value"] == child_span["inputs"]["value"]
        assert child_span["status"] == parent_span["status"] == "completed"


@pytest.mark.asyncio
async def test_stream_consumed_by_different_tasks_keeps_nested_parent(recorder):
    @recorder.traceable(name="nested")
    async def nested(session_id):
        return {"result": "observed"}

    @recorder.traceable(name="stream", kind="llm")
    async def stream(session_id):
        await nested(session_id)
        yield IterationChunk(token="first")
        await nested(session_id)
        yield IterationChunk(token="second", usage={"total_tokens": 2})

    iterator = stream("streamed")
    assert (await asyncio.wait_for(anext(iterator), 1)).token == "first"
    assert (await asyncio.wait_for(anext(iterator), 1)).token == "second"
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(iterator), 1)
    spans = recorder.read("streamed")
    outer = next(span for span in spans if span["name"] == "stream")
    assert outer["outputs"]["text"] == "firstsecond"
    assert outer["outputs"]["usage"]["total_tokens"] == 2
    assert all(span["parent_span_id"] == outer["span_id"] for span in spans if span["name"] == "nested")


@pytest.mark.asyncio
async def test_error_and_cancelled_generator_finish_without_turn_end_hook(recorder):
    @recorder.traceable(name="failing")
    async def failing(session_id):
        yield IterationChunk(token="partial")
        raise RuntimeError("provider disconnected")

    iterator = failing("failure")
    await anext(iterator)
    with pytest.raises(RuntimeError, match="provider disconnected"):
        await anext(iterator)
    failed = next(span for span in recorder.read("failure") if span["name"] == "failing")
    assert failed["status"] == "failed"
    assert failed["ended_at"] is not None
    assert failed["outputs"]["text"] == "partial"

    @recorder.traceable(name="cancelled")
    async def cancelled(session_id):
        yield IterationChunk(token="partial")
        await asyncio.Event().wait()

    iterator = cancelled("cancel")
    await anext(iterator)
    await iterator.aclose()
    span = next(span for span in recorder.read("cancel") if span["name"] == "cancelled")
    assert span["status"] == "cancelled"
    assert span["ended_at"] is not None


@pytest.mark.asyncio
async def test_queue_parent_roundtrips_into_independent_worker_context(recorder):
    class Queue:
        async def enqueue(self, job_data):
            self.payload = dict(job_data)
            return "redis-job"

    class Worker:
        async def _process_job_data(self, group_name, message_id, job_data):
            return True

    queue, worker = Queue(), Worker()
    recorder.wrap(queue, "enqueue", "queue")
    recorder.wrap(worker, "_process_job_data", "worker", "task")
    await queue.enqueue({"session_id": "queued", "turn_id": "turn_001"})
    # Deliberately outside the producer ContextVar scope, like a Redis consumer.
    await worker._process_job_data("assessors", "redis-job", queue.payload)
    spans = recorder.read("queued")
    enqueue = next(span for span in spans if span["name"] == "queue")
    worker_span = next(span for span in spans if span["name"] == "worker")
    assert worker_span["parent_span_id"] == enqueue["span_id"]
    assert worker_span["attributes"]["turn_id"] == "turn_001"


@pytest.mark.asyncio
async def test_task_schedule_before_execution_and_cancellation_before_start(recorder):
    ran = False

    async def operation():
        nonlocal ran
        ran = True

    task = recorder.spawn_task("scheduled", operation(), "background")
    before = next(span for span in recorder.read("scheduled") if span["name"] == "background")
    assert before["status"] == "queued" and before["started_at"] is None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    after = next(span for span in recorder.read("scheduled") if span["name"] == "background")
    assert after["status"] == "cancelled" and after["started_at"] is None
    assert ran is False


@pytest.mark.asyncio
async def test_deleted_session_does_not_create_new_span(recorder):
    recorder.store.purge_session("deleted")

    @recorder.traceable(name="late")
    async def late(session_id):
        raise AssertionError("deleted session must not execute")

    with pytest.raises(ValueError):
        await late("deleted")
    assert recorder.read("deleted") == []


def test_rejections_and_returned_fallbacks_have_honest_status():
    from core.interview.controller import TurnRequestResult
    assert RuntimeTraceRecorder._result_status(TurnRequestResult(status="rejected", error_code="INVALID_STATE")) == ("failed", "INVALID_STATE")
    assert RuntimeTraceRecorder._result_status("fallback") == ("degraded", None)
