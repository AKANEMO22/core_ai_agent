"""LLM provider contract suite.

The adapter behavior is verified against a scripted fake of the OpenAI SDK on every run.
The same contract runs against the real provider when explicitly enabled:

    RUN_PROVIDER_CONTRACT=1 uv run pytest tests/contract/test_provider_contract.py   (key/model/url from .env)

Contract: streaming tokens, tool-call assembly, usage events, cancellation closes the stream,
timeouts/deadlines, malformed tool arguments, and transient-error retry before streaming.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from config.settings import get_settings
from core.runtime.client import ProviderCapabilityError, ProviderCapabilities, verify_capabilities
from core.runtime.providers.openai import OpenAIRuntimeLLMClient, is_transient_error
from core.runtime.structured import StructuredOutputError, run_structured


def _chunk(content: Optional[str] = None, tool_calls: Any = None, finish: Optional[str] = None, usage: Any = None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage)


def _tool_delta(index: int, call_id: Optional[str], name: Optional[str], args: Optional[str]):
    return [SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=args))]


class FakeStream:
    def __init__(self, chunks: List[Any], delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def close(self) -> None:
        self.closed = True


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, streams: List[Any]) -> None:
        self.streams = list(streams)
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.streams.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _adapter(streams: List[Any]) -> OpenAIRuntimeLLMClient:
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(streams)))
    return OpenAIRuntimeLLMClient(client=fake, model="test-model", retry_base_delay_seconds=0.0)


async def _drain(client: OpenAIRuntimeLLMClient, **kwargs: Any) -> List[Any]:
    return [c async for c in client.stream_chat(messages=[{"role": "user", "content": "hi"}], **kwargs)]


@pytest.mark.asyncio
async def test_streams_tokens_usage_and_request_shape():
    stream = FakeStream([
        _chunk("Hel"),
        _chunk("lo", finish="stop"),
        SimpleNamespace(choices=[], usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 5, "completion_tokens": 2})),
    ])
    client = _adapter([stream])
    chunks = await _drain(client, temperature=0.2)
    assert "".join(c.token for c in chunks if c.token) == "Hello"
    assert any(c.usage == {"prompt_tokens": 5, "completion_tokens": 2} for c in chunks)
    call = client.client.chat.completions.calls[0]
    assert call["stream"] is True and call["stream_options"] == {"include_usage": True}
    assert call["model"] == "test-model" and call["temperature"] == 0.2 and call["timeout"] > 0
    assert stream.closed


@pytest.mark.asyncio
async def test_assembles_streamed_tool_call_deltas_and_tolerates_malformed_args():
    stream = FakeStream([
        _chunk(tool_calls=_tool_delta(0, "call_1", "retrieve", '{"collection": "rub')),
        _chunk(tool_calls=_tool_delta(0, None, None, 'ric", "query": "design"}')),
        _chunk(tool_calls=_tool_delta(1, "call_2", "broken", "{not json")),
        _chunk(finish="tool_calls"),
    ])
    chunks = await _drain(_adapter([stream]), tools=[{"name": "retrieve", "param_schema": {"type": "object"}}])
    calls = [c for c in chunks if c.tool_calls][0].tool_calls
    assert calls[0] == {"id": "call_1", "name": "retrieve", "arguments": {"collection": "rubric", "query": "design"}}
    assert calls[1]["arguments"] == "{not json"


@pytest.mark.asyncio
async def test_consumer_stop_closes_stream():
    stream = FakeStream([_chunk(str(i)) for i in range(50)], delay=0.01)
    client = _adapter([stream])
    generator = client.stream_chat(messages=[{"role": "user", "content": "x"}])
    received = 0
    async for _ in generator:
        received += 1
        if received == 3:
            break
    await generator.aclose()
    assert stream.closed


@pytest.mark.asyncio
async def test_retries_transient_error_once_before_streaming():
    client = _adapter([StatusError(429), FakeStream([_chunk("ok", finish="stop")])])
    chunks = await _drain(client)
    assert "".join(c.token for c in chunks if c.token) == "ok"
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_does_not_retry_client_errors():
    client = _adapter([StatusError(400), FakeStream([_chunk("never")])])
    with pytest.raises(StatusError):
        await _drain(client)
    assert len(client.client.chat.completions.calls) == 1


def test_transient_classification():
    assert is_transient_error(StatusError(503)) and is_transient_error(StatusError(429))
    assert not is_transient_error(StatusError(401))
    assert is_transient_error(asyncio.TimeoutError())


def test_exhausted_quota_is_not_retried():
    quota = StatusError(429)
    quota.args = ("[429]: would exceed your account's rate limit (reset after 59m 58s)",)
    assert not is_transient_error(quota)
    burst = StatusError(429)
    burst.args = ("[429]: slow down (reset after 2s)",)
    assert is_transient_error(burst)


def test_capability_verification():
    verify_capabilities(_adapter([]))
    weak = SimpleNamespace(capabilities=ProviderCapabilities(streaming=True, usage_events=False, cancellation=True))
    with pytest.raises(ProviderCapabilityError):
        verify_capabilities(weak)


@pytest.mark.asyncio
async def test_structured_output_uses_json_mode_and_repairs_once():
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    client = _adapter([
        FakeStream([_chunk('{"score": "high"}', finish="stop")]),
        FakeStream([_chunk('{"score": 3}', finish="stop")]),
    ])
    result = await run_structured(client, [{"role": "user", "content": "rate"}], schema, max_repairs=1)
    assert result == {"score": 3}
    calls = client.client.chat.completions.calls
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "did not match" in calls[1]["messages"][-1]["content"]

    failing = _adapter([FakeStream([_chunk("nope", finish="stop")]), FakeStream([_chunk("still no", finish="stop")])])
    with pytest.raises(StructuredOutputError):
        await run_structured(failing, [{"role": "user", "content": "rate"}], schema, max_repairs=1)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("RUN_PROVIDER_CONTRACT") == "1" and get_settings().llm_api_key),
    reason="real provider contract runs only with RUN_PROVIDER_CONTRACT=1 and LLM_API_KEY/OPENAI_API_KEY",
)
async def test_real_provider_contract():
    client = OpenAIRuntimeLLMClient()
    verify_capabilities(client)
    chunks = [c async for c in client.stream_chat(messages=[{"role": "user", "content": "Say OK."}], max_tokens=5)]
    assert any(c.token for c in chunks)
    assert any(c.usage for c in chunks)

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    result = await run_structured(
        client, [{"role": "user", "content": 'Return {"ok": true} as JSON.'}], schema, deadline_seconds=30
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_empty_completion_is_retried_once():
    client = _adapter([FakeStream([_chunk(finish="stop")]), FakeStream([_chunk("hi", finish="stop")])])
    chunks = await _drain(client)
    assert "".join(c.token for c in chunks if c.token) == "hi"
    assert len(client.client.chat.completions.calls) == 2


class RejectError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400


@pytest.mark.asyncio
async def test_rejected_parameter_is_dropped_and_remembered():
    client = _adapter([
        RejectError("[400]: `temperature` is deprecated for this model."),
        FakeStream([_chunk("ok", finish="stop")]),
        FakeStream([_chunk("again", finish="stop")]),
    ])
    first = await _drain(client, temperature=0.4)
    assert "".join(c.token for c in first if c.token) == "ok"
    second = await _drain(client, temperature=0.4)
    assert "".join(c.token for c in second if c.token) == "again"
    calls = client.client.chat.completions.calls
    assert "temperature" in calls[0] and "temperature" not in calls[1] and "temperature" not in calls[2]
