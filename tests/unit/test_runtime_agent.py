"""Unit tests for RuntimeAgent execution loop, streaming protocols, tools, and providers."""

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.runtime.agent import RuntimeAgent
from core.runtime.client import ProviderCapabilities, RuntimeLLMClient
from core.runtime.providers.mock import MockLLMClient
from core.runtime.providers.openai import OpenAIRuntimeLLMClient
from core.runtime.tooling import RuntimeTool, ToolSchemaValidator
from core.runtime.types import IterationChunk, LoopEvent, LoopEventType, LoopStatus
from core.workspace.session import SessionWorkspace
from core.workspace.spill import WorkspaceSpillManager


@pytest.mark.asyncio
async def test_agent_streaming_tokens():
    """Verify agent fast-paths token streaming and produces complete loop events."""
    mock_client = MockLLMClient(canned_tokens=["Hello", " ", "candidate!"])
    agent = RuntimeAgent(client=mock_client)
    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        events.append(event)

    token_events = [e for e in events if e.type == LoopEventType.RESPONSE_TOKEN]
    assert len(token_events) == 3
    assert "".join([e.payload["token"] for e in token_events]) == "Hello candidate!"
    complete_events = [e for e in events if e.type == LoopEventType.COMPLETE]
    assert len(complete_events) == 1

    # Verify standard envelope fields across all events
    for idx, e in enumerate(events, start=1):
        assert e.schema_version == "1.0"
        assert e.event_id.startswith("evt_")
        assert e.session_id == "s1"
        assert e.turn_id == "turn_001"
        assert e.sequence == idx
        assert isinstance(e.timestamp, float)
        assert isinstance(e.payload, dict)


@pytest.mark.asyncio
async def test_agent_tool_call_and_soft_recovery():
    """Verify tool invocation lifecycle: start, execute, result, and resume streaming."""
    def get_weather(location: str) -> str:
        return f"Weather in {location} is Sunny"

    tool = RuntimeTool.from_callable(
        name="get_weather",
        description="Get weather for location",
        func=get_weather,
        param_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "get_weather", "arguments": {"location": "Hanoi"}}],
        final_response="The weather is Sunny.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool])
    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Check weather in Hanoi"}],
    ):
        events.append(event)

    tool_starts = [e for e in events if e.type == LoopEventType.TOOL_CALL_START]
    assert len(tool_starts) == 1
    assert tool_starts[0].payload["tool_name"] == "get_weather"
    assert tool_starts[0].payload["arguments"] == {"location": "Hanoi"}

    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) == 1
    assert "Sunny" in tool_results[0].payload["result"]

    complete_events = [e for e in events if e.type == LoopEventType.COMPLETE]
    assert len(complete_events) == 1
    assert complete_events[0].payload["final_response"] == "The weather is Sunny."


@pytest.mark.asyncio
async def test_tool_schema_soft_validation_and_repair():
    """Verify that schema violations are detected and fed back to the model for self-repair."""
    def calculate_tax(income: float, rate: float = 0.1) -> float:
        return income * rate

    tool = RuntimeTool.from_callable(
        name="calculate_tax",
        description="Calculate income tax",
        func=calculate_tax,
        param_schema={
            "type": "object",
            "properties": {
                "income": {"type": "number"},
                "rate": {"type": "number"},
            },
            "required": ["income"],
        },
    )

    # 1st call has invalid argument (income missing, invalid string key), 2nd call fixes it
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "calculate_tax", "arguments": {"invalid_arg": "abc"}}],
        final_response="Tax calculated.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool], schema_repair_budget=1)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Compute tax"}],
    ):
        events.append(event)

    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) >= 1
    # Schema validation error returned as tool result without crashing
    assert "validation" in tool_results[0].payload["result"].lower() or "required" in tool_results[0].payload["result"].lower()


@pytest.mark.asyncio
async def test_tool_callable_auto_schema_inspection():
    """Verify RuntimeTool inspects python signatures to build JSON schema if not provided."""
    def search_docs(query: str, limit: int = 5) -> str:
        """Search documentation index."""
        return f"Found results for {query} with limit {limit}"

    tool = RuntimeTool.from_callable(
        name="search_docs",
        description="Search documentation index",
        func=search_docs,
    )

    schema = tool.param_schema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["limit"]["type"] == "integer"
    assert "query" in schema["required"]
    assert "limit" not in schema["required"]

    is_valid, errors = tool.validate_arguments({"query": "python"})
    assert is_valid is True
    assert len(errors) == 0

    is_valid, errors = tool.validate_arguments({})
    assert is_valid is False
    assert len(errors) > 0


@pytest.mark.asyncio
async def test_async_and_sync_tool_execution():
    """Verify RuntimeTool seamlessly executes both sync functions and async coroutines."""
    def sync_tool(x: int) -> int:
        return x * 2

    async def async_tool(x: int) -> int:
        await asyncio.sleep(0.01)
        return x + 10

    t_sync = RuntimeTool.from_callable(name="sync_fn", description="Sync tool", func=sync_tool)
    t_async = RuntimeTool.from_callable(name="async_fn", description="Async tool", func=async_tool)

    res_sync = await t_sync.execute({"x": 5})
    assert res_sync == 10

    res_async = await t_async.execute({"x": 5})
    assert res_async == 15


@pytest.mark.asyncio
async def test_budget_control_max_iterations():
    """Verify agent enforces max_iterations limit and stops looping on repetitive tool calls."""
    def endless_probe() -> str:
        return "Still analyzing..."

    tool = RuntimeTool.from_callable(name="probe", description="Endless probe", func=endless_probe)

    # Client always requests the tool
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "probe", "arguments": {}}],
        repeat_tool_calls=True,
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool], max_iterations=3)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Analyze"}],
    ):
        events.append(event)

    # Should have terminated with error or nudged completion within budget
    iteration_events = [e for e in events if e.type in (LoopEventType.ERROR, LoopEventType.COMPLETE)]
    assert len(iteration_events) >= 1
    # Check that tool call results do not exceed max_iterations
    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) <= 3


@pytest.mark.asyncio
async def test_tool_output_spillover_integration(tmp_path: Path):
    """Verify tool output > 32KB is spilled to workspace disk and preview returned."""
    ws = SessionWorkspace(session_id="sess_spill", root_path=tmp_path / "sess_spill")
    ws.initialize()

    # Tool returns 50KB payload
    large_output = "X" * (35 * 1024)

    def large_tool() -> str:
        return large_output

    tool = RuntimeTool.from_callable(name="large_tool", description="Large tool", func=large_tool)
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "large_tool", "arguments": {}}],
        final_response="Done analyzing large file.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool], workspace=ws)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="sess_spill",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Fetch big payload"}],
    ):
        events.append(event)

    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) == 1
    payload = tool_results[0].payload
    assert payload.get("is_spilled") is True or "spill" in str(payload.get("result", ""))
    assert "spill_path" in payload or "spill" in payload.get("preview", "")

    # Verify spilled file actually exists in workspace spill/
    spill_files = list(ws.spill_dir.glob("*.json"))
    assert len(spill_files) == 1


@pytest.mark.asyncio
async def test_agent_cancellation():
    """Verify agent responds cleanly to cancellation token without hanging."""
    cancel_event = asyncio.Event()

    # Pre-cancel before run or during streaming
    cancel_event.set()

    mock_client = MockLLMClient(canned_tokens=["Never", "seen"])
    agent = RuntimeAgent(client=mock_client)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Hi"}],
        cancellation_token=cancel_event,
    ):
        events.append(event)

    # Must contain error or cancelled status
    assert len(events) >= 1
    assert events[-1].type == LoopEventType.ERROR
    assert events[-1].payload.get("code") == "CANCELLED"


@pytest.mark.asyncio
async def test_reasoning_summary_streaming():
    """Verify provider reasoning summaries stream as REASONING_SUMMARY loop events."""
    mock_client = MockLLMClient(
        canned_reasoning_summaries=["Analyzing applicant response...", "Formulating follow-up question."],
        canned_tokens=["Can you elaborate on your experience?"],
    )
    agent = RuntimeAgent(client=mock_client)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Here is my answer."}],
    ):
        events.append(event)

    reasoning_events = [e for e in events if e.type == LoopEventType.REASONING_SUMMARY]
    assert len(reasoning_events) == 2
    assert reasoning_events[0].payload["summary"] == "Analyzing applicant response..."
    assert reasoning_events[1].payload["summary"] == "Formulating follow-up question."


@pytest.mark.asyncio
async def test_openai_runtime_client_adapter():
    """Verify OpenAIRuntimeLLMClient adapter translates OpenAI streaming events correctly."""
    mock_openai_client = MagicMock()

    # Simulate chunk stream from AsyncOpenAI
    class MockChoiceDelta:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class MockChoice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class MockChunk:
        def __init__(self, choices, usage=None):
            self.choices = choices
            self.usage = usage

    async def fake_stream():
        yield MockChunk([MockChoice(MockChoiceDelta(content="Hello"))])
        yield MockChunk([MockChoice(MockChoiceDelta(content=" world!"), finish_reason="stop")])

    mock_openai_client.chat.completions.create = AsyncMock(return_value=fake_stream())

    client = OpenAIRuntimeLLMClient(client=mock_openai_client, model="gpt-4o")
    chunks: List[IterationChunk] = []
    async for chunk in client.stream_chat(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].token == "Hello"
    assert chunks[1].token == " world!"
    assert chunks[1].finish_reason == "stop"
    assert client.capabilities.streaming is True
    assert client.capabilities.tool_calls is True


@pytest.mark.asyncio
async def test_secret_scrubbing_in_errors_and_logs():
    """Verify that raw secret tokens (sk-..., Bearer ...) are never emitted in error events."""
    secret_key = "sk-proj-supersecretkey1234567890abcdef"
    mock_client = MockLLMClient(
        raise_error=RuntimeError(f"Connection failed to provider with token {secret_key}"),
    )
    agent = RuntimeAgent(client=mock_client)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        events.append(event)

    error_events = [e for e in events if e.type == LoopEventType.ERROR]
    assert len(error_events) == 1
    error_msg = str(error_events[0].payload)
    assert secret_key not in error_msg
    assert "[REDACTED" in error_msg or "***" in error_msg


@pytest.mark.asyncio
async def test_unknown_tool_call_handling():
    """Verify agent handles hallucinated/unknown tool calls gracefully without crashing."""
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "non_existent_tool", "arguments": {"foo": "bar"}}],
        final_response="Handled unknown tool.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[])

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Run non-existent tool"}],
    ):
        events.append(event)

    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) == 1
    assert "not found" in tool_results[0].payload["result"].lower()

    complete_events = [e for e in events if e.type == LoopEventType.COMPLETE]
    assert len(complete_events) == 1


@pytest.mark.asyncio
async def test_tool_transient_retry_budget():
    """Verify transient tool errors are retried according to transient_retry_budget."""
    call_attempts = 0

    def flaky_tool(param: str) -> str:
        nonlocal call_attempts
        call_attempts += 1
        if call_attempts == 1:
            raise ConnectionResetError("Connection reset by peer")
        return f"Success with {param}"

    tool = RuntimeTool.from_callable(name="flaky_tool", description="Flaky", func=flaky_tool)
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "flaky_tool", "arguments": {"param": "test"}}],
        final_response="Tool succeeded after retry.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool], transient_retry_budget=1)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Run flaky"}],
    ):
        events.append(event)

    assert call_attempts == 2
    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) == 1
    assert "Success with test" in tool_results[0].payload["result"]


@pytest.mark.asyncio
async def test_agent_lifecycle_status_transitions():
    """Verify agent status transitions from IDLE to RUNNING to COMPLETED."""
    mock_client = MockLLMClient(canned_tokens=["Status", " check"])
    agent = RuntimeAgent(client=mock_client)
    assert agent.status == LoopStatus.IDLE

    async for _ in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        pass

    assert agent.status == LoopStatus.COMPLETED


@pytest.mark.asyncio
async def test_openai_client_streaming_tool_calls_assembly():
    """Verify OpenAIRuntimeLLMClient correctly reassembles chunked tool calls."""
    mock_openai_client = MagicMock()

    class MockFnDelta:
        def __init__(self, name=None, arguments=None):
            self.name = name
            self.arguments = arguments

    class MockToolCallDelta:
        def __init__(self, index, id=None, function=None):
            self.index = index
            self.id = id
            self.function = function

    class MockChoiceDelta:
        def __init__(self, tool_calls=None):
            self.content = None
            self.tool_calls = tool_calls

    class MockChoice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class MockChunk:
        def __init__(self, choices):
            self.choices = choices

    async def fake_tool_stream():
        # Chunk 1: Tool call id and function name
        yield MockChunk([MockChoice(MockChoiceDelta(
            tool_calls=[MockToolCallDelta(index=0, id="call_abc123", function=MockFnDelta(name="fetch_user", arguments=""))]
        ))])
        # Chunk 2: Tool call partial argument
        yield MockChunk([MockChoice(MockChoiceDelta(
            tool_calls=[MockToolCallDelta(index=0, function=MockFnDelta(arguments='{"user_id": '))]
        ))])
        # Chunk 3: Tool call remaining argument and finish_reason
        yield MockChunk([MockChoice(
            MockChoiceDelta(tool_calls=[MockToolCallDelta(index=0, function=MockFnDelta(arguments='42}'))]),
            finish_reason="tool_calls",
        )])

    mock_openai_client.chat.completions.create = AsyncMock(return_value=fake_tool_stream())
    client = OpenAIRuntimeLLMClient(client=mock_openai_client)

    chunks: List[IterationChunk] = []
    async for chunk in client.stream_chat(messages=[{"role": "user", "content": "lookup"}]):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].finish_reason == "tool_calls"
    assert chunks[0].tool_calls is not None
    assert len(chunks[0].tool_calls) == 1
    tc = chunks[0].tool_calls[0]
    assert tc["id"] == "call_abc123"
    assert tc["name"] == "fetch_user"
    assert tc["arguments"] == {"user_id": 42}


@pytest.mark.asyncio
async def test_tool_call_id_synchronization():
    """Verify tool call IDs are synchronized between assistant message and tool response."""
    def sample_tool() -> str:
        return "result_ok"

    tool = RuntimeTool.from_callable(name="sample_tool", description="Sample", func=sample_tool)
    # Canned tool call with no 'id' field
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "sample_tool", "arguments": {}}],
        final_response="Completed sample.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool])

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Execute sample"}],
    ):
        events.append(event)

    start_events = [e for e in events if e.type == LoopEventType.TOOL_CALL_START]
    result_events = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]

    assert len(start_events) == 1
    assert len(result_events) == 1
    # Ensure call_id was generated and matches perfectly between start and result
    call_id = start_events[0].payload["call_id"]
    assert call_id is not None
    assert call_id.startswith("call_")
    assert result_events[0].payload["call_id"] == call_id


def test_default_max_iterations_is_five():
    """Verify default max_iterations is 5 per Global Constraints."""
    mock_client = MockLLMClient(canned_tokens=["ok"])
    agent = RuntimeAgent(client=mock_client)
    assert agent.max_iterations == 5


@pytest.mark.asyncio
async def test_schema_repair_budget_ceiling_enforced():
    """Verify that when schema repair budget is exhausted, the model is nudged to synthesize final answer."""
    def strict_tool(req: str) -> str:
        return f"Got {req}"

    tool = RuntimeTool.from_callable(
        name="strict_tool",
        description="Strict",
        func=strict_tool,
        param_schema={"type": "object", "properties": {"req": {"type": "string"}}, "required": ["req"]},
    )

    # Repeat tool calls with invalid arguments
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "strict_tool", "arguments": {}}],
        repeat_tool_calls=True,
    )
    # Set repair budget to 1 and max_iterations to 3
    agent = RuntimeAgent(client=mock_client, tools=[tool], schema_repair_budget=1, max_iterations=3)

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Run strict"}],
    ):
        events.append(event)

    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    # In iteration 1: repair allowed
    assert len(tool_results) >= 2
    # In iteration 2: repair budget exceeded message returned
    assert "schema repair budget exceeded" in tool_results[1].payload["result"].lower()


@pytest.mark.asyncio
async def test_null_tool_arguments_handled_safely():
    """Verify tool execution gracefully handles null (None) arguments without crashing."""
    def no_arg_tool() -> str:
        return "Executed without arguments"

    tool = RuntimeTool.from_callable(name="no_arg_tool", description="No args", func=no_arg_tool)
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "no_arg_tool", "arguments": None}],
        final_response="Success with null arguments.",
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool])

    events: List[LoopEvent] = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Call tool"}],
    ):
        events.append(event)

    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) == 1
    assert "Executed without arguments" in tool_results[0].payload["result"]


@pytest.mark.asyncio
async def test_sync_tool_executed_in_thread():
    """Verify synchronous tools are offloaded to worker threads via asyncio.to_thread."""
    import threading

    main_thread_ident = threading.get_ident()
    executed_thread_ident = None

    def blocking_sync_tool() -> str:
        nonlocal executed_thread_ident
        executed_thread_ident = threading.get_ident()
        return "done"

    tool = RuntimeTool.from_callable(name="blocking_sync_tool", description="Sync", func=blocking_sync_tool)
    result = await tool.execute({})

    assert result == "done"
    # Thread should be a worker thread, distinct from the asyncio event loop thread
    assert executed_thread_ident != main_thread_ident
