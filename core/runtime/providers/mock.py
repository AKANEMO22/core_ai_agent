"""Deterministic mock LLM provider client for unit and integration testing."""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from core.runtime.client import ProviderCapabilities, RuntimeLLMClient
from core.runtime.types import IterationChunk


class MockLLMClient:
    """Mock implementation of RuntimeLLMClient for testing execution loops and agent behaviors.

    Provides programmable token streams, simulated tool calling responses, reasoning summaries,
    error injection, and iteration state tracking.
    """

    def __init__(
        self,
        canned_tokens: Optional[List[str]] = None,
        canned_tool_calls: Optional[List[Dict[str, Any]]] = None,
        final_response: Optional[str] = None,
        canned_reasoning_summaries: Optional[List[str]] = None,
        canned_turns: Optional[List[Dict[str, Any]]] = None,
        repeat_tool_calls: bool = False,
        stream_delay: float = 0.0,
        raise_error: Optional[Exception] = None,
        capabilities: Optional[ProviderCapabilities] = None,
    ) -> None:
        self.canned_tokens = canned_tokens or []
        self.canned_tool_calls = canned_tool_calls
        self.final_response = final_response
        self.canned_reasoning_summaries = canned_reasoning_summaries or []
        self.canned_turns = canned_turns
        self.repeat_tool_calls = repeat_tool_calls
        self.stream_delay = stream_delay
        self.raise_error = raise_error
        self.capabilities = capabilities or ProviderCapabilities(
            streaming=True,
            tool_calls=True,
            usage_events=True,
            cancellation=True,
            safe_reasoning_summary=True,
        )

        self.call_count = 0
        self._tool_calls_consumed = False

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        """Stream simulated response chunks according to programmed configuration."""
        if self.raise_error is not None:
            raise self.raise_error

        self.call_count += 1

        if self.stream_delay > 0:
            await asyncio.sleep(self.stream_delay)

        # 1. Emit reasoning summaries if provided
        for summary in self.canned_reasoning_summaries:
            if self.stream_delay > 0:
                await asyncio.sleep(self.stream_delay)
            yield IterationChunk(reasoning_summary=summary)

        # 2. Check if scripted multi-turn responses provided
        if self.canned_turns and self.call_count <= len(self.canned_turns):
            turn_spec = self.canned_turns[self.call_count - 1]
            if "tool_calls" in turn_spec:
                yield IterationChunk(
                    tool_calls=turn_spec["tool_calls"],
                    finish_reason="tool_calls",
                )
                return
            if "tokens" in turn_spec:
                for tok in turn_spec["tokens"]:
                    yield IterationChunk(token=tok)
                yield IterationChunk(finish_reason="stop")
                return

        # 3. Handle canned tool calls on first turn (or repeatedly if repeat_tool_calls=True)
        if self.canned_tool_calls and (not self._tool_calls_consumed or self.repeat_tool_calls):
            if not self.repeat_tool_calls:
                self._tool_calls_consumed = True

            normalized_tool_calls: List[Dict[str, Any]] = []
            for idx, tc in enumerate(self.canned_tool_calls):
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                normalized_tool_calls.append(
                    {
                        "id": call_id,
                        "name": name,
                        "arguments": args,
                    }
                )

            yield IterationChunk(
                tool_calls=normalized_tool_calls,
                finish_reason="tool_calls",
            )
            return

        # 4. Handle final response after tool calls
        if self.final_response is not None and self._tool_calls_consumed:
            words = self.final_response.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                if self.stream_delay > 0:
                    await asyncio.sleep(self.stream_delay)
                yield IterationChunk(token=token)
            yield IterationChunk(finish_reason="stop")
            return

        # 5. Handle canned token stream
        if self.canned_tokens:
            for tok in self.canned_tokens:
                if self.stream_delay > 0:
                    await asyncio.sleep(self.stream_delay)
                yield IterationChunk(token=tok)
            yield IterationChunk(finish_reason="stop")
            return

        # 6. Fallback empty completion
        yield IterationChunk(token="", finish_reason="stop")
