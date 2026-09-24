"""OpenAI provider client adapter for streaming responses and tool calling."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import AsyncOpenAI

from config.settings import get_settings
from core.runtime.client import ProviderCapabilities, RuntimeLLMClient
from core.runtime.types import IterationChunk


class OpenAIRuntimeLLMClient:
    """Production provider adapter wrapping official AsyncOpenAI streaming client.

    Translates OpenAI Chat Completion SSE chunks into standardized IterationChunks,
    aggregating streamed tool call deltas and capturing token usage telemetry.
    """

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        capabilities: Optional[ProviderCapabilities] = None,
    ) -> None:
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = model or settings.DEFAULT_LLM_MODEL
        self.capabilities = capabilities or ProviderCapabilities(
            streaming=True,
            tool_calls=True,
            usage_events=True,
            cancellation=True,
            safe_reasoning_summary=True,
        )

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize tool specifications into OpenAI function calling format."""
        formatted: List[Dict[str, Any]] = []
        for t in tools:
            if "type" in t and t["type"] == "function":
                formatted.append(t)
            elif "name" in t:
                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("param_schema") or t.get("parameters") or {},
                        },
                    }
                )
        return formatted

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        """Stream chat completions from OpenAI, yielding standardized IterationChunk objects."""
        create_kwargs: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }

        if tools:
            create_kwargs["tools"] = self._format_tools(tools)

        response_stream = await self.client.chat.completions.create(**create_kwargs)

        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        async for chunk in response_stream:
            # 1. Process usage if present in chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage_dict = (
                    chunk.usage.model_dump()
                    if hasattr(chunk.usage, "model_dump")
                    else dict(chunk.usage)
                )
                yield IterationChunk(usage=usage_dict)

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue

            for choice in choices:
                finish_reason = getattr(choice, "finish_reason", None)
                delta = getattr(choice, "delta", None)
                if not delta:
                    if finish_reason:
                        yield IterationChunk(finish_reason=finish_reason)
                    continue

                # 2. Extract content token
                content = getattr(delta, "content", None)
                if content:
                    # If this chunk already carries finish_reason without tool calls, attach it directly
                    eff_finish = finish_reason if (finish_reason and finish_reason != "tool_calls" and not tool_calls_accumulator) else None
                    yield IterationChunk(token=content, finish_reason=eff_finish)

                # 3. Extract reasoning/thought preview if available (o1/o3 or custom models)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    yield IterationChunk(reasoning_summary=reasoning)

                # 4. Accumulate tool call deltas
                raw_tool_calls = getattr(delta, "tool_calls", None)
                if raw_tool_calls:
                    for tc in raw_tool_calls:
                        idx = getattr(tc, "index", 0)
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {
                                "id": getattr(tc, "id", None) or f"call_{idx}",
                                "name": "",
                                "arguments": "",
                            }
                        if getattr(tc, "id", None):
                            tool_calls_accumulator[idx]["id"] = tc.id

                        fn = getattr(tc, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                tool_calls_accumulator[idx]["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                tool_calls_accumulator[idx]["arguments"] += fn.arguments

                # 5. Emit completed tool calls if finish_reason signals tool invocation
                if finish_reason == "tool_calls" or (
                    finish_reason and tool_calls_accumulator
                ):
                    assembled_calls: List[Dict[str, Any]] = []
                    for idx in sorted(tool_calls_accumulator.keys()):
                        item = tool_calls_accumulator[idx]
                        arg_str = item["arguments"]
                        try:
                            parsed_args = json.loads(arg_str) if arg_str else {}
                        except json.JSONDecodeError:
                            parsed_args = arg_str
                        assembled_calls.append(
                            {
                                "id": item["id"],
                                "name": item["name"],
                                "arguments": parsed_args,
                            }
                        )
                    yield IterationChunk(
                        tool_calls=assembled_calls,
                        finish_reason="tool_calls",
                    )
                    tool_calls_accumulator.clear()
                elif finish_reason and not content:
                    yield IterationChunk(finish_reason=finish_reason)
