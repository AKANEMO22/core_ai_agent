"""OpenAI provider adapter: streaming, tool-call assembly, deadlines, retries and cancellation."""

import asyncio
import json
import logging
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from config.settings import get_settings
from core.runtime.client import ProviderCapabilities
from core.runtime.sanitize import sanitize_secrets
from core.runtime.types import IterationChunk

logger = logging.getLogger(__name__)

NO_TEMPERATURE_SUFFIXES = ("claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-1")
# Sampling/format parameters a model may reject with HTTP 400; dropped and remembered per model.
DROPPABLE_PARAMS = ("temperature", "top_p", "response_format", "stream_options")
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


MAX_RETRYABLE_RESET_SECONDS = 30.0
_RESET_PATTERN = re.compile(r"reset after\s+((?:\d+\s*[hms]\s*)+)", re.IGNORECASE)


def rate_limit_reset_seconds(error: BaseException) -> Optional[float]:
    """Seconds until a 429 quota resets, from a Retry-After header or a 'reset after 1h 2m' message."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            return float(retry_after)
    except (TypeError, ValueError, AttributeError):
        pass
    match = _RESET_PATTERN.search(str(error))
    if not match:
        return None
    units = {"h": 3600, "m": 60, "s": 1}
    return float(sum(int(n) * units[u.lower()] for n, u in re.findall(r"(\d+)\s*([hms])", match.group(1))))


def is_transient_error(error: BaseException) -> bool:
    """True for errors worth one retry: rate limits, 5xx, timeouts, connection failures.

    A 429 whose quota resets later than MAX_RETRYABLE_RESET_SECONDS is exhausted quota, not a
    burst limit; retrying only adds latency, so it fails fast.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        if status == 429:
            reset = rate_limit_reset_seconds(error)
            if reset is not None and reset > MAX_RETRYABLE_RESET_SECONDS:
                return False
        return status in TRANSIENT_STATUS
    name = type(error).__name__
    return name in {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"} or isinstance(
        error, (asyncio.TimeoutError, ConnectionError)
    )


class OpenAIRuntimeLLMClient:
    """Production adapter over the official AsyncOpenAI client.

    - Translates Chat Completions SSE chunks into IterationChunks, assembling tool-call deltas.
    - Applies a request timeout (`request_timeout_seconds`, overridable per call via `deadline_s`).
    - Retries a transient failure once (with jitter) only before the first chunk is produced, so
      a partially streamed answer is never duplicated.
    - Closes the HTTP stream when the consumer stops early (cancellation, deadline), which stops
      provider-side generation and billing.
    """

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        capabilities: Optional[ProviderCapabilities] = None,
        request_timeout_seconds: Optional[float] = None,
        max_retries: int = 1,
        base_url: Optional[str] = None,
        retry_base_delay_seconds: float = 0.5,
    ) -> None:
        settings = get_settings()
        # The SDK's own retries are disabled; this adapter decides what is safe to retry.
        self.client = client or AsyncOpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=base_url or settings.LLM_API_URL or None,
            max_retries=0,
        )
        self.model = model or settings.llm_model
        self.request_timeout_seconds = request_timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self._unsupported_params: Dict[str, set] = {}
        self.capabilities = capabilities or ProviderCapabilities(
            streaming=True,
            tool_calls=True,
            usage_events=True,
            cancellation=True,
            safe_reasoning_summary=True,
            json_mode=True,
        )

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize tool specifications into OpenAI function calling format."""
        formatted: List[Dict[str, Any]] = []
        for t in tools:
            if t.get("type") == "function":
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

    def _rejected_param(self, error: BaseException, create_kwargs: Dict[str, Any]) -> Optional[str]:
        """Name of a droppable parameter that a 400 response says the model does not accept."""
        status = getattr(error, "status_code", None)
        if status != 400:
            return None
        message = str(error).lower()
        for param in DROPPABLE_PARAMS:
            if param in create_kwargs and param in message:
                return param
        return None

    async def _open_stream(self, create_kwargs: Dict[str, Any]) -> Any:
        attempt = 0
        model_name = str(create_kwargs.get("model"))
        for param in self._unsupported_params.get(model_name, ()):
            create_kwargs.pop(param, None)
        while True:
            try:
                return await self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                rejected = self._rejected_param(e, create_kwargs)
                if rejected is not None:
                    # Remember the incompatibility so later calls skip the failing round trip.
                    self._unsupported_params.setdefault(model_name, set()).add(rejected)
                    create_kwargs.pop(rejected, None)
                    logger.warning("Model rejected parameter '%s'; retrying without it", rejected)
                    continue
                if attempt >= self.max_retries or not is_transient_error(e):
                    raise
                attempt += 1
                delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0, delay / 2)
                logger.warning("Transient provider error, retrying in %.2fs: %s", delay, sanitize_secrets(str(e)))
                await asyncio.sleep(delay)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        """Stream chat completions as IterationChunks.

        A completion that ends without any content or tool call (some gateways return empty
        completions under concurrency) is retried once; nothing has been streamed to the consumer
        in that case, so the retry cannot duplicate output.
        """
        for attempt in range(1 + self.max_retries):
            produced = False
            inner = self._stream_once(messages, tools, dict(kwargs))
            try:
                async for chunk in inner:
                    if chunk.token or chunk.tool_calls:
                        produced = True
                    yield chunk
            finally:
                await inner.aclose()  # closes the HTTP stream when the consumer stops early
            if produced:
                return
            if attempt < self.max_retries:
                logger.warning("Provider returned an empty completion; retrying once")
                await asyncio.sleep(self.retry_base_delay_seconds * random.uniform(0.5, 1.5))

    async def _stream_once(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> AsyncIterator[IterationChunk]:
        """One streaming request."""
        deadline_s = kwargs.pop("deadline_s", None)
        create_kwargs: Dict[str, Any] = {
            "model": kwargs.pop("model", None) or self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": deadline_s or self.request_timeout_seconds,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = self._format_tools(tools)
        model_name = str(create_kwargs["model"])
        if any(model_name.endswith(suffix) for suffix in NO_TEMPERATURE_SUFFIXES):
            create_kwargs.pop("temperature", None)  # some models reject sampling parameters

        response_stream = await self._open_stream(create_kwargs)
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
        try:
            async for chunk in response_stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                    yield IterationChunk(usage=usage.model_dump() if hasattr(usage, "model_dump") else dict(usage))

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

                    content = getattr(delta, "content", None)
                    if content:
                        eff_finish = (
                            finish_reason
                            if finish_reason and finish_reason != "tool_calls" and not tool_calls_accumulator
                            else None
                        )
                        yield IterationChunk(token=content, finish_reason=eff_finish)

                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    if reasoning:
                        yield IterationChunk(reasoning_summary=reasoning)

                    for tc in getattr(delta, "tool_calls", None) or []:
                        idx = getattr(tc, "index", 0)
                        entry = tool_calls_accumulator.setdefault(
                            idx, {"id": getattr(tc, "id", None) or f"call_{idx}", "name": "", "arguments": ""}
                        )
                        if getattr(tc, "id", None):
                            entry["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                entry["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                entry["arguments"] += fn.arguments

                    if finish_reason == "tool_calls" or (finish_reason and tool_calls_accumulator):
                        assembled: List[Dict[str, Any]] = []
                        for idx in sorted(tool_calls_accumulator):
                            item = tool_calls_accumulator[idx]
                            try:
                                parsed_args: Any = json.loads(item["arguments"]) if item["arguments"] else {}
                            except json.JSONDecodeError:
                                parsed_args = item["arguments"]
                            assembled.append({"id": item["id"], "name": item["name"], "arguments": parsed_args})
                        yield IterationChunk(tool_calls=assembled, finish_reason="tool_calls")
                        tool_calls_accumulator.clear()
                    elif finish_reason and not content:
                        yield IterationChunk(finish_reason=finish_reason)
        finally:
            close = getattr(response_stream, "close", None) or getattr(response_stream, "aclose", None)
            if close is not None:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
