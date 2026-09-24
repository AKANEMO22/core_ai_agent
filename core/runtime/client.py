"""Protocol definitions and capability declarations for LLM providers."""

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from core.runtime.types import IterationChunk


class ProviderCapabilities(BaseModel):
    """Declared capabilities supported by a specific LLM provider adapter.

    Attributes:
        streaming: Supports token-by-token streaming response generation.
        tool_calls: Supports function/tool invocation schemas and deltas.
        usage_events: Emits prompt and completion token accounting metrics.
        cancellation: Supports cleanly cancelling in-flight request streams.
        safe_reasoning_summary: Emits sanitized chain-of-thought summaries.
    """

    streaming: bool = Field(default=True, description="Supports streaming response tokens")
    tool_calls: bool = Field(default=True, description="Supports function/tool calls")
    usage_events: bool = Field(default=True, description="Provides token usage statistics")
    cancellation: bool = Field(default=True, description="Supports async cancellation")
    safe_reasoning_summary: bool = Field(
        default=False, description="Supports safe reasoning preview/summary"
    )


@runtime_checkable
class RuntimeLLMClient(Protocol):
    """Protocol defining the interface required for all runtime LLM client adapters."""

    capabilities: ProviderCapabilities

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        """Stream conversational model completions with optional tool declarations.

        Args:
            messages: List of chat message dictionaries with 'role' and 'content'.
            tools: Optional tool definitions formatted according to function calling schema.
            **kwargs: Additional provider-specific generation parameters (temperature, model, etc.).

        Yields:
            IterationChunk: Incremental token, reasoning, tool call, or usage chunk.
        """
        ...
