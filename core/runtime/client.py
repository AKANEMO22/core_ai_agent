"""Protocol definitions and capability declarations for LLM providers.

Luồng sử dụng: Hợp đồng giữa agent và backend LLM: stream_chat nhận messages/tools và trả IterationChunk.
Provider tự khai báo khả năng để bước khởi động kiểm tra; ProfiledLLMClient áp model của
vai trò làm mặc định cho các agent gọi provider trực tiếp, vẫn giữ tham số gọi tường minh.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple, runtime_checkable
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
    json_mode: bool = Field(default=False, description="Supports constrained JSON object output")


class ProfiledLLMClient:
    """Wraps a client so every call defaults to one agent profile's model.

    Components that call the provider directly (structured agents) get the model configured for
    their profile without threading model ids through every call site.
    """

    def __init__(self, inner: Any, model: str = "") -> None:
        self.inner = inner
        self.model = model or getattr(inner, "model", "")
        self._override = model
        self.capabilities = inner.capabilities

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[IterationChunk]:
        if self._override:
            kwargs.setdefault("model", self._override)
        return self.inner.stream_chat(messages=messages, tools=tools, **kwargs)


class ProviderCapabilityError(RuntimeError):
    """Raised at startup when a provider lacks a capability the runtime requires."""


REQUIRED_CAPABILITIES = ("streaming", "usage_events", "cancellation")


def verify_capabilities(client: Any, required: Tuple[str, ...] = REQUIRED_CAPABILITIES) -> None:
    """Fail fast if the configured provider cannot serve the runtime's contract."""
    capabilities = getattr(client, "capabilities", None)
    if capabilities is None:
        raise ProviderCapabilityError("provider does not declare capabilities")
    # Kiểm tra khi lắp client để lỗi thiếu khả năng xuất hiện trước khi bắt đầu lượt interview.
    missing = [name for name in required if not getattr(capabilities, name, False)]
    if missing:
        raise ProviderCapabilityError(f"provider lacks required capabilities: {missing}")


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
