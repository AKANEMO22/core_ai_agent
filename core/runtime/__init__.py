"""Agent runtime execution loop, protocols, tools, and provider adapters.

Luồng sử dụng: Điểm nhập chung của runtime: xuất agent loop, hợp đồng stream, tool và provider.
Các nơi lắp ghép ứng dụng có thể nhập từ package này; logic xử lý nằm ở các module con.
"""

from core.runtime.agent import RuntimeAgent
from core.runtime.client import ProviderCapabilities, RuntimeLLMClient
from core.runtime.providers.mock import MockLLMClient
from core.runtime.providers.openai import OpenAIRuntimeLLMClient
from core.runtime.tooling import RuntimeTool, ToolSchemaValidator
from core.runtime.types import (
    IterationChunk,
    LoopEvent,
    LoopEventType,
    LoopStatus,
)

__all__ = [
    "IterationChunk",
    "LoopEvent",
    "LoopEventType",
    "LoopStatus",
    "MockLLMClient",
    "OpenAIRuntimeLLMClient",
    "ProviderCapabilities",
    "RuntimeAgent",
    "RuntimeLLMClient",
    "RuntimeTool",
    "ToolSchemaValidator",
]
