"""Explicit unavailable provider for studio deployments lacking real LLM credentials."""
from core.runtime.client import ProviderCapabilities


class UnconfiguredLLMClient:
    capabilities = ProviderCapabilities(streaming=True, usage_events=True, cancellation=True)
    model = None

    async def stream_chat(self, messages, tools=None, **kwargs):
        raise RuntimeError("No real LLM provider configured. Set LLM_API_KEY, LLM_API_URL and LLM_MODEL.")
        yield
