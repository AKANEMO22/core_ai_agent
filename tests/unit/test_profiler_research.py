from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.profiler.agent import ProfilerAgent
from core.profiler.tools import SafeWebSearchTool
from core.runtime.providers.mock import MockLLMClient


@pytest.mark.asyncio
async def test_brave_search_uses_supported_global_country_and_keeps_page_age():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "web": {
            "results": [
                {
                    "url": "https://careers.example.com/interview",
                    "title": "Software engineer interview process",
                    "description": "The technical interview takes 45 minutes.",
                    "page_age": "3 weeks ago",
                }
            ]
        }
    }
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=client):
        result = await SafeWebSearchTool(api_key="test-secret").search(
            "Example software engineer interview", max_results=1
        )

    assert result["status"] == "success"
    assert client.get.await_args.kwargs["params"]["country"] == "ALL"
    assert result["results"][0]["source_age"] == "3 weeks ago"
    assert "test-secret" not in repr(result)


@pytest.mark.asyncio
async def test_company_recon_reports_provider_errors_instead_of_empty_results():
    class FailingSearchTool:
        name = "safe_web_search"

        async def execute(self, _payload, workspace=None):
            return {"status": "error", "reason": "provider rejected request"}

    profiler = ProfilerAgent(llm_client=MockLLMClient(), tools=[FailingSearchTool()])
    recon = await profiler._gather_recon(
        "Example", "Software Engineer Intern", workspace=None
    )

    assert recon is not None
    assert recon["web_recon"] == []
    assert recon["web_recon_status"] == "error"
