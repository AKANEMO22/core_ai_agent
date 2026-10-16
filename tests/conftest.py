"""Test isolation: never call a real LLM unless explicitly requested.

Real-provider suites opt in with RUN_PROVIDER_CONTRACT=1 or RUN_LIVE_LLM=1 (credentials from .env).
"""

import os

if os.getenv("RUN_PROVIDER_CONTRACT") != "1" and os.getenv("RUN_LIVE_LLM") != "1":
    os.environ["LLM_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""

# The developer's .env may point DATABASE_URL at the real Postgres; tests must never write there.
# Postgres suites use TEST_DATABASE_URL explicitly.
os.environ["DATABASE_URL"] = ""

from config.settings import get_settings  # noqa: E402

get_settings.cache_clear()
