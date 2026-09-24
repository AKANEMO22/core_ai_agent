"""Global configuration and settings management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configurations.

    Attributes:
        WORKSPACE_ROOT_BASE: Base directory path for isolated session workspaces.
        MAX_TOOL_RESULT_BYTES: Maximum tool execution output size before spilling (32 KB).
        MAX_UPLOAD_SIZE_BYTES: Maximum allowed file upload size (default 10 MB).
        MAX_JSON_DEPTH: Maximum allowed nesting depth for uploaded JSON structures.
        SESSION_RETENTION_HOURS: Inactivity duration before workspaces become eligible for cleanup.
        REDIS_URL: Connection URL for Redis Streams event queues.
        JWT_SECRET: Shared secret key for HMAC SHA-256 JWT tokens.
        JWT_ALGORITHM: JWT signing algorithm.
        SERVICE_TOKEN_EXPIRE_MINUTES: Expiration window for backend service tokens.
        SESSION_TOKEN_EXPIRE_MINUTES: Expiration window for candidate session tokens.
        OPENAI_API_KEY: API key for OpenAI LLM services.
        DEFAULT_LLM_MODEL: Default language model name for runtime agents.
        LLM_TIMEOUT_SECONDS: Per-request timeout for LLM inferences.
        MAX_CONCURRENT_SESSIONS: Max parallel sessions supported on single node.
        SSRF_TIMEOUT_SECONDS: Timeout for external web queries in profiler.
        MAX_SUBAGENT_PROMPT_TOKENS: ContextAssembler envelope ceiling for subagents.
        ASSESSMENT_STREAM_KEY: Redis stream key for assessment jobs.
        ASSESSMENT_CONSUMER_GROUP: Redis consumer group name for assessor workers.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Workspace & Isolation
    WORKSPACE_ROOT_BASE: Path = Field(
        default=Path("./data/workspaces"),
        description="Root directory for session workspaces",
    )
    MAX_TOOL_RESULT_BYTES: int = Field(
        default=32768,
        description="Max tool result size in bytes before spilling to spill/ directory (32 KB)",
    )
    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=10 * 1024 * 1024,
        description="Max upload size in bytes (10 MB)",
    )
    MAX_JSON_DEPTH: int = Field(
        default=10,
        description="Max allowed nesting depth for JSON inputs",
    )
    SESSION_RETENTION_HOURS: int = Field(
        default=24,
        description="Retention duration for session workspaces in hours",
    )

    # Message Queue & State
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for streams and state",
    )
    ASSESSMENT_STREAM_KEY: str = Field(
        default="stream:assessment_jobs",
        description="Redis Stream key for asynchronous assessment jobs",
    )
    ASSESSMENT_CONSUMER_GROUP: str = Field(
        default="group:assessor_workers",
        description="Consumer group name for assessor workers",
    )

    # Authentication & Security
    JWT_SECRET: str = Field(
        default="supersecretjwtkeyforinterviewsdefaultchangeinproduction",
        description="Secret key for signing and verifying JWT tokens",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Algorithm used for JWT signing",
    )
    SERVICE_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        description="Service JWT expiration period in minutes",
    )
    SESSION_TOKEN_EXPIRE_MINUTES: int = Field(
        default=180,
        description="Candidate session JWT expiration period in minutes",
    )

    # Runtime & LLM
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key",
    )
    DEFAULT_LLM_MODEL: str = Field(
        default="gpt-4o",
        description="Default model for interview agents",
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        description="Timeout for LLM inference calls in seconds",
    )
    MAX_CONCURRENT_SESSIONS: int = Field(
        default=100,
        description="Maximum concurrent active interview sessions",
    )
    SSRF_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        description="Network timeout for outbound reconnaissance requests",
    )
    MAX_SUBAGENT_PROMPT_TOKENS: int = Field(
        default=1500,
        description="Maximum token budget for subagent context envelope",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return the cached singleton instance of application settings.

    Returns:
        Settings: The singleton configuration instance.
    """
    return Settings()
