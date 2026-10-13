"""Global configuration and settings management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import List

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
        RETRIEVAL_BACKEND: Retrieval adapter name ("rag" by default; "mock" is explicit).
        RETRIEVAL_TIMEOUT_SECONDS: Deadline for each retrieval call during profiling.
        RETRIEVAL_QUESTIONS_PER_COMPETENCY: Question bank items retrieved per competency.
        RETRIEVAL_COMPANY_TOP_K: Company knowledge passages retrieved per session.

    Lifecycle, realtime, assessor, orchestration, decision and retention settings are documented
    by their field descriptions below and in docs/runbook.md. `validate_settings` lists the
    requirements enforced when APP_ENV=production.
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
    WORKSPACE_FSYNC: bool = Field(
        default=True,
        description="Enforce os.fsync on atomic file writes",
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
    LLM_API_KEY: str = Field(
        default="",
        description="API key of an OpenAI-compatible endpoint (takes precedence over OPENAI_API_KEY)",
    )
    LLM_MODEL: str = Field(
        default="",
        description="Model id for the OpenAI-compatible endpoint (takes precedence over DEFAULT_LLM_MODEL)",
    )
    LLM_API_URL: str = Field(
        default="",
        description="Base URL of an OpenAI-compatible endpoint, e.g. https://host/v1",
    )

    @property
    def llm_api_key(self) -> str:
        """Effective provider key."""
        return self.LLM_API_KEY or self.OPENAI_API_KEY

    @property
    def llm_model(self) -> str:
        """Effective model id."""
        return self.LLM_MODEL or self.DEFAULT_LLM_MODEL
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

    # Environment & product
    APP_ENV: str = Field(
        default="development",
        description="'development' or 'production'; production enables fail-fast config validation",
    )
    PRODUCT_MODE: str = Field(
        default="practice",
        description="'practice' (readiness coaching) or 'hiring_assist' (requires human review)",
    )
    CORS_ALLOWED_ORIGINS: List[str] = Field(
        default_factory=list,
        description="Explicit CORS origins; empty in development allows all without credentials",
    )
    DECISION_BORDERLINE_BAND: float = Field(
        default=0.25,
        description="Distance from the pass threshold (1-4 scale) that yields NEEDS_REVIEW",
    )
    ALLOW_MOCK_RETRIEVAL: bool = Field(
        default=False,
        description="Permit RETRIEVAL_BACKEND=mock when APP_ENV=production",
    )

    # Lifecycle persistence & retention
    SESSION_STATE_TTL_SECONDS: int = Field(
        default=7 * 24 * 3600,
        description="TTL for Redis session state, refreshed on every transition",
    )
    IDEMPOTENCY_TTL_SECONDS: int = Field(
        default=24 * 3600,
        description="TTL for idempotency records and message-to-turn mappings",
    )
    EVENT_TTL_SECONDS: int = Field(
        default=24 * 3600,
        description="Retention for replayable session events",
    )
    EVENT_STREAM_MAX_LENGTH: int = Field(
        default=20000,
        description="Maximum retained events per session",
    )
    EVENT_MEMORY_MAX_SESSIONS: int = Field(
        default=1000,
        description="Sessions kept in the in-memory event cache (LRU)",
    )
    RAW_UPLOAD_RETENTION_HOURS: int = Field(
        default=24,
        description="Hours after report generation before raw uploads are deleted",
    )
    SESSION_ARTIFACT_RETENTION_DAYS: int = Field(
        default=30,
        description="Days before remaining session artifacts are deleted",
    )
    RETENTION_SWEEP_INTERVAL_SECONDS: float = Field(
        default=3600.0,
        description="Interval of the background retention sweeper",
    )

    # Profiling & turns
    PROFILING_DEADLINE_SECONDS: float = Field(
        default=90.0,
        description="Deadline for pre-flight profiling before the session fails",
    )
    TURN_DEADLINE_SECONDS: float = Field(
        default=30.0,
        description="Hard deadline for one interviewer generation",
    )
    TURN_LOCK_TTL_MS: int = Field(
        default=60000,
        description="Lease of the per-session turn lock",
    )
    MAX_INTERACTIVE_GENERATIONS: int = Field(
        default=20,
        description="Concurrent interviewer generations across sessions",
    )
    MAX_INTERACTIVE_QUEUE_MS: int = Field(
        default=1500,
        description="Maximum wait for an interactive generation slot before replying busy",
    )
    BUSY_RETRY_AFTER_MS: int = Field(
        default=1000,
        description="retry_after_ms hint sent with busy responses",
    )
    MAX_ANSWER_CHARS: int = Field(default=20000, description="Maximum characters in one candidate answer")
    MAX_INBOUND_FRAME_BYTES: int = Field(default=65536, description="Maximum inbound WebSocket frame size")
    ANSWER_RATE_PER_SECOND: float = Field(default=1.0, description="Answer submissions per session per second")
    ANSWER_BURST: float = Field(default=3.0, description="Answer submission burst per session")
    RECONNECT_RATE_PER_MINUTE: float = Field(default=5.0, description="WebSocket connects per session per minute")
    SESSION_CREATE_RATE_PER_SECOND: float = Field(default=5.0, description="Session creations per caller per second")
    SESSION_CREATE_BURST: float = Field(default=20.0, description="Session creation burst per caller")

    # Assessor pool & report pipeline
    ASSESSOR_POOL_SIZE: int = Field(default=4, description="Assessor consumer tasks per process")
    ASSESSOR_BLOCK_MS: int = Field(default=1000, description="XREADGROUP block time per poll")
    ASSESSOR_LEASE_MS: int = Field(default=60000, description="Idle time before a pending job is reclaimed")
    ASSESSOR_RECOVERY_INTERVAL_SECONDS: float = Field(default=15.0, description="XAUTOCLAIM interval")
    ASSESSOR_HEARTBEAT_TTL_SECONDS: int = Field(default=15, description="Worker heartbeat key TTL")
    REPORT_SWEEP_INTERVAL_SECONDS: float = Field(default=60.0, description="Barrier/synthesis recovery sweep interval")
    SYNTHESIS_LOCK_TTL_SECONDS: int = Field(default=120, description="Lease of the per-session synthesis lock")
    MAX_ASSESSOR_LLM_CALLS: int = Field(default=8, description="Concurrent evaluator LLM calls")
    MAX_PROFILER_LLM_CALLS: int = Field(default=8, description="Concurrent profiler LLM calls")
    MAX_SYNTHESIS_LLM_CALLS: int = Field(default=4, description="Concurrent report synthesis LLM calls")

    # Orchestration (control plane)
    ORCHESTRATOR_ENABLED: bool = Field(default=True, description="Run an orchestration cycle after each turn")
    ORCHESTRATOR_DEADLINE_MS: int = Field(default=20000, description="Deadline of one orchestration cycle (runs while the candidate speaks)")
    ORCHESTRATOR_JOIN_MS: int = Field(default=6000, description="Wait for fast advisory agents (follow-up, guard) within a cycle")
    SPEAKER_HEDGE_AFTER_MS: int = Field(
        default=2500, ge=0, description="Start a second speaker request if no first token by then (0 disables)"
    )
    BACKGROUND_YIELD_MAX_MS: int = Field(
        default=3000, description="Max wait of a background LLM call for in-flight speaker turns to finish"
    )
    COVERAGE_MIN_CONFIDENCE: float = Field(default=0.6, description="Summed confidence that marks a criterion covered")
    STAGE_MAX_EXTRA_TURNS: int = Field(
        default=1, ge=0, description="Turns a stage may run past its budget while a must-have lacks evidence"
    )
    ROLLING_SUMMARY_ENABLED: bool = Field(default=True, description="Maintain the speaker rolling summary")
    SPEAKER_RECENT_MESSAGES: int = Field(default=6, description="Raw messages always kept in the speaker prompt")
    BRIEF_ENABLED: bool = Field(default=True, description="Write the interview brief after profiling")
    MAX_ORCHESTRATOR_LLM_CALLS: int = Field(default=16, description="Concurrent orchestrator/subagent LLM calls")
    AGENT_PROFILES_OVERRIDES: str = Field(default="", description="JSON overrides merged into agent profiles")

    # Retrieval (RAG)
    RETRIEVAL_BACKEND: str = Field(
        default="rag",
        description="Retrieval adapter name registered in core/retrieval/factory.py",
    )
    RAG_DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data",
        description="Directory containing the sanitized interview framework corpus",
    )
    RAG_INDEX_PATH: Path = Field(
        default=Path("./data/rag/index.db"),
        description="Writable SQLite path for versioned RAG index snapshots",
    )
    RAG_EMBEDDING_PROVIDER: str = Field(
        default="lsa",
        description="Embedding provider used to bootstrap the offline RAG index",
    )
    RETRIEVAL_TIMEOUT_SECONDS: float = Field(
        default=3.0,
        description="Deadline for each retrieval call during profiling",
    )
    RETRIEVAL_QUESTIONS_PER_COMPETENCY: int = Field(
        default=4,
        description="Question bank items retrieved per rubric competency",
    )
    RETRIEVAL_COMPANY_TOP_K: int = Field(
        default=5,
        description="Company knowledge passages retrieved per session",
    )


DEFAULT_JWT_SECRET = "supersecretjwtkeyforinterviewsdefaultchangeinproduction"


def validate_settings(settings: Settings) -> List[str]:
    """Return configuration errors that must block startup when APP_ENV=production.

    Development mode reports nothing so local runs and tests keep working with defaults.
    """
    if settings.APP_ENV.strip().lower() != "production":
        return []
    errors: List[str] = []
    if settings.JWT_SECRET == DEFAULT_JWT_SECRET or len(settings.JWT_SECRET.encode("utf-8")) < 32:
        errors.append("JWT_SECRET must be set to a non-default value of at least 32 bytes")
    if not settings.CORS_ALLOWED_ORIGINS or "*" in settings.CORS_ALLOWED_ORIGINS:
        errors.append("CORS_ALLOWED_ORIGINS must list explicit origins in production")
    if not settings.llm_api_key:
        errors.append("An LLM provider key (LLM_API_KEY or OPENAI_API_KEY) is required in production")
    if settings.RETRIEVAL_BACKEND.strip().lower() == "mock" and not settings.ALLOW_MOCK_RETRIEVAL:
        errors.append("RETRIEVAL_BACKEND=mock requires ALLOW_MOCK_RETRIEVAL=true in production")
    if settings.RETRIEVAL_BACKEND.strip().lower() == "rag":
        if settings.RAG_EMBEDDING_PROVIDER.strip().lower() not in {"lsa", "gemini", "neural"}:
            errors.append("RAG_EMBEDDING_PROVIDER must be lsa, gemini, or neural")
        if not (settings.RAG_DATA_DIR / "interview_frameworks.json").is_file():
            errors.append("RAG_DATA_DIR must contain interview_frameworks.json")
    if settings.PRODUCT_MODE not in ("practice", "hiring_assist"):
        errors.append("PRODUCT_MODE must be 'practice' or 'hiring_assist'")
    if not settings.REDIS_URL:
        errors.append("REDIS_URL is required")
    return errors


@lru_cache()
def get_settings() -> Settings:
    """Return the cached singleton instance of application settings.

    Returns:
        Settings: The singleton configuration instance.
    """
    return Settings()
