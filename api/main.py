"""FastAPI application factory, CORS configuration, health probes, and runtime service assembly."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import fakeredis.aioredis
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router as sessions_router
from api.websocket import ConnectionManager, ws_router
from config.settings import Settings, get_settings
from core.assessor.queue import RedisAssessmentQueue
from core.interview.context import ContextAssembler
from core.interview.director import StageDirector
from core.interview.runner import InteractiveSubagentRunner
from core.profiler.agent import ProfilerAgent
from core.runtime.agent import RuntimeAgent, sanitize_secrets
from core.runtime.client import RuntimeLLMClient
from core.runtime.providers.mock import MockLLMClient
from core.runtime.providers.openai import OpenAIRuntimeLLMClient
from core.session.events import SessionEventStore
from core.session.idempotency import IdempotencyManager, TokenBucket
from core.session.state import LifecycleStatus, SessionStateManager
from core.workspace.cleanup import WorkspaceCleanupManager
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager

logger = logging.getLogger(__name__)


def create_app(
    workspace_base: Optional[Union[str, Path]] = None,
    fake_redis: bool = False,
    redis_client: Optional[Any] = None,
    llm_client: Optional[RuntimeLLMClient] = None,
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> FastAPI:
    """Create and configure a production-ready FastAPI application gateway.

    Args:
        workspace_base: Base directory path for isolated session workspaces.
        fake_redis: If True, uses fakeredis aioredis client (ideal for tests).
        redis_client: Optional pre-configured Redis client.
        llm_client: Optional LLM runtime client adapter.
        settings: Application settings configuration instance.
        **kwargs: Additional keyword arguments.

    Returns:
        FastAPI: Fully wired FastAPI application instance.
    """
    app_settings = settings or get_settings()

    app = FastAPI(
        title="AI Interview Platform Core Engine",
        version="1.0.0",
        description="External Access Gateway, Auth, WebSocket Stream & Connection Manager",
    )

    # 1. CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Workspace Management
    ws_base_path = Path(workspace_base) if workspace_base else app_settings.WORKSPACE_ROOT_BASE
    ws_base_path.mkdir(parents=True, exist_ok=True)
    workspace_mgr = SessionWorkspaceManager(root_base=str(ws_base_path))
    cleanup_mgr = WorkspaceCleanupManager(root_base=str(ws_base_path))

    # 3. Redis Client & Event Storage
    active_redis = redis_client
    if active_redis is None:
        if fake_redis:
            active_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        else:
            try:
                import redis.asyncio as aioredis
                active_redis = aioredis.from_url(
                    app_settings.REDIS_URL,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning(
                    "Could not initialize connection to Redis URL '%s': %s. Falling back to FakeRedis.",
                    app_settings.REDIS_URL,
                    sanitize_secrets(str(e)),
                )
                active_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    event_store = SessionEventStore(redis_client=active_redis)
    connection_mgr = ConnectionManager(event_store=event_store)

    # 4. Assessment Queue & State Management
    assessment_queue = RedisAssessmentQueue(
        redis_client=active_redis,
        stream_key=app_settings.ASSESSMENT_STREAM_KEY,
    )
    state_mgr = SessionStateManager()
    idempotency_mgr = IdempotencyManager(
        rate_limiter=TokenBucket(rate=10.0, capacity=20.0),
    )

    # 5. LLM Client and Profiler Setup
    active_llm = llm_client
    if active_llm is None:
        if not fake_redis and app_settings.OPENAI_API_KEY:
            active_llm = OpenAIRuntimeLLMClient(
                api_key=app_settings.OPENAI_API_KEY,
                model=app_settings.DEFAULT_LLM_MODEL,
            )
        else:
            active_llm = MockLLMClient(
                canned_tokens=["Hello", " candidate,", " welcome to your", " interview."],
                canned_json_responses={
                    "company_intel": {
                        "company": "Target Company",
                        "culture": "Engineering Excellence",
                        "tech_stack": ["Python", "FastAPI", "Redis"],
                    },
                    "candidate_profile": {
                        "strengths": ["Backend Design"],
                        "gaps": ["Distributed Locks"],
                        "probes": ["Concurrency"],
                    },
                    "interview_spec": {
                        "interviewer_persona": "Tech Lead",
                        "tone": "rigorous_challenger",
                        "knowledge_scope": ["Concurrency", "Systems"],
                    },
                    "stage_plan": {
                        "stages": [
                            {"stage_id": "warmup", "name": "Warmup", "target_turns": 2},
                            {"stage_id": "technical", "name": "Technical Depth", "target_turns": 3},
                        ]
                    },
                },
            )

    profiler_agent = ProfilerAgent(
        llm_client=active_llm,
        tools=[] if isinstance(active_llm, MockLLMClient) else None,
    )

    # 6. Concurrency Limiter
    concurrency_semaphore = asyncio.Semaphore(app_settings.MAX_CONCURRENT_SESSIONS)

    # 7. Runner Registry / Resolver
    active_runners: Dict[str, InteractiveSubagentRunner] = {}

    async def get_or_create_runner(session_id: str) -> Optional[InteractiveSubagentRunner]:
        if session_id in active_runners:
            return active_runners[session_id]

        ws = workspace_mgr.get_or_create_workspace(session_id)
        if not ws.root_path.exists():
            return None

        # Load stage plan if generated by profiler
        stage_plan_file = "context/stage_plan.json"
        stages = [
            {"stage_id": "warmup", "name": "Warmup", "target_turns": 2},
            {"stage_id": "technical", "name": "Technical Depth", "target_turns": 3},
        ]
        if ws.exists(stage_plan_file):
            try:
                plan_data = ws.read_json(stage_plan_file)
                if "stages" in plan_data and isinstance(plan_data["stages"], list):
                    stages = plan_data["stages"]
            except Exception as e:
                logger.warning("Could not read stage plan for '%s': %s", session_id, sanitize_secrets(str(e)))

        director = StageDirector(stages=stages, state_mgr=state_mgr)
        assembler = ContextAssembler(max_tokens=app_settings.MAX_SUBAGENT_PROMPT_TOKENS)
        agent = RuntimeAgent(client=active_llm, workspace=ws)

        runner = InteractiveSubagentRunner(
            workspace=ws,
            runtime_agent=agent,
            context_assembler=assembler,
            stage_director=director,
            assessment_queue=assessment_queue,
            state_mgr=state_mgr,
        )
        active_runners[session_id] = runner
        return runner

    # Wire state onto application
    app.state.settings = app_settings
    app.state.workspace_mgr = workspace_mgr
    app.state.cleanup_mgr = cleanup_mgr
    app.state.redis_client = active_redis
    app.state.event_store = event_store
    app.state.connection_mgr = connection_mgr
    app.state.assessment_queue = assessment_queue
    app.state.state_mgr = state_mgr
    app.state.idempotency_mgr = idempotency_mgr
    app.state.llm_client = active_llm
    app.state.profiler_agent = profiler_agent
    app.state.concurrency_semaphore = concurrency_semaphore
    app.state.get_runner = get_or_create_runner

    # 8. Health Routes
    @app.get("/health/live", tags=["health"])
    async def health_live() -> Dict[str, Any]:
        """Liveness probe: verifies process is alive and accepting connections."""
        return {"status": "ok", "live": True}

    @app.get("/health/ready", tags=["health"])
    async def health_ready() -> Any:
        """Readiness probe: validates Redis connectivity, workspace disk write, and worker pool."""
        details: Dict[str, Any] = {}
        is_ready = True

        # Probe Redis
        if active_redis is not None:
            try:
                ping_res = active_redis.ping()
                if hasattr(ping_res, "__await__"):
                    await ping_res
                details["redis"] = "connected"
            except Exception as e:
                details["redis"] = f"unreachable: {sanitize_secrets(str(e))}"
                is_ready = False
        else:
            details["redis"] = "not_configured"

        # Probe Workspace writable
        try:
            probe_file = ws_base_path / f".health_probe_{uuid.uuid4().hex[:8]}"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
            details["workspace"] = "writable"
        except Exception as e:
            details["workspace"] = f"unwritable: {sanitize_secrets(str(e))}"
            is_ready = False

        details["max_concurrent_sessions"] = app_settings.MAX_CONCURRENT_SESSIONS

        if not is_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "details": details},
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "details": details},
        )

    # 9. Mount Routers
    app.include_router(sessions_router)
    app.include_router(ws_router)

    return app


# Default ASGI application instance
app = create_app()
