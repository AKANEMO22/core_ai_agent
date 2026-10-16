"""FastAPI application factory: service graph assembly, lifecycle, health and metrics."""

import asyncio
import contextlib
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

import fakeredis.aioredis
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api.auth import SessionRevocationList, require_service_token
from api.inspect import router as inspect_router
from api.routes import router as sessions_router
from api.websocket import ConnectionManager, ws_router
from config.settings import Settings, get_settings, validate_settings
from core.assessor.pipeline import AssessmentPipeline
from core.assessor.pool import AssessorWorkerPool
from core.assessor.queue import RedisAssessmentQueue
from core.agents.narrative import NarrativeWriter
from core.assessor.decision import DecisionConfig
from core.assessor.synthesizer import ReportSynthesizer, ReviewService
from core.assessor.worker import AssessorWorker
from core.interview.context import ContextAssembler
from core.interview.controller import InterviewSessionController, RunnerRegistry
from core.interview.director import StageDirector
from core.interview.runner import InteractiveSubagentRunner
from core.interview.speaker import SpeakerContextBuilder
from core.interview.summary import RollingSummarizer
from core.agents import CriterionEvaluator, register_subagents
from core.orchestration.blackboard import Blackboard
from core.orchestration.brief import InterviewBriefWriter
from core.orchestration.delegate import DelegationRuntime
from core.orchestration.orchestrator import InterviewOrchestrator
from core.runtime.hooks import HookBus
from core.runtime.policy import ToolPolicy
from core.runtime.priority import BackgroundLLMClient, HedgedLLMClient, HotPathGate
from core.runtime.profiles import load_profiles
from core.runtime.skills import SkillRegistry
from core.observability import build_default_metrics, configure_logging, correlation_id_var
from fe_inspect.instrumentation import RuntimeTraceRecorder
from fe_inspect.provider import UnconfiguredLLMClient
from core.profiler.agent import ProfilerAgent
from core.profiler.tools import SafeWebSearchTool
from core.retrieval.factory import build_knowledge_service
from core.retrieval.port import RetrievalClient
from core.runtime.agent import RuntimeAgent, sanitize_secrets
from core.runtime.client import ProfiledLLMClient, RuntimeLLMClient, verify_capabilities
from core.runtime.providers.mock import MockLLMClient
from core.runtime.providers.openai import OpenAIRuntimeLLMClient
from core.runtime.providers.scripted import ScriptedInterviewLLM
from core.session.cancellation import CancellationRegistry
from core.session.events import SessionEventStore
from core.session.idempotency import MessageLedger, RedisIdempotencyStore, RedisLease, TokenBucket
from core.session.store import RedisSessionStateStore
from core.storage import ArtifactArchive, S3ObjectStore
from core.storage.database import Database
from core.workspace.cleanup import WorkspaceCleanupManager
from core.workspace.retention import RetentionService
from core.workspace.session import SessionWorkspaceManager, normalize_session_id

logger = logging.getLogger(__name__)

# Upper bound for /end waiting on an in-flight turn/assessment lock; the ENDING sweep retries.
END_PREPARE_LOCK_TIMEOUT_SECONDS = 5.0

DEFAULT_STAGES = [
    {"stage_id": "warmup", "name": "Warmup", "target_turns": 2},
    {"stage_id": "technical", "name": "Technical Depth", "target_turns": 3},
]


def _default_mock_llm() -> MockLLMClient:
    """Deterministic scripted provider used when no API key is configured (development only).

    It answers every structured prompt kind, so the whole pipeline runs end to end locally.
    """
    return ScriptedInterviewLLM(
        speaker_tokens=["Hello", " candidate,", " welcome to your", " interview."],
        stage_plan={"stages": DEFAULT_STAGES},
    )


def create_app(
    workspace_base: Optional[Union[str, Path]] = None,
    fake_redis: bool = False,
    redis_client: Optional[Any] = None,
    llm_client: Optional[RuntimeLLMClient] = None,
    settings: Optional[Settings] = None,
    retrieval_client: Optional[RetrievalClient] = None,
    artifact_archive: Optional[ArtifactArchive] = None,
    **kwargs: Any,
) -> FastAPI:
    """Create the AI runtime application.

    Args:
        workspace_base: Base directory path for isolated session workspaces.
        fake_redis: If True, uses an in-process fakeredis client (tests and local runs).
        redis_client: Optional pre-configured async Redis client.
        llm_client: Optional LLM runtime client adapter.
        settings: Application settings (defaults to environment settings).
        retrieval_client: Optional retrieval adapter; defaults to `RETRIEVAL_BACKEND`.

    Raises:
        RuntimeError: If APP_ENV=production and the configuration is unsafe.
    """
    app_settings = settings or get_settings()
    config_errors = validate_settings(app_settings)
    if config_errors:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(config_errors))
    if app_settings.APP_ENV.strip().lower() == "production" and app_settings.INSPECTOR_LOCAL_AUTH:
        raise RuntimeError("Core Studio local token issuing is development-only")

    # 1. Redis
    active_redis = redis_client
    if active_redis is None:
        if fake_redis:
            active_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        else:
            import redis.asyncio as aioredis

            active_redis = aioredis.from_url(app_settings.REDIS_URL, decode_responses=True)

    # 2. Workspace
    ws_base_path = Path(workspace_base) if workspace_base else Path(app_settings.WORKSPACE_ROOT_BASE)
    ws_base_path.mkdir(parents=True, exist_ok=True)
    workspace_mgr = SessionWorkspaceManager(root_base=str(ws_base_path), fsync=app_settings.WORKSPACE_FSYNC)
    cleanup_mgr = WorkspaceCleanupManager(root_base=str(ws_base_path))
    # Durable stores: PostgreSQL when DATABASE_URL is set, otherwise local SQLite files.
    owned_databases: list[Database] = []

    def open_store_database(schema: str) -> Optional[Database]:
        if not app_settings.DATABASE_URL:
            return None
        database = Database(
            app_settings.DATABASE_URL,
            schema=schema,
            min_size=app_settings.DATABASE_POOL_MIN_SIZE,
            max_size=app_settings.DATABASE_POOL_MAX_SIZE,
        )
        owned_databases.append(database)
        return database

    active_artifact_archive = artifact_archive
    if active_artifact_archive is None and app_settings.ARTIFACT_ARCHIVE_ENABLED:
        if not all(
            (
                app_settings.OBJECT_STORE_ENDPOINT,
                app_settings.OBJECT_STORE_ACCESS_KEY,
                app_settings.OBJECT_STORE_SECRET_KEY,
            )
        ):
            raise RuntimeError("S3-compatible object store endpoint and credentials are required when artifact archiving is enabled")
        active_artifact_archive = ArtifactArchive(
            object_store=S3ObjectStore(
                endpoint_url=app_settings.OBJECT_STORE_ENDPOINT,
                access_key=app_settings.OBJECT_STORE_ACCESS_KEY,
                secret_key=app_settings.OBJECT_STORE_SECRET_KEY,
                region=app_settings.OBJECT_STORE_REGION,
                verify_tls=app_settings.OBJECT_STORE_VERIFY_TLS,
                timeout_seconds=app_settings.OBJECT_STORE_TIMEOUT_SECONDS,
            ),
            bucket=app_settings.ARTIFACT_ARCHIVE_BUCKET,
            catalog_path=app_settings.ARTIFACT_ARCHIVE_CATALOG_PATH,
            max_restore_bytes=app_settings.ARTIFACT_ARCHIVE_MAX_RESTORE_BYTES,
            database=open_store_database("artifact_archive"),
        )
    if app_settings.APP_ENV.strip().lower() == "production" and active_artifact_archive is not None:
        raise RuntimeError(
            "Durable artifact archives require user ownership authorization before production use"
        )
    studio_enabled = app_settings.INSPECTOR_ENABLED or app_settings.TEST_UI_ENABLED
    strict_studio = studio_enabled and app_settings.STUDIO_REQUIRE_REAL_PROVIDER
    inspect_trace_recorder = (
        RuntimeTraceRecorder(workspace_mgr, database=open_store_database("inspect_trace")) if studio_enabled else None
    )

    # 3. Durable session primitives
    metrics = build_default_metrics()
    event_store = SessionEventStore(
        redis_client=active_redis,
        ttl_seconds=app_settings.EVENT_TTL_SECONDS,
        max_length=app_settings.EVENT_STREAM_MAX_LENGTH,
        memory_max_sessions=app_settings.EVENT_MEMORY_MAX_SESSIONS,
    )
    connection_mgr = ConnectionManager(event_store=event_store)
    state_store = RedisSessionStateStore(active_redis, ttl_seconds=app_settings.SESSION_STATE_TTL_SECONDS)
    idempotency = RedisIdempotencyStore(active_redis, ttl_seconds=app_settings.IDEMPOTENCY_TTL_SECONDS)
    ledger = MessageLedger(active_redis, ttl_seconds=app_settings.IDEMPOTENCY_TTL_SECONDS)
    lease = RedisLease(active_redis)
    revocations = SessionRevocationList(active_redis, ttl_seconds=app_settings.SESSION_STATE_TTL_SECONDS)
    cancellations = CancellationRegistry()
    assessment_queue = RedisAssessmentQueue(
        redis_client=active_redis,
        stream_key=app_settings.ASSESSMENT_STREAM_KEY,
        dead_letter_stream_key=f"{app_settings.ASSESSMENT_STREAM_KEY}:dead_letter",
        counter_ttl_seconds=app_settings.SESSION_STATE_TTL_SECONDS,
    )

    # 4. LLM provider, retrieval and profiler
    active_llm = llm_client
    if active_llm is None:
        if not fake_redis and app_settings.llm_api_key:
            active_llm = OpenAIRuntimeLLMClient(
                api_key=app_settings.llm_api_key,
                model=app_settings.llm_model,
                base_url=app_settings.LLM_API_URL or None,
            )
        elif strict_studio:
            active_llm = UnconfiguredLLMClient()
        else:
            active_llm = _default_mock_llm()
    verify_capabilities(active_llm)
    if inspect_trace_recorder is not None:
        inspect_trace_recorder.instrument_events(event_store)
        inspect_trace_recorder.instrument_state(state_store)
        if hasattr(active_llm, "_stream_once"):
            inspect_trace_recorder.wrap(active_llm, "_stream_once", "Provider HTTP attempt", "llm")

    # Agent profiles: each structured agent calls the provider with its own profile model.
    profiles = load_profiles(overrides_json=app_settings.AGENT_PROFILES_OVERRIDES, app_env=app_settings.APP_ENV)

    hot_path_gate = HotPathGate()
    # Hedging targets real network providers; scripted/mock clients have no latency tail.
    speaker_llm: RuntimeLLMClient = (
        HedgedLLMClient(active_llm, app_settings.SPEAKER_HEDGE_AFTER_MS / 1000, metrics=metrics)
        if isinstance(active_llm, OpenAIRuntimeLLMClient) and app_settings.SPEAKER_HEDGE_AFTER_MS > 0
        else active_llm
    )
    if inspect_trace_recorder is not None:
        # A separate wrapper prevents tracing a shared provider as "speaker" during
        # background agent calls. HTTP attempts remain children of the active caller.
        speaker_llm = ProfiledLLMClient(speaker_llm)

    def llm_for(profile_name: str) -> RuntimeLLMClient:
        """Client for a background agent: its profile model, yielding to in-flight speaker turns."""
        profile = profiles.get(profile_name)
        client = BackgroundLLMClient(
            ProfiledLLMClient(active_llm, profile.model if profile else ""),
            hot_path_gate,
            max_wait_seconds=app_settings.BACKGROUND_YIELD_MAX_MS / 1000,
        )
        if inspect_trace_recorder is not None:
            inspect_trace_recorder.wrap(client, "stream_chat", f"{profile_name} model", "llm", profile_name)
        return client

    knowledge_service = build_knowledge_service(app_settings, client=retrieval_client)
    profiler_tools = (
        [SafeWebSearchTool(api_key=app_settings.BRAVE_SEARCH_API_KEY)]
        if app_settings.BRAVE_SEARCH_API_KEY and not isinstance(active_llm, MockLLMClient)
        else []
    )
    profiler_agent = ProfilerAgent(
        llm_client=llm_for("profiler"),
        tools=profiler_tools,
        knowledge_service=knowledge_service,
    )

    # 5. Concurrency governors
    concurrency_semaphore = asyncio.Semaphore(app_settings.MAX_CONCURRENT_SESSIONS)
    generation_slots = asyncio.Semaphore(app_settings.MAX_INTERACTIVE_GENERATIONS)
    profiler_semaphore = asyncio.Semaphore(app_settings.MAX_PROFILER_LLM_CALLS)
    assessor_semaphore = asyncio.Semaphore(app_settings.MAX_ASSESSOR_LLM_CALLS)

    # 6. Assessment pipeline
    synthesis_semaphore = asyncio.Semaphore(app_settings.MAX_SYNTHESIS_LLM_CALLS)
    synthesizer = ReportSynthesizer(
        redis_client=active_redis,
        workspace_resolver=workspace_mgr,
        state_manager=state_store,
        event_store=event_store,
        llm_client=active_llm,
        connection_manager=connection_mgr,
        decision_config=DecisionConfig(
            min_confidence=app_settings.COVERAGE_MIN_CONFIDENCE,
            borderline_band=app_settings.DECISION_BORDERLINE_BAND,
            product_mode=app_settings.PRODUCT_MODE,
        ),
        narrative_writer=NarrativeWriter(
            llm_for("narrative"),
            deadline_seconds=profiles["narrative"].turn_deadline_ms / 1000,
            semaphore=synthesis_semaphore,
        ),
        extra_versions={"narrative_prompt": "report_narrative@1", "orchestrator_prompt": "orchestrator_decision@1"},
        artifact_archive=active_artifact_archive,
    )
    pipeline = AssessmentPipeline(
        redis_client=active_redis,
        state_store=state_store,
        synthesizer=synthesizer,
        counter_ttl_seconds=app_settings.SESSION_STATE_TTL_SECONDS,
        synthesis_lock_ttl_seconds=app_settings.SYNTHESIS_LOCK_TTL_SECONDS,
    )
    worker = AssessorWorker(
        redis_client=active_redis,
        queue=assessment_queue,
        llm_client=active_llm,
        workspace_resolver=workspace_mgr,
        state_manager=state_store,
        event_store=event_store,
        connection_manager=connection_mgr,
        on_assessment_completed=pipeline.maybe_synthesize,
        llm_semaphore=assessor_semaphore,
        evaluator=CriterionEvaluator(
            llm_for("evaluator"),
            workspace_mgr,
            deadline_seconds=profiles["evaluator"].turn_deadline_ms / 1000,
        ),
    )
    worker_pool = AssessorWorkerPool(
        worker=worker,
        queue=assessment_queue,
        redis_client=active_redis,
        size=app_settings.ASSESSOR_POOL_SIZE,
        group_name=app_settings.ASSESSMENT_CONSUMER_GROUP,
        block_ms=app_settings.ASSESSOR_BLOCK_MS,
        lease_ms=app_settings.ASSESSOR_LEASE_MS,
        recovery_interval_seconds=app_settings.ASSESSOR_RECOVERY_INTERVAL_SECONDS,
        heartbeat_ttl_seconds=app_settings.ASSESSOR_HEARTBEAT_TTL_SECONDS,
        sweeper=pipeline.sweep,
        sweep_interval_seconds=app_settings.REPORT_SWEEP_INTERVAL_SECONDS,
    )
    retention = RetentionService(
        state_store=state_store,
        workspace_mgr=workspace_mgr,
        cleanup_mgr=cleanup_mgr,
        raw_upload_retention_hours=app_settings.RAW_UPLOAD_RETENTION_HOURS,
        artifact_retention_days=app_settings.SESSION_ARTIFACT_RETENTION_DAYS,
        artifact_archive=active_artifact_archive,
    )

    # 7. Harness: profiles, hooks, skills
    hooks = HookBus()
    skills = SkillRegistry()
    tool_policy = ToolPolicy()
    orchestration_semaphore = asyncio.Semaphore(app_settings.MAX_ORCHESTRATOR_LLM_CALLS)

    # 8. Interview runners and controller
    async def build_runner(session_id: str) -> Optional[InteractiveSubagentRunner]:
        ws = workspace_mgr.get_or_create_workspace(session_id)
        if not ws.root_path.exists():
            return None
        stages = list(DEFAULT_STAGES)
        if ws.exists("context/stage_plan.json"):
            try:
                plan_data = ws.read_json("context/stage_plan.json")
                if isinstance(plan_data.get("stages"), list) and plan_data["stages"]:
                    stages = plan_data["stages"]
            except Exception as e:
                logger.warning("Could not read stage plan: %s", sanitize_secrets(str(e)))
        speaker = profiles["speaker"]
        runtime_agent = RuntimeAgent(
            client=speaker_llm, workspace=ws, profile=speaker, hooks=hooks,
            policy=tool_policy, metrics=metrics,
        )
        if inspect_trace_recorder is not None:
            inspect_trace_recorder.bind_agent(runtime_agent, "speaker", "session", session_id)
            inspect_trace_recorder.wrap(runtime_agent, "run_loop_streaming", "Speaker turn", "agent")
            if runtime_agent.compactor is not None:
                inspect_trace_recorder.wrap(runtime_agent.compactor, "maybe_compact", "Context compaction", "operation", "speaker")
            for tool in runtime_agent.tools:
                inspect_trace_recorder.wrap(tool, "execute", f"Tool: {tool.name}", "tool", "speaker")
        return InteractiveSubagentRunner(
            workspace=ws,
            runtime_agent=runtime_agent,
            context_assembler=ContextAssembler(max_tokens=app_settings.MAX_SUBAGENT_PROMPT_TOKENS),
            stage_director=StageDirector(
                stages=stages, state_mgr=state_store, max_extra_turns=app_settings.STAGE_MAX_EXTRA_TURNS
            ),
            assessment_queue=assessment_queue,
            state_mgr=state_store,
            turn_deadline_seconds=app_settings.TURN_DEADLINE_SECONDS,
            context_builder=SpeakerContextBuilder(
                workspace=ws,
                skills=skills,
                budget_tokens=speaker.context_budget_tokens,
                recent_messages=app_settings.SPEAKER_RECENT_MESSAGES,
            ),
        )

    # 9. Control plane: blackboard, delegation, orchestrator, rolling summary, brief
    blackboard = Blackboard(active_redis, ttl_seconds=app_settings.SESSION_STATE_TTL_SECONDS)
    delegation = DelegationRuntime(
        publisher=connection_mgr,
        cancellations=cancellations,
        workspace_resolver=workspace_mgr.get_or_create_workspace,
        metrics=metrics,
    )

    registered_agents = register_subagents(delegation, llm_for, workspace_mgr, retrieval_client=knowledge_service.client,
                       profiles=profiles, semaphore=orchestration_semaphore)

    orchestrator = InterviewOrchestrator(
        llm=llm_for("orchestrator"),
        state_store=state_store,
        workspace_resolver=workspace_mgr,
        delegation=delegation,
        blackboard=blackboard,
        publisher=connection_mgr,
        metrics=metrics,
        min_confidence=app_settings.COVERAGE_MIN_CONFIDENCE,
        cycle_deadline_seconds=app_settings.ORCHESTRATOR_DEADLINE_MS / 1000,
        join_deadline_seconds=app_settings.ORCHESTRATOR_JOIN_MS / 1000,
        llm_semaphore=orchestration_semaphore,
        max_extra_turns=app_settings.STAGE_MAX_EXTRA_TURNS,
    )
    summarizer = RollingSummarizer(
        llm=llm_for("summary"),
        redis_client=active_redis,
        deadline_seconds=profiles["summary"].turn_deadline_ms / 1000,
        semaphore=orchestration_semaphore,
    )
    brief_writer = InterviewBriefWriter(
        llm=llm_for("brief"), skills=skills, deadline_seconds=profiles["brief"].turn_deadline_ms / 1000
    )
    if inspect_trace_recorder is not None:
        recorder = inspect_trace_recorder
        recorder.wrap(speaker_llm, "stream_chat", "Speaker model", "llm", "speaker")
        for owner, name, method in [
            (profiler_agent, "profiler", "run_profiling"),
            (worker.evaluator, "evaluator", "evaluate_turn"),
            (synthesizer.narrative_writer, "narrative", "write"),
            (orchestrator, "orchestrator", "run_cycle"),
            (summarizer, "summary", "update"),
            (brief_writer, "brief", "write"),
            (registered_agents["followup"], "followup", "plan"),
            (registered_agents["claims"], "claims", "check"),
            (registered_agents["guard"], "guard", "check"),
        ]:
            recorder.bind_agent(owner, name)
            recorder.wrap(owner, method, f"{name}.{method}", "agent")
        recorder.wrap(knowledge_service, "build_pack", "Build retrieval knowledge pack", "retrieval")
        recorder.wrap(profiler_agent, "_query_model", "Profiler structured query", "operation", "profiler")
        recorder.wrap(profiler_agent, "_retrieve_knowledge", "Profiler retrieval", "retrieval", "profiler")
        recorder.wrap(orchestrator, "_decide", "Choose next directive", "operation", "orchestrator")
        recorder.wrap(knowledge_service, "_search", "Retrieval query", "retrieval")
        recorder.wrap(knowledge_service.client, "search", "Retrieval backend search", "retrieval")
        recorder.wrap(delegation, "delegate", "Schedule delegation", "operation", "orchestrator")
        recorder.wrap(delegation, "_run", "Execute delegated task", "task")
        delegation.trace_recorder = recorder
        recorder.wrap(assessment_queue, "enqueue", "Enqueue assessment", "operation")
        recorder.wrap(worker, "_process_job_data", "Assessment worker delivery", "task", "assessor")
        recorder.wrap(synthesizer, "synthesize", "Report synthesis", "operation")

    async def after_turn_orchestrate(session_id: str, turn_number: int) -> None:
        await orchestrator.run_cycle(session_id, turn_number)

    async def after_turn_summarize(session_id: str, turn_number: int) -> None:
        await summarizer.update(session_id, workspace_mgr.get_or_create_workspace(session_id), turn_number)

    async def persist_blackboard(session_id: str) -> None:
        await blackboard.persist(session_id, workspace_mgr.get_or_create_workspace(session_id))

    async def write_brief(session_id: str, workspace: Any) -> Dict[str, Any]:
        return {"brief_status": await brief_writer.write(workspace)}

    runners = RunnerRegistry(build_runner)
    controller = InterviewSessionController(
        state_store=state_store,
        publisher=connection_mgr,
        runners=runners,
        ledger=ledger,
        lease=lease,
        cancellations=cancellations,
        generation_slots=generation_slots,
        answer_limiter=TokenBucket(rate=app_settings.ANSWER_RATE_PER_SECOND, capacity=app_settings.ANSWER_BURST),
        queue_timeout_ms=app_settings.MAX_INTERACTIVE_QUEUE_MS,
        busy_retry_after_ms=app_settings.BUSY_RETRY_AFTER_MS,
        turn_lock_ttl_ms=app_settings.TURN_LOCK_TTL_MS,
        metrics=metrics,
        hot_path_gate=hot_path_gate,
    )
    if inspect_trace_recorder is not None:
        inspect_trace_recorder.instrument_controller(controller)
        inspect_trace_recorder.instrument_registry(runners)
    if app_settings.ORCHESTRATOR_ENABLED:
        controller.after_turn_hooks.append(after_turn_orchestrate)
    if app_settings.ROLLING_SUMMARY_ENABLED:
        controller.after_turn_hooks.append(after_turn_summarize)
    controller.dialogue_completed_hooks.append(persist_blackboard)
    controller.dialogue_completed_hooks.append(pipeline.on_dialogue_completed)

    async def prepare_ended_report(session_id: str) -> None:
        # Stop child agent calls, never cancel shared WebSocket/request/pool tasks.
        controller.cancel_session(session_id)
        worker.cancellations.cancel_session(session_id)
        await cancellations.cancel_session_and_wait(session_id)
        await worker.cancellations.cancel_session_and_wait(session_id)
        # A remote gateway observes the durable fence before releasing its turn lease. The wait is
        # bounded so /end answers promptly; on timeout the ENDING sweep retries this preparation.
        async with contextlib.AsyncExitStack() as locks:
            await asyncio.wait_for(
                locks.enter_async_context(assessment_queue.session_mutation(session_id, purpose="turn_lock")),
                timeout=END_PREPARE_LOCK_TIMEOUT_SECONDS,
            )
            await asyncio.wait_for(
                locks.enter_async_context(assessment_queue.session_mutation(session_id)),
                timeout=END_PREPARE_LOCK_TIMEOUT_SECONDS,
            )
            workspace = workspace_mgr.get_or_create_workspace(session_id)
            turns = [int(path.name.split("_")[1]) for path in workspace.list_artifacts("turns")
                if path.name.startswith("turn_") and path.name.split("_")[1].isdigit()]
            await summarizer.update(session_id, workspace, max(turns, default=0))
        runners.evict(session_id)

    async def forget_ended_session(session_id: str) -> None:
        # The durable fence and final status now refuse new work; drop per-process markers.
        cancellations.forget(session_id)
        worker.cancellations.forget(session_id)

    pipeline.end_prepare = prepare_ended_report
    pipeline.end_finished = forget_ended_session

    background_tasks: Dict[str, asyncio.Task] = {}

    async def _retention_loop() -> None:
        while True:
            try:
                await retention.sweep()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Retention sweep failed: %s", sanitize_secrets(str(e)))
            await asyncio.sleep(app_settings.RETENTION_SWEEP_INTERVAL_SECONDS)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if app_settings.APP_ENV.strip().lower() == "production":
            configure_logging(json_format=True)
        if active_artifact_archive is not None:
            await asyncio.to_thread(active_artifact_archive.initialize)
            app.state.artifact_archive_initialized = True
        if app_settings.ASSESSOR_POOL_SIZE > 0:
            await worker_pool.start()
        background_tasks["retention"] = asyncio.create_task(_retention_loop(), name="retention")
        try:
            yield
        finally:
            for task in background_tasks.values():
                task.cancel()
            await asyncio.gather(*background_tasks.values(), return_exceptions=True)
            background_tasks.clear()
            await controller.drain(timeout=10.0)
            await hooks.drain(timeout=5.0)
            await worker_pool.stop(grace_seconds=30.0)
            if inspect_trace_recorder is not None:
                await inspect_trace_recorder.close()
            for database in owned_databases:
                await asyncio.to_thread(database.close)

    app = FastAPI(
        title="AI Interview Platform Core Engine",
        version="2.0.0",
        description="AI runtime: profiling, realtime interviewer, assessment and reporting",
        lifespan=lifespan,
    )

    if app_settings.CORS_ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.CORS_ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Correlation-ID"],
        )
    else:
        # Development: any origin, never with credentials.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next: Any) -> Response:
        if strict_studio and request.method == "POST" and request.url.path.startswith("/api/v1/sessions") and not request.url.path.endswith(("/cancel", "/review")):
            parts = request.url.path.rstrip("/").split("/")
            session_id = parts[4] if len(parts) > 4 else None
            if not studio_execution_allowed(session_id):
                return JSONResponse(status_code=503, content={"detail": "Real LLM/retrieval required; legacy simulation archives are read-only."})
        incoming = request.headers.get("x-correlation-id", "")
        correlation_id = incoming if 0 < len(incoming) <= 64 and incoming.isascii() else uuid.uuid4().hex[:16]
        token = correlation_id_var.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    # Wire state onto the application (names kept stable for adapters and tests)
    app.state.settings = app_settings
    app.state.workspace_mgr = workspace_mgr
    app.state.cleanup_mgr = cleanup_mgr
    app.state.artifact_archive = active_artifact_archive
    app.state.artifact_archive_initialized = active_artifact_archive is None
    app.state.redis_client = active_redis
    app.state.event_store = event_store
    app.state.connection_mgr = connection_mgr
    app.state.assessment_queue = assessment_queue
    app.state.state_mgr = state_store
    app.state.idempotency_mgr = idempotency
    app.state.message_ledger = ledger
    app.state.revocations = revocations
    app.state.llm_client = active_llm
    app.state.knowledge_service = knowledge_service
    app.state.profiler_agent = profiler_agent
    app.state.profiler_semaphore = profiler_semaphore
    app.state.concurrency_semaphore = concurrency_semaphore
    app.state.synthesizer = synthesizer
    app.state.review_service = ReviewService(workspace_mgr)
    app.state.pipeline = pipeline
    app.state.assessor_worker = worker
    app.state.worker_pool = worker_pool
    app.state.retention = retention
    app.state.runners = runners
    app.state.get_runner = runners.get
    app.state.controller = controller
    app.state.cancellations = cancellations
    app.state.metrics = metrics
    app.state.post_profiling_hooks = [write_brief] if app_settings.BRIEF_ENABLED else []
    app.state.profiles = profiles
    app.state.hot_path_gate = hot_path_gate
    app.state.hooks = hooks
    app.state.inspect_trace_recorder = inspect_trace_recorder
    app.state.storage_backend = "postgres" if app_settings.DATABASE_URL else "sqlite"
    app.state.databases = owned_databases
    app.state.studio_real_provider = not isinstance(active_llm, (MockLLMClient, UnconfiguredLLMClient))
    def studio_execution_allowed(session_id=None):
        if not app.state.studio_real_provider or knowledge_service.backend_name == "mock":
            return False
        if session_id and (ws_base_path / normalize_session_id(session_id) / "inspect/simulation_result.json").is_file():
            return False
        return True
    app.state.studio_execution_allowed = studio_execution_allowed
    if inspect_trace_recorder is not None:
        inspect_trace_recorder.provider_metadata = {
            "client": type(active_llm).__name__,
            "model": app_settings.llm_model if app.state.studio_real_provider else None,
            "real_provider": app.state.studio_real_provider,
            "retrieval_backend": knowledge_service.backend_name,
        }
    app.state.skills = skills
    app.state.blackboard = blackboard
    app.state.delegation = delegation
    app.state.orchestrator = orchestrator
    app.state.summarizer = summarizer
    app.state.brief_writer = brief_writer
    app.state.reconnect_limiter = TokenBucket(
        rate=app_settings.RECONNECT_RATE_PER_MINUTE / 60.0, capacity=app_settings.RECONNECT_RATE_PER_MINUTE
    )
    app.state.session_create_limiter = TokenBucket(
        rate=app_settings.SESSION_CREATE_RATE_PER_SECOND, capacity=app_settings.SESSION_CREATE_BURST
    )

    @app.get("/health/live", tags=["health"])
    async def health_live() -> Dict[str, Any]:
        """Liveness probe: the process is serving requests."""
        return {"status": "ok", "live": True}

    @app.get("/health/ready", tags=["health"])
    async def health_ready() -> Any:
        """Readiness: Redis, Postgres (if configured), workspace write, assessor heartbeat and provider configuration."""
        details: Dict[str, Any] = {}
        is_ready = True

        try:
            await active_redis.ping()
            details["redis"] = "connected"
        except Exception as e:
            details["redis"] = f"unreachable: {sanitize_secrets(str(e))}"
            is_ready = False

        try:
            probe_file = ws_base_path / f".health_probe_{uuid.uuid4().hex[:8]}"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
            details["workspace"] = "writable"
        except Exception as e:
            details["workspace"] = f"unwritable: {sanitize_secrets(str(e))}"
            is_ready = False

        if owned_databases:
            def ping_databases() -> None:
                for database in owned_databases:
                    with database.connection() as connection:
                        connection.execute("SELECT 1")

            try:
                await asyncio.wait_for(asyncio.to_thread(ping_databases), timeout=3)
                details["database"] = "connected"
            except Exception as e:
                # Exceptions from the pool never carry the URL; sanitize anyway.
                details["database"] = f"unreachable: {sanitize_secrets(type(e).__name__)}"
                is_ready = False

        if app_settings.ASSESSOR_POOL_SIZE > 0 and worker_pool.status()["running_tasks"] > 0:
            healthy = details.get("redis") == "connected" and await worker_pool.is_healthy()
            details["assessor_pool"] = "healthy" if healthy else "stale_heartbeat"
            is_ready = is_ready and healthy
        else:
            details["assessor_pool"] = "not_running"
            if app_settings.APP_ENV.strip().lower() == "production":
                is_ready = False

        provider_ok = not isinstance(active_llm, (MockLLMClient, UnconfiguredLLMClient)) or (
            not strict_studio and app_settings.APP_ENV != "production"
        )
        details["provider"] = "unconfigured" if isinstance(active_llm, UnconfiguredLLMClient) else "mock" if isinstance(active_llm, MockLLMClient) else "configured"
        is_ready = is_ready and provider_ok
        details["retrieval_backend"] = knowledge_service.backend_name
        details["max_concurrent_sessions"] = app_settings.MAX_CONCURRENT_SESSIONS
        if active_artifact_archive is not None:
            details["artifact_archive"] = (
                "initialized" if app.state.artifact_archive_initialized else "unavailable"
            )
            is_ready = is_ready and app.state.artifact_archive_initialized
        else:
            details["artifact_archive"] = "disabled"

        return JSONResponse(
            status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ready" if is_ready else "not_ready", "details": details},
        )

    @app.get("/metrics", tags=["health"])
    async def metrics_endpoint(_: Dict[str, Any] = Depends(require_service_token)) -> PlainTextResponse:
        """Prometheus metrics (service token required)."""
        try:
            metrics.set_gauge("assessment_pending", float(await assessment_queue.get_queue_size()))
        except Exception:
            pass
        metrics.set_gauge("active_connections", float(connection_mgr.active_connection_count()))
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    app.include_router(sessions_router)
    app.include_router(ws_router)
    if studio_enabled:
        app.include_router(inspect_router)
    if app_settings.INSPECTOR_ENABLED:
        inspect_dir = Path(__file__).resolve().parent.parent / "FE-inspect"
        if not inspect_dir.is_dir():
            raise RuntimeError(f"FE-inspect assets are missing: {inspect_dir}")
        app.mount("/inspect", StaticFiles(directory=str(inspect_dir), html=True), name="fe-inspect")
    if app_settings.TEST_UI_ENABLED:
        test_dir = Path(__file__).resolve().parent.parent / "FE-test"
        if not test_dir.is_dir():
            raise RuntimeError(f"FE-test assets are missing: {test_dir}")
        app.mount("/test", StaticFiles(directory=str(test_dir), html=True), name="fe-test")
    return app


def build_default_app() -> FastAPI:
    """Factory for `uvicorn api.main:build_default_app --factory`."""
    settings = get_settings()
    return create_app(settings=settings, fake_redis=settings.APP_ENV != "production" and not settings.REDIS_URL)


# Default ASGI application instance (development convenience)
app = build_default_app()
