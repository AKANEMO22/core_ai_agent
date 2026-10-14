"""Archive behavior using isolated real core primitives, without provider requests."""
import httpx
import pytest
from redis.exceptions import ConnectionError

from api.auth import require_service_token
from api.main import create_app
from config.settings import Settings
from core.runtime.types import LoopEvent, LoopEventType
from core.session.state import LifecycleStatus


@pytest.fixture
def studio(tmp_path):
    settings = Settings(INSPECTOR_ENABLED=True, TEST_UI_ENABLED=False, INSPECTOR_LOCAL_AUTH=True,
                        LLM_API_KEY="", OPENAI_API_KEY="", RETRIEVAL_BACKEND="rag", ASSESSOR_POOL_SIZE=0)
    app = create_app(settings=settings, fake_redis=True, workspace_base=str(tmp_path))
    app.dependency_overrides[require_service_token] = lambda: {"iss": "archive-check", "aud": "ai-runtime"}
    return app


async def create_workspace(app, session_id):
    state = await app.state.state_mgr.acreate(session_id)
    workspace = app.state.workspace_mgr.get_or_create_workspace(session_id)
    workspace.initialize()
    workspace.write_artifact("inputs/form_input.json", "input", {"candidate_name": "Archive fixture", "target_role": "Persistence validation"})
    return state, workspace


@pytest.mark.asyncio
async def test_archive_survives_redis_loss_without_reconstructed_fake_graph(studio):
    state, _ = await create_workspace(studio, "durable")
    state = await studio.state.state_mgr.atransition(state, state.state_version, LifecycleStatus.PROFILING)
    event = LoopEvent(session_id="durable", turn_id="turn_000", sequence=0, type=LoopEventType.RESPONSE_TOKEN,
                      payload={"token": "Persisted response"})
    await studio.state.event_store.append_async("durable", event)
    await studio.state.redis_client.flushall()
    studio.state.event_store._events.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        detail = (await client.get("/api/v1/inspect/runs/durable")).json()
        assert detail["session"]["status"] == "PROFILING"
        assert detail["session"]["state_source"] == "sqlite"
        assert detail["events"][0]["payload"]["token"] == "Persisted response"
        assert detail["trace"]["coverage"] == "not_recorded"
        assert all(span["kind"] == "run" for span in detail["spans"])
        assert (await client.get("/api/v1/inspect/runs")).json()["total"] == 1


@pytest.mark.asyncio
async def test_inspect_does_not_acquire_or_displace_candidate_lease(studio):
    await create_workspace(studio, "lease")

    class CandidateSocket:
        async def close(self, **kwargs):
            raise AssertionError("Inspector must not close candidate socket")

    lease = await studio.state.connection_mgr.acquire_lease("lease", CandidateSocket())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        assert (await client.get("/api/v1/inspect/runs/lease/trace")).status_code == 200
        assert (await client.get("/api/v1/inspect/runs/lease")).status_code == 200
    assert studio.state.connection_mgr.get_active_lease("lease") is lease


@pytest.mark.asyncio
async def test_unconfigured_provider_is_not_scripted_and_generation_is_blocked(studio):
    assert type(studio.state.llm_client).__name__ == "UnconfiguredLLMClient"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        config = (await client.get("/api/v1/inspect/config")).json()
        assert config["real_provider"] is False and config["provider_model"] is None
        response = await client.post("/api/v1/sessions", json={"candidate_name": "Blocked"})
        assert response.status_code == 503
        assert (await client.get("/api/v1/inspect/runs")).json()["total"] == 0
        assert (await client.post("/api/v1/inspect/runs/simulate", json={})).status_code in {404, 405}


@pytest.mark.asyncio
async def test_auth_and_artifact_path_security(studio):
    _, workspace = await create_workspace(studio, "safe")
    workspace.write_artifact("reports/report.json", "report", {"authorization": "Bearer sensitive-value", "content": "visible"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        preview = (await client.get("/api/v1/inspect/runs/safe/artifacts/reports/report.json")).json()
        assert preview["content"]["payload"]["authorization"] == "[REDACTED]"
        assert preview["content"]["payload"]["content"] == "visible"
        response = await client.get("/api/v1/inspect/runs/safe/artifacts/%2e%2e%2foutside.json")
        assert response.status_code == 404
    studio.dependency_overrides.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        assert (await client.get("/api/v1/inspect/runs/safe")).status_code == 401


@pytest.mark.asyncio
async def test_task_and_worker_parent_survive_serialized_redis_job(studio):
    await create_workspace(studio, "queue_parent")
    queue = studio.state.assessment_queue
    await queue.enqueue({"session_id": "queue_parent", "turn_id": "turn_001", "question": "Input", "candidate_text": "Answer"})
    await queue.setup_consumer_group("trace_check", start_id="0")
    deliveries = await queue.read_jobs("trace_check", "consumer", count=1, block_ms=0)
    assert deliveries
    _, payload = deliveries[0]
    spans = studio.state.inspect_trace_recorder.read("queue_parent")
    dispatch = next(span for span in spans if span["name"] == "Enqueue assessment")
    assert payload["inspect_trace_parent_id"] == dispatch["span_id"]


@pytest.mark.asyncio
async def test_event_pagination_retains_real_text(studio):
    await create_workspace(studio, "paged")
    for i in range(5):
        await studio.state.event_store.append_async("paged", LoopEvent(
            session_id="paged", turn_id="turn_000", sequence=0, type=LoopEventType.RESPONSE_TOKEN, payload={"token": str(i)}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        first = (await client.get("/api/v1/inspect/runs/paged/events?limit=2")).json()
        second = (await client.get(f"/api/v1/inspect/runs/paged/events?limit=2&after_sequence={first['next_sequence']}")).json()
        assert [row["payload"]["token"] for row in first["events"] + second["events"]] == ["0", "1", "2", "3"]
        assert first["has_more"] and second["has_more"]


@pytest.mark.asyncio
async def test_inspector_reads_archive_while_redis_is_unreachable(studio, monkeypatch):
    await create_workspace(studio, "offline")
    await studio.state.event_store.append_async("offline", LoopEvent(
        session_id="offline", turn_id="turn_000", sequence=0, type=LoopEventType.RESPONSE_TOKEN, payload={"token": "saved"}))
    async def unavailable(*args, **kwargs):
        raise ConnectionError("Redis unreachable")
    monkeypatch.setattr(studio.state.state_mgr, "aget", unavailable)
    monkeypatch.setattr(studio.state.event_store, "get_events_async", unavailable)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        assert (await client.get("/api/v1/inspect/runs")).status_code == 200
        detail = (await client.get("/api/v1/inspect/runs/offline")).json()
        assert detail["session"]["state_source"] == "sqlite"
        assert "Redis unreachable" in detail["session"]["state_refresh_error"]
        assert detail["events"][0]["payload"]["token"] == "saved"


@pytest.mark.asyncio
async def test_legacy_redis_history_imported_even_after_new_sqlite_event(studio):
    await create_workspace(studio, "legacy")
    old_event = LoopEvent(session_id="legacy", turn_id="turn_000", sequence=1,
                         type=LoopEventType.RESPONSE_TOKEN, payload={"token": "older"})
    new_event = LoopEvent(session_id="legacy", turn_id="turn_000", sequence=2,
                         type=LoopEventType.RESPONSE_TOKEN, payload={"token": "newer"})
    # Redis already holds older events from before tracing was deployed.
    await studio.state.redis_client.rpush(studio.state.event_store._list_key("legacy"), old_event.model_dump_json())
    await studio.state.redis_client.set(studio.state.event_store._seq_key("legacy"), 1)
    await studio.state.event_store.append_async("legacy", new_event)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio), base_url="http://archive") as client:
        detail = (await client.get("/api/v1/inspect/runs/legacy")).json()
        assert [row["payload"]["token"] for row in detail["events"]] == ["older", "newer"]
