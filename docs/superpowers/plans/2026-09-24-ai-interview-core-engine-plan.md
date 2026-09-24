# AI Interview Platform Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, standalone AI Agent Runtime Engine for an AI Mock Interview platform on a single-node deployment with isolated workspaces, streaming agent loop, profiler agent, stage director, Redis-backed async assessor worker pool, and FastAPI REST/WebSocket gateway supporting 100 concurrent sessions.

**Architecture:** Loosely coupled modular runtime engine consisting of: (1) isolated workspace sandbox with atomic writes and path traversal defense; (2) streaming-first core agent loop with soft schema repair; (3) pre-flight company/CV recon profiler; (4) stage-delegated turn runner with tokenizer-aware context compression; (5) background idempotent Redis Streams evaluation and report synthesis barrier; and (6) authenticated FastAPI REST and WebSocket streaming gateway.

**Tech Stack:** Python 3.14+, FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings, Redis / Fakeredis, Pytest, Pytest-Asyncio, WebSockets, PyJWT, HTTPX.

## Global Constraints

- Deployment target is single-node runtime; no horizontal multi-node cluster dependencies.
- Session workspace root base is configurable via `WORKSPACE_ROOT_BASE` (default `./data/workspaces`).
- All session workspaces are isolated with normalized naming `session_{clean_id}-{sha256(session_id)[:12]}`.
- Path traversal outside workspace root MUST raise `WorkspacePathError`.
- Max tool output size before spilling to `spill/` is `MAX_TOOL_RESULT_BYTES = 32768` (32 KB).
- ContextAssembler envelope MUST NOT exceed 1500 tokens for subagent prompt.
- Event schemas MUST include `schema_version`, `event_id`, `session_id`, `turn_id`, `sequence`, `timestamp`, `type`, `payload`.
- Outbound research requests MUST enforce HTTPS, DNS/IP validation rejecting loopback/private/link-local IPs (SSRF defense), size limits, and timeouts.
- At-least-once delivery for assessment jobs with idempotency key (`job_id + input_sha256 + rubric_version`); XACK only after atomic write.
- Assessment barrier: `final_assessment.json` and `interview_report.md` are generated only when `completed_assessments == expected_assessments` and `failed_assessments == 0`.
- Logs, errors, and public metrics MUST NEVER contain CV text, raw answers, prompt secrets, or tokens.
- Complete, runnable production-ready code with no TODOs or pseudo-code.

---

### Task 1: Configuration & Workspace Isolation

**Files:**
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `core/__init__.py`
- Create: `core/workspace/__init__.py`
- Create: `core/workspace/security.py`
- Create: `core/workspace/session.py`
- Create: `core/workspace/spill.py`
- Create: `core/workspace/cleanup.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_workspace.py`

**Interfaces:**
- Consumes: None (base foundational layer).
- Produces:
  - `config.settings.Settings` (singleton `get_settings()`)
  - `core.workspace.security.WorkspacePathError`, `validate_upload_content()`, `validate_path_safety()`
  - `core.workspace.session.SessionWorkspace`, `SessionWorkspaceManager`
  - `core.workspace.spill.WorkspaceSpillManager`
  - `core.workspace.cleanup.WorkspaceCleanupManager`

- [ ] **Step 1: Write the failing tests for workspace isolation, path traversal, atomic write, and spillover**

```python
# tests/unit/test_workspace.py
import pytest
from pathlib import Path
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager
from core.workspace.security import WorkspacePathError, validate_upload_content
from core.workspace.spill import WorkspaceSpillManager

def test_normalized_session_id():
    manager = SessionWorkspaceManager(root_base="./test_data/workspaces")
    ws = manager.get_or_create_workspace("user/session@123")
    assert "user_session_123" in ws.normalized_session_id
    assert len(ws.normalized_session_id.split("-")[-1]) == 12

def test_path_traversal_detection(tmp_path):
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    with pytest.raises(WorkspacePathError):
        ws.resolve_path("../outside.txt")
    with pytest.raises(WorkspacePathError):
        ws.resolve_path("/etc/passwd")

def test_atomic_artifact_write(tmp_path):
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    data = {"status": "ok", "value": 42}
    rel_path = "context/test_artifact.json"
    written_path = ws.write_artifact(rel_path=rel_path, artifact_type="test", payload=data)
    assert written_path.exists()
    read_data = ws.read_json(rel_path)
    assert read_data["value"] == 42
    assert "checksum_sha256" in read_data

def test_spillover_exceeding_threshold(tmp_path):
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    spill_mgr = WorkspaceSpillManager(ws, max_bytes=100)
    large_text = "A" * 200
    res = spill_mgr.maybe_spill(tool_name="test_tool", payload=large_text)
    assert res["is_spilled"] is True
    assert "spill_path" in res
    assert len(res["preview"]) <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_workspace.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing imports.

- [ ] **Step 3: Implement settings, workspace security, session workspace, spill, and cleanup**

Implement:
- `config/settings.py`: Pydantic settings loading `WORKSPACE_ROOT_BASE`, `REDIS_URL`, `JWT_SECRET`, timeouts, thresholds.
- `core/workspace/security.py`: `WorkspacePathError`, `validate_path_safety()`, `validate_upload_content()` checking mime, size, magic bytes, JSON depth.
- `core/workspace/session.py`: `SessionWorkspace` directory layout (`inputs/`, `context/`, `turns/`, `evaluations/`, `reports/`, `spill/`), atomic write via temp file + replace + sha256 checksum, and `SessionWorkspaceManager`.
- `core/workspace/spill.py`: `WorkspaceSpillManager` spilling payload > 32KB to `spill/spill_tool_{uuid}.json`.
- `core/workspace/cleanup.py`: `WorkspaceCleanupManager` retention and idempotent deletion.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_workspace.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/ core/ tests/unit/test_workspace.py
git commit -m "feat(workspace): implement isolated session workspace, path security, spillover and cleanup"
```

---

### Task 2: Streaming Core Loop, Protocols & Provider Adapters

**Files:**
- Create: `core/runtime/__init__.py`
- Create: `core/runtime/types.py`
- Create: `core/runtime/client.py`
- Create: `core/runtime/tooling.py`
- Create: `core/runtime/providers/__init__.py`
- Create: `core/runtime/providers/mock.py`
- Create: `core/runtime/providers/openai.py`
- Create: `core/runtime/agent.py`
- Create: `tests/unit/test_runtime_agent.py`

**Interfaces:**
- Consumes: `core.workspace.session.SessionWorkspace`, `core.workspace.spill.WorkspaceSpillManager`
- Produces:
  - `core.runtime.types.LoopEvent`, `LoopEventType`, `LoopStatus`, `IterationChunk`
  - `core.runtime.client.RuntimeLLMClient`, `ProviderCapabilities`
  - `core.runtime.tooling.RuntimeTool`, `ToolSchemaValidator`
  - `core.runtime.providers.mock.MockLLMClient`
  - `core.runtime.providers.openai.OpenAIRuntimeLLMClient`
  - `core.runtime.agent.RuntimeAgent`

- [ ] **Step 1: Write the failing tests for RuntimeAgent execution loop and streaming**

```python
# tests/unit/test_runtime_agent.py
import pytest
from core.runtime.types import LoopEventType, LoopEvent
from core.runtime.agent import RuntimeAgent
from core.runtime.providers.mock import MockLLMClient
from core.runtime.tooling import RuntimeTool

@pytest.mark.asyncio
async def test_agent_streaming_tokens():
    mock_client = MockLLMClient(canned_tokens=["Hello", " ", "candidate!"])
    agent = RuntimeAgent(client=mock_client)
    events = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        events.append(event)
    token_events = [e for e in events if e.type == LoopEventType.RESPONSE_TOKEN]
    assert len(token_events) == 3
    assert "".join([e.payload["token"] for e in token_events]) == "Hello candidate!"
    complete_events = [e for e in events if e.type == LoopEventType.COMPLETE]
    assert len(complete_events) == 1

@pytest.mark.asyncio
async def test_agent_tool_call_and_soft_recovery():
    def get_weather(location: str) -> str:
        return f"Weather in {location} is Sunny"

    tool = RuntimeTool.from_callable(
        name="get_weather",
        description="Get weather for location",
        func=get_weather,
        param_schema={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    )
    mock_client = MockLLMClient(
        canned_tool_calls=[{"name": "get_weather", "arguments": {"location": "Hanoi"}}],
        final_response="The weather is Sunny."
    )
    agent = RuntimeAgent(client=mock_client, tools=[tool])
    events = []
    async for event in agent.run_loop_streaming(
        session_id="s1",
        turn_id="turn_001",
        messages=[{"role": "user", "content": "Check weather in Hanoi"}],
    ):
        events.append(event)
    tool_results = [e for e in events if e.type == LoopEventType.TOOL_CALL_RESULT]
    assert len(tool_results) == 1
    assert "Sunny" in tool_results[0].payload["result"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_runtime_agent.py -v`
Expected: FAIL with missing classes/modules.

- [ ] **Step 3: Implement streaming agent loop, types, tooling, mock provider & OpenAI provider**

Implement:
- `core/runtime/types.py`: `LoopEvent`, `LoopEventType` (`PROGRESS`, `REASONING_SUMMARY`, `RESPONSE_TOKEN`, `TOOL_CALL_START`, `TOOL_CALL_RESULT`, `COMPLETE`, `ERROR`), `LoopStatus`, `IterationChunk`.
- `core/runtime/client.py`: Protocol `RuntimeLLMClient` with `stream_chat` and `capabilities: ProviderCapabilities`.
- `core/runtime/tooling.py`: `RuntimeTool` wrapper, soft schema validation collecting violations to self-correct.
- `core/runtime/providers/mock.py`: Deterministic programmable mock client for tests.
- `core/runtime/providers/openai.py`: OpenAIRuntimeLLMClient wrapping async openai streaming.
- `core/runtime/agent.py`: `RuntimeAgent.run_loop_streaming()` with fast-path tokens, budget control, tool nudge, schema retry budget (1 repair + 1 transient retry), and cancellation handling.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_runtime_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/runtime/ tests/unit/test_runtime_agent.py
git commit -m "feat(runtime): implement streaming core loop, tool execution and provider adapters"
```

---

### Task 3: Pre-flight Profiler & Recon Agent

**Files:**
- Create: `core/profiler/__init__.py`
- Create: `core/profiler/prompts.py`
- Create: `core/profiler/tools.py`
- Create: `core/profiler/agent.py`
- Create: `tests/unit/test_profiler.py`
- Create: `tests/security/__init__.py`
- Create: `tests/security/test_untrusted_inputs.py`

**Interfaces:**
- Consumes: `core.workspace.session.SessionWorkspace`, `core.runtime.client.RuntimeLLMClient`, `core.runtime.agent.RuntimeAgent`
- Produces:
  - `core.profiler.tools.SafeWebSearchTool`, `MarketIntelTool` (with SSRF protection & IP validation)
  - `core.profiler.agent.ProfilerAgent`
  - Workspace artifacts: `context/company_intel.json`, `context/candidate_profile.json`, `context/interview_spec.json`, `context/stage_plan.json`

- [ ] **Step 1: Write the failing tests for ProfilerAgent and SSRF protection**

```python
# tests/unit/test_profiler.py
import pytest
from core.profiler.agent import ProfilerAgent
from core.runtime.providers.mock import MockLLMClient

@pytest.mark.asyncio
async def test_profiler_generates_required_artifacts(tmp_path):
    from core.workspace.session import SessionWorkspace
    ws = SessionWorkspace(session_id="sess_p1", root_path=tmp_path / "sess_p1")
    ws.initialize()
    # Write a mock form input
    form_data = {
        "candidate_name": "Nguyen Van A",
        "experience_years": 5,
        "target_company": "TechCorp",
        "target_role": "Senior Backend Engineer",
        "skills": ["Python", "Distributed Systems", "Redis"]
    }
    ws.write_artifact("inputs/form_input.json", artifact_type="input", payload=form_data)

    mock_client = MockLLMClient(
        canned_json_responses={
            "company_intel": {"company": "TechCorp", "culture": "High Bar", "tech_stack": ["Python", "Kafka"]},
            "candidate_profile": {"strengths": ["Python"], "gaps": ["Kubernetes"], "probes": ["Concurrency"]},
            "interview_spec": {"interviewer_persona": "Tech Lead", "tone": "rigorous_challenger", "knowledge_scope": ["Concurrency"]},
            "stage_plan": {"stages": [{"stage_id": "stage_1", "name": "Warmup", "turns": 2}]}
        }
    )
    profiler = ProfilerAgent(llm_client=mock_client)
    res = await profiler.run_profiling(ws)
    assert res["status"] == "SUCCESS"
    assert (ws.root_path / "context" / "company_intel.json").exists()
    assert (ws.root_path / "context" / "candidate_profile.json").exists()
    assert (ws.root_path / "context" / "interview_spec.json").exists()
    assert (ws.root_path / "context" / "stage_plan.json").exists()

# tests/security/test_untrusted_inputs.py
import pytest
from core.profiler.tools import validate_outbound_url

def test_ssrf_blocking_private_and_loopback_ips():
    assert validate_outbound_url("http://127.0.0.1/admin") is False
    assert validate_outbound_url("http://localhost:8000") is False
    assert validate_outbound_url("http://192.168.1.100") is False
    assert validate_outbound_url("http://169.254.169.254/latest/meta-data") is False
    assert validate_outbound_url("https://example.com/api") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_profiler.py tests/security/test_untrusted_inputs.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ProfilerAgent, SSRF-safe recon tools, and prompt templates**

Implement:
- `core/profiler/prompts.py`: Prompts enforcing untrusted boundary around CV/form input and producing structured JSON specs.
- `core/profiler/tools.py`: `validate_outbound_url` checking schemes (HTTPS only in prod), DNS resolution checking against loopback/private/link-local IPv4/IPv6, `SafeWebSearchTool` and `MarketIntelTool`.
- `core/profiler/agent.py`: `ProfilerAgent` reading `inputs/cv_input.*` or `inputs/form_input.json`, executing recon, and atomically persisting the 4 artifacts.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_profiler.py tests/security/test_untrusted_inputs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/profiler/ tests/unit/test_profiler.py tests/security/
git commit -m "feat(profiler): implement pre-flight recon agent with SSRF protection and blueprint generation"
```

---

### Task 4: Session State, Idempotency & Replayable Events

**Files:**
- Create: `core/session/__init__.py`
- Create: `core/session/state.py`
- Create: `core/session/events.py`
- Create: `core/session/idempotency.py`
- Create: `tests/unit/test_session_state.py`

**Interfaces:**
- Consumes: `core.runtime.types.LoopEvent`
- Produces:
  - `core.session.state.LifecycleStatus`, `SessionState`, `SessionStateManager` (CAS transitions via monotonic `state_version`)
  - `core.session.events.SessionEventStore`, `EventReplayer`
  - `core.session.idempotency.IdempotencyManager` (per-session async lock, message_id deduplication, idempotency key caching)

- [ ] **Step 1: Write the failing tests for FSM CAS transitions and idempotency**

```python
# tests/unit/test_session_state.py
import pytest
from core.session.state import LifecycleStatus, SessionState, SessionStateManager, InvalidStateTransitionError
from core.session.idempotency import IdempotencyManager

def test_cas_state_transition():
    state = SessionState(session_id="s1", status=LifecycleStatus.INITIALIZED, state_version=1)
    mgr = SessionStateManager()
    # Valid transition with matching version
    updated = mgr.transition(state, expected_version=1, new_status=LifecycleStatus.PROFILING)
    assert updated.status == LifecycleStatus.PROFILING
    assert updated.state_version == 2

    # Stale transition should fail
    with pytest.raises(InvalidStateTransitionError):
        mgr.transition(updated, expected_version=1, new_status=LifecycleStatus.READY)

def test_idempotency_cache():
    idemp = IdempotencyManager()
    idemp.record("msg_123", {"result": "turn_001_complete"})
    cached = idemp.get("msg_123")
    assert cached is not None
    assert cached["result"] == "turn_001_complete"
    assert idemp.get("unknown") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_session_state.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SessionState, CAS transitions, EventReplayer, and IdempotencyManager**

Implement:
- `core/session/state.py`: `LifecycleStatus` (`INITIALIZED`, `PROFILING`, `READY`, `STAGE_ACTIVE`, `STAGE_TRANSITIONING`, `DIALOGUE_COMPLETED`, `ASSESSING`, `REPORT_READY`, `ASSESSMENT_FAILED`, `FAILED`, `CANCELLED`, `EXPIRED`), `SessionState` dataclass, `SessionStateManager.transition` with CAS validation.
- `core/session/events.py`: In-memory and Redis-compatible `SessionEventStore` appending structured events with monotonic sequence numbers, and `EventReplayer` slicing from `last_event_id`.
- `core/session/idempotency.py`: Per-session async lock, token bucket, `IdempotencyManager` storing cached results with payload hashing to detect 409 mismatch.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_session_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/session/ tests/unit/test_session_state.py
git commit -m "feat(session): implement versioned FSM lifecycle, CAS transitions, replay events and idempotency"
```

---

### Task 5: Stage Director, Context Assembler & Interactive Subagent Runner

**Files:**
- Create: `core/interview/__init__.py`
- Create: `core/interview/context.py`
- Create: `core/interview/director.py`
- Create: `core/interview/queue.py`
- Create: `core/interview/runner.py`
- Create: `tests/unit/test_context_assembler.py`
- Create: `tests/unit/test_stage_director.py`

**Interfaces:**
- Consumes: `core.workspace.session.SessionWorkspace`, `core.runtime.agent.RuntimeAgent`, `core.session.state.SessionState`
- Produces:
  - `core.interview.context.ContextAssembler` (< 1500 tokens budget, deterministic truncation priority)
  - `core.interview.director.StageDirector` (stage transitions, turn counts, completion triggers)
  - `core.interview.queue.AssessmentQueue` protocol and in-memory test queue
  - `core.interview.runner.InteractiveSubagentRunner` (turn-by-turn streaming, turn persistence, assessment enqueueing)

- [ ] **Step 1: Write the failing tests for ContextAssembler truncation and StageDirector transitions**

```python
# tests/unit/test_context_assembler.py
from core.interview.context import ContextAssembler

def test_context_assembler_under_token_budget():
    assembler = ContextAssembler(max_tokens=1500)
    spec = {"interviewer_persona": "Tech Lead", "tone": "rigorous", "knowledge_scope": ["Distributed Systems"]}
    stage = {"stage_id": "stage_2", "name": "Technical", "goals": ["Probe concurrency"]}
    profile = {"strengths": ["Python"], "claims": ["Built distributed lock"]}
    turns = [{"role": "interviewer", "content": "Tell me about Redis."}, {"role": "candidate", "content": "I used redlock."}]

    prompt = assembler.assemble_prompt(spec=spec, stage=stage, profile=profile, recent_turns=turns)
    assert len(prompt) > 0
    token_est = assembler.estimate_tokens(prompt)
    assert token_est <= 1500

# tests/unit/test_stage_director.py
import pytest
from core.interview.director import StageDirector
from core.session.state import SessionState, LifecycleStatus, SessionStateManager

def test_stage_progression_on_budget():
    state_mgr = SessionStateManager()
    stages = [
        {"stage_id": "s1", "target_turns": 2},
        {"stage_id": "s2", "target_turns": 2}
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    state = SessionState(session_id="sess_1", status=LifecycleStatus.STAGE_ACTIVE, state_version=1)

    # Turn 1 in s1
    res1 = director.record_turn(state)
    assert res1.current_stage_id == "s1"
    assert res1.is_stage_complete is False

    # Turn 2 in s1 -> triggers stage transition
    res2 = director.record_turn(res1)
    assert res2.is_stage_complete is True
    next_state = director.advance_stage(res2)
    assert next_state.current_stage_id == "s2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_context_assembler.py tests/unit/test_stage_director.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ContextAssembler, StageDirector, AssessmentQueue protocol, and InteractiveSubagentRunner**

Implement:
- `core/interview/context.py`: `ContextAssembler` keeping prompt under 1500 tokens, priority: drop old raw turns -> drop low-confidence research -> drop non-critical profile.
- `core/interview/director.py`: `StageDirector` managing current stage index, turn counters, checking turn budget or tool `complete_stage`, advancing stages through CAS.
- `core/interview/queue.py`: `AssessmentQueue` abstract protocol + `InMemoryAssessmentQueue`.
- `core/interview/runner.py`: `InteractiveSubagentRunner` driving candidate turn, running `RuntimeAgent`, recording turn to `turns/turn_{n}_{role}.json`, and enqueuing assessment job.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_context_assembler.py tests/unit/test_stage_director.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/interview/ tests/unit/test_context_assembler.py tests/unit/test_stage_director.py
git commit -m "feat(interview): implement stage director, context assembler and interactive runner"
```

---

### Task 6: Background Assessor Worker Pool & Final Report Synthesizer

**Files:**
- Create: `core/assessor/__init__.py`
- Create: `core/assessor/queue.py`
- Create: `core/assessor/subagent.py`
- Create: `core/assessor/worker.py`
- Create: `core/assessor/synthesizer.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_assessment_queue.py`

**Interfaces:**
- Consumes: `core.workspace.session.SessionWorkspace`, `core.runtime.client.RuntimeLLMClient`, `core.session.state.SessionStateManager`
- Produces:
  - `core.assessor.queue.RedisAssessmentQueue` (Redis Streams `XADD`, `XREADGROUP`, `XACK`, `XAUTOCLAIM`, dead-letter)
  - `core.assessor.subagent.TurnEvaluator` (Content Score 1-10, Communication Structure 1-10, descriptive VAD Delivery Observations)
  - `core.assessor.worker.AssessorWorker`
  - `core.assessor.synthesizer.ReportSynthesizer` (barrier check, `final_assessment.json`, `interview_report.md`, transition to `REPORT_READY`)

- [ ] **Step 1: Write the failing tests for Redis Streams assessment queue, worker idempotency, and report barrier**

```python
# tests/integration/test_assessment_queue.py
import pytest
import fakeredis.aioredis
from core.assessor.queue import RedisAssessmentQueue
from core.assessor.worker import AssessorWorker
from core.runtime.providers.mock import MockLLMClient
from core.workspace.session import SessionWorkspace

@pytest.mark.asyncio
async def test_redis_assessment_queue_and_worker_persistence(tmp_path):
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis_client=r, stream_key="test:tasks:assessment")
    await queue.setup_consumer_group(group_name="workers")

    ws = SessionWorkspace(session_id="sess_eval1", root_path=tmp_path / "sess_eval1")
    ws.initialize()

    job_data = {
        "schema_version": "1.0",
        "job_id": "sess_eval1:turn_001:rubric_v1",
        "session_id": "sess_eval1",
        "turn_id": "turn_001",
        "stage_id": "stage_1",
        "question": "Explain Redis Pub/Sub.",
        "candidate_text": "Redis Pub/Sub sends messages to channels.",
        "vad_metrics": {"speech_duration_ms": 5000, "pause_count": 1, "total_silence_ms": 400, "words_per_minute": 120},
        "rubric_version": "rubric_v1",
        "rubric_target": ["Redis", "PubSub"]
    }
    await queue.enqueue(job_data)

    mock_llm = MockLLMClient(canned_json_responses={
        "assessment": {
            "content_score": 8,
            "communication_score": 7,
            "strengths": ["Clear explanation"],
            "improvements": ["Mention persistent streams vs pub/sub"]
        }
    })
    worker = AssessorWorker(redis_client=r, queue=queue, llm_client=mock_llm, workspace_resolver=lambda sid: ws)
    processed = await worker.process_next_job(group_name="workers", consumer_name="worker_1")
    assert processed is True
    eval_file = ws.root_path / "evaluations" / "turn_001_eval.json"
    assert eval_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_assessment_queue.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RedisAssessmentQueue, TurnEvaluator, AssessorWorker, and ReportSynthesizer**

Implement:
- `core/assessor/queue.py`: Producer and consumer for Redis Streams `interview:tasks:assessment`, handling `XADD`, `XREADGROUP`, `XACK`, `XAUTOCLAIM`, dead-letter stream `interview:tasks:assessment:dead_letter`.
- `core/assessor/subagent.py`: `TurnEvaluator` scoring Content (1-10), Communication Structure (1-10), and formatting non-scoring descriptive delivery observations.
- `core/assessor/worker.py`: `AssessorWorker` checking idempotency key (`job_id + input_sha256 + rubric_version`), executing evaluation, atomic write to `evaluations/turn_{n}_eval.json`, XACK, and updating Redis set cardinality.
- `core/assessor/synthesizer.py`: `ReportSynthesizer` checking barrier `completed_assessments == expected_assessments` and `failed_assessments == 0`, generating `final_assessment.json` and `interview_report.md`, and transitioning state to `REPORT_READY`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_assessment_queue.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/assessor/ tests/integration/test_assessment_queue.py
git commit -m "feat(assessor): implement Redis Streams worker pool, rubric evaluation and report synthesizer barrier"
```

---

### Task 7: Gateway, Auth, WebSocket Stream & Connection Manager

**Files:**
- Create: `api/__init__.py`
- Create: `api/schemas.py`
- Create: `api/auth.py`
- Create: `api/routes.py`
- Create: `api/websocket.py`
- Create: `api/main.py`
- Create: `tests/integration/test_auth.py`
- Create: `tests/integration/test_event_replay.py`

**Interfaces:**
- Consumes: All `core.*` modules, `config.settings.Settings`
- Produces:
  - `api.auth.create_service_token()`, `verify_service_token()`, `create_session_token()`, `verify_session_token()`
  - `api.schemas.*` (Pydantic REST request/response schemas and WebSocket protocol envelopes)
  - `api.websocket.ConnectionManager` (active lease management, replay on reconnect)
  - `api.routes.router` (REST endpoints `/api/v1/sessions`)
  - `api.main.app` (FastAPI app factory with health probes `/health/live`, `/health/ready`)

- [ ] **Step 1: Write the failing tests for JWT auth, REST endpoints, and WebSocket replay**

```python
# tests/integration/test_auth.py
import pytest
from api.auth import create_service_token, verify_service_token, create_session_token, verify_session_token, AuthError

def test_service_token_validation():
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=secret)
    claims = verify_service_token(token, secret=secret, expected_audience="ai-runtime")
    assert claims["iss"] == "backend"

    with pytest.raises(AuthError):
        verify_service_token(token, secret=secret, expected_audience="wrong-aud")

# tests/integration/test_event_replay.py
import pytest
from api.websocket import ConnectionManager
from core.runtime.types import LoopEvent, LoopEventType

@pytest.mark.asyncio
async def test_reconnection_and_event_replay():
    mgr = ConnectionManager()
    mgr.record_event("sess_1", LoopEvent(
        schema_version="1.0",
        event_id="e1",
        session_id="sess_1",
        turn_id="turn_001",
        sequence=1,
        timestamp="2026-09-24T00:00:00Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "Hello"}
    ))
    mgr.record_event("sess_1", LoopEvent(
        schema_version="1.0",
        event_id="e2",
        session_id="sess_1",
        turn_id="turn_001",
        sequence=2,
        timestamp="2026-09-24T00:00:01Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": " World"}
    ))

    replayed = mgr.get_events_after("sess_1", last_sequence=1)
    assert len(replayed) == 1
    assert replayed[0].event_id == "e2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_auth.py tests/integration/test_event_replay.py -v`
Expected: FAIL

- [ ] **Step 3: Implement schemas, auth, routes, websocket gateway, connection manager, and app**

Implement:
- `api/schemas.py`: Versioned REST request/response schemas, WebSocket protocol messages (`candidate_answer`, `progress`, `response_token`, etc.).
- `api/auth.py`: PyJWT generation and verification for service token and session token (audience, expiry, session_id scope).
- `api/websocket.py`: `ConnectionManager` tracking active connection leases, handling `last_event_id` replay, and forwarding incoming turns to `InteractiveSubagentRunner`.
- `api/routes.py`: REST routes (`POST /sessions` with `Idempotency-Key`, `GET /sessions/{id}`, `GET /report`, `POST /cancel`, `DELETE /sessions/{id}`).
- `api/main.py`: FastAPI app assembly with CORS, health routes `/health/live`, `/health/ready`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_auth.py tests/integration/test_event_replay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/ tests/integration/test_auth.py tests/integration/test_event_replay.py
git commit -m "feat(api): implement REST and WebSocket gateway, JWT auth, and ConnectionManager with event replay"
```

---

### Task 8: End-to-End Simulation, Recovery Flows & 100-Session Load Benchmark

**Files:**
- Create: `tests/integration/test_mock_interview_flow.py`
- Create: `tests/integration/test_recovery_flow.py`
- Create: `tests/load/__init__.py`
- Create: `tests/load/stress_test_100_sessions.py`

**Interfaces:**
- Consumes: Entire system (`api`, `core`, `config`)
- Produces:
  - Comprehensive full lifecycle integration tests.
  - Process restart / crash recovery verification.
  - 100-session concurrent load benchmark verifying TTFT P95 < 1.2s and zero dropped turns.

- [ ] **Step 1: Write the end-to-end integration and recovery tests**

```python
# tests/integration/test_mock_interview_flow.py
import pytest
from httpx import AsyncClient, ASGITransport
import fakeredis.aioredis
from api.main import create_app
from api.auth import create_service_token

@pytest.mark.asyncio
async def test_full_mock_interview_lifecycle(tmp_path):
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        token = create_service_token(issuer="backend", audience="ai-runtime", secret="test_jwt_secret_32_chars_long_1234")
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "req_001"}

        # 1. Create session
        resp = await ac.post("/api/v1/sessions", json={
            "candidate_name": "Tran Van B",
            "experience_years": 4,
            "target_company": "Acme Corp",
            "target_role": "Backend Engineer",
            "skills": ["Python", "FastAPI"]
        }, headers=headers)
        assert resp.status_code == 202
        sess_id = resp.json()["session_id"]

        # 2. Check session status
        status_resp = await ac.get(f"/api/v1/sessions/{sess_id}", headers=headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["session_id"] == sess_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_mock_interview_flow.py -v`
Expected: FAIL

- [ ] **Step 3: Implement full end-to-end orchestration, recovery test, and 100-session load test script**

Implement:
- `tests/integration/test_mock_interview_flow.py`: Full interview flow across all 4 stages, verifying turn persistence, assessor execution, and report barrier.
- `tests/integration/test_recovery_flow.py`: Verifying idempotency on restart, replaying pending tasks, and cleanup.
- `tests/load/stress_test_100_sessions.py`: High-concurrency async simulation spawning 100 WebSocket client sessions transmitting turns with VAD metrics, asserting TTFT, zero drop rate, zero assessment lag.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_mock_interview_flow.py tests/integration/test_recovery_flow.py -v`
Expected: PASS

- [ ] **Step 5: Run complete test suite and load simulation**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test(e2e): implement full lifecycle simulation, crash recovery tests, and 100-session load benchmark"
```
