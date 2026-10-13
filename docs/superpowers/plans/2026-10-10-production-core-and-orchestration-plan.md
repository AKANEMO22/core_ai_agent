# Production Core & Interview Orchestration Plan

- **Date:** 2026-10-10
- **Status:** Implemented (2026-10-11) — see "Implementation status" at the end
- **Builds on:** `docs/superpowers/specs/2026-09-24-ai-interview-core-engine-design.md` (v2) and the implementation delivered by `docs/superpowers/plans/2026-09-24-ai-interview-core-engine-plan.md`
- **Scope:** the AI module only. STT, TTS and VAD remain in the backend service; this module consumes final transcripts (optionally partial ones) and streams text tokens.

**Goal:** take the existing engine from "all components exist and pass tests" to a product-ready core that runs an interview end to end with no manual wiring, survives restarts and duplicate deliveries, keeps the realtime budget, grounds every question and score in retrieved knowledge, and produces an auditable, reproducible verdict.

**Architecture (target):** a two-plane design on top of the existing runtime. The **hot path** is one tool-free LLM call that streams the interviewer's words. The **control plane** is an orchestrator that runs while the candidate is talking: it delegates evaluation, follow-up planning and claim checking to subagents, keeps a code-generated notebook as its memory, and prepares a directive for the next turn. Retrieval happens before the interview (profiling) and inside background subagents, never on the hot path. Scores and the pass/fail decision are computed in code from structured evidence; the LLM writes only narrative.

**Tech stack:** unchanged (Python 3.11+, FastAPI, Pydantic v2, Redis Streams, PyJWT, httpx, pytest). Additions: `prometheus-client` (metrics), optional `tiktoken` (token counting), no new infrastructure.

---

## 1. Definition of "Product Ready" (release gates)

Every gate is measurable and is verified by a test, a benchmark, or a checklist item named in a task below. The core is not product ready until all gates hold.

| # | Gate | Measure |
|---|------|---------|
| G1 | **End-to-end without manual wiring** | A test drives the real app (`create_app`) from `POST /sessions` to `GET /report` returning 200, with no test code starting workers, setting counters, or transitioning state (Task 0.12). |
| G2 | **Crash and restart safety** | Process restart during profiling, generation, assessment, synthesis and deletion resumes to `REPORT_READY` or an explicit failure state; no duplicate LLM call for the same `message_id`/`job_id`, no lost completed turn (Task 0.12). |
| G3 | **Duplicate and ordering safety** | Repeated `Idempotency-Key`, repeated `message_id`, repeated Redis delivery, displaced connections and out-of-order turns create no duplicate turns, evaluations or reports (Tasks 0.2, 0.8). |
| G4 | **Realtime budget** | Against the real provider profile: accepted-turn TTFT P95 < 1.2 s, P99 < 2.5 s; `busy` rate < 1%; directive ready before the next answer in ≥ 95% of turns (Tasks 3.1, 3.5, 6.3). |
| G5 | **Security** | No `bash` or free-form file tools in production agent profiles; per-agent tool allowlists enforced in the runtime; automated test proves candidate text, CV text, tokens and prompts never reach logs; startup fails on unsafe config; delete purges Redis and workspace; SSRF suite passes (Tasks 0.11, 2.2, 2.3). |
| G6 | **Assessment quality** | On the golden set: pass/fail agreement with a human panel κ ≥ 0.6; per-criterion MAE ≤ 0.5 level; must-have coverage ≥ 95% of sessions; invented-evidence rate = 0 (Task 6.2). |
| G7 | **Reproducibility** | Same evaluations + same rubric version ⇒ byte-identical `decision.json`; every decision records code, model, prompt and skill versions (Task 5.1). |
| G8 | **Operability** | `/health/ready` reflects Redis, workspace, worker heartbeat and provider config; metrics exported for TTFT, turn duration, queue lag, dead letters, compactions, cost; alerts and runbook exist (Tasks 0.10, 6.4). |
| G9 | **Load and soak** | Section 4.2 profile of the spec (100 connected sessions, 20 answers/s burst) passes with orchestrator and evaluator fan-out enabled; 2-hour soak with zero unresolved jobs and zero unexpected dead letters (Task 6.3). |
| G10 | **Code quality** | ≥ 90% line coverage on `core/` and `api/`; type hints on public APIs; no TODO/pseudo-code; provider and retrieval contract suites pass for every enabled backend. |

---

## 2. Current State vs Target

What exists is solid infrastructure. What is missing is the wiring that makes it a product and the AI layer that makes it an interviewer.

| Area | Today (verified in code) | Target |
|------|--------------------------|--------|
| Assessor worker | `AssessorWorker` exists but nothing starts it in `api/main.py` (no lifespan); jobs are enqueued and never consumed | Worker pool managed by the app lifecycle with recovery loop and heartbeat |
| Completion → report | Nothing moves `DIALOGUE_COMPLETED → ASSESSING`, nothing sets `expected_assessments`, `ReportSynthesizer` is never called | Hook-driven pipeline; barrier check after every assessment; idempotent synthesis |
| Session state | `SessionStateManager` is an in-memory dict (`core/session/state.py:164`); restart loses every session | Redis-backed store with Lua CAS; in-memory implementation kept for unit tests |
| Idempotency | `IdempotencyManager` is in-memory; `session_lock` and `TokenBucket` are never used by `api/`; `CandidateAnswerMessage` has no `message_id` | Redis-backed keys with TTL; per-session turn lock; `message_id` dedupe; rate limits on answers and reconnects |
| Event retention | Redis list `session:events:{id}` has no TTL or trim | 24 h TTL, length cap, `resync_required` on expired cursor |
| Interview opening | The interviewer never speaks first; `READY → STAGE_ACTIVE` happens on the first candidate answer (`api/websocket.py:408`) | `start_interview` produces the opening question (turn 0) |
| Cancellation | `POST /cancel` only changes state; `cancellation_token` is never passed to `run_turn` (`api/websocket.py:469`); no barge-in | Session cancellation registry; `cancel_generation` message; partial turns persisted as interrupted |
| Deadlines | No per-turn deadline in `RuntimeAgent`, no timeout in the OpenAI adapter | Per-call and per-turn deadlines; transient retry with jitter |
| Config safety | CORS `allow_origins=["*"]` with credentials; default `JWT_SECRET` accepted; `DELETE` removes the workspace but not Redis keys | Fail-fast validation in production mode; full purge on delete |
| Interviewer memory | `ContextAssembler` capped at 1,500 tokens drops old turns, so the interviewer forgets | Rolling summary + larger budget + prompt-cache-friendly ordering |
| Orchestration | `StageDirector` counts turns; `complete_stage` tool is never registered (`api/main.py:186`); no follow-up logic | Orchestrator off the hot path with directives, coverage tracking, delegation |
| Retrieval | **Done:** `core/retrieval/` port, mock adapter, knowledge service, grounded stage plan, rubric per session | Contract kit for the real RAG; runtime `retrieve` tool for subagents |
| Evaluation | Two generic scores per turn; `rubric_target` only now populated; `confidence_score` hard-coded 0.95 | Per-criterion levels with anchors and verified evidence quotes |
| Report | Template markdown; `problem_solving` derived as `0.7·content + 0.3·comm`; no verdict | Code-computed decision + LLM narrative with validated citations |

---

## 3. Target Architecture

### 3.1 Planes and timing

```
   backend STT ──final transcript──► candidate_answer
                                          │
            ┌─────────────────────────────┴──────────────────────────────┐
            ▼ HOT PATH (≤ 1 LLM call, 0 tools, ≤ ~700 ms to first token)  ▼ CONTROL PLANE (background, ~5–120 s available)
  ┌───────────────────────────────┐                      ┌──────────────────────────────────────────────┐
  │ Speaker                       │                      │ Orchestrator cycle (stateless per turn)       │
  │ stable prefix (persona,       │                      │  input: brief + notebook + latest turn        │
  │   playbook, brief excerpt)    │                      │  delegate async: Evaluator(criteria)           │
  │ + rolling summary             │                      │                  Follow-up planner             │
  │ + last K turns                │                      │                  Claim checker                 │
  │ + directive                   │                      │  join → Blackboard → notebook regenerate       │
  │ streams response_token ───────┼──► backend TTS       │  → Directive for next turn                     │
  └───────────────────────────────┘                      └──────────────────────┬───────────────────────┘
                 ▲                                                               │
                 └──────────────── reads directive on next turn ◄────────────────┘
```

- The speaker never waits for the orchestrator. If no directive is ready it uses the **fallback directive** (next unasked question candidate of the stage).
- Every delegation, directive and compaction is an event in the replayable session stream.

### 3.2 Module map

| Layer | Module | Status | Responsibility |
|-------|--------|--------|----------------|
| Core runtime | `core/runtime/agent.py`, `types.py`, `client.py`, `providers/` | exists, harden | Streaming loop, tool execution, schema repair |
| Core runtime | `core/runtime/hooks.py`, `policy.py`, `profiles.py`, `compaction.py`, `structured.py`, `skills.py` | new | Hook bus, tool policy, agent profiles, auto-compaction, structured output, skill registry |
| Session | `core/session/state.py`, `events.py`, `idempotency.py`, `cancellation.py` | harden / new | Redis state with CAS, event retention, Redis idempotency, cancellation registry |
| Retrieval | `core/retrieval/` | done | Port, mock, knowledge service, stage-plan grounding |
| Interview | `core/interview/runner.py` (→ speaker), `context.py`, `director.py`, `summary.py` | refactor / new | Hot-path speaker, rolling summary |
| Orchestration | `core/orchestration/orchestrator.py`, `delegate.py`, `blackboard.py`, `directive.py`, `notebook.py`, `brief.py`, `coverage.py` | new | Control plane |
| Agents | `core/agents/evaluator.py`, `followup.py`, `claims.py`, `guard.py`, `narrative.py` | new (evaluator replaces `core/assessor/subagent.py`) | Specialized subagents |
| Assessor | `core/assessor/worker.py`, `queue.py`, `synthesizer.py`, `pool.py`, `decision.py` | harden / new | Worker pool lifecycle, barrier pipeline, decision engine |
| Gateway | `api/main.py`, `routes.py`, `websocket.py`, `schemas.py`, `auth.py`, `metrics.py`, `lifespan.py` | harden / new | Lifecycle, new messages, metrics, config validation |
| Skills | `skills/playbooks/<role_family>/<stage_type>.md`, `skills/rubric_packs/` | new | Versioned prompts selected by code |

### 3.3 Workspace layout (additions in bold)

```
/workspaces/{normalized_session_id}/
├── inputs/            form_input.json | cv_input.*
├── context/           company_intel.json, candidate_profile.json, interview_spec.json, stage_plan.json,
│                      retrieved_knowledge.json, rubric.json,
│                      **interview_brief.md** (written once by the brief skill, read-only afterwards)
│                      **interview_notebook.md** (regenerated by code each orchestrator cycle; two LLM sections)
├── turns/             turn_{n}_candidate.json, turn_{n}_interviewer.json (**turn_0_interviewer.json** = opening),
│                      **rolling_summary.json**
├── **orchestration/** directive_{n}.json, blackboard.json (snapshot at session end), delegation_{id}.json
├── evaluations/       turn_{n}_eval.json (**schema 2.0**, per-criterion)
├── **compaction/**    compaction_{n}.json (summary + dropped ranges, audit only)
├── reports/           final_assessment.json, interview_report.md, **decision.json**
└── spill/
```

### 3.4 Memory model (three files, three owners)

| File | Owner | Mutation | Purpose |
|------|-------|----------|---------|
| `interview_brief.md` | brief skill (once, after profiling) | write-once | Persona, stage frame, criteria with level anchors, question candidates, CV points to probe. Expected answers are **not** included (they stay in `retrieved_knowledge.json` for evaluators only). |
| `evaluations/*.json` | evaluator subagents | append-only, idempotent by `job_id` | Structured evidence: criterion, level, confidence, verified quotes. Source of truth for scores and the verdict. |
| `interview_notebook.md` | code (`NotebookRenderer`) + orchestrator via `update_notes` | sections 1–4 and 7 regenerated from state; sections 5–6 LLM-editable with length caps | The orchestrator's working memory. Candidate speech appears only as quoted blocks in the evidence log, never in the LLM sections. |

The orchestrator is a fresh LLM call every cycle with brief + notebook + latest turn as input, so it holds no long conversation and never needs compaction.

### 3.5 Event and message additions

- `LoopEventType` additions: `INTERVIEW_STARTED`, `BUSY`, `RESYNC_REQUIRED`, `GENERATION_CANCELLED`, `DIRECTIVE_READY`, `DELEGATION_STARTED`, `DELEGATION_COMPLETED`, `COMPACTION`, `DECISION_READY`.
- Inbound WebSocket messages: `start_interview`, `candidate_answer` (**now with `message_id`**), `cancel_generation`, `ack`, `ping`, `replay`, optional `candidate_partial`.
- REST additions: `POST /api/v1/sessions/{id}/start`, `POST /api/v1/sessions/{id}/assessments/requeue` (operator, service token).

---

## 4. Global Constraints

Carried over from the v2 spec and the previous plan, plus:

- The speaker agent profile has **zero tools** and a hard per-turn deadline. Anything that needs a tool runs in the control plane.
- No production agent profile may include `bash` or unrestricted file access. Workspace access goes through typed tools over `SessionWorkspace`.
- Scores, coverage and the verdict are computed in code. LLM output that feeds them must pass JSON-schema validation and evidence verification.
- Every prompt, playbook and rubric pack is versioned (content hash) and recorded in events and in `decision.json`.
- `ContextAssembler`'s inviolable core (persona, stage goals, criteria, key questions, current Q&A) is never truncated; the budget default rises from 1,500 to 6,000 tokens for the speaker.
- Candidate text enters LLM prompts only inside delimited untrusted blocks. It never enters notebook sections 5–6, logs or metrics.
- Redis is the source of truth for live state, counters, idempotency and events; the workspace for artifacts. Redis writes happen after the artifact write succeeds.
- All new keys carry a TTL: live state and counters 7 days after terminal state, events 24 h, idempotency 24 h, blackboard until session deletion.
- Hot-path hooks are non-blocking or carry a budget ≤ 20 ms.
- Code style: match existing modules (docstrings, type hints, `sanitize_secrets` on every error path, atomic artifact writes).

---

## 5. Work Breakdown

Sizes are relative (S ≈ a day, M ≈ 2–3 days, L ≈ a week, XL ≈ 2 weeks) for one engineer familiar with the code. Each task lists files, interfaces, steps and its exit test.

### Phase 0: Production Wiring and Hardening of the Existing Core

Goal: gates G1, G2, G3, G5, G8 for the current feature set, before any new AI behavior is added.

#### Task 0.1: Redis-backed session state with CAS (L)

**Files:** modify `core/session/state.py`; create `core/session/store.py`; modify call sites (`api/routes.py`, `api/websocket.py`, `core/interview/director.py`, `core/interview/runner.py`, `core/assessor/worker.py`, `core/assessor/synthesizer.py`) and their tests.

**Interfaces:**
```python
class SessionStateStore(Protocol):
    async def create(self, session_id: str, initial: SessionState) -> SessionState: ...
    async def get(self, session_id: str) -> Optional[SessionState]: ...
    async def transition(self, state: SessionState, expected_version: int,
                         new_status: Optional[LifecycleStatus] = None, **updates) -> SessionState: ...
    async def delete(self, session_id: str) -> None: ...

class InMemorySessionStateStore(SessionStateStore): ...      # unit tests
class RedisSessionStateStore(SessionStateStore): ...         # key session:{id}:state, Lua CAS on state_version
```

- [ ] Keep FSM graph and `SessionState` as they are; move CAS into a Lua script (`GET` → compare `state_version` → `SET` with `EX`).
- [ ] Convert `StageDirector.record_turn/advance_stage/complete_current_stage` and all call sites to `async`.
- [ ] TTL: 7 days after entering a terminal status; refreshed on every transition.
- [ ] Tests: CAS conflict under concurrent transitions (two coroutines, one must fail), restart reload, TTL set on terminal state. Run existing `test_session_state.py` against both stores (parametrized fixture).

#### Task 0.2: Redis idempotency, `message_id`, turn lock, rate limits (M)

**Files:** modify `core/session/idempotency.py`, `api/schemas.py`, `api/websocket.py`, `api/routes.py`.

- [ ] `RedisIdempotencyStore`: `SET NX` for in-flight, result cache with payload hash, 24 h TTL; same API as today.
- [ ] Add `message_id: str` (UUID) to `CandidateAnswerMessage`; reject frames without it with a typed `error` (`code=MISSING_MESSAGE_ID`).
- [ ] Mapping `session:{id}:message_turns` (hash `message_id → turn_id`, TTL 24 h). A repeated `message_id` replays that turn's events from the store instead of calling the LLM.
- [ ] Wrap turn execution in `idempotency_mgr.session_lock(session_id)`; a second in-flight answer gets `error` with `code=TURN_IN_PROGRESS`. This closes the race where a displaced connection's `run_turn` keeps running while the new connection submits.
- [ ] Token buckets: answers (default 1/s, burst 3), reconnects (default 5/min), session creation per service caller; exceeded → `BUSY`/`429` with `retry_after_ms`.
- [ ] Tests: duplicate `message_id` → exactly one LLM call (mock `call_count`), concurrent submit → one accepted one rejected, bucket exhaustion.

#### Task 0.3: Assessor worker pool lifecycle (M)

**Files:** create `core/assessor/pool.py`, `api/lifespan.py`; modify `api/main.py`.

**Interfaces:**
```python
class AssessorWorkerPool:
    def __init__(self, worker_factory, size: int, group: str, recovery_interval_s: float, lease_ms: int): ...
    async def start(self) -> None     # N consumer tasks (block_ms=1000) + 1 recovery task (XAUTOCLAIM) + heartbeat
    async def stop(self, grace_s: float = 30) -> None   # stop reading, finish in-flight, cancel after grace
    def status(self) -> Dict[str, Any]                  # for /health/ready
```

- [ ] FastAPI `lifespan` creates the consumer group, starts the pool, stops it on shutdown; `ASSESSOR_POOL_SIZE` setting (default 4).
- [ ] Heartbeat key `assessor:{consumer}:heartbeat` (EX 15 s); `/health/ready` fails if no heartbeat is fresh.
- [ ] Tests: pool processes enqueued jobs through the app with no manual worker; stop waits for in-flight job; recovery task reclaims a job abandoned by a crashed consumer.

#### Task 0.4: Completion → assessing → report pipeline (M)

**Files:** create `core/assessor/pipeline.py`; modify `core/interview/runner.py`, `core/assessor/worker.py`, `core/assessor/synthesizer.py`.

- [ ] On enqueue, `SADD session:{id}:enqueued_assessments job_id` (same transaction as the stream `XADD`, via pipeline/MULTI).
- [ ] On `DIALOGUE_COMPLETED`: `expected = SCARD(enqueued)`, store `session:{id}:expected_assessments`, transition to `ASSESSING`, then run `maybe_synthesize`.
- [ ] `maybe_synthesize(session_id)`: `SET session:{id}:synth_lock NX EX 120` → state is `ASSESSING` → `check_barrier` → `synthesize` → release. Called after every `assessment_completed`, after the `ASSESSING` transition, and by a 60 s sweeper for recovery.
- [ ] Sessions with zero candidate turns go straight to `REPORT_READY` with an "insufficient evidence" report.
- [ ] Remove the hard-coded `confidence_score`; replace with `None` until Task 5.1 computes it.
- [ ] Tests: full flow through the real app reaches `REPORT_READY`; duplicate `assessment_completed` triggers exactly one synthesis; sweeper recovers after a crash between barrier and synthesis.

#### Task 0.5: Profiling failure handling and deadline (S)

**Files:** modify `api/routes.py`, `core/profiler/agent.py`.

- [ ] Profiling runs under `PROFILING_DEADLINE_SECONDS` (default 90). Failure or timeout → `FAILED` with `failure_code` (`PROFILING_FAILED`, `PROFILING_TIMEOUT`), safe client message, correlation id in metadata.
- [ ] `GET /sessions/{id}` exposes `failure_code` and the safe message; `GET /report` returns 409 as already specified.
- [ ] Tests: profiler exception → `FAILED`; timeout → `FAILED` with `PROFILING_TIMEOUT`.

#### Task 0.6: Interview opening (turn 0) (M)

**Files:** modify `api/websocket.py`, `api/routes.py`, `core/interview/runner.py`, `api/schemas.py`.

- [ ] `start_interview` WebSocket message and `POST /sessions/{id}/start` (backend-driven flows). Valid only in `READY`; idempotent (second call replays the opening events).
- [ ] Transition `READY → STAGE_ACTIVE` with the first stage; speaker streams the opening question with `turn_id="turn_000"`, persists `turns/turn_0_interviewer.json`, emits `INTERVIEW_STARTED` then `TURN_COMPLETE` (`turn_number=0`). `_get_previous_question(1)` then finds the real opening question.
- [ ] Remove the implicit `READY → STAGE_ACTIVE` on the first answer; an answer in `READY` returns `error` `code=INTERVIEW_NOT_STARTED`.
- [ ] Tests: start twice → one LLM call; answer before start → typed error; e2e flow updated.

#### Task 0.7: Cancellation and barge-in (M)

**Files:** create `core/session/cancellation.py`; modify `api/websocket.py`, `api/routes.py`, `core/interview/runner.py`, `core/runtime/providers/openai.py`.

- [ ] `CancellationRegistry`: `token_for(session_id) -> asyncio.Event`; `cancel(session_id, reason)`; cleared on turn completion.
- [ ] `cancel_generation` message (barge-in) sets the token; `POST /cancel` sets it and cancels delegated tasks (Task 3.6).
- [ ] Runner passes the token; on `CANCELLED` the partial interviewer text is persisted with `interrupted: true`, `GENERATION_CANCELLED` and `TURN_COMPLETE` (`interrupted=true`) are emitted, and the stage turn counter still advances.
- [ ] OpenAI adapter closes the HTTP stream on cancellation (`await stream.close()`), so provider tokens stop being billed.
- [ ] Tests: cancel mid-stream → no further `response_token`, partial artifact written, next answer accepted; `POST /cancel` during generation → `CANCELLED` state and closed stream.

#### Task 0.8: Backpressure and interactive concurrency (S)

**Files:** modify `api/websocket.py`, `api/main.py`, `config/settings.py`.

- [ ] `MAX_INTERACTIVE_GENERATIONS` semaphore (separate from the connection ceiling) and `MAX_INTERACTIVE_QUEUE_MS`; waiting longer emits `BUSY` with bounded `retry_after_ms` and does not queue the turn.
- [ ] Separate semaphores for profiler, evaluator and synthesis calls (defaults documented as derived from provider quotas).
- [ ] Tests: semaphore saturated → `BUSY` within the queue budget; queue wait included in TTFT measurement.

#### Task 0.9: Event retention and resync (S)

**Files:** modify `core/session/events.py`, `api/websocket.py`.

- [ ] On append: `EXPIRE session:events:{id} 86400` and `LTRIM` to `EVENT_STREAM_MAX_LENGTH` (default 20,000).
- [ ] Replay request with `last_sequence` older than the oldest retained event → `RESYNC_REQUIRED` carrying the latest durable turn snapshot (`current_turn`, `current_stage_id`, last interviewer text).
- [ ] Tests: TTL present, trim keeps the newest, resync emitted on expired cursor.

#### Task 0.10: Observability (M)

**Files:** create `api/metrics.py`, `core/observability.py`; modify `api/main.py`, `core/runtime/agent.py`, `core/assessor/worker.py`.

- [ ] Structured JSON logs with `correlation_id`, `session_hash` (SHA-256 prefix, never the raw id), event type, latency, retry count, error code.
- [ ] Prometheus metrics: `ttft_seconds` (histogram), `turn_duration_seconds`, `provider_errors_total`, `assessment_lag_seconds`, `assessment_pending`, `dead_letters_total`, `busy_total`, `compactions_total`, `llm_tokens_total{agent,model}`, `llm_cost_usd_total`, `active_connections`.
- [ ] `/metrics` endpoint behind the service token; `/health/ready` adds worker heartbeat and provider configuration checks.
- [ ] Test: log capture over a full e2e run contains no candidate answer text, CV text, token or prompt fragment (assert against the inputs used).

#### Task 0.11: Configuration and security hardening (M)

**Files:** modify `config/settings.py`, `api/main.py`, `api/routes.py`, `api/auth.py`, `core/workspace/cleanup.py`.

- [ ] `APP_ENV` (`development` | `production`). In production, startup fails if `JWT_SECRET` is the default or shorter than 32 bytes, if `CORS_ALLOWED_ORIGINS` is empty or contains `*`, if no provider key is configured, or if `RETRIEVAL_BACKEND=mock` without `ALLOW_MOCK_RETRIEVAL=true`.
- [ ] CORS origins from settings; `allow_credentials` only with explicit origins.
- [ ] `DELETE /sessions/{id}`: cancel active work, purge Redis keys (state, events, counters, idempotency, blackboard, message map), delete the workspace, add the session to `revoked_sessions` (checked by `verify_session_token`), record a content-free audit event.
- [ ] Retention sweeper task in the lifespan (`cleanup_expired_workspaces`, raw-upload deletion 24 h after report, as specified).
- [ ] Tests: startup refuses unsafe production config; delete purges every key (assert `SCAN` finds none); revoked token rejected on reconnect.

#### Task 0.12: End-to-end and recovery suites through the real app (M)

**Files:** rewrite `tests/integration/test_mock_interview_flow.py`, extend `tests/integration/test_recovery_flow.py`.

- [ ] E2E: create → profiling → `start_interview` → N answers (with `message_id`) → `DIALOGUE_COMPLETED` → pool evaluates → `REPORT_READY` → `GET /report` 200. No worker, counter or state manipulation in the test.
- [ ] Recovery: simulate restart (new `create_app` on the same Redis and workspace) during profiling, streaming, assessment, synthesis and deletion; assert resumption and no duplicate LLM calls.
- [ ] Both suites run against fakeredis in CI and against a real Redis in the nightly job.

**Phase 0 exit:** G1, G2, G3, G5 (config part), G8 hold for the current feature set.

---

### Phase 1: Retrieval (done) and RAG Contract Kit

Delivered in this change set: `core/retrieval/port.py`, `mock.py`, `fixtures/mock_corpus.json`, `knowledge.py`, `factory.py`, `RETRIEVAL_*` settings; profiler integration; stage-plan grounding; `rubric_target` from stage criteria; context assembler rendering of competencies and key questions. Tests: `tests/unit/test_retrieval.py` and additions in `test_profiler.py`, `test_context_assembler.py`, `test_stage_director.py`.

#### Task 1.1: Retrieval contract kit (S)

- [ ] `tests/contract/test_retrieval_contract.py` parametrized by `RETRIEVAL_BACKEND`: every collection returns the documented metadata fields, filters behave per the port contract, `top_k` honored, timeouts raise `RetrievalError`, no candidate identity in queries.
- [ ] Document the plug-in steps in `core/retrieval/README.md`.

#### Task 1.2: Runtime `retrieve` tool (S)

- [ ] `core/retrieval/tool.py`: `RuntimeTool` wrapping `RetrievalClient.search` with a fixed collection allowlist, `top_k ≤ 5`, deadline 2 s, results stripped to content + metadata subset. Available only to control-plane profiles (Task 2.2).

---

### Phase 2: Core Harness (RuntimeAgent v2)

Goal: the minimal core you described, with the production guardrails the interview needs.

#### Task 2.1: Hook bus (M)

**Files:** create `core/runtime/hooks.py`; modify `core/runtime/agent.py`.

**Interfaces:**
```python
class HookEvent(str, Enum):
    TURN_START, BEFORE_MODEL, AFTER_MODEL, BEFORE_TOOL, AFTER_TOOL, BEFORE_COMPACTION, AFTER_COMPACTION,
    TURN_END, STAGE_TRANSITION, DIALOGUE_COMPLETED, ASSESSMENT_COMPLETED, REPORT_READY

class HookBus:
    def register(self, event: HookEvent, fn, *, blocking: bool = False, budget_ms: int = 20, name: str) -> None
    async def emit(self, event: HookEvent, ctx: HookContext) -> HookOutcome   # blocking hooks run inline under budget; others as tasks
```

- [ ] Hook failures are isolated and logged; a `BEFORE_TOOL` hook that fails or times out **denies** the tool (fail-closed); all other hooks fail-open.
- [ ] `RuntimeAgent` emits every event; `HookContext` carries session/turn ids, agent profile name, messages (read-only view) and mutable `extra`.
- [ ] Tests: ordering, budget timeout, fail-closed on `BEFORE_TOOL`, non-blocking hooks do not delay token streaming (measure).

#### Task 2.2: Agent profiles and tool policy (M)

**Files:** create `core/runtime/profiles.py`, `core/runtime/policy.py`; modify `core/runtime/agent.py`, `api/main.py`.

```python
@dataclass(frozen=True)
class AgentProfile:
    name: str                      # "speaker" | "orchestrator" | "evaluator" | "followup" | "claims" | "profiler" | "narrative" | "brief"
    model: str
    temperature: float
    allowed_tools: frozenset[str]  # speaker: frozenset()
    max_iterations: int
    turn_deadline_ms: int
    context_budget_tokens: int
    compaction: CompactionPolicy   # enabled, threshold_ratio, keep_last_turns, summary_model
    output_schema: Optional[dict]  # for structured agents
```

- [ ] `ToolPolicy.check(profile, tool_name) -> Decision`; the runtime converts a denial into a `TOOL_CALL_RESULT` with `code=TOOL_DENIED` and an audit event, never an exception.
- [ ] Profiles are loaded from `config/agent_profiles.yaml` with env overrides; production validation rejects any profile containing `bash`, `write_file`, `edit_file` unless `APP_ENV=development`.
- [ ] Per-turn deadline enforced in `run_loop_streaming` (`asyncio.timeout`), emitting `ERROR code=TURN_DEADLINE` with partial text.
- [ ] Tests: denied tool, deadline, profile validation.

#### Task 2.3: Typed workspace tools (M)

**Files:** create `core/runtime/tools/workspace.py`, `core/runtime/tools/notes.py`, `core/runtime/tools/dev.py`.

- [ ] `read_artifact(path)`: read-only, allowlisted prefixes (`context/`, `turns/`, `evaluations/`, `orchestration/`), spill-aware.
- [ ] `write_artifact(artifact_type, payload)`: JSON-schema per `artifact_type` (`interview_brief`, `directive`, `delegation_result`); path derived from type, never from the model.
- [ ] `update_notes(section, text)`: only sections 5–6 of the notebook, per-section cap (1,200 / 600 chars), rejects text containing verbatim candidate sentences longer than 12 words (checked against `turns/*`), appends a change record.
- [ ] `dev.py`: `read_file`, `write_file`, `edit_file`, `bash` inside a temp sandbox for offline/dev agents only; import guarded by `APP_ENV`.
- [ ] Tests: path traversal rejected, schema rejection, notes cap and quote rejection, dev tools unavailable in production.

#### Task 2.4: Context budgeting and auto-compaction (L)

**Files:** create `core/runtime/compaction.py`, `core/runtime/tokens.py`; modify `core/runtime/agent.py`.

```python
class TokenCounter:            # tiktoken when installed, else chars/3.5 (Vietnamese) or chars/4 (English) by detected script
class ContextCompactor:
    async def maybe_compact(self, messages, profile, llm) -> CompactionResult
```

- [ ] Trigger: before each model call when `count(messages) > threshold_ratio × context_budget_tokens`.
- [ ] Step 1 (no LLM): replace tool results older than `keep_last_turns` with their spill preview or a `<result trimmed, see spill/...>` stub.
- [ ] Step 2: summarize the middle block with `summary_model` into a fixed schema (`goals`, `decisions`, `open_items`, `facts_with_ids`); insert as one user-role message tagged `compaction_summary`.
- [ ] Step 3 (fallback on summary failure): drop oldest non-pinned turns until under budget.
- [ ] Invariants (tested): system message untouched; messages with `pinned=True` untouched; last `keep_last_turns` untouched; an assistant `tool_calls` message and its `tool` results are kept or removed together; token count after compaction < budget.
- [ ] Emits `COMPACTION` event, `BEFORE/AFTER_COMPACTION` hooks, and writes `compaction/compaction_{n}.json`.
- [ ] Profile defaults: speaker **disabled** (rolling summary instead, Task 3.2), orchestrator safety-only (threshold 0.9), profiler / narrative / brief **enabled** (threshold 0.7).
- [ ] Tests: each step, each invariant, a 50-tool-call profiler session staying under budget, fallback path.

#### Task 2.5: Structured output helper (S)

**Files:** create `core/runtime/structured.py`; modify `core/runtime/providers/openai.py`, `providers/mock.py`.

- [ ] `run_structured(llm, messages, schema, *, max_repairs=1, deadline_ms)`: uses provider JSON mode when the capability flag says so, validates with `jsonschema`, one repair round with the violation list, raises `StructuredOutputError` after that.
- [ ] Mock provider gains `canned_structured` support; capability `json_mode` added to `ProviderCapabilities`.
- [ ] Tests: valid first try, repaired second try, failure after budget.

#### Task 2.6: Provider hardening and contract suite (M)

**Files:** modify `core/runtime/providers/openai.py`, `core/runtime/client.py`; create `tests/contract/test_provider_contract.py`.

- [ ] `stream_chat(..., deadline_s)`; transient retry (429, 5xx, connection errors) once with jitter, never for 4xx; usage accounting into metrics; stream closed on cancellation.
- [ ] Capability check at startup against the configured model (streaming, tool calls, usage, json mode); unsupported combinations fail startup, as the spec requires.
- [ ] Contract suite (run against each enabled provider in the nightly job): streaming, tool call assembly, usage event, cancellation, timeout, malformed event, rate limit.

#### Task 2.7: Skill registry (S)

**Files:** create `core/runtime/skills.py`, `skills/playbooks/{backend,frontend,any}/{warmup,technical,system_design,behavioral,wrapup}.md`, `skills/rubric_packs/README.md`.

- [ ] `SkillRegistry.get_playbook(role_family, stage_type) -> Skill(content, version, sha256)` with fallback to `any`; loaded at startup, validated (front-matter with `version`, size cap).
- [ ] Selected by code at stage transition; the sha256 is recorded in the `STAGE_TRANSITION` event and in `decision.json`.
- [ ] Tests: selection and fallback, hash stability.

**Phase 2 exit:** the runtime enforces profiles, deadlines and policies; compaction invariants hold; provider contract passes.

---

### Phase 3: Orchestrator and Hot-Path Split

#### Task 3.1: Speaker refactor (M)

**Files:** refactor `core/interview/runner.py` → `core/interview/speaker.py` (keep the public `run_turn` signature); modify `core/interview/context.py`.

- [ ] Prompt order for cache hits: `[system: persona + playbook + brief excerpt (stable)] [rolling summary] [last K turns] [directive] [candidate answer]`.
- [ ] Profile `speaker`: no tools, `max_iterations=1`, deadline 8 s, budget 6,000 tokens; the 1,500-token truncation logic stays as the last resort.
- [ ] Directive consumption: read `orchestration/directive_{n}.json` if present, else fallback directive from the stage plan (next unasked `question_candidates` entry), recorded in the turn artifact (`directive_source`).
- [ ] Tests: no tool definitions sent to the provider (assert `tools is None`), prefix stability across turns (byte-equal system message), fallback directive used when none exists.

#### Task 3.2: Rolling summary (S)

**Files:** create `core/interview/summary.py`.

- [ ] After each `TURN_COMPLETE`, a background job updates `turns/rolling_summary.json` (`covered_topics`, `candidate_claims`, `open_threads`, `last_turn_included`) with the summary model under a 5 s deadline; failure keeps the previous summary.
- [ ] Speaker includes the summary only if `last_turn_included ≥ current_turn − K`; otherwise it falls back to more raw turns.
- [ ] Tests: summary updated asynchronously, stale summary ignored, interviewer does not repeat an already-asked question across 12 turns (mock provider scripted).

#### Task 3.3: Interview brief skill (M)

**Files:** create `core/orchestration/brief.py`, `skills/brief_template.md`.

- [ ] Runs once after profiling (profile `brief`, tools: `read_artifact`, `write_artifact`); input: `retrieved_knowledge.json`, `stage_plan.json`, `candidate_profile.json`, `interview_spec.json`.
- [ ] Output `context/interview_brief.md` with fixed sections (frame, persona, stages with criteria and anchors, question candidates with follow-ups, CV points to probe, risks). Validator checks sections, size cap (≤ 4,000 tokens) and that no `expected_points` text leaked.
- [ ] If the brief fails, the session continues with a code-generated minimal brief; `retrieval_status` and `brief_status` exposed on `GET /sessions/{id}`.
- [ ] Tests: generated, validated, fallback.

#### Task 3.4: Notebook (M)

**Files:** create `core/orchestration/notebook.py`.

- [ ] `NotebookRenderer.render(state, blackboard, evaluations, notes) -> str` regenerates sections 1–4 and 7 from data; sections 5–6 come from `update_notes` records.
- [ ] Written atomically after every orchestrator cycle; previous version kept as `interview_notebook.prev.md` for audit.
- [ ] Section 7 (evidence log) quotes candidate text inside fenced blocks labeled `candidate_quote`; sections 5–6 are validated by the same quote rule as `update_notes`.
- [ ] Tests: deterministic render, LLM sections preserved across renders, quote rule.

#### Task 3.5: Orchestrator cycle (L)

**Files:** create `core/orchestration/orchestrator.py`, `directive.py`, `coverage.py`.

```python
class InterviewOrchestrator:
    async def run_cycle(self, session_id: str, turn_number: int) -> Directive
```

- [ ] Triggered after each candidate answer (background task with deadline `ORCHESTRATOR_DEADLINE_MS`, default 8,000). Steps: delegate evaluator / follow-up / claims (Task 3.6) → join → `CoverageTracker.update` → render notebook → one structured LLM call (profile `orchestrator`, tools `set_directive`, `complete_stage`, `retrieve`, `read_artifact`) → write `orchestration/directive_{n+1}.json` → emit `DIRECTIVE_READY`.
- [ ] Stage transition rule: `complete_stage` is accepted when every must-have criterion of the stage has coverage confidence ≥ `COVERAGE_MIN_CONFIDENCE` (0.6) or the stage budget is exhausted; otherwise the orchestrator is told why and continues. `StageDirector` keeps the CAS and budget logic.
- [ ] Deadline or failure → fallback directive (`generated_by="fallback"`) and a `DIRECTIVE_READY` event with `degraded=true`; the speaker is never blocked.
- [ ] Metrics: `directive_ready_before_next_answer_ratio`, cycle duration.
- [ ] Tests: directive produced, coverage gate blocks premature transition, fallback on timeout, event trail complete.

#### Task 3.6: Delegation runtime and blackboard (M)

**Files:** create `core/orchestration/delegate.py`, `blackboard.py`.

```python
async def delegate(agent: str, task: dict, *, mode: Literal["async","sync"], deadline_ms: int, budget: dict) -> DelegationHandle
```

- [ ] Registered subagent types with their profiles; per-type semaphores from settings; results written to the blackboard (`session:{id}:blackboard` hash) and `orchestration/delegation_{id}.json`; `DELEGATION_STARTED/COMPLETED` events with duration and token usage.
- [ ] `mode="sync"` is rejected when called from the speaker profile (policy), allowed in the control plane.
- [ ] Session cancellation and deletion cancel outstanding delegations.
- [ ] Blackboard keys: `coverage:<criterion>`, `asked_questions`, `pending_follow_ups`, `flags`, `directive:<turn>`; snapshot to `orchestration/blackboard.json` on dialogue completion.
- [ ] Tests: parallel delegations, deadline, cancellation, semaphore bound.

#### Task 3.7: Degradation matrix (S)

- [ ] Document and test each failure: retrieval degraded → generic playbook and legacy two-score evaluation; brief failed → minimal brief; orchestrator timeout → fallback directive; evaluator failure → existing retry / DLQ; summary failure → raw turns. Each path emits a `degraded` flag on the relevant event and increments a metric.

**Phase 3 exit:** with the real provider, speaker TTFT P95 within +50 ms of the Phase 0 baseline; directive-ready ratio ≥ 95% on the simulated-candidate suite (Task 6.1, run early with the mock provider).

---

### Phase 4: Specialized Subagents

#### Task 4.1: Criterion evaluator v2 (L)

**Files:** create `core/agents/evaluator.py`; modify `core/assessor/worker.py`, `core/assessor/synthesizer.py`; delete `core/assessor/subagent.py` after migration.

Evaluation artifact schema 2.0:
```json
{
  "schema_version": "2.0", "job_id": "...", "session_id": "...", "turn_id": "turn_003", "stage_id": "...",
  "input_sha256": "...", "rubric_version": "backend_senior_v1",
  "criteria": [
    {"competency_id": "be_distributed_systems", "status": "scored", "level": 3, "confidence": 0.8,
     "anchor_matched": "3", "rationale": "...",
     "evidence": [{"quote": "...", "turn_id": "turn_003", "verified": true}]},
    {"competency_id": "be_data_storage", "status": "insufficient_evidence"}
  ],
  "answer_quality": {"content_score": 7, "communication_score": 6},
  "delivery_observations": {"...": "unchanged, non-scoring"},
  "flags": {"off_topic": false, "injection_suspected": false},
  "model": {"id": "...", "prompt_sha256": "..."},
  "evaluated_at": 0
}
```

- [ ] One structured call per turn covering all `rubric_target` criteria, with the rubric anchors and the question's `expected_points` from `retrieved_knowledge.json`; the transcript is wrapped as untrusted and labeled as ASR output.
- [ ] Evidence verification: each quote must be a normalized substring (case, whitespace, punctuation folded) of the candidate turn; unverifiable quotes are dropped and the criterion's confidence halved; artifacts record `verified`.
- [ ] Idempotency key unchanged (`job_id + input_sha256 + rubric_version`); schema 1.0 artifacts still readable by the synthesizer (legacy path).
- [ ] Tests: schema validation, quote verification (positive, fuzzy, invented), `insufficient_evidence`, legacy compatibility.

#### Task 4.2: Follow-up planner (S)

- [ ] `core/agents/followup.py`: input = question `expected_points`, retrieved `follow_ups`, the evaluation gaps; output = ranked probe hints and at most one follow-up question; written to the blackboard for the orchestrator. May call `retrieve` (question bank, same competency, higher difficulty).

#### Task 4.3: Claim checker (S)

- [ ] `core/agents/claims.py`: compares `candidate_profile.claims` with what the candidate said; output `supported | contradicted | untested` per claim with evidence; feeds notebook section 7 and the report.

#### Task 4.4: Input guard (S)

- [ ] `core/agents/guard.py`: non-blocking `TURN_START` hook (budget 20 ms for the heuristic part; the LLM part runs as a delegation) flagging injection attempts and off-topic answers; flags land on the blackboard; the orchestrator's directive `mode="redirect"` handles them. Candidate text with flagged injection is still evaluated but marked.

#### Task 4.5: Partial transcripts (optional, M; needs a backend contract)

- [ ] `candidate_partial` frames update a `turns/partial_{n}.txt` buffer; the follow-up planner may start on partials ≥ 80 words; the final answer supersedes. Gate: measured reduction in silence after end-of-speech.

**Phase 4 exit:** zero invented-evidence quotes on the golden set; evaluator cost per turn within budget (one call).

---

### Phase 5: Decision and Report

#### Task 5.1: Decision engine (M)

**Files:** create `core/assessor/decision.py`; modify `core/assessor/synthesizer.py`, `core/assessor/pipeline.py`.

Aggregation (deterministic, no LLM):
1. Per criterion, `final_level = Σ(level_i · conf_i) / Σ(conf_i)` over `scored` entries; `covered = Σ(conf_i) ≥ COVERAGE_MIN_CONFIDENCE`.
2. Any must-have criterion not covered → `NEEDS_REVIEW` (missing evidence is not failure).
3. Any must-have criterion with `final_level < pass_level` → `FAIL`.
4. `weighted_score` = weight-renormalized mean over covered criteria; `|weighted_score − pass_threshold| ≤ BORDERLINE_BAND` (0.25) → `NEEDS_REVIEW`; else `PASS`/`FAIL`.

`reports/decision.json` records verdict, scores, gates, per-criterion detail, uncovered criteria, and `versions` (`code` git SHA, model ids, prompt hashes, skill hashes, rubric version).

- [ ] Tests: property test that decision is a pure function of its inputs; each rule with fixtures; borderline band; must-have gate ordering.

#### Task 5.2: Narrative report v2 (M)

**Files:** create `core/agents/narrative.py`, `skills/report_template.md`; modify `core/assessor/synthesizer.py`.

- [ ] Input: `decision.json`, notebook, evaluations; profile `narrative` (tools `read_artifact`, compaction enabled). Output sections: verdict and summary, per-criterion findings with quotes, strengths, improvement areas with study plan, claim verification, uncovered criteria, limitations and disclaimer.
- [ ] Every quote in the narrative is re-verified against `turns/*`; unverifiable quotes are removed before writing; `final_assessment.json` gains `decision` and `narrative_version`.
- [ ] Disclaimer text selected by `PRODUCT_MODE` (`practice` | `hiring_assist`); `hiring_assist` additionally forces `NEEDS_REVIEW` to require a human reviewer field before the report is marked final (API `POST /sessions/{id}/review`).
- [ ] Tests: citation verification, both product modes, report renders all criteria.

#### Task 5.3: API surface for decisions and operators (S)

- [ ] `GET /report` returns `decision` and `verdict`; `SessionReportResponse` updated.
- [ ] `POST /sessions/{id}/assessments/requeue` (service token) re-enqueues DLQ jobs idempotently and moves `ASSESSMENT_FAILED → ASSESSING`.
- [ ] `GET /sessions/{id}` exposes `retrieval_status`, `brief_status`, coverage summary.

---

### Phase 6: Evaluation Harness, Load, Soak and Release

#### Task 6.1: Simulated candidates (M)

- [ ] `tests/simulation/`: candidate personas (target level, scripted weaknesses, one injection attempt, one off-topic answer) driven by an LLM through the real app; runs with the mock provider in CI (scripted answers) and with real providers nightly.

#### Task 6.2: Golden set and quality metrics (M)

- [ ] `tests/quality/`: human-graded transcripts (mock-generated at first; real ones only with consent), scorer computing κ, per-criterion MAE, coverage, invented-evidence rate; thresholds from G6 enforced in the nightly job.

#### Task 6.3: Load, burst, soak and chaos (M)

- [ ] Extend `tests/load/stress_test_100_sessions.py` with orchestrator and evaluator fan-out enabled; report TTFT, directive-ready ratio, assessment lag, Redis memory, event-loop lag.
- [ ] 30-minute burst and 2-hour soak jobs; chaos cases: Redis restart, provider 429 storm, worker crash, workspace disk full.

#### Task 6.4: Release checklist (S)

- [ ] `docs/runbook.md`: configuration reference, dashboards, alerts (TTFT P95, busy rate, DLQ > 0, assessment lag, heartbeat missing), on-call procedures (requeue, resync, delete).
- [ ] Config validation output printed at startup; prompts, playbooks and rubric packs frozen by hash for the release.

---

## 6. Sequencing

```
Phase 0 (wiring)  ──────────────────────────┐
Phase 2 (harness) ───── can start in parallel ┼──► Phase 3 (orchestrator) ──► Phase 4 (subagents) ──► Phase 5 (decision) ──► Phase 6 (gates)
Phase 1 (retrieval) done; 1.1–1.2 any time ─┘                 ▲                       ▲
                                                 Task 6.1 simulation harness built early (mock provider) and reused from here on
```

Dependencies worth noting: Task 3.1 needs 2.2 (profiles); 3.5 needs 2.5 (structured output) and 3.6; 4.1 needs 2.5; 5.1 needs 4.1's schema; 6.3 needs 3.5 and 4.1.

---

## 7. Open Decisions (need an owner before the dependent task starts)

| Decision | Blocks | Default if undecided |
|----------|--------|----------------------|
| Pass/fail semantics and legal review (the spec lists employment decisions as a non-goal; EU AI Act high-risk classification and NYC Local Law 144 bias audits apply if used for hiring) | Task 5.1, 5.2 | `PRODUCT_MODE=practice`, verdict phrased as readiness; `NEEDS_REVIEW` requires a human |
| Interview language(s) (Vietnamese changes token estimates, tokenizer choice, ASR vocabulary, playbook text) | Task 2.4, 2.7, 4.1 | English playbooks; Vietnamese token factor in `TokenCounter` |
| Partial-transcript contract with the backend | Task 4.5 | Not implemented; follow-up decided inline by the speaker |
| Provider and model per profile (speaker needs the lowest TTFT; evaluators need the best calibration) | Task 2.6, 6.2 | OpenAI adapter for all; profiles configurable |
| Real RAG delivery date | Task 1.1 contract run | Mock backend with `ALLOW_MOCK_RETRIEVAL=true` outside production |
| Human review workflow for `NEEDS_REVIEW` (who, where, SLA) | Task 5.2 | API field only; no UI |

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Provider latency spikes break G4 at P99 | Per-call deadlines, fallback model in the speaker profile, instant backchannel text while waiting (configurable) |
| Evaluator cost grows with criteria × turns | One structured call per turn; smaller calibrated model for evaluators; batch criteria |
| Async state refactor (Task 0.1) touches many files | Do it first, behind the `SessionStateStore` protocol, with parametrized tests over both stores |
| Notebook drift or injected content | Code-generated sections; LLM sections capped and quote-checked; previous version retained |
| Compaction loses a fact the orchestrator needed | Facts referenced by id in summaries; artifacts remain readable via `read_artifact`; audit file per compaction |
| Mock corpus shapes behavior that the real RAG will not reproduce | Contract kit, quality metrics measured only on the real backend before release |

---

## 9. Implementation status (2026-10-11)

| Phase | Status | Where | Verified by |
|---|---|---|---|
| 0 Wiring & hardening | Done | `core/session/{store,events,idempotency,cancellation}.py`, `core/interview/controller.py`, `core/assessor/{pool,pipeline}.py`, `api/*`, `core/observability.py`, `core/workspace/retention.py` | `tests/integration/test_mock_interview_flow.py`, `test_production_hardening.py`, `tests/unit/test_session_primitives.py` |
| 1 Retrieval contract kit | Done | `core/retrieval/{tool.py,README.md}` | `tests/contract/test_retrieval_contract.py` |
| 2 Core harness | Done | `core/runtime/{hooks,profiles,policy,compaction,tokens,structured,skills}.py`, `core/runtime/tools/`, `core/runtime/providers/openai.py`, `skills/` | `tests/unit/test_harness.py`, `tests/contract/test_provider_contract.py` |
| 3 Orchestrator & hot path | Done | `core/interview/{speaker,summary}.py`, `core/orchestration/` | `tests/integration/test_orchestration.py` |
| 4 Subagents | Done (4.5 partial transcripts not implemented; needs a backend contract) | `core/agents/` | `tests/unit/test_agents.py` |
| 5 Decision & report | Done | `core/assessor/{decision,synthesizer}.py`, `core/agents/narrative.py` | `tests/unit/test_decision.py` |
| 6 Harness & release | Done (tooling); gates needing real providers/human labels pending | `core/simulation/`, `tests/simulation/`, `tests/load/`, `docs/runbook.md` | `tests/simulation/test_simulated_candidates.py` |

Implementation notes and deviations:
- CAS uses Redis WATCH/MULTI instead of Lua (equivalent optimistic concurrency; works with fakeredis without `lupa`).
- Metrics use an in-house Prometheus text exporter (`prometheus_client` not installed); same exposition format.
- Agent profiles are configured in `config/agent_profiles.json` (JSON instead of YAML) plus `AGENT_PROFILES_OVERRIDES`.
- The orchestrator decides with one structured call per cycle instead of a tool loop; delegation fan-out is code-driven. This keeps cycle latency bounded; the `orchestrator` profile and typed tools exist for tool-loop use.
- `decision.json` is written without the artifact envelope so it stays byte-identical for identical inputs; human reviews are stored separately in `reports/review.json`.
- Release gates still requiring external resources: G4/G9 against the real provider profile (load + soak), G6 on a human-graded golden set, G10 line coverage (no coverage tool installed), provider contract run with a real key.
