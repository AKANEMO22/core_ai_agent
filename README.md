# AI Mock Interview Platform Core Engine

Standalone AI runtime for realtime mock interviews. The backend handles STT/TTS; this service
prepares the interview (retrieval + profiling), runs the interviewer in realtime, orchestrates
evaluation and follow-up in the background, and produces an auditable report with a verdict.

## Architecture at a glance

- **Hot path (speaker):** one tool-free LLM call per turn with a cache-friendly prompt
  (persona + stage playbook + rolling summary + recent turns + directive). Deadlines, barge-in,
  backpressure (`busy`), exactly-once answers by `message_id`.
- **Control plane (orchestrator):** after each answer, delegates in parallel (evaluation wait,
  follow-up planner, claim checker, input guard), rebuilds criterion coverage, renders its
  notebook, and writes the directive for the next turn, gated by coverage and stage budget.
- **Assessment:** Redis Streams worker pool in the app lifecycle; per-criterion evaluator with
  rubric anchors and verified evidence quotes; barrier → deterministic decision
  (`PASS` / `FAIL` / `NEEDS_REVIEW`) → narrative with verified citations.
- **Retrieval:** the profiler and control-plane agents share the offline-capable RAG backend via
  `core/retrieval/`; the deterministic mock remains available by explicit configuration (see
  `core/retrieval/README.md`).
- **Harness:** hook bus, agent profiles with tool allowlists, typed workspace tools, auto-compaction,
  structured output with repair, skill registry (`skills/`).
- **Durability:** Redis-backed state (CAS), events (monotonic, TTL), idempotency, message ledger,
  leases; restart-safe.

Plans and design: `docs/superpowers/specs/`, `docs/superpowers/plans/`. Operations: `docs/runbook.md`.

## Setup

```bash
uv sync
```

## Tests

```bash
uv run pytest                                   # unit, integration, contract, simulation
RUN_PROVIDER_CONTRACT=1 uv run pytest tests/contract/test_provider_contract.py   # uses LLM_* from .env
RUN_LIVE_LLM=1 uv run python -m core.simulation.live --persona strong --model <model>  # live interview
RETRIEVAL_BACKEND=<name> uv run pytest tests/contract/test_retrieval_contract.py
uv run python tests/load/stress_test_100_sessions.py --concurrency 100 --workers 5
uv run python -m core.simulation.quality --golden golden.jsonl --decisions-dir data/workspaces
```

The default backend is `rag`. On first startup it indexes `data/interview_frameworks.json` into
`data/rag/index.db`; Compose stores that generated index in the `rag_index` volume. A separate
`data/it_seniority_rubrics.json` supplies generic Mid/Senior competency anchors. Intern, Fresher and
Junior questions come from the role corpus; Mid/Senior questions are generated from the candidate
profile, JD and generic rubric because the corpus has no reviewed question bank for those levels.
The rubric and generated questions are not company policy or human-validated interview material.
To use the deterministic fixture backend, set `RETRIEVAL_BACKEND=mock`.

## Interview preparation inputs and evidence

FE-test accepts CV and job-description files in PDF, DOCX or TXT format, with a 12 MiB limit per
file. Text PDFs, DOCX text and UTF-8 text are extracted; scanned PDFs are rejected because OCR is
not configured. Candidates can review and correct the extracted profile and interview plan before
starting; a correction refreshes retrieval and the stage plan. Company-specific web research requires `DOCKER_BRAVE_SEARCH_API_KEY`; without it, the
plan shows that company research is unavailable. Search results and fetched page excerpts remain
unreviewed leads unless explicitly verified; the interview must not present them as official policy.
The planned duration shapes the stage and turn budget; it is an estimate, not a wall-clock cutoff.

## Running

```bash
# Development (no API key: scripted provider + in-process fake Redis when REDIS_URL is empty)
uv run uvicorn api.main:build_default_app --factory --reload

# Production
APP_ENV=production uv run uvicorn api.main:build_default_app --factory --host 0.0.0.0 --port 8000 --no-access-log
```

### Durable storage: SQLite (default) or PostgreSQL

The artifact archive catalog and the inspector trace store are SQLite files by default
(`ARTIFACT_ARCHIVE_CATALOG_PATH`, `<WORKSPACE_ROOT_BASE>/.inspect/traces.sqlite3`). Set
`DATABASE_URL=postgresql://user:password@host:5432/db` to keep both in PostgreSQL instead
(schemas `artifact_archive` and `inspect_trace`, created idempotently at startup and versioned in
each schema's `schema_version` table). Pool size: `DATABASE_POOL_MIN_SIZE` / `DATABASE_POOL_MAX_SIZE`
(per store). URL-encode special characters in the password; the URL is never logged. The RAG
index/cache stays in SQLite (it is derived, per-container data). Copy existing SQLite rows once
with the API stopped (idempotent, `ON CONFLICT DO NOTHING`):

```bash
DATABASE_URL=postgresql://... uv run python scripts/migrate_sqlite_to_postgres.py   --catalog data/archive/catalog.sqlite3 --traces data/workspaces/.inspect/traces.sqlite3
```

PostgreSQL store tests run only when `TEST_DATABASE_URL` is set:

```bash
docker run -d --rm --name dpl-pg-test -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:17-alpine
TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres uv run pytest tests/integration/test_postgres_stores.py
docker stop dpl-pg-test
```

## Core Studio: FE-test and FE-inspect

Two separate frontends share the core API and persisted workspaces:

- `FE-test/` at `/test/` creates real interview sessions from your input, sends your answers,
  resumes saved conversations and links to the matching inspection view.
- `FE-inspect/` at `/inspect/` is read-only: recorded agent call tree/flow, waterfall, input,
  output, usage, errors, events, artifacts and observed agent instance lifetimes.

FE-test provides **Kết thúc và tạo báo cáo** for ending a session early. It calls
`POST /api/v1/sessions/{session_id}/end` with the existing session or service bearer token.
The core stops session work, rejects further turns, and starts final summary/report generation
from saved evidence without waiting for unfinished assessments. The early-end report records
incomplete evidence and uses `NEEDS_REVIEW`; it does not score answers that were never evaluated.
The FE opens the saved report when it becomes ready. The separate **Hủy phiên** action cancels
the session without requesting a report.

```bash
docker compose up --build -d
# Test:    http://127.0.0.1:8001/test/
# Inspect: http://127.0.0.1:8001/inspect/
# Redis UI: http://127.0.0.1:5540/
```

Compose also runs `postgres` (postgres:17-alpine, `postgres_data` volume, bound to
`127.0.0.1:${DPL_POSTGRES_PORT:-5432}`) and points the app's `DATABASE_URL` at it, so the archive
catalog and inspector traces live there. Override `POSTGRES_USER`, `POSTGRES_PASSWORD` and
`POSTGRES_DB` in `.env`; the defaults are local development credentials only.

Redis Insight is preconfigured with the `dpl-pvan` connection to `redis:6379`, database 0.
On first use, complete Redis Insight's welcome/terms screen to enable connection discovery.
It binds to loopback only and persists its settings in the `redisinsight_data` volume.
To start only the Redis UI and its Redis dependency, run `docker compose up -d redisinsight`.

The local stack also builds MinIO Community from its pinned upstream source release and archives
completed session workspaces with checksums. Open the MinIO object browser at
`http://127.0.0.1:9001/`; the S3 endpoint is `http://127.0.0.1:9000/`. For local use, the
Compose defaults work as-is. Set `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` in `.env` to use
your own credentials. These are local development credentials; production object storage is
blocked until a maintained MinIO distribution and durable user ownership/authentication are
configured. The MinIO Community upstream was archived in April 2026, so its local source build
does not receive upstream security fixes.

Archived objects and the SQLite archive catalog have separate named volumes (`minio_data` and
`artifact_catalog`). A named volume is not a backup. Back up and restore both the MinIO data and
catalog together before relying on the archive for long-term retention. Session deletion removes
all stored object versions and records a durable tombstone before local/Redis cleanup.

Compose binds to loopback and enables short-lived development tokens. Both screens connect
automatically to the same origin. Advanced connection settings allow a different core with a
service token. Production requires existing service auth; local token issuing is rejected.

The interface defaults to Vietnamese and supports English through the language selector.
This browser preference is independent of **interview language**: choose `vi` or `en` when
creating a session. REST clients send `interview_language` in `POST /api/v1/sessions`;
omitting it defaults to `vi`. The selection is saved in session metadata and
`inputs/form_input.json`, then used as language guidance for profiling, speaker questions,
follow-ups and orchestration. Source documents, transcript text, identifiers and stored
artifacts remain as recorded; switching the interface does not translate historical runs.
Model compliance with the prompt remains subject to the configured provider.

Desktop views stay inside the viewport, with independent scrolling for the run list,
trace/conversation and detail panels. The waterfall can zoom its time scale and scroll
horizontally while keeping span names visible. Artifact and structured-data previews
support readable JSON/Markdown and source views; raw HTML is displayed as text and unsafe
link protocols are blocked. The Markdown viewer supports common document syntax and is
not a full CommonMark implementation.

Local development tokens expire after one hour. Both frontends refresh them before expiry
and retry an unauthorized HTTP request once when local issuing is enabled. Production
service tokens must be supplied by your authentication system.

Set `DOCKER_LLM_API_KEY`, `DOCKER_LLM_API_URL` and `DOCKER_LLM_MODEL` in `.env` for your real
provider (see `.env.docker.example`). Without a provider, the screens remain accessible and
show execution unavailable; no scripted LLM fallback, synthetic candidate or demo run is
used. Core readiness returns 503 until a real provider is configured. Compose uses the real
local `rag` adapter; fabricated recon tools are disabled in strict Studio mode.

`INSPECTOR_ENABLED=true`, `TEST_UI_ENABLED=true`, `INSPECTOR_LOCAL_AUTH=true` enable the
screens outside Compose. `STUDIO_REQUIRE_REAL_PROVIDER=true` is the default for Studio.

SQLite at `/data/workspaces/.inspect/traces.sqlite3` uses WAL and `synchronous=FULL` in the
`interview_workspaces` volume. Each span start, task scheduling, completion, state snapshot
and event is committed. Redis still owns the core's live CAS/queue protocol, with AOF and
its own volume; the inspector archive remains readable after Redis expiry or app restart.
Do not use `docker compose down -v` if you want to retain history. Inspector reads its own
API; it never acquires the candidate WebSocket lease. Parent IDs are propagated through
Redis assessment jobs, so worker calls link to their actual enqueue span.

Agent instance observations distinguish application-scoped reused agents, session-scoped
speaker objects, and per-delegation tasks. Missing completion after process restart is
marked `abandoned` with unknown end time. Old runs have no reconstructed/fabricated graph:
surviving events/artifacts can be viewed, with missing trace coverage explicitly indicated.
The current core/SQLite integration enforces a single API process per workspace volume
with an OS writer lock, as configured by Compose. A second live writer fails at startup;
restart recovery cannot incorrectly abandon the first process's running agents.

Capture includes actual prompts, model output, tool arguments/results and usage under service
authentication. Credential fields and recognized credential strings are redacted; this is
pattern filtering, so arbitrary secrets split across streamed chunks may evade it.
Reasoning fields are omitted. Large previews are bounded
with explicit truncation markers; the workspace preserves the core's original artifacts.

Backend extensions can use the same persistence and parent context:

```python
recorder = app.state.inspect_trace_recorder

@recorder.traceable(name="Backend preparation", kind="operation")
async def prepare(session_id: str, request_data: dict):
    return await existing_backend_operation(request_data)
```

GET `/api/v1/inspect/runs/{session_id}/trace` reads a coherent graph snapshot and revision.
GET `/api/v1/inspect/runs/{session_id}/events?after_sequence=0&limit=1000` pages durable events.
GET `/api/v1/inspect/runs/{session_id}/export` exports the inspected snapshot. All require
a service token. Automatic transcript/scripted simulation endpoints have been removed.

## Protocol summary

REST (`/api/v1/sessions`): create (202, `Idempotency-Key`), status, `POST /{id}/start`, report,
cancel, delete, `POST /{id}/assessments/requeue`, `POST /{id}/review`.

WebSocket (`/ws/sessions/{id}/stream`), inbound: `start_interview`, `candidate_answer`
(`message_id` required), `cancel_generation`, `replay`, `ack`, `ping`. Outbound events include
`interview_started`, `response_token`, `turn_complete`, `stage_transition`, `generation_cancelled`,
`busy`, `resync_required`, `directive_ready`, `assessment_update`, `decision_ready`, `report_ready`.
