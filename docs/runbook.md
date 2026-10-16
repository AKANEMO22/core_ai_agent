# AI Interview Runtime: Operations Runbook

## 1. Configuration reference

All settings are environment variables (or `.env`), defined in `config/settings.py`. Production
startup fails (`RuntimeError: Unsafe production configuration`) unless:

| Setting | Production requirement |
|---|---|
| `APP_ENV` | `production` |
| `JWT_SECRET` | non-default, ≥ 32 bytes |
| `CORS_ALLOWED_ORIGINS` | explicit list (JSON array), no `*` |
| `LLM_API_KEY` (or `OPENAI_API_KEY`) | set (provider capability check runs at startup); `LLM_MODEL` and `LLM_API_URL` select an OpenAI-compatible endpoint. The speaker needs an endpoint that streams tokens incrementally and a low-latency, low-reasoning model (see §7) |
| `RETRIEVAL_BACKEND` | a registered production adapter, or `mock` with `ALLOW_MOCK_RETRIEVAL=true` |
| `PRODUCT_MODE` | `practice` or `hiring_assist` |
| `REDIS_URL` | reachable Redis with persistence (AOF recommended) on an encrypted volume |
| `DATABASE_URL` | optional; `postgresql://` URL for the archive catalog and inspector traces (empty = SQLite files). Secret: never logged, shown redacted (`user@host:port/db`) by `/api/v1/inspect/system` |

Key tuning knobs:

| Area | Settings |
|---|---|
| Realtime | `MAX_INTERACTIVE_GENERATIONS`, `MAX_INTERACTIVE_QUEUE_MS`, `TURN_DEADLINE_SECONDS`, `SPEAKER_HEDGE_AFTER_MS` (second speaker request if no first token by then; 0 disables), `BACKGROUND_YIELD_MAX_MS`, speaker profile `turn_deadline_ms` / `context_budget_tokens` (`config/agent_profiles.json` or `AGENT_PROFILES_OVERRIDES`) |
| Control plane | `ORCHESTRATOR_ENABLED`, `ORCHESTRATOR_DEADLINE_MS`, `ORCHESTRATOR_JOIN_MS`, `COVERAGE_MIN_CONFIDENCE`, `STAGE_MAX_EXTRA_TURNS` (extra turns a stage may take while a must-have lacks evidence or is below its pass level), `ROLLING_SUMMARY_ENABLED`, `BRIEF_ENABLED` |
| Assessment | `ASSESSOR_POOL_SIZE`, `ASSESSOR_LEASE_MS`, `MAX_ASSESSOR_LLM_CALLS`, `REPORT_SWEEP_INTERVAL_SECONDS` |
| Decision | `DECISION_BORDERLINE_BAND` (also routes a must-have within the band below its pass level to NEEDS_REVIEW instead of FAIL), `COVERAGE_MIN_CONFIDENCE`, `PRODUCT_MODE` |
| Rate limits | `ANSWER_RATE_PER_SECOND`, `ANSWER_BURST`, `RECONNECT_RATE_PER_MINUTE`, `SESSION_CREATE_*` |
| Retention | `EVENT_TTL_SECONDS`, `SESSION_STATE_TTL_SECONDS`, `RAW_UPLOAD_RETENTION_HOURS`, `SESSION_ARTIFACT_RETENTION_DAYS` |

Agent profiles are validated at startup; any production profile that enables `bash`,
`read_file`, `write_file` or `edit_file`, or gives the speaker any tool, is refused.

## 2. Running

```bash
uv sync
# Production (one process per node; Redis reachable via REDIS_URL)
APP_ENV=production uv run uvicorn api.main:build_default_app --factory --host 0.0.0.0 --port 8000 --no-access-log
```

Use `--no-access-log` (or a proxy that strips paths): HTTP access logs contain session ids in URLs.
Application logs never contain candidate text, CV text, tokens or raw session ids
(`core/observability.py` adds a redaction filter as defense in depth).

Set `BUILD_SHA` so every `decision.json` records the deployed code version.

## 3. Health and metrics

| Endpoint | Meaning |
|---|---|
| `GET /health/live` | process is up |
| `GET /health/ready` | Redis reachable, workspace writable, assessor heartbeat fresh, provider configured; 503 otherwise |
| `GET /metrics` | Prometheus text format; requires a service token |

Key metrics (prefix `ai_interview_`):

| Metric | Use |
|---|---|
| `ttft_seconds` (histogram) | realtime SLO: accepted answer (incl. queue wait) → first token |
| `turn_duration_seconds` | full interviewer turn |
| `busy_total` | capacity or rate-limit refusals |
| `provider_errors_total{agent}` | LLM failures by agent profile |
| `assessment_pending` | jobs waiting in the stream |
| `directive_ready_total{source,on_time}` | control-plane health: share of directives ready before the next answer |
| `orchestrator_cycle_seconds` | control-plane latency |
| `degraded_total{component}` | fallbacks (orchestrator, delegations) |
| `compactions_total{agent}` | context compaction activity |
| `llm_tokens_total{agent,kind}` | cost tracking |
| `active_connections` | WebSocket leases |

## 4. Alerts (suggested thresholds)

| Alert | Condition | First action |
|---|---|---|
| TTFT regression | P95 `ttft_seconds` > 1.2 s for 10 min | Check provider latency and `busy_total`; raise `MAX_INTERACTIVE_GENERATIONS` only within provider quota |
| Busy rate | `busy_total` / answers > 1 % | Capacity or quota issue; scale or tune semaphores |
| Dead letters | any increase of dead-letter stream length | §5.1 |
| Assessment lag | `assessment_pending` growing for 15 min | Check `/health/ready` assessor heartbeat and provider errors |
| Heartbeat missing | `/health/ready` reports `stale_heartbeat` | Restart the process; pending jobs are reclaimed via XAUTOCLAIM |
| Control plane degraded | on-time directive ratio < 95 % or `degraded_total{component="orchestrator"}` rising | Check orchestrator latency; the interview continues on fallback directives |

## 5. Procedures

### 5.1 Requeue failed assessments
A session in `ASSESSMENT_FAILED` has jobs in `<ASSESSMENT_STREAM_KEY>:dead_letter`. After fixing the cause:

```bash
curl -X POST -H "Authorization: Bearer $SERVICE_TOKEN" \
  https://runtime/api/v1/sessions/<session_id>/assessments/requeue
```

Idempotent: requeued jobs leave the dead-letter stream; the session returns to `ASSESSING` and the
report is produced automatically when the barrier is met.

### 5.2 Stuck sessions
The pool's sweeper (`REPORT_SWEEP_INTERVAL_SECONDS`) advances sessions stuck in
`DIALOGUE_COMPLETED` or `ASSESSING`. No manual action is normally needed; restarting a process is safe.

### 5.3 Client resync
Clients that reconnect with an expired cursor receive `resync_required` with a snapshot
(status, turn, stage, last interviewer text) and should re-render from it.

### 5.4 Deletion (retention or user request)
`DELETE /api/v1/sessions/{id}` cancels running work and waits for it, closes the connection,
revokes tokens, deletes every Redis key and the workspace, and appends a content-free entry to
`audit:session_deletions`.

### 5.5 Durable stores on PostgreSQL
With `DATABASE_URL` set, the archive catalog (schema `artifact_archive`) and inspector traces
(schema `inspect_trace`) use PostgreSQL through a psycopg pool per store
(`DATABASE_POOL_MIN_SIZE`/`DATABASE_POOL_MAX_SIZE`). Schemas are created idempotently at startup
and recorded in `<schema>.schema_version`; the role needs `CREATE` on the database for first start.
Writers are serialized with transaction advisory locks (traces: one per store, like SQLite
`BEGIN IMMEDIATE`; catalog: one per session). One API process owns a trace schema at a time
(session advisory lock on a dedicated connection; a second process fails at startup with
"already has an active core process"). Docker Compose ships a `postgres` service for this.
To move existing SQLite data, stop the API and run
`uv run python scripts/migrate_sqlite_to_postgres.py` (idempotent; `--catalog`/`--traces` paths).
Back up with `pg_dump -n artifact_archive -n inspect_trace`.

### 5.6 Human review (`PRODUCT_MODE=hiring_assist`)
When `requires_human_review` is true, a reviewer records the final verdict:

```bash
curl -X POST -H "Authorization: Bearer $SERVICE_TOKEN" -H "Content-Type: application/json" \
  -d '{"reviewer_id": "hr-42", "final_verdict": "PASS", "notes": "..."}' \
  https://runtime/api/v1/sessions/<session_id>/review
```

The computed `decision.json` is never modified; the review is stored next to it.

## 6. Release checklist

- [ ] `uv run pytest` green (unit, integration, contract, simulation).
- [ ] Provider contract with the production model: `RUN_PROVIDER_CONTRACT=1 OPENAI_API_KEY=... uv run pytest tests/contract/test_provider_contract.py`.
- [ ] Retrieval contract with the production RAG: `RETRIEVAL_BACKEND=<name> uv run pytest tests/contract/test_retrieval_contract.py`.
- [ ] Quality gates on the golden set: `uv run python -m core.simulation.quality --golden golden.jsonl --decisions-dir <workspaces>` exits 0 (κ ≥ 0.6, criterion MAE ≤ 0.5, must-have coverage ≥ 95 %).
- [ ] Load benchmark against the staging provider profile: `uv run python tests/load/stress_test_100_sessions.py --concurrency 100 --workers 5` meets the TTFT and directive on-time targets.
- [ ] 2-hour soak with zero unresolved jobs and zero unexpected dead letters.
- [ ] Playbooks, templates and rubric packs frozen: record `SkillRegistry().list()` refs in the release notes.
- [ ] `BUILD_SHA` set; production config validation passes on startup.

## 7. Provider requirements for realtime

Measured on 2026-10-11 against the OpenAI-compatible gateway configured in `.env`:

| Observation | Impact |
|---|---|
| Responses arrive in 1-2 chunks (the gateway buffers instead of streaming tokens) | TTFT equals full generation time; streaming to TTS cannot start early |
| `gemini-3.8-flash-high` spends ~250-500 reasoning tokens even on trivial prompts; TTFT 3.8-7 s | Speaker misses the 1.2 s P95 target by 3-6x |
| Best TTFT across fast models on this gateway: ~2.6 s (`gemini-3.5-flash-lite`) | Still above target because of buffering |
| Several routes return 5xx/400 intermittently | Adapter retries once before streaming; persistent failures surface as `TURN_FAILED` (retryable) |

Requirements for production:
- An endpoint that streams tokens incrementally (SSE deltas), measured TTFT P95 < 0.8 s for the speaker prompt.
- A low-latency, non-reasoning (or minimal-reasoning) model for the `speaker` profile; reasoning models are acceptable for `evaluator`, `orchestrator` and `narrative`, whose deadlines are longer. Set per-profile models in `config/agent_profiles.json`.
- An official provider API under a data-processing agreement: candidate transcripts and CV-derived data are personal data.
- Validate with `RUN_PROVIDER_CONTRACT=1 uv run pytest tests/contract/test_provider_contract.py` and `RUN_LIVE_LLM=1 uv run python -m core.simulation.live --persona strong`.
