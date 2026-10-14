# Core Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate real core execution in FE-test from durable, detailed execution inspection in FE-inspect.

**Architecture:** SQLite in the shared workspace volume is the authoritative archive for spans, events, state snapshots and observed agent instances. Invocation wrappers propagate parent span IDs through context variables; Redis assessment jobs carry the parent explicitly across worker tasks. The inspector reads this archive and never acquires the interview WebSocket's exclusive lease.

**Tech Stack:** Existing FastAPI/core/Redis, Python sqlite3, dependency-free HTML/CSS/JavaScript.

## Delivery tasks

- [x] Build `fe_inspect/store.py`: transactional span start/end, persistent event deduplication, state snapshots, instance observation/release, honest crash recovery, deletion tombstones and recognized credential redaction.
- [x] Replace `fe_inspect/instrumentation.py`: trace real async functions/generators, model attempts, tools, task scheduling and Redis queue parent propagation. Distinguish reused application agents, per-session speakers and per-call tasks.
- [x] Integrate at `api/main.py` assembly without overwriting current core edits. Use an unavailable provider instead of scripted fallback when real execution is required. Disable fabricated profiler recon tools in that mode.
- [x] Replace `api/inspect.py`: read archived trace/event/state, artifact details, export and pagination. Remove simulation jobs and synthetic personas. Trace history remains available after Redis expiration and process restart.
- [x] Rebuild `FE-inspect/`: actual run browser, recorded flow/tree, waterfall and node details with inputs/outputs/errors/lifetime; events/artifacts; explicit missing-history state.
- [x] Build `FE-test/`: blank input form, real candidate answer submission, persisted session resume, live core stream, inspect links, hard block missing/mock provider.
- [x] Update Dockerfile/Compose/docs for `/test/` and `/inspect/`, persistent SQLite volume and real provider configuration.
- [x] Verify persistence/redaction/parent concurrency/deletion/restart with targeted tests; Python/JS syntax and Docker build; browser inspect both UIs. Do not claim a real LLM conversation without actually configured credentials and observed provider response.

## Completion evidence

- Targeted trace/archive and existing affected runtime suites: 92 passed.
- Docker image builds with both frontends; `/test/` and `/inspect/` return 200; configured real provider and core readiness observed.
- Isolated real-provider session `sess_3147ebe2f6de`: actual browser create/start/one answer, 76 recorded spans, 9 observed agent instances, 14 HTTP provider attempts, 89 persisted events.
- After restarting the isolated app: same 76 spans and 89 events, zero missing parent IDs, all nine previous instances recorded released with process_shutdown. Waterfall and evaluator input/output/lifetime verified in browser. No sample data was seeded into the user's main volume.
- Full live interview final report was not exercised by this one-answer validation. Secret scrubbing recognizes fields/patterns; arbitrary split streaming secrets may evade pattern recognition. Studio's SQLite writer enforces one API process per workspace volume.

## Data contract

Each span has `span_id`, `parent_span_id`, `session_id`, `name`, `kind`, `agent`, `instance_id`, `lifetime_scope`, `scheduled_at`, `started_at`, `ended_at`, `duration_ms`, `status`, `inputs`, `outputs`, `attributes`, `error`, and revision sequence. No historical node, timestamp, token count or provider result is inferred from artifact counts. Prior runs without recorded spans display missing coverage explicitly.

## Acceptance

Both frontends share the configured core and persisted sessions; all run control lives in FE-test. Inspector refresh and reload read durable data, including while another browser owns the candidate socket. Graph edges follow recorded parent IDs, the waterfall follows real start/end times, and application agent reuse is visible. Missing provider credentials produce an honest unavailable state. User's existing dirty files remain intact except minimal tracing integration where required.
