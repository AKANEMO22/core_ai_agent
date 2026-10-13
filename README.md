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

The default backend is `rag`. On first startup it indexes only
`data/interview_frameworks.json` into `data/rag/index.db`; Compose stores that generated index in
the `rag_index` volume. To use the deterministic fixture backend, set `RETRIEVAL_BACKEND=mock`.

## Running

```bash
# Development (no API key: scripted provider + in-process fake Redis when REDIS_URL is empty)
uv run uvicorn api.main:build_default_app --factory --reload

# Production
APP_ENV=production uv run uvicorn api.main:build_default_app --factory --host 0.0.0.0 --port 8000 --no-access-log
```

## Protocol summary

REST (`/api/v1/sessions`): create (202, `Idempotency-Key`), status, `POST /{id}/start`, report,
cancel, delete, `POST /{id}/assessments/requeue`, `POST /{id}/review`.

WebSocket (`/ws/sessions/{id}/stream`), inbound: `start_interview`, `candidate_answer`
(`message_id` required), `cancel_generation`, `replay`, `ack`, `ping`. Outbound events include
`interview_started`, `response_token`, `turn_complete`, `stage_transition`, `generation_cancelled`,
`busy`, `resync_required`, `directive_ready`, `assessment_update`, `decision_ready`, `report_ready`.
