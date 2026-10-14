# RAG Interview Product Readiness and E2E Plan

> **For agentic workers:** Execute inline in this existing worktree and preserve all carried user changes. The user authorized implementation and E2E verification. Do not print or persist API-key values.

**Goal:** Deliver and verify a source-grounded interview flow for IT Intern through Senior that uses candidate CV/JD evidence, exposes a reviewable interview plan, runs a complete interview, and persists report artifacts.

**Architecture:** Keep `rag/` as the in-process retrieval implementation behind `core/retrieval` contracts. Keep candidate profiling, live company research, and interview-plan resolution in `core/profiler`; retain user documents and finished artifacts in the existing workspace/MinIO archive path. Company-specific assertions must carry URLs and source status; unreviewed public anecdotes must remain explicitly unverified.

**Tech Stack:** Existing FastAPI, local RAG/LSA+BM25, Brave Search, configured LLM provider, Redis, MinIO, pytest, Docker Compose.

## Acceptance

- Compose passes the configured Brave key to the running app without exposing it; source-search status is observable.
- CV and JD uploads are extracted, retained as originals, represented in the candidate profile, and used as retrieval/planning inputs.
- Supported seniorities are Intern, Fresher, Junior, Mid, Senior. Unsupported or ambiguous profiles require confirmation and never receive mislabeled questions.
- The plan records selected or inferred round/duration, confidence, evidence IDs, source URLs/status, and limitations. User edits trigger retrieval and plan regeneration before start.
- The first interviewer turn addresses the candidate by name, names the role/round/time budget, and invites an introduction. The plan covers its required stages and ends with candidate questions and a clear close.
- A complete interview reaches `REPORT_READY`; the report and source artifacts are present in the verified MinIO archive and can be inspected/restored.
- A live Compose E2E run confirms the real provider, RAG backend, Brave research, WebSocket interview, report, archive, and both frontends. Automated deterministic tests cover failure and fallback paths.
- No claim of human-validated company policy or real-world quality is made for unreviewed public anecdotes or generic rubrics.

## Work plan

### 1. Close the live-search configuration and provenance gap

Inspect `compose.yaml`, `api/main.py`, `core/profiler/tools.py`, `core/profiler/agent.py`, and `core/profiler/brief.py`. Recreate the app after confirming the Brave key is present by name only. Add or adjust tests for search-disabled, successful search/fetch, unsafe URL rejection, source deduplication, source status, duration inference, and safe low-confidence fallback. Keep search excerpts bounded and preserve source URLs and trust status in session artifacts.

### 2. Close retrieval/profile/plan gaps

Inspect `rag/adapter.py`, `core/retrieval/knowledge.py`, `core/profiler/documents.py`, `core/profiler/prompts.py`, `core/profiler/agent.py`, and `data/it_seniority_rubrics.json`. Verify role-family coverage and the five supported levels. Ensure CV and JD facts, projects, technologies, education, certificates, awards, goals, and skills are available to the profile and plan without promoting inferred facts to verified facts. Add focused tests for extraction and plan review/regeneration.

### 3. Close the interview lifecycle E2E gap

Add a deterministic integration scenario using the real RAG adapter and a scripted LLM: upload synthetic CV/JD documents, wait for profiling, inspect and edit the plan, start the interview, submit answers over WebSocket, and wait for the report. Assert the opening, required stage coverage, close, source/rubric provenance, and report artifacts. Run the existing integration, retrieval-contract, security, and full test suites.

### 4. Deploy and accept the real Compose stack

Rebuild/recreate the app with the configured Brave key. Run a synthetic-candidate live E2E session against Compose using the configured LLM and Brave services; verify live research sources and statuses, complete the session, verify the MinIO object/catalog and report retrieval, then remove only the synthetic acceptance session and verify its deletion. Confirm `/health/ready`, `/test/`, `/inspect/`, RedisInsight, and MinIO health/UI. Record commands and observed results in the handoff without exposing credentials.

## Stop condition

Stop only when all acceptance checks pass, or report a specific external dependency/quality gate that cannot be met with available evidence. A healthy container alone is not E2E acceptance. A passing synthetic E2E proves integration behavior, not human validation of interview quality.

## Execution evidence — local Compose acceptance

Verified on 2026-10-14 after rebuilding and recreating `app`:

- `uv run pytest -q`: **358 passed, 1 skipped**. Focused turn-recovery and interview-quality checks: **30 passed**.
- `docker compose config --quiet`, `git diff --check`, and `uv run python -m py_compile scripts/e2e_live_compose.py` completed successfully; `node --check FE-test/app.js` passed.
- `/health/ready` reported Redis connected, workspace writable, assessor healthy, provider configured, RAG selected, and archive initialized. `/health/live` returned `ok`; `redis-cli ping` returned `PONG`.
- `/test/`, `/inspect/`, RedisInsight, and MinIO Console returned HTTP 200; MinIO `/minio/health/ready` returned HTTP 200.
- Live Compose E2E on the final rebuilt image: **PASS** using actual Brave and LLM providers, actual multipart TXT uploads for CV/JD, and local RAG. It extracted all seven standard CV sections, retained both original uploads with matching SHA-256 metadata, produced 10 source cards with URL/trust provenance, inferred 90 minutes from framework evidence, applied a reviewed 10-minute plan, and reached the candidate Q&A stage after six scored answers. Observed stages: opening, stage_1, stage_2, stage_3, motivation_behavioral, candidate_questions.
- The opening was verified against the reviewed candidate name, role, round, duration, and introduction request. A deterministic localized opening is used when generated wording omits any required item or asks for age.
- The interview reached `REPORT_READY`; MinIO archive revision 4 restored the report, decision, profile, and both uploaded originals. The live run required zero transient-busy retries; the backend retry contract is separately covered by the focused turn-recovery test. The synthetic session was deleted after verification.

Acceptance boundary: all Brave source cards observed in the live run had trust tier `unreviewed`; this verifies retrieval and provenance plumbing, not company policy or human-validated interview accuracy. Production startup intentionally rejects `ARTIFACT_ARCHIVE_ENABLED` until durable user identity and owner authorization are configured, so these results establish local Compose/product-flow readiness, not production deployment approval.
