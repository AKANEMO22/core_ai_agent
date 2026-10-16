# RAG Review Remediation Implementation Plan

**Goal:** Correct the seven reproduced review findings and re-accept the complete CV/JD-to-interview flow without treating authored prompts or web anecdotes as verified company evidence.

**Architecture:** Keep `core/retrieval` as the port and `rag/` as the in-process adapter. Validate question metadata before filtering; add explicitly project-authored role/level scaffolds where source evidence is absent. Keep research inference deterministic, move confirmation into the shared controller, and preserve CV/JD evidence IDs through the generated plan and interviewer artifacts.

**Execution:** User authorized the reviewed fixes. Execute in the current shared checkout, preserve carried FE/Compose changes, do not print secrets, do not commit unrelated work. Independent file-owned lanes may run in parallel. Each lane records a failing regression before implementation; spec and quality reviews follow integration.

## Tasks and acceptance

- [ ] Question mapping and capability: `rag/adapter.py`, `rag/indexing.py`, `rag/ontology.py`, `core/retrieval/knowledge.py`, new focused role-level scaffolds and retrieval regression tests. Hash-table/recursion questions must not become Communication by query assignment; technical questions must not inherit warmup from a shared stage list. Every known framework title resolves across five advertised levels. Missing source questions use actual authored, calibrated probes with explicit provenance, never relabeled junior source evidence. A 25-role x 5-level acceptance matrix checks nonempty suitable rubric/questions and calibration metadata.
- [ ] Source and duration inference: `core/profiler/brief.py` plus focused helpers/tests. Search query text cannot establish source relevance; unrelated pages cannot set round/duration. Duration claims must match selected round, recognize hour/minute ranges, reject unrelated numbers, and allocate stage minutes to the exact plan total. Test wrong-company pages, mixed HR/coding durations, 1-2 hours, conflicts, and generic framework fallback.
- [ ] Shared confirmation and runtime budgets: `core/interview/controller.py`, `director.py`, `runner.py`, `api/routes.py`, `api/websocket.py`, `api/schemas.py`, focused lifecycle tests. REST, WS and direct controller entry must enforce the same needs-confirmation guard. Explicit confirmation is bound to the current brief. Record start/deadline and reserve candidate Q&A near elapsed budget without delaying fast sessions to fill time or interrupting an accepted answer; test clock-controlled transitions and retry/replay behavior.
- [ ] CV/JD evidence probes: new `core/profiler/evidence.py`, `core/profiler/agent.py`, `core/interview/speaker.py`, focused tests. Build stable evidence/probe IDs from substantive CV fields and JD requirements, attach traceable probes after generic stage grounding, preserve original evidence, choose a bounded plan based on duration, and expose asked/unasked evidence coverage from real question IDs. Missing/uncertain claims remain marked unverified. Preserve reviewed profile corrections.
- [ ] Integration acceptance: targeted regressions, role-level matrix, full suite, diff/config/syntax checks, rebuild/recreate current Compose app, live synthetic upload/review/WebSocket/report/MinIO restore and cleanup. Strengthen `scripts/e2e_live_compose.py` to compare extracted CV facts, selected probes and source/criterion consistency instead of only nonempty fields. No claims of expert validation or verified company policy from unreviewed data.
- [ ] Independent spec-compliance review, then code-quality review. Resolve findings, rerun affected checks, record exact evidence and remaining external quality gates.

## Validation commands

`uv run pytest tests/integration/test_rag_retrieval_integration.py tests/unit/test_interview_brief.py tests/unit/test_interview_quality.py tests/integration/test_production_hardening.py -q`

`uv run pytest -q`

`docker compose config --quiet`; `git diff --check`; `node --check FE-test/app.js`; `uv run python -m py_compile scripts/e2e_live_compose.py`

`docker compose build app`; `docker compose up -d --force-recreate app`

`Get-Content scripts/e2e_live_compose.py -Raw | docker compose exec -T app python -`

## Stop condition

The reproduced semantic defects have negative regressions and passing fixes, supported role/level probes have truthful provenance, both transports enforce confirmation, CV/JD question coverage is inspectable, and the final built runtime passes the live synthetic lifecycle and artifact restore. Report any unmet human/external evidence gate explicitly.
