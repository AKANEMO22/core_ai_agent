# End session and final report implementation plan

**Goal:** Add an authenticated API and FE action that stop interview work for one session and immediately start a final summary/report using saved evidence.

**Architecture:** A durable `ENDING` state and stop fence reject new work and late assessment writes. Session-scoped cancellation stops local model and agent tasks without terminating shared assessor consumers. The existing report synthesizer gains an explicit early-end path, retaining the normal assessment barrier for normal completion. Early-end reports disclose incomplete evidence and require review.

**Tech stack:** FastAPI, asyncio, Redis, existing report narrative and summary agents, vanilla JavaScript FE.

User authorized autonomous implementation. Preserve pre-existing FE and Compose changes; do not commit or deploy as part of this request.

## Backend

- [ ] Write and run failing integration tests for early end, pending assessment, active generation, repeated/concurrent requests and access control.
- [ ] Add lifecycle transition, durable work fence and cancellation coverage for profiling, speaker and evaluator tasks.
- [ ] Implement authenticated `POST /api/v1/sessions/{session_id}/end`, with idempotent report generation and recovery after interruption.
- [ ] Include saved transcript in final summary; disclose missing assessments and preserve normal report semantics.
- [ ] Run targeted integration and lifecycle tests, then repository test suite.

## Frontend

- [ ] Add “Kết thúc và tạo báo cáo” beside the existing cancellation action, with Vietnamese and English copy.
- [ ] Block submissions and retries during ending, protect async callbacks against session selection changes, and open the report when ready.
- [ ] Verify JavaScript syntax and the button/request/report flow.

## Integration

- [ ] Review stop fencing, shared worker survival and incomplete evidence behavior.
- [ ] Document API response semantics and local validation limits.
- [ ] Check the final diff for whitespace errors and unrelated changes.
