# AI Mock Interview Platform Core Engine

Production-grade runtime engine for AI Mock Interview platform with high-concurrency streaming, isolated session sandboxing, pre-flight reconnaissance profiler, FSM state management, background Redis Streams assessor worker pool, and FastAPI REST / WebSocket streaming gateway.

## Features
- **Isolated Session Workspace:** Automatic path traversal defense, SHA-256 normalized workspace directories, atomic artifact writes (tempfile + fsync + atomic rename), and spillover storage for large payloads.
- **Streaming Core Runtime:** Multi-stage dialogue loop, soft schema repair, fast-path token streaming, and budget control.
- **Pre-flight Reconnaissance Profiler:** SSRF-protected web search, candidate CV parsing, and blueprint generation (`company_intel.json`, `candidate_profile.json`, `interview_spec.json`, `stage_plan.json`).
- **State & Idempotency:** 12-state lifecycle FSM with Compare-And-Set (CAS) versioning, monotonic sequence replay, and SHA-256 payload conflict detection.
- **1,500-Token Context Assembler & Stage Director:** Strict token ceiling, priority-based context pruning, multi-stage curriculum coordination.
- **Background Assessor Worker Pool:** Redis Streams at-least-once evaluation queue, non-scoring descriptive speech observation, and report barrier synthesis.
- **FastAPI REST & WebSocket Gateway:** Connection leases (`ConnectionLease`), PyJWT service and session tokens, event resumption, and concurrency semaphore.
- **100-Session Benchmark:** TTFT P95 = 225.6 ms (< 1.2s target), 0% dropped turns, 127.69 jobs/s evaluation throughput.

## Installation & Environment Setup

Using [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Sync dependencies and create virtual environment (.venv)
uv sync

# 2. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

## Running Tests

```bash
# Run all unit and integration tests (132 tests)
uv run pytest tests/ -v

# Run 100-session high-concurrency load benchmark
uv run python tests/load/stress_test_100_sessions.py --concurrency 100 --workers 5
```

## Starting the Gateway Server

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
