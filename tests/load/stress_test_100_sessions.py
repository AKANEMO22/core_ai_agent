"""High-concurrency async load simulation benchmark (100 concurrent sessions).

Validates single-node capacity:
- Spawns 100 concurrent mock candidate sessions on ASGI/WebSocket gateway
- Transmits concurrent turns with simulated VAD audio metrics
- Measures Time-To-First-Token (TTFT) distribution asserting P95 < 1.2s
- Verifies 0% dropped turns and 0 unhandled connection drops
- Verifies background assessor queue throughput under 100 concurrent sessions without deadlock or unbounded memory leak
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when executed directly as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fakeredis.aioredis
import httpx
import pytest

from api.auth import create_service_token
from api.main import create_app
from config.settings import get_settings
from core.assessor.worker import AssessorWorker
from core.runtime.providers.scripted import ScriptedInterviewLLM
from core.session.state import LifecycleStatus

logger = logging.getLogger(__name__)


class AsyncASGIWebSocket:
    """Async in-memory ASGI WebSocket client for high-concurrency simulation."""

    def __init__(self, app: Any, path: str, headers: Optional[List[Tuple[str, str]]] = None) -> None:
        self.app = app
        self.path = path
        self.headers = headers or []
        self._to_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._from_app: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._app_task: Optional[asyncio.Task[Any]] = None
        self.closed = False

    async def connect(self, timeout: float = 10.0) -> None:
        """Establish ASGI WebSocket connection."""
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": self.path,
            "raw_path": self.path.encode("ascii"),
            "query_string": b"",
            "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in self.headers],
            "client": ("127.0.0.1", 54321),
            "server": ("127.0.0.1", 80),
            "subprotocols": [],
        }

        async def receive() -> Dict[str, Any]:
            return await self._to_app.get()

        async def send(message: Dict[str, Any]) -> None:
            await self._from_app.put(message)

        self._app_task = asyncio.create_task(self.app(scope, receive, send))
        await self._to_app.put({"type": "websocket.connect"})

        msg = await asyncio.wait_for(self._from_app.get(), timeout=timeout)
        if msg.get("type") != "websocket.accept":
            raise ConnectionRefusedError(f"Expected websocket.accept, got: {msg}")

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON text frame to ASGI server."""
        await self._to_app.put({
            "type": "websocket.receive",
            "text": json.dumps(data),
        })

    async def receive_json(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Receive parsed JSON text frame from ASGI server."""
        msg = await asyncio.wait_for(self._from_app.get(), timeout=timeout)
        if msg.get("type") == "websocket.send":
            return json.loads(msg["text"])
        elif msg.get("type") == "websocket.close":
            self.closed = True
            raise ConnectionResetError(f"WebSocket closed by server with code: {msg.get('code')}")
        return msg

    async def close(self, code: int = 1000) -> None:
        """Disconnect WebSocket cleanly."""
        if not self.closed:
            self.closed = True
            await self._to_app.put({"type": "websocket.disconnect", "code": code})
            if self._app_task:
                try:
                    await asyncio.wait_for(self._app_task, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass


def calculate_percentiles(values: List[float], percentiles: List[int]) -> Dict[int, float]:
    """Calculate specified percentiles from a list of numerical values."""
    if not values:
        return {p: 0.0 for p in percentiles}
    sorted_vals = sorted(values)
    results = {}
    for p in percentiles:
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            results[p] = round(sorted_vals[int(k)], 4)
        else:
            d0 = sorted_vals[int(f)] * (c - k)
            d1 = sorted_vals[int(c)] * (k - f)
            results[p] = round(d0 + d1, 4)
    return results


async def prepare_session(
    session_index: int,
    app: Any,
    client: httpx.AsyncClient,
    service_headers: Dict[str, str],
) -> Tuple[Optional[str], Optional[AsyncASGIWebSocket], Optional[str]]:
    """Initialize session via REST endpoint and establish authenticated WebSocket stream."""
    payload = {
        "candidate_name": f"Candidate_{session_index:03d}",
        "target_role": "Backend Engineer",
        "target_company": "Scale Systems",
        "skills": ["Python", "FastAPI", "Redis"],
        "metadata": {"session_index": session_index, "expected_assessments": 1},
    }

    try:
        resp = await client.post("/api/v1/sessions", json=payload, headers=service_headers)
        if resp.status_code != 202:
            return None, None, f"Session creation failed with status {resp.status_code}"

        data = resp.json()
        session_id = data["session_id"]
        session_token = data["session_token"]

        # Wait for background profiling to reach READY
        state_mgr = app.state.state_mgr
        for _ in range(100):
            st = await state_mgr.aget(session_id)
            if st and st.status == LifecycleStatus.READY:
                break
            await asyncio.sleep(0.01)

        # Connect WebSocket
        ws = AsyncASGIWebSocket(
            app=app,
            path=f"/ws/sessions/{session_id}/stream",
            headers=[("Authorization", f"Bearer {session_token}")],
        )
        await ws.connect(timeout=10.0)

        # Start the interview (opening question, turn 0) before the measured turn
        await ws.send_json({"type": "start_interview"})
        for _ in range(50):
            frame = await ws.receive_json(timeout=10.0)
            if frame.get("type") in ("turn_complete", "error", "busy"):
                if frame.get("type") != "turn_complete":
                    return None, None, f"Opening failed: {frame.get('payload')}"
                break

        return session_id, ws, None
    except Exception as e:
        return None, None, f"Setup error for session {session_index}: {str(e)}"


async def execute_session_turn(
    session_index: int,
    session_id: str,
    ws: AsyncASGIWebSocket,
    barrier: asyncio.Barrier,
) -> Dict[str, Any]:
    """Execute concurrent candidate turn simultaneously and measure TTFT."""
    result: Dict[str, Any] = {
        "index": session_index,
        "session_id": session_id,
        "success": False,
        "ttft": None,
        "total_duration": None,
        "error": None,
        "tokens_count": 0,
    }

    # Synchronize all concurrent sessions to transmit at the exact same moment
    await barrier.wait()

    try:
        turn_start = time.perf_counter()
        await ws.send_json({
            "type": "candidate_answer",
            "message_id": f"load-{session_index}-1",
            "candidate_answer": (
                f"Candidate {session_index} answering: in distributed systems, "
                "we handle consensus via Raft and use Redis for fast atomic locking."
            ),
            "vad_metrics": {
                "speech_duration_ms": 4200,
                "pause_count": 1,
                "total_silence_ms": 350,
                "words_per_minute": 128,
            },
            "expected_turn": 1,
        })

        ttft: Optional[float] = None
        tokens_received = 0

        for _ in range(50):
            try:
                frame = await ws.receive_json(timeout=10.0)
                frame_type = frame.get("type", "")

                if frame_type == "response_token":
                    if ttft is None:
                        ttft = time.perf_counter() - turn_start
                    tokens_received += 1
                elif frame_type in ("complete", "turn_complete"):
                    break
                elif frame_type in ("error", "busy"):
                    result["error"] = f"Turn {frame_type} frame: {frame.get('payload')}"
                    break
            except (asyncio.TimeoutError, ConnectionResetError) as e:
                result["error"] = f"WebSocket receive error: {e}"
                break

        turn_duration = time.perf_counter() - turn_start
        await ws.close()

        if ttft is not None and tokens_received > 0 and result["error"] is None:
            result["success"] = True
            result["ttft"] = ttft
            result["total_duration"] = turn_duration
            result["tokens_count"] = tokens_received
        elif result["error"] is None:
            result["error"] = "No response tokens received before completion"

    except Exception as e:
        result["error"] = f"Unhandled exception: {str(e)}"
        try:
            await ws.close()
        except Exception:
            pass

    return result


async def run_benchmark(
    concurrency: int = 100,
    worker_pool_size: int = 5,
    max_ttft_p95: float = 1.2,
    workspace_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete 100-concurrent session load benchmark."""
    settings = get_settings()
    app_settings = settings.model_copy(
        update={
            "MAX_CONCURRENT_SESSIONS": concurrency + 20,
            "WORKSPACE_FSYNC": False,
        }
    )

    temp_dir_ctx = None
    if workspace_base is None:
        temp_dir_ctx = tempfile.TemporaryDirectory()
        ws_base = Path(temp_dir_ctx.name)
    else:
        ws_base = workspace_base

    fake_r = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Scripted provider: answers every structured prompt (orchestrator, evaluator v2, subagents),
    # so the control plane and evaluator fan-out run during the benchmark.
    mock_llm = ScriptedInterviewLLM(
        speaker_tokens=["Concurrency", " test", " response", " verified."],
        stage_plan={"stages": [{"stage_id": "warmup", "name": "Warmup", "target_turns": 1}]},
    )

    app = create_app(
        workspace_base=str(ws_base),
        redis_client=fake_r,
        llm_client=mock_llm,
        settings=app_settings,
    )

    service_token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    service_headers = {"Authorization": f"Bearer {service_token}"}

    # Setup assessment queue consumer group
    assessment_queue = app.state.assessment_queue
    await assessment_queue.setup_consumer_group(group_name="load_workers")

    # Start memory monitoring
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        # Phase 1: Provision sessions and establish WebSockets
        print(f"\n[Load Benchmark] Phase 1: Provisioning {concurrency} candidate sessions & establishing WebSockets...")
        prep_tasks = [
            prepare_session(
                session_index=i + 1,
                app=app,
                client=http_client,
                service_headers=service_headers,
            )
            for i in range(concurrency)
        ]
        prepared_results = await asyncio.gather(*prep_tasks)

        setup_errors = [p[2] for p in prepared_results if p[2] is not None]
        if setup_errors:
            print(f"[Load Benchmark] Warning: {len(setup_errors)} sessions encountered setup issues: {setup_errors[:3]}")

        # Phase 2: Transmit concurrent turns simultaneously
        print(f"[Load Benchmark] Phase 2: Transmitting {concurrency} candidate turns simultaneously...")
        barrier = asyncio.Barrier(concurrency)
        turn_start_phase = time.perf_counter()

        turn_tasks = [
            execute_session_turn(
                session_index=i + 1,
                session_id=prepared_results[i][0] or f"sess_{i+1}",
                ws=prepared_results[i][1],
                barrier=barrier,
            )
            for i in range(concurrency)
            if prepared_results[i][1] is not None
        ]

        results = await asyncio.gather(*turn_tasks)
        turn_phase_duration = time.perf_counter() - turn_start_phase

    # Phase 3: Process background assessment queue using worker pool
    print(f"[Load Benchmark] Phase 3: Processing assessment queue with {worker_pool_size} background workers...")
    assessor_start = time.perf_counter()

    workers = [
        AssessorWorker(
            redis_client=fake_r,
            queue=assessment_queue,
            llm_client=mock_llm,
            workspace_resolver=app.state.workspace_mgr,
            state_manager=app.state.state_mgr,
            event_store=app.state.event_store,
        )
        for _ in range(worker_pool_size)
    ]

    total_processed = 0
    worker_errors = 0

    async def worker_loop(w_idx: int, worker: AssessorWorker) -> int:
        nonlocal total_processed, worker_errors
        count = 0
        while total_processed < concurrency:
            try:
                processed = await worker.process_next_job(
                    group_name="load_workers",
                    consumer_name=f"worker_{w_idx}",
                    block_ms=50,
                )
                if processed:
                    count += 1
                    total_processed += 1
                else:
                    if total_processed >= concurrency:
                        break
                    await asyncio.sleep(0.01)
            except Exception:
                worker_errors += 1
                break
        return count

    worker_tasks = [worker_loop(idx, w) for idx, w in enumerate(workers)]
    await asyncio.wait_for(asyncio.gather(*worker_tasks), timeout=30.0)

    assessor_duration = time.perf_counter() - assessor_start
    total_benchmark_duration = turn_phase_duration + assessor_duration

    # Stop memory monitoring
    mem_after, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_delta_mb = (mem_after - mem_before) / (1024 * 1024)
    mem_peak_mb = mem_peak / (1024 * 1024)

    # Clean temporary directory if created
    if temp_dir_ctx:
        temp_dir_ctx.cleanup()

    # Calculate metrics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    ttft_values = [r["ttft"] for r in successful if r["ttft"] is not None]

    ttft_stats = calculate_percentiles(ttft_values, [50, 90, 95, 99])
    drop_rate = (len(failed) / concurrency) * 100.0
    assessor_throughput = total_processed / assessor_duration if assessor_duration > 0 else 0

    metrics = app.state.metrics
    directive_on_time = sum(
        metrics.counter_value("directive_ready_total", {"source": src, "on_time": "true"})
        for src in ("orchestrator", "fallback")
    )
    directive_total = directive_on_time + sum(
        metrics.counter_value("directive_ready_total", {"source": src, "on_time": "false"})
        for src in ("orchestrator", "fallback")
    )
    report = {
        "directive_ready_total": directive_total,
        "directive_on_time_ratio": round(directive_on_time / directive_total, 4) if directive_total else None,
        "server_ttft_p95_sec": metrics.quantile("ttft_seconds", 0.95),
        "concurrency": concurrency,
        "total_sessions": concurrency,
        "successful_sessions": len(successful),
        "failed_sessions": len(failed),
        "drop_rate_pct": drop_rate,
        "ttft_p50_sec": ttft_stats[50],
        "ttft_p90_sec": ttft_stats[90],
        "ttft_p95_sec": ttft_stats[95],
        "ttft_p99_sec": ttft_stats[99],
        "ttft_avg_sec": round(sum(ttft_values) / len(ttft_values), 4) if ttft_values else 0,
        "ttft_max_sec": round(max(ttft_values), 4) if ttft_values else 0,
        "ttft_min_sec": round(min(ttft_values), 4) if ttft_values else 0,
        "turn_phase_duration_sec": round(turn_phase_duration, 2),
        "assessor_jobs_processed": total_processed,
        "assessor_duration_sec": round(assessor_duration, 2),
        "assessor_throughput_jobs_per_sec": round(assessor_throughput, 2),
        "total_benchmark_duration_sec": round(total_benchmark_duration, 2),
        "memory_delta_mb": round(mem_delta_mb, 2),
        "memory_peak_mb": round(mem_peak_mb, 2),
        "failures": [f["error"] for f in failed[:5]],
    }

    # Print summary
    print("\n" + "=" * 70)
    print("           HIGH-CONCURRENCY LOAD BENCHMARK RESULTS")
    print("=" * 70)
    print(f" Total Sessions:            {report['total_sessions']}")
    print(f" Successful Turns:          {report['successful_sessions']} / {report['total_sessions']} ({100 - drop_rate:.1f}%)")
    print(f" Dropped Turns:             {report['failed_sessions']} ({drop_rate:.1f}%)")
    print("-" * 70)
    print(f" TTFT Min:                  {report['ttft_min_sec'] * 1000:.1f} ms")
    print(f" TTFT Avg:                  {report['ttft_avg_sec'] * 1000:.1f} ms")
    print(f" TTFT P50:                  {report['ttft_p50_sec'] * 1000:.1f} ms")
    print(f" TTFT P90:                  {report['ttft_p90_sec'] * 1000:.1f} ms")
    print(f" TTFT P95:                  {report['ttft_p95_sec'] * 1000:.1f} ms  (Target: < {max_ttft_p95 * 1000:.0f} ms)")
    print(f" TTFT P99:                  {report['ttft_p99_sec'] * 1000:.1f} ms")
    print(f" TTFT Max:                  {report['ttft_max_sec'] * 1000:.1f} ms")
    print("-" * 70)
    print(f" Turn Streaming Phase:      {report['turn_phase_duration_sec']} s")
    print(f" Assessor Queue Drained:    {report['assessor_jobs_processed']} jobs in {report['assessor_duration_sec']} s")
    print(f" Assessor Throughput:       {report['assessor_throughput_jobs_per_sec']} jobs/s")
    print(f" Directives (on time):      {report['directive_ready_total']} ({report['directive_on_time_ratio']})")
    print(f" Memory Delta:              +{report['memory_delta_mb']} MB (Peak: {report['memory_peak_mb']} MB)")
    print(f" Total Benchmark Time:      {report['total_benchmark_duration_sec']} s")
    print("=" * 70 + "\n")

    return report


@pytest.mark.asyncio
async def test_100_concurrent_sessions_load_benchmark(tmp_path: Path):
    """Pytest integration test executing 100 concurrent session load benchmark."""
    report = await run_benchmark(
        concurrency=100,
        worker_pool_size=5,
        max_ttft_p95=1.2,
        workspace_base=tmp_path / "load_workspaces",
    )

    # Core performance assertions
    assert report["successful_sessions"] == 100, f"Expected 100 successful sessions, got {report['successful_sessions']}"
    assert report["drop_rate_pct"] == 0.0, f"Expected 0% drop rate, got {report['drop_rate_pct']}%"
    assert report["ttft_p95_sec"] < 1.2, f"P95 TTFT exceeded 1.2s threshold: {report['ttft_p95_sec']}s"
    assert report["assessor_jobs_processed"] == 100, f"Expected 100 assessment jobs processed, got {report['assessor_jobs_processed']}"
    assert report["memory_delta_mb"] < 200.0, f"Unbounded memory growth detected: +{report['memory_delta_mb']} MB"


def main() -> None:
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(description="Run 100-session load benchmark")
    parser.add_argument("--concurrency", type=int, default=100, help="Number of concurrent sessions (default 100)")
    parser.add_argument("--workers", type=int, default=5, help="Number of assessor workers in pool (default 5)")
    parser.add_argument("--max-ttft-p95", type=float, default=1.2, help="P95 TTFT threshold in seconds (default 1.2)")
    args = parser.parse_args()

    report = asyncio.run(
        run_benchmark(
            concurrency=args.concurrency,
            worker_pool_size=args.workers,
            max_ttft_p95=args.max_ttft_p95,
        )
    )

    if report["drop_rate_pct"] > 0:
        print(f"FAILED: Dropped turns detected: {report['drop_rate_pct']}%")
        sys.exit(1)
    if report["ttft_p95_sec"] >= args.max_ttft_p95:
        print(f"FAILED: P95 TTFT exceeded threshold: {report['ttft_p95_sec']}s >= {args.max_ttft_p95}s")
        sys.exit(1)

    print("SUCCESS: All load benchmark assertions satisfied!")
    sys.exit(0)


if __name__ == "__main__":
    main()
