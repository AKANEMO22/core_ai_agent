"""End-to-end interview lifecycle through the real application, with no manual wiring.

Covers gate G1: session creation → profiling → start_interview → answers over WebSocket →
dialogue completion → assessor pool (started by the app lifespan) → barrier → report,
plus exactly-once answers, retry after a failed generation, and log hygiene.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from api.auth import create_service_token
from api.main import create_app
from config.settings import Settings, get_settings
from tests.fakes import ScriptedInterviewLLM
from core.runtime.types import IterationChunk
from core.session.state import LifecycleStatus

CV_SECRET_LINE = "Proprietary ledger reconciliation engine at Contoso Bank"

STAGE_PLAN = {
    "stages": [
        {"stage_id": "warmup", "name": "Warmup", "target_turns": 1},
        {"stage_id": "technical", "name": "Technical Depth", "target_turns": 1},
        {"stage_id": "system_design", "name": "System Design", "target_turns": 1},
        {"stage_id": "qa_close", "name": "Q&A and Close", "target_turns": 1},
    ]
}

ANSWERS = [
    "Hello! I am a senior engineer with six years in distributed services.",
    "We use Redis Streams for at-least-once messaging and distributed locks for idempotency.",
    "For high throughput we partition streams, use read replicas and enforce TTL cache tiers.",
    "I reduce production risk with staged releases, latency alerts, rollback criteria and a post-incident review.",
    "I owned the service migration, compared consistency and availability trade-offs, and validated it with load tests.",
    "How does the team approach observability and distributed tracing?",
]


class RecordingMockLLM(ScriptedInterviewLLM):
    """Scripted provider that records every final user message and can fail selected answers once."""

    def __init__(self, fail_once_for: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_messages: List[str] = []
        self.fail_once_for = fail_once_for

    async def stream_chat(self, messages, tools=None, **kwargs):  # type: ignore[override]
        last_user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
        self.user_messages.append(last_user)
        if self.fail_once_for is not None and last_user == self.fail_once_for:
            self.fail_once_for = None
            raise RuntimeError("Provider 503 with api_key=sk-test1234567890abcdef")
        async for chunk in super().stream_chat(messages, tools=tools, **kwargs):
            yield chunk

    def interviewer_calls_for(self, answer: str) -> int:
        return sum(1 for m in self.user_messages if m == answer)


def _llm(**kwargs: Any) -> RecordingMockLLM:
    return RecordingMockLLM(stage_plan=STAGE_PLAN, **kwargs)


def _settings() -> Settings:
    return get_settings().model_copy(
        update={
            "RETRIEVAL_BACKEND": "mock",
            "ASSESSOR_BLOCK_MS": 50,
            "ASSESSOR_RECOVERY_INTERVAL_SECONDS": 0.2,
            "REPORT_SWEEP_INTERVAL_SECONDS": 0.2,
            "ANSWER_BURST": 100.0,
            "ORCHESTRATOR_JOIN_MS": 300,
        }
    )


def _create_session(client: TestClient, headers: Dict[str, str]) -> Dict[str, Any]:
    res = client.post(
        "/api/v1/sessions",
        json={
            "candidate_name": "Nguyen Van B",
            "experience_years": 6,
            "target_company": "Acme Systems",
            "target_role": "Senior Backend Engineer",
            "duration_minutes": 10,
            "skills": ["Python", "Redis", "Distributed Systems"],
            "cv_text": f"Experience:\n{CV_SECRET_LINE}\nSkills: Python, Redis",
        },
        headers={**headers, "Idempotency-Key": f"e2e-{time.time_ns()}"},
    )
    assert res.status_code == 202, res.text
    return res.json()


def _drain_until(ws: Any, terminal_types: tuple, limit: int = 80) -> List[Dict[str, Any]]:
    frames = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        assert frame.get("type") != "busy", frame
        if frame.get("type") in terminal_types:
            break
    return frames


def _wait_for_report(client: TestClient, session_id: str, headers: Dict[str, str], timeout: float = 20.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/v1/sessions/{session_id}/report", headers=headers)
        if res.status_code == 200:
            return res.json()
        assert res.status_code == 202, res.text
        time.sleep(0.1)
    raise AssertionError("report was not produced in time")


def test_full_lifecycle_through_real_app(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """G1 + G3 + log hygiene: the app alone drives a session from creation to REPORT_READY."""
    caplog.set_level(logging.DEBUG)
    settings = _settings()
    llm = _llm()
    app = create_app(
        workspace_base=str(tmp_path / "workspaces"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=llm,
        settings=settings,
    )
    service_headers = {
        "Authorization": f"Bearer {create_service_token(issuer='backend', audience='ai-runtime', secret=settings.JWT_SECRET)}"
    }

    with TestClient(app) as client:
        created = _create_session(client, service_headers)
        session_id, session_token = created["session_id"], created["session_token"]

        status_res = client.get(f"/api/v1/sessions/{session_id}", headers=service_headers).json()
        assert status_res["status"] == "READY"
        assert status_res["retrieval_status"] == "ok"

        seen_types: List[str] = []
        with client.websocket_connect(
            f"/ws/sessions/{session_id}/stream",
            headers={"Authorization": f"Bearer {session_token}"},
        ) as ws:
            ws.send_json({"type": "start_interview"})
            opening = _drain_until(ws, ("turn_complete", "error"))
            assert opening[-1]["type"] == "turn_complete"
            seen_types += [f["type"] for f in opening]

            for index, answer in enumerate(ANSWERS, start=1):
                ws.send_json(
                    {
                        "type": "candidate_answer",
                        "message_id": f"msg-{index}",
                        "candidate_answer": answer,
                        "vad_metrics": {"speech_duration_ms": 9000, "words_per_minute": 120},
                        "expected_turn": index,
                    }
                )
                frames = _drain_until(ws, ("turn_complete", "error"))
                assert frames[-1]["type"] == "turn_complete", frames[-1]
                assert frames[-1]["payload"]["turn_number"] == index
                seen_types += [f["type"] for f in frames]

            # Duplicate delivery of an answered message replays without a new LLM call
            calls_before = llm.interviewer_calls_for(ANSWERS[1])
            ws.send_json({"type": "candidate_answer", "message_id": "msg-2", "candidate_answer": ANSWERS[1]})
            replayed = _drain_until(ws, ("turn_complete", "error"))
            assert replayed[-1]["type"] == "turn_complete"
            assert replayed[-1]["turn_id"] == "turn_002"
            assert llm.interviewer_calls_for(ANSWERS[1]) == calls_before == 1

        assert {"interview_started", "response_token", "turn_complete", "stage_transition"} <= set(seen_types)

        report = _wait_for_report(client, session_id, service_headers)
        assert report["status"] == "REPORT_READY"
        assert report["final_assessment"]["metrics"]["total_turns_evaluated"] == len(ANSWERS)
        assert report["report_markdown"]

        final_status = client.get(f"/api/v1/sessions/{session_id}", headers=service_headers).json()
        assert final_status["metadata"]["expected_assessments"] == len(ANSWERS)

        ws_dir = app.state.workspace_mgr.get_or_create_workspace(session_id)
        for index in range(1, len(ANSWERS) + 1):
            evaluation = ws_dir.read_json(f"evaluations/turn_{index:03d}_eval.json")
            if evaluation["stage_id"] in {"opening", "candidate_questions"}:
                assert evaluation["rubric_target"] == [], "intro and candidate Q&A must not be scored"
            else:
                assert evaluation["rubric_target"], (
                    f"assessable stage {evaluation['stage_id']} must target retrieved criteria"
                )
        assert ws_dir.exists("turns/turn_0_interviewer.json")

    # Log hygiene: candidate answers, CV text and provider keys never reach logs
    # (only application loggers; HTTP access logs of the test client include URL paths)
    app_records = [r for r in caplog.records if r.name.split(".")[0] in ("api", "core", "config")]
    assert app_records
    log_text = "\n".join(r.getMessage() + " " + str(r.__dict__) for r in app_records)
    for answer in ANSWERS:
        assert answer not in log_text
    assert CV_SECRET_LINE not in log_text
    assert session_id not in log_text


def test_failed_generation_is_retryable_with_same_message_id(tmp_path: Path) -> None:
    """A provider failure does not advance the turn; resending the same message_id succeeds."""
    settings = _settings().model_copy(update={"ASSESSOR_POOL_SIZE": 0})
    llm = _llm(fail_once_for=ANSWERS[0])
    app = create_app(
        workspace_base=str(tmp_path / "workspaces"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        llm_client=llm,
        settings=settings,
    )
    service_headers = {
        "Authorization": f"Bearer {create_service_token(issuer='backend', audience='ai-runtime', secret=settings.JWT_SECRET)}"
    }
    with TestClient(app) as client:
        created = _create_session(client, service_headers)
        with client.websocket_connect(
            f"/ws/sessions/{created['session_id']}/stream",
            headers={"Authorization": f"Bearer {created['session_token']}"},
        ) as ws:
            ws.send_json({"type": "start_interview"})
            _drain_until(ws, ("turn_complete", "error"))

            message = {"type": "candidate_answer", "message_id": "retry-1", "candidate_answer": ANSWERS[0]}
            ws.send_json(message)
            failed: List[Dict[str, Any]] = []
            while not (failed and failed[-1]["type"] == "error" and failed[-1]["payload"].get("code") == "TURN_FAILED"):
                failed.append(ws.receive_json())
                assert len(failed) < 20
            codes = [f["payload"].get("code") for f in failed if f["type"] == "error"]
            assert "PROVIDER_ERROR" in codes
            assert failed[-1]["payload"]["retryable"] is True
            # The controller also answers the requester directly with the retryable failure
            direct = ws.receive_json()
            assert direct["type"] == "error" and direct["payload"]["code"] == "TURN_FAILED"
            assert all("sk-test" not in str(f) for f in failed)

            status_mid = client.get(f"/api/v1/sessions/{created['session_id']}", headers=service_headers).json()
            assert status_mid["current_turn"] == 0

            ws.send_json(message)
            retried = _drain_until(ws, ("turn_complete", "error"))
            assert retried[-1]["type"] == "turn_complete"
            assert retried[-1]["payload"]["turn_number"] == 1


def test_report_blocks_until_barrier_and_respects_failures(tmp_path: Path) -> None:
    """GET /report stays 202 while assessments are outstanding; synthesis waits for the barrier."""
    settings = _settings().model_copy(update={"ASSESSOR_POOL_SIZE": 0})
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app = create_app(
        workspace_base=str(tmp_path / "workspaces"),
        redis_client=redis,
        llm_client=_llm(),
        settings=settings,
    )
    service_headers = {
        "Authorization": f"Bearer {create_service_token(issuer='backend', audience='ai-runtime', secret=settings.JWT_SECRET)}"
    }
    with TestClient(app) as client:
        created = _create_session(client, service_headers)
        session_id = created["session_id"]
        with client.websocket_connect(
            f"/ws/sessions/{session_id}/stream",
            headers={"Authorization": f"Bearer {created['session_token']}"},
        ) as ws:
            ws.send_json({"type": "start_interview"})
            _drain_until(ws, ("turn_complete", "error"))
            for index, answer in enumerate(ANSWERS, start=1):
                ws.send_json({"type": "candidate_answer", "message_id": f"b-{index}", "candidate_answer": answer})
                _drain_until(ws, ("turn_complete", "error"))

        # Dialogue completed, no workers running: the session waits in ASSESSING
        deadline = time.time() + 5
        while time.time() < deadline:
            current = client.get(f"/api/v1/sessions/{session_id}", headers=service_headers).json()
            if current["status"] == "ASSESSING":
                break
            time.sleep(0.05)
        assert current["status"] == "ASSESSING"
        assert current["metadata"]["expected_assessments"] == len(ANSWERS)
        assert client.get(f"/api/v1/sessions/{session_id}/report", headers=service_headers).status_code == 202

        # The pipeline refuses to synthesize before the barrier is met
        synthesized = client.portal.call(app.state.pipeline.maybe_synthesize, session_id)
        assert synthesized is False
        assert client.get(f"/api/v1/sessions/{session_id}/report", headers=service_headers).status_code == 202

        # Workers catch up -> report produced through the normal callbacks
        async def _drain_queue() -> None:
            await app.state.assessment_queue.setup_consumer_group(group_name="manual", start_id="0")
            while await app.state.assessor_worker.process_next_job(group_name="manual", consumer_name="m"):
                pass

        client.portal.call(_drain_queue)
        report = _wait_for_report(client, session_id, service_headers, timeout=5)
        assert report["status"] == LifecycleStatus.REPORT_READY.value
