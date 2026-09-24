"""Integration test for full end-to-end interview lifecycle simulation.

Covers:
- Session creation (POST /api/v1/sessions) with Idempotency-Key
- Pre-flight reconnaissance profiling and artifact verification
- WebSocket connection and turn exchange across all curriculum stages (Warmup, Technical, System Design, Q&A / Close)
- Real-time event streaming (progress, response_token, turn_complete, stage_transition, assessment_update)
- Background assessor worker pool processing turn evaluations into evaluations/turn_{n}_eval.json
- Assessment barrier enforcement (completed == expected, failed == 0) and report generation
- Verification of reports/final_assessment.json and reports/interview_report.md via GET /api/v1/sessions/{id}/report
"""

import asyncio
import json
import time
from pathlib import Path
import pytest
import fakeredis.aioredis
from fastapi.testclient import TestClient

from api.auth import create_service_token
from api.main import create_app
from config.settings import get_settings
from core.assessor.queue import RedisAssessmentQueue
from core.assessor.synthesizer import ReportSynthesizer
from core.assessor.worker import AssessorWorker
from core.runtime.providers.mock import MockLLMClient
from core.runtime.types import LoopEventType
from core.session.state import LifecycleStatus


@pytest.mark.asyncio
async def test_full_mock_interview_lifecycle_simulation(tmp_path: Path):
    """Simulate complete interview lifecycle from session intake to final report."""
    settings = get_settings()
    workspace_base = tmp_path / "workspaces"
    workspace_base.mkdir(parents=True, exist_ok=True)

    fake_r = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # 4-stage curriculum plan
    stage_plan_data = {
        "stages": [
            {"stage_id": "warmup", "name": "Warmup", "target_turns": 1},
            {"stage_id": "technical", "name": "Technical Depth", "target_turns": 1},
            {"stage_id": "system_design", "name": "System Design", "target_turns": 1},
            {"stage_id": "qa_close", "name": "Q&A and Close", "target_turns": 1},
        ]
    }

    mock_llm = MockLLMClient(
        canned_tokens=["Thank", " you", " for", " that", " answer.", " Let's", " proceed."],
        canned_json_responses={
            "company_intel": {
                "company": "Acme Systems",
                "culture": "Engineering Rigor",
                "tech_stack": ["Python", "FastAPI", "Redis", "Distributed Systems"],
            },
            "candidate_profile": {
                "strengths": ["Backend Architecture", "Event Streaming"],
                "gaps": ["Capacity Modeling"],
                "probes": ["Distributed Locking", "Consensus"],
            },
            "interview_spec": {
                "interviewer_persona": "Principal Staff Engineer",
                "tone": "rigorous_challenger",
                "knowledge_scope": ["Distributed Systems", "Concurrency", "FastAPI"],
            },
            "stage_plan": stage_plan_data,
            "assessment": {
                "content_score": 9,
                "communication_score": 8,
                "strengths": ["Demonstrates strong conceptual grasp of distributed consensus"],
                "improvements": ["Could elaborate further on network failure modes"],
            },
        },
    )

    app = create_app(
        workspace_base=str(workspace_base),
        redis_client=fake_r,
        llm_client=mock_llm,
        settings=settings,
    )

    client = TestClient(app)
    service_token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    service_headers = {
        "Authorization": f"Bearer {service_token}",
        "Idempotency-Key": "e2e_interview_key_001",
    }

    # --------------------------------------------------------------------------
    # 1. Session Creation (POST /api/v1/sessions) with Idempotency-Key
    # --------------------------------------------------------------------------
    session_payload = {
        "candidate_name": "Nguyen Van B",
        "experience_years": 6,
        "target_company": "Acme Systems",
        "target_role": "Senior Backend Architect",
        "skills": ["Python", "FastAPI", "Redis", "Distributed Systems"],
        "metadata": {"expected_assessments": 4},
    }

    create_res = client.post("/api/v1/sessions", json=session_payload, headers=service_headers)
    assert create_res.status_code == 202
    res_json = create_res.json()
    assert res_json["status"] == "PROFILING"
    assert "session_id" in res_json
    assert "session_token" in res_json
    session_id = res_json["session_id"]
    session_token = res_json["session_token"]

    # Verify idempotency cache returns identical response
    repeat_res = client.post("/api/v1/sessions", json=session_payload, headers=service_headers)
    assert repeat_res.status_code == 202
    assert repeat_res.json()["session_id"] == session_id

    # --------------------------------------------------------------------------
    # 2. Pre-flight Profiling Artifact Verification
    # --------------------------------------------------------------------------
    session_ws = app.state.workspace_mgr.get_or_create_workspace(session_id)
    assert session_ws.exists("context/company_intel.json")
    assert session_ws.exists("context/candidate_profile.json")
    assert session_ws.exists("context/interview_spec.json")
    assert session_ws.exists("context/stage_plan.json")

    # Session state should transition to READY after profiling
    session_state = app.state.state_mgr.get_state(session_id)
    assert session_state is not None
    assert session_state.status == LifecycleStatus.READY

    # --------------------------------------------------------------------------
    # 3. Setup Background Assessor Worker Pool
    # --------------------------------------------------------------------------
    assessment_queue = app.state.assessment_queue
    await assessment_queue.setup_consumer_group(group_name="assessor_group")

    worker = AssessorWorker(
        redis_client=fake_r,
        queue=assessment_queue,
        llm_client=mock_llm,
        workspace_resolver=app.state.workspace_mgr,
        state_manager=app.state.state_mgr,
        event_store=app.state.event_store,
        connection_manager=getattr(app.state, "connection_mgr", None),
    )

    # --------------------------------------------------------------------------
    # 4. WebSocket Connection & Turn Exchange across All 4 Stages
    # --------------------------------------------------------------------------
    turn_answers = [
        # Stage 1: Warmup
        (1, "warmup", "Hello! I am Nguyen Van B, senior engineer with 6 years in distributed services.", {"wpm": 130}),
        # Stage 2: Technical Depth
        (2, "technical", "We use Redis Streams for at-least-once messaging and distributed locks for idempotency.", {"wpm": 125}),
        # Stage 3: System Design
        (3, "system_design", "For high throughput, we partition streams, use read replicas, and enforce TTL cache tiers.", {"wpm": 120}),
        # Stage 4: Q&A / Close
        (4, "qa_close", "I would like to know how the team approaches observability and distributed tracing.", {"wpm": 115}),
    ]

    all_emitted_event_types = []

    with client.websocket_connect(
        f"/ws/sessions/{session_id}/stream",
        headers={"Authorization": f"Bearer {session_token}"},
    ) as ws:
        # Ping check
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"

        for turn_num, expected_stage, answer_text, vad in turn_answers:
            # Send candidate answer
            ws.send_json({
                "type": "candidate_answer",
                "candidate_answer": answer_text,
                "vad_metrics": vad,
                "expected_turn": turn_num,
            })

            # Receive streamed response frames
            turn_frames = []
            for _ in range(30):
                try:
                    frame = ws.receive_json()
                    turn_frames.append(frame)
                    all_emitted_event_types.append(frame["type"])
                    if frame["type"] in ("complete", "turn_complete", "error"):
                        break
                except Exception:
                    break

            assert len(turn_frames) > 0
            # Verify candidate and interviewer artifacts persisted
            assert session_ws.exists(f"turns/turn_{turn_num}_candidate.json")
            assert session_ws.exists(f"turns/turn_{turn_num}_interviewer.json")

            cand_art = session_ws.read_json(f"turns/turn_{turn_num}_candidate.json")
            assert cand_art["content"] == answer_text
            assert cand_art["turn_number"] == turn_num

            # Process background evaluation job for this turn
            job_processed = await worker.process_next_job(group_name="assessor_group", consumer_name="worker_test")
            assert job_processed is True

            # Verify evaluation artifact written
            turn_id_str = f"turn_{turn_num:03d}"
            assert session_ws.exists(f"evaluations/{turn_id_str}_eval.json")
            eval_art = session_ws.read_json(f"evaluations/{turn_id_str}_eval.json")
            payload = eval_art.get("payload", eval_art)
            assert payload["content_score"] == 9
            assert payload["communication_score"] == 8

    # --------------------------------------------------------------------------
    # 5. Verify Real-Time Event Streaming Types
    # --------------------------------------------------------------------------
    assert "response_token" in all_emitted_event_types
    assert "progress" in all_emitted_event_types
    assert "turn_complete" in all_emitted_event_types
    assert "stage_transition" in all_emitted_event_types
    assert "assessment_update" in all_emitted_event_types

    # --------------------------------------------------------------------------
    # 6. Verify Dialogue Completion & State Progression
    # --------------------------------------------------------------------------
    final_state = app.state.state_mgr.get_state(session_id)
    assert final_state.current_turn == 4
    assert final_state.status == LifecycleStatus.DIALOGUE_COMPLETED

    # --------------------------------------------------------------------------
    # 7. Barrier Check & Report Synthesis
    # --------------------------------------------------------------------------
    synthesizer = ReportSynthesizer(
        redis_client=fake_r,
        workspace_resolver=app.state.workspace_mgr,
        state_manager=app.state.state_mgr,
        event_store=app.state.event_store,
        llm_client=mock_llm,
    )

    # Check barrier satisfies: 4 completed, 0 failed, 4 expected
    is_ready, completed, failed, expected = await synthesizer.check_barrier(session_id)
    assert is_ready is True
    assert completed == 4
    assert failed == 0
    assert expected == 4

    # Advance state to ASSESSING and execute synthesize
    state_assessing = app.state.state_mgr.transition(
        final_state,
        expected_version=final_state.state_version,
        new_status=LifecycleStatus.ASSESSING,
    )
    final_assessment = await synthesizer.synthesize(session_id)
    assert final_assessment["session_id"] == session_id
    assert final_assessment["metrics"]["total_turns_evaluated"] == 4

    # State transitioned to REPORT_READY
    ready_state = app.state.state_mgr.get_state(session_id)
    assert ready_state.status == LifecycleStatus.REPORT_READY

    # Verify report artifacts on disk
    assert session_ws.exists("reports/final_assessment.json")
    assert session_ws.exists("reports/interview_report.md")

    report_md = (session_ws.root_path / "reports" / "interview_report.md").read_text(encoding="utf-8")
    assert "AI Interview Practice & Coaching Report" in report_md
    assert "Disclaimer" in report_md

    # --------------------------------------------------------------------------
    # 8. Verification via GET /api/v1/sessions/{id}/report
    # --------------------------------------------------------------------------
    report_res = client.get(f"/api/v1/sessions/{session_id}/report", headers=service_headers)
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert report_data["status"] == "REPORT_READY"
    assert report_data["final_assessment"]["session_id"] == session_id
    assert "AI Interview Practice & Coaching Report" in report_data["report_markdown"]


@pytest.mark.asyncio
async def test_report_barrier_blocks_when_incomplete_in_lifecycle(tmp_path: Path):
    """Verify that report synthesis and GET /report strictly block when assessment barrier is not met."""
    from core.assessor.synthesizer import AssessmentBarrierError

    settings = get_settings()
    fake_r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app = create_app(
        workspace_base=str(tmp_path / "workspaces_barrier"),
        redis_client=fake_r,
        settings=settings,
    )
    client = TestClient(app)
    service_token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    headers = {"Authorization": f"Bearer {service_token}"}

    create_res = client.post(
        "/api/v1/sessions",
        json={"candidate_name": "Barrier Candidate", "metadata": {"expected_assessments": 2}},
        headers=headers,
    )
    assert create_res.status_code == 202
    sess_id = create_res.json()["session_id"]

    # In PROFILING or STAGE_ACTIVE -> GET /report returns 202
    rep_202 = client.get(f"/api/v1/sessions/{sess_id}/report", headers=headers)
    assert rep_202.status_code == 202
    assert rep_202.json()["status"] == "PROFILING"

    synthesizer = ReportSynthesizer(
        redis_client=fake_r,
        workspace_resolver=app.state.workspace_mgr,
        state_manager=app.state.state_mgr,
    )

    # Barrier check fails when 0 / 2 completed
    is_ready, completed, failed, expected = await synthesizer.check_barrier(sess_id)
    assert is_ready is False
    assert completed == 0
    assert expected == 2

    with pytest.raises(AssessmentBarrierError):
        await synthesizer.synthesize(sess_id)

