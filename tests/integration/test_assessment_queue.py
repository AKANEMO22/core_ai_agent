"""Integration tests for background assessment queue, assessor worker, and report synthesizer."""

import json
import pytest
import fakeredis.aioredis
from core.assessor.queue import RedisAssessmentQueue
from core.assessor.subagent import TurnEvaluator
from core.assessor.worker import AssessorWorker
from core.assessor.synthesizer import ReportSynthesizer, AssessmentBarrierError
from core.runtime.providers.mock import MockLLMClient
from core.session.events import SessionEventStore
from core.session.state import LifecycleStatus, SessionState, SessionStateManager
from core.workspace.session import SessionWorkspace


@pytest.mark.asyncio
async def test_redis_assessment_queue_and_worker_persistence(tmp_path):
    """Test full flow: enqueueing turn assessment to Redis stream, worker processing, atomic write and ACK."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis_client=r, stream_key="test:tasks:assessment")
    await queue.setup_consumer_group(group_name="workers")

    ws = SessionWorkspace(session_id="sess_eval1", root_path=tmp_path / "sess_eval1")
    ws.initialize()

    job_data = {
        "schema_version": "1.0",
        "job_id": "sess_eval1:turn_001:rubric_v1",
        "session_id": "sess_eval1",
        "turn_id": "turn_001",
        "stage_id": "stage_1",
        "question": "Explain Redis Pub/Sub.",
        "candidate_text": "Redis Pub/Sub sends messages to channels.",
        "vad_metrics": {"speech_duration_ms": 5000, "pause_count": 1, "total_silence_ms": 400, "words_per_minute": 120},
        "rubric_version": "rubric_v1",
        "rubric_target": ["Redis", "PubSub"],
    }
    await queue.enqueue(job_data)

    mock_llm = MockLLMClient(canned_json_responses={
        "assessment": {
            "content_score": 8,
            "communication_score": 7,
            "strengths": ["Clear explanation"],
            "improvements": ["Mention persistent streams vs pub/sub"],
        }
    })
    worker = AssessorWorker(redis_client=r, queue=queue, llm_client=mock_llm, workspace_resolver=lambda sid: ws)
    processed = await worker.process_next_job(group_name="workers", consumer_name="worker_1")
    assert processed is True

    # Verify evaluation artifact on disk
    eval_file = ws.root_path / "evaluations" / "turn_001_eval.json"
    assert eval_file.exists()

    data = ws.read_json("evaluations/turn_001_eval.json")
    payload = data.get("payload", data)
    assert payload["content_score"] == 8
    assert payload["communication_score"] == 7
    assert "Clear explanation" in payload["strengths"]
    assert "delivery_observations" in payload

    # Verify Redis set cardinality and completed set
    completed_card = await r.scard("session:sess_eval1:completed_assessments")
    assert completed_card == 1
    assert await r.sismember("session:sess_eval1:completed_assessments", "sess_eval1:turn_001:rubric_v1")


@pytest.mark.asyncio
async def test_worker_idempotency_avoids_duplicate_evaluation(tmp_path):
    """Test that duplicate or re-delivered job does not re-invoke LLM and keeps set cardinality at 1."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis_client=r, stream_key="test:tasks:assessment:idemp")
    await queue.setup_consumer_group(group_name="workers")

    ws = SessionWorkspace(session_id="sess_idemp", root_path=tmp_path / "sess_idemp")
    ws.initialize()

    job_data = {
        "schema_version": "1.0",
        "job_id": "sess_idemp:turn_001:rubric_v1",
        "session_id": "sess_idemp",
        "turn_id": "turn_001",
        "stage_id": "stage_1",
        "question": "What is an index?",
        "candidate_text": "An index accelerates database lookups using B-trees.",
        "vad_metrics": {"speech_duration_ms": 4000, "pause_count": 0, "total_silence_ms": 0, "words_per_minute": 130},
        "rubric_version": "rubric_v1",
        "rubric_target": ["Databases", "Indexing"],
    }
    await queue.enqueue(job_data)

    mock_llm = MockLLMClient(canned_json_responses={
        "assessment": {
            "content_score": 9,
            "communication_score": 8,
            "strengths": ["Concise definition"],
            "improvements": ["Mention write overhead"],
        }
    })
    worker = AssessorWorker(redis_client=r, queue=queue, llm_client=mock_llm, workspace_resolver=lambda sid: ws)

    # First execution
    assert await worker.process_next_job(group_name="workers", consumer_name="w1") is True
    assert mock_llm.call_count == 1
    assert await r.scard("session:sess_idemp:completed_assessments") == 1

    # Enqueue duplicate job
    await queue.enqueue(job_data)

    # Second execution should hit idempotency check and skip LLM call
    assert await worker.process_next_job(group_name="workers", consumer_name="w1") is True
    assert mock_llm.call_count == 1  # Still 1, NOT 2
    assert await r.scard("session:sess_idemp:completed_assessments") == 1


@pytest.mark.asyncio
async def test_turn_evaluator_scores_and_descriptive_observations():
    """Test TurnEvaluator scoring bounds (1-10) and non-scoring descriptive delivery observations."""
    mock_llm = MockLLMClient(canned_json_responses={
        "assessment": {
            "content_score": 10,
            "communication_score": 9,
            "strengths": ["Deep expertise"],
            "improvements": [],
        }
    })
    evaluator = TurnEvaluator(llm_client=mock_llm)
    vad_metrics = {
        "speech_duration_ms": 12500,
        "pause_count": 3,
        "total_silence_ms": 1500,
        "words_per_minute": 135,
    }

    result = await evaluator.evaluate_turn(
        question="How does Raft handle network partitions?",
        candidate_text="Raft maintains consistency via leader leases and majority quorum...",
        rubric_target=["Consensus", "Raft", "Fault Tolerance"],
        rubric_version="rubric_v1",
        vad_metrics=vad_metrics,
    )

    assert result["content_score"] == 10
    assert result["communication_score"] == 9
    assert "delivery_observations" in result
    obs = result["delivery_observations"]
    # Descriptive observations should contain timing metrics and explicit non-evaluative phrasing
    assert obs["speech_duration_ms"] == 12500
    assert obs["pause_count"] == 3
    assert "observations" in obs
    assert "not used to infer" in obs["observations"].lower() or "descriptive" in obs["observations"].lower()


@pytest.mark.asyncio
async def test_worker_transient_failure_and_dead_letter_queue(tmp_path):
    """Test that permanent failure moves job to dead letter stream and records failure in Redis."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(
        redis_client=r,
        stream_key="test:tasks:assessment:fail",
        dead_letter_stream_key="test:tasks:assessment:fail:dead_letter",
    )
    await queue.setup_consumer_group(group_name="workers")

    ws = SessionWorkspace(session_id="sess_fail", root_path=tmp_path / "sess_fail")
    ws.initialize()

    state_mgr = SessionStateManager()
    state = state_mgr.create_session("sess_fail", initial_status=LifecycleStatus.ASSESSING)

    job_data = {
        "schema_version": "1.0",
        "job_id": "sess_fail:turn_001:rubric_v1",
        "session_id": "sess_fail",
        "turn_id": "turn_001",
        "stage_id": "stage_1",
        "question": "Tell me about CAP theorem.",
        "candidate_text": "Consistency, Availability, Partition tolerance.",
        "vad_metrics": {},
        "rubric_version": "rubric_v1",
        "rubric_target": ["Distributed Systems"],
        "retry_count": 2,  # Already tried twice, this will be 3rd (and final) attempt
    }
    await queue.enqueue(job_data)

    failing_llm = MockLLMClient(raise_error=RuntimeError("Provider API timeout"))
    worker = AssessorWorker(
        redis_client=r,
        queue=queue,
        llm_client=failing_llm,
        workspace_resolver=lambda sid: ws,
        state_manager=state_mgr,
        max_retries=3,
    )

    processed = await worker.process_next_job(group_name="workers", consumer_name="w1")
    assert processed is False

    # Check dead letter queue
    dead_letters = await r.xrange("test:tasks:assessment:fail:dead_letter", "-", "+")
    assert len(dead_letters) == 1

    # Check failed set
    failed_card = await r.scard("session:sess_fail:failed_assessments")
    assert failed_card == 1

    # Check state transitioned to ASSESSMENT_FAILED
    updated_state = state_mgr.get_state("sess_fail")
    assert updated_state.status == LifecycleStatus.ASSESSMENT_FAILED
    assert updated_state.metadata.get("failure_code") == "ASSESSMENT_EXHAUSTED"


@pytest.mark.asyncio
async def test_redis_assessment_queue_autoclaim():
    """Test claiming idle jobs using XAUTOCLAIM."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis_client=r, stream_key="test:tasks:assessment:claim")
    await queue.setup_consumer_group(group_name="claim_group")

    job_data = {
        "job_id": "sess_claim:turn_001:rubric_v1",
        "session_id": "sess_claim",
        "turn_id": "turn_001",
        "question": "What is concurrency?",
        "candidate_text": "Executing multiple tasks in overlapping time periods.",
    }
    await queue.enqueue(job_data)

    # Worker 1 reads job but crashes before ACK
    jobs = await queue.read_jobs(group_name="claim_group", consumer_name="worker_1", count=1)
    assert len(jobs) == 1
    msg_id, _ = jobs[0]

    # Worker 2 attempts to claim idle jobs with 0 min_idle_time_ms
    claimed = await queue.claim_idle_jobs(
        group_name="claim_group",
        consumer_name="worker_2",
        min_idle_time_ms=0,
        count=10,
    )
    assert len(claimed) == 1
    assert claimed[0][0] == msg_id


@pytest.mark.asyncio
async def test_report_synthesizer_barrier_blocks_when_incomplete(tmp_path):
    """Test that ReportSynthesizer refuses to generate report if completed_assessments < expected_assessments."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ws = SessionWorkspace(session_id="sess_barrier", root_path=tmp_path / "sess_barrier")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        "sess_barrier",
        initial_status=LifecycleStatus.ASSESSING,
        metadata={"expected_assessments": 2},
    )

    # Only 1 completed assessment recorded in Redis
    await r.sadd("session:sess_barrier:completed_assessments", "job_1")

    synthesizer = ReportSynthesizer(
        redis_client=r,
        workspace_resolver=lambda sid: ws,
        state_manager=state_mgr,
    )

    is_ready, completed, failed, expected = await synthesizer.check_barrier("sess_barrier")
    assert is_ready is False
    assert completed == 1
    assert expected == 2
    assert failed == 0

    with pytest.raises(AssessmentBarrierError):
        await synthesizer.synthesize("sess_barrier")


@pytest.mark.asyncio
async def test_report_synthesizer_barrier_blocks_when_failed(tmp_path):
    """Test that ReportSynthesizer refuses to generate report if failed_assessments > 0."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ws = SessionWorkspace(session_id="sess_barrier_fail", root_path=tmp_path / "sess_barrier_fail")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        "sess_barrier_fail",
        initial_status=LifecycleStatus.ASSESSING,
        metadata={"expected_assessments": 2},
    )

    # 1 completed, 1 failed
    await r.sadd("session:sess_barrier_fail:completed_assessments", "job_1")
    await r.sadd("session:sess_barrier_fail:failed_assessments", "job_2")

    synthesizer = ReportSynthesizer(
        redis_client=r,
        workspace_resolver=lambda sid: ws,
        state_manager=state_mgr,
    )

    is_ready, completed, failed, expected = await synthesizer.check_barrier("sess_barrier_fail")
    assert is_ready is False
    assert failed == 1

    with pytest.raises(AssessmentBarrierError):
        await synthesizer.synthesize("sess_barrier_fail")


@pytest.mark.asyncio
async def test_report_synthesizer_success_generates_artifacts_and_transitions_state(tmp_path):
    """Test full report synthesis: barrier satisfied, artifact write, disclaimer, event emission, state transition."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ws = SessionWorkspace(session_id="sess_synth", root_path=tmp_path / "sess_synth")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        "sess_synth",
        initial_status=LifecycleStatus.ASSESSING,
        metadata={"expected_assessments": 2},
    )

    event_store = SessionEventStore(redis_client=r)

    # Write evaluation artifacts for turn 1 and turn 2
    ws.write_artifact(
        "evaluations/turn_001_eval.json",
        artifact_type="turn_evaluation",
        payload={
            "job_id": "sess_synth:turn_001:rubric_v1",
            "session_id": "sess_synth",
            "turn_id": "turn_001",
            "stage_id": "stage_tech",
            "content_score": 8,
            "communication_score": 7,
            "strengths": ["Good technical clarity"],
            "improvements": ["Give more concrete examples"],
            "delivery_observations": {"observations": "Moderate pace."},
            "rubric_target": ["System Design"],
        },
    )
    ws.write_artifact(
        "evaluations/turn_002_eval.json",
        artifact_type="turn_evaluation",
        payload={
            "job_id": "sess_synth:turn_002:rubric_v1",
            "session_id": "sess_synth",
            "turn_id": "turn_002",
            "stage_id": "stage_tech",
            "content_score": 9,
            "communication_score": 8,
            "strengths": ["Excellent trade-off analysis"],
            "improvements": [],
            "delivery_observations": {"observations": "Confident delivery."},
            "rubric_target": ["Concurrency"],
        },
    )

    # Record completed jobs in Redis
    await r.sadd("session:sess_synth:completed_assessments", "sess_synth:turn_001:rubric_v1")
    await r.sadd("session:sess_synth:completed_assessments", "sess_synth:turn_002:rubric_v1")

    synthesizer = ReportSynthesizer(
        redis_client=r,
        workspace_resolver=lambda sid: ws,
        state_manager=state_mgr,
        event_store=event_store,
    )

    is_ready, completed, failed, expected = await synthesizer.check_barrier("sess_synth")
    assert is_ready is True
    assert completed == 2
    assert expected == 2
    assert failed == 0

    report_result = await synthesizer.synthesize("sess_synth")
    assert report_result is not None

    # Verify final_assessment.json exists and has practice metrics and disclaimer
    final_json_path = ws.root_path / "reports" / "final_assessment.json"
    assert final_json_path.exists()
    final_data = ws.read_json("reports/final_assessment.json")
    payload = final_data.get("payload", final_data)
    assert "metrics" in payload
    assert "disclaimer" in payload
    assert "coaching feedback" in payload["disclaimer"].lower()
    assert "employment decision" in payload["disclaimer"].lower()

    # Verify interview_report.md exists and has markdown report with disclaimer
    report_md_path = ws.root_path / "reports" / "interview_report.md"
    assert report_md_path.exists()
    report_md = ws.read_text("reports/interview_report.md")
    assert "coaching feedback" in report_md.lower()
    assert "employment decision" in report_md.lower()

    # Verify session transitioned to REPORT_READY
    final_state = state_mgr.get_state("sess_synth")
    assert final_state.status == LifecycleStatus.REPORT_READY

    # Verify report_ready event was emitted
    events = event_store.get_events("sess_synth")
    event_types = [e.type.value if hasattr(e.type, "value") else str(e.type) for e in events]
    assert "report_ready" in event_types


@pytest.mark.asyncio
async def test_worker_idempotency_detects_changed_rubric_or_candidate_text(tmp_path):
    """Test that worker re-evaluates if rubric_version or candidate_text differs despite matching job_id."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisAssessmentQueue(redis_client=r, stream_key="test:tasks:assessment:strict_idemp")
    await queue.setup_consumer_group(group_name="workers")

    ws = SessionWorkspace(session_id="sess_strict", root_path=tmp_path / "sess_strict")
    ws.initialize()

    # Pre-populate an existing evaluation with rubric_v1 and text A
    ws.write_artifact(
        "evaluations/turn_001_eval.json",
        artifact_type="turn_evaluation",
        payload={
            "job_id": "sess_strict:turn_001:rubric_v1",
            "session_id": "sess_strict",
            "turn_id": "turn_001",
            "input_sha256": "sha256_text_A",
            "rubric_version": "rubric_v1",
            "content_score": 7,
            "communication_score": 7,
            "strengths": ["Initial eval"],
            "improvements": [],
            "delivery_observations": {},
        },
    )

    mock_llm = MockLLMClient(canned_json_responses={
        "assessment": {
            "content_score": 9,
            "communication_score": 9,
            "strengths": ["Re-evaluated with new rubric"],
            "improvements": [],
        }
    })
    worker = AssessorWorker(redis_client=r, queue=queue, llm_client=mock_llm, workspace_resolver=lambda sid: ws)

    # Job with different rubric_version (rubric_v2) -> must NOT trigger false cache hit
    job_new_rubric = {
        "job_id": "sess_strict:turn_001:rubric_v1",  # Same job_id
        "session_id": "sess_strict",
        "turn_id": "turn_001",
        "question": "What is Raft?",
        "candidate_text": "A consensus algorithm.",
        "input_sha256": "sha256_text_A",  # Same sha256
        "rubric_version": "rubric_v2",   # Different rubric_version!
    }
    await queue.enqueue(job_new_rubric)
    assert await worker.process_next_job(group_name="workers", consumer_name="w1") is True
    assert mock_llm.call_count == 1  # Re-evaluation occurred!


@pytest.mark.asyncio
async def test_report_synthesizer_barrier_fails_safe_when_expected_is_unknown(tmp_path):
    """Test that check_barrier returns is_ready=False when expected count cannot be resolved."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ws = SessionWorkspace(session_id="sess_unknown_exp", root_path=tmp_path / "sess_unknown_exp")
    ws.initialize()

    state_mgr = SessionStateManager()
    # Create session WITHOUT expected_assessments in metadata
    state_mgr.create_session("sess_unknown_exp", initial_status=LifecycleStatus.ASSESSING)

    # 1 completed assessment in Redis
    await r.sadd("session:sess_unknown_exp:completed_assessments", "job_1")

    synthesizer = ReportSynthesizer(
        redis_client=r,
        workspace_resolver=lambda sid: ws,
        state_manager=state_mgr,
    )

    # Barrier check must NOT default to completed_count and must fail safe
    is_ready, completed, failed, expected = await synthesizer.check_barrier("sess_unknown_exp")
    assert is_ready is False
    assert completed == 1
    assert expected == 0
    assert failed == 0

    with pytest.raises(AssessmentBarrierError):
        await synthesizer.synthesize("sess_unknown_exp")

