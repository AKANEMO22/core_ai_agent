"""Unit tests for StageDirector, AssessmentQueue, and InteractiveSubagentRunner."""

import asyncio
import pytest
from pathlib import Path

from core.interview.director import StageDirector, StageSessionConflictError
from core.interview.queue import InMemoryAssessmentQueue
from core.interview.runner import InteractiveSubagentRunner
from core.interview.context import ContextAssembler
from core.runtime.agent import RuntimeAgent
from core.runtime.providers.mock import MockLLMClient
from core.session.state import (
    SessionState,
    LifecycleStatus,
    SessionStateManager,
    InvalidStateTransitionError,
)
from core.workspace.session import SessionWorkspace


@pytest.mark.asyncio
async def test_stage_progression_on_budget():
    """Verify turn recording advances turn counters and flags stage completion on budget."""
    state_mgr = SessionStateManager()
    stages = [
        {"stage_id": "s1", "target_turns": 2},
        {"stage_id": "s2", "target_turns": 2},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    state = state_mgr.create_session("sess_1", initial_status=LifecycleStatus.STAGE_ACTIVE)

    # Turn 1 in s1
    res1 = await director.record_turn(state)
    assert res1.current_stage_id == "s1"
    assert res1.is_stage_complete is False

    # Turn 2 in s1 -> triggers stage transition
    res2 = await director.record_turn(res1)
    assert res2.is_stage_complete is True
    next_state = await director.advance_stage(res2)
    assert next_state.current_stage_id == "s2"


@pytest.mark.asyncio
async def test_stage_progression_complete_stage_flag():
    """Verify complete_stage flag prematurely triggers stage completion before budget."""
    state_mgr = SessionStateManager()
    stages = [
        {"stage_id": "s1", "target_turns": 5},
        {"stage_id": "s2", "target_turns": 3},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    state = state_mgr.create_session("sess_2", initial_status=LifecycleStatus.STAGE_ACTIVE)

    # Turn 1 with explicit complete_stage
    res1 = await director.record_turn(state, complete_stage=True)
    assert res1.current_stage_id == "s1"
    assert res1.is_stage_complete is True

    next_state = await director.advance_stage(res1)
    assert next_state.current_stage_id == "s2"
    assert next_state.is_stage_complete is False


@pytest.mark.asyncio
async def test_stage_advance_to_dialogue_completed():
    """Verify that advancing past the final stage transitions session to DIALOGUE_COMPLETED."""
    state_mgr = SessionStateManager()
    stages = [
        {"stage_id": "final_stage", "target_turns": 1},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    state = state_mgr.create_session("sess_final", initial_status=LifecycleStatus.STAGE_ACTIVE)

    res1 = await director.record_turn(state)
    assert res1.is_stage_complete is True

    final_state = await director.advance_stage(res1)
    assert final_state.status == LifecycleStatus.DIALOGUE_COMPLETED
    assert final_state.is_stage_complete is True


@pytest.mark.asyncio
async def test_stage_director_rejects_non_stage_active_status():
    """Verify recording turns in non-STAGE_ACTIVE states raises StageSessionConflictError."""
    state_mgr = SessionStateManager()
    stages = [{"stage_id": "s1", "target_turns": 2}]
    director = StageDirector(stages=stages, state_mgr=state_mgr)

    invalid_state = state_mgr.create_session("sess_err", initial_status=LifecycleStatus.INITIALIZED)
    with pytest.raises(StageSessionConflictError):
        await director.record_turn(invalid_state)


@pytest.mark.asyncio
async def test_stage_director_cas_protection():
    """Verify CAS version mismatch during advance_stage raises InvalidStateTransitionError."""
    state_mgr = SessionStateManager()
    stages = [
        {"stage_id": "s1", "target_turns": 1},
        {"stage_id": "s2", "target_turns": 1},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    state = state_mgr.create_session("sess_cas", initial_status=LifecycleStatus.STAGE_ACTIVE)

    res1 = await director.record_turn(state)
    assert res1.state_version == 2

    # Stale snapshot with version 1
    stale_state = SessionState(
        session_id="sess_cas",
        status=LifecycleStatus.STAGE_ACTIVE,
        state_version=1,
        current_stage_id="s1",
        is_stage_complete=True,
    )
    with pytest.raises(InvalidStateTransitionError):
        await director.advance_stage(stale_state)


@pytest.mark.asyncio
async def test_in_memory_assessment_queue():
    """Verify InMemoryAssessmentQueue enqueue, dequeue, and size tracking."""
    queue = InMemoryAssessmentQueue()
    assert await queue.get_queue_size() == 0

    job_data = {
        "schema_version": "1.0",
        "job_id": "sess_1:turn_001:rubric_v1",
        "session_id": "sess_1",
        "turn_id": "turn_001",
        "stage_id": "stage_1",
        "question": "Tell me about your experience.",
        "candidate_text": "I worked on distributed systems.",
        "vad_metrics": {"speech_duration_ms": 3000, "pause_count": 0},
        "rubric_version": "rubric_v1",
        "rubric_target": ["Distributed Systems"],
    }

    job_id = await queue.enqueue(job_data)
    assert job_id == "sess_1:turn_001:rubric_v1"
    assert await queue.get_queue_size() == 1

    dequeued = await queue.dequeue(timeout_seconds=0.1)
    assert dequeued is not None
    assert dequeued["candidate_text"] == "I worked on distributed systems."
    assert await queue.get_queue_size() == 0


@pytest.mark.asyncio
async def test_interactive_subagent_runner_turn_execution(tmp_path: Path):
    """Verify runner streams tokens, persists candidate and interviewer artifacts, and enqueues job."""
    # 1. Setup workspace
    ws = SessionWorkspace(session_id="sess_runner_1", root_path=tmp_path / "sess_runner_1")
    ws.initialize()

    # Pre-populate context artifacts
    ws.write_artifact(
        "context/interview_spec.json",
        artifact_type="interview_spec",
        payload={
            "interviewer_persona": "Bar Raiser",
            "tone": "probing",
            "knowledge_scope": ["Distributed Systems"],
        },
    )
    ws.write_artifact(
        "context/stage_plan.json",
        artifact_type="stage_plan",
        payload={
            "stages": [
                {"stage_id": "stage_1", "name": "Intro", "target_turns": 1, "goals": ["Probe basics"]},
                {"stage_id": "stage_2", "name": "Deep Dive", "target_turns": 2, "goals": ["Probe concurrency"]},
            ]
        },
    )

    # 2. Setup state manager and director
    state_mgr = SessionStateManager()
    stages = [
        {"stage_id": "stage_1", "name": "Intro", "target_turns": 1},
        {"stage_id": "stage_2", "name": "Deep Dive", "target_turns": 2},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    init_state = state_mgr.create_session(
        session_id="sess_runner_1",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="stage_1",
    )

    # 3. Setup LLM client, runtime agent, queue, assembler, and runner
    mock_llm = MockLLMClient(canned_tokens=["Hello", " candidate,", " let's", " begin!"])
    runtime_agent = RuntimeAgent(client=mock_llm, workspace=ws)
    assembler = ContextAssembler(max_tokens=1500)
    queue = InMemoryAssessmentQueue()

    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=assembler,
        stage_director=director,
        assessment_queue=queue,
        state_mgr=state_mgr,
    )

    # 4. Run turn
    events = []
    async for event in runner.run_turn(
        session_id="sess_runner_1",
        candidate_answer="My name is Alice and I build high-concurrency backends.",
        vad_metrics={"speech_duration_ms": 4500, "pause_count": 1, "words_per_minute": 130},
    ):
        events.append(event)

    # Verify event stream
    assert len(events) > 0
    token_events = [e for e in events if e.type.value == "response_token"]
    assert len(token_events) == 4
    assert "".join(e.payload["token"] for e in token_events) == "Hello candidate, let's begin!"

    # Verify turn persistence
    candidate_file = ws.root_path / "turns" / "turn_1_candidate.json"
    interviewer_file = ws.root_path / "turns" / "turn_1_interviewer.json"
    assert candidate_file.exists()
    assert interviewer_file.exists()

    cand_data = ws.read_json("turns/turn_1_candidate.json")
    assert cand_data["content"] == "My name is Alice and I build high-concurrency backends."
    assert cand_data["vad_metrics"]["words_per_minute"] == 130

    interviewer_data = ws.read_json("turns/turn_1_interviewer.json")
    assert "Hello candidate, let's begin!" in interviewer_data["content"]

    # Verify assessment queue job
    assert await queue.get_queue_size() == 1
    job = await queue.dequeue()
    assert job["session_id"] == "sess_runner_1"
    assert job["candidate_text"] == "My name is Alice and I build high-concurrency backends."
    assert job["vad_metrics"]["speech_duration_ms"] == 4500

    # Verify stage progression: target_turns was 1, so runner advanced stage to stage_2
    curr_state = state_mgr.get_state("sess_runner_1")
    assert curr_state.current_stage_id == "stage_2"


@pytest.mark.asyncio
async def test_interactive_subagent_runner_rejects_inactive_session(tmp_path: Path):
    """Verify runner immediately raises StageSessionConflictError if session is not STAGE_ACTIVE."""
    ws = SessionWorkspace(session_id="sess_runner_err", root_path=tmp_path / "sess_runner_err")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        session_id="sess_runner_err",
        initial_status=LifecycleStatus.INITIALIZED,
    )
    director = StageDirector(stages=[{"stage_id": "s1"}], state_mgr=state_mgr)
    mock_llm = MockLLMClient(canned_tokens=["Hi"])
    runtime_agent = RuntimeAgent(client=mock_llm)

    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )

    with pytest.raises(StageSessionConflictError):
        async for _ in runner.run_turn(
            session_id="sess_runner_err",
            candidate_answer="Test answer",
        ):
            pass


@pytest.mark.asyncio
async def test_interactive_subagent_runner_rejects_out_of_order_turn(tmp_path: Path):
    """Verify runner rejects stale or future turns when expected_turn does not match."""
    ws = SessionWorkspace(session_id="sess_order", root_path=tmp_path / "sess_order")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        session_id="sess_order",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="s1",
    )
    director = StageDirector(stages=[{"stage_id": "s1", "target_turns": 3}], state_mgr=state_mgr)
    mock_llm = MockLLMClient(canned_tokens=["Hi"])
    runtime_agent = RuntimeAgent(client=mock_llm)

    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )

    # Next expected turn is 1, but candidate submits expected_turn=5
    with pytest.raises(StageSessionConflictError):
        async for _ in runner.run_turn(
            session_id="sess_order",
            candidate_answer="Out of order answer",
            expected_turn=5,
        ):
            pass


@pytest.mark.asyncio
async def test_interactive_subagent_runner_tool_complete_stage(tmp_path: Path):
    """Verify complete_stage tool emitted by agent triggers immediate stage advancement."""
    from core.runtime.tooling import RuntimeTool

    ws = SessionWorkspace(session_id="sess_tool_stage", root_path=tmp_path / "sess_tool_stage")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        session_id="sess_tool_stage",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="s1",
    )
    stages = [
        {"stage_id": "s1", "target_turns": 10},  # budget is high (10 turns)
        {"stage_id": "s2", "target_turns": 5},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)

    complete_tool = RuntimeTool.from_callable(
        name="complete_stage",
        description="Mark current stage complete",
        func=lambda reason="": f"Stage finished: {reason}",
    )

    mock_llm = MockLLMClient(
        canned_tool_calls=[{"name": "complete_stage", "arguments": {"reason": "All criteria met"}}],
        final_response="Great, let us move to the next stage.",
    )
    runtime_agent = RuntimeAgent(client=mock_llm, tools=[complete_tool], workspace=ws)

    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )

    events = []
    async for event in runner.run_turn(
        session_id="sess_tool_stage",
        candidate_answer="Here is my comprehensive design answer.",
    ):
        events.append(event)

    # State should advance to s2 even though turn was only 1/10
    final_state = state_mgr.get_state("sess_tool_stage")
    assert final_state.current_stage_id == "s2"
    assert final_state.current_turn == 1


@pytest.mark.asyncio
async def test_runner_complete_stage_no_double_increment_and_sequential_turn(tmp_path: Path):
    """Verify complete_stage does not double increment current_turn, allowing turn 2 to succeed."""
    from core.runtime.tooling import RuntimeTool

    ws = SessionWorkspace(session_id="sess_double_inc", root_path=tmp_path / "sess_double_inc")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        session_id="sess_double_inc",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="s1",
    )
    stages = [
        {"stage_id": "s1", "target_turns": 10},
        {"stage_id": "s2", "target_turns": 5},
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)

    complete_tool = RuntimeTool.from_callable(
        name="complete_stage",
        description="Mark current stage complete",
        func=lambda reason="": f"Done: {reason}",
    )

    mock_llm = MockLLMClient(
        canned_tool_calls=[{"name": "complete_stage", "arguments": {"reason": "Met"}}],
        final_response="Moving to s2.",
    )
    runtime_agent = RuntimeAgent(client=mock_llm, tools=[complete_tool], workspace=ws)
    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )

    # Turn 1: trigger complete_stage
    async for _ in runner.run_turn(
        session_id="sess_double_inc",
        candidate_answer="Answer turn 1",
    ):
        pass

    state_after_turn_1 = state_mgr.get_state("sess_double_inc")
    # Must be 1, NOT 2!
    assert state_after_turn_1.current_turn == 1
    assert state_after_turn_1.current_stage_id == "s2"

    # Turn 2: next candidate turn arrives with expected_turn=2
    mock_llm_turn2 = MockLLMClient(canned_tokens=["Welcome", " to", " stage", " 2!"])
    runner.runtime_agent = RuntimeAgent(client=mock_llm_turn2, workspace=ws)

    events_turn2 = []
    async for event in runner.run_turn(
        session_id="sess_double_inc",
        candidate_answer="Answer turn 2",
        expected_turn=2,
    ):
        events_turn2.append(event)

    state_after_turn_2 = state_mgr.get_state("sess_double_inc")
    assert state_after_turn_2.current_turn == 2


@pytest.mark.asyncio
async def test_runner_no_candidate_duplication_in_system_prompt(tmp_path: Path):
    """Verify that current candidate answer is passed as user message only, not duplicated in system prompt."""
    ws = SessionWorkspace(session_id="sess_no_dup", root_path=tmp_path / "sess_no_dup")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        session_id="sess_no_dup",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="s1",
    )
    director = StageDirector(stages=[{"stage_id": "s1", "target_turns": 5}], state_mgr=state_mgr)

    # Track messages passed to stream_chat
    recorded_messages = []

    class InspectingMockLLM(MockLLMClient):
        async def stream_chat(self, messages, tools=None, **kwargs):
            recorded_messages.extend(messages)
            async for chunk in super().stream_chat(messages, tools, **kwargs):
                yield chunk

    mock_llm = InspectingMockLLM(canned_tokens=["Got it!"])
    runtime_agent = RuntimeAgent(client=mock_llm, workspace=ws)
    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )

    unique_candidate_phrase = "UNIQUE_PHRASE_NEVER_DUPLICATED_12345"
    async for _ in runner.run_turn(
        session_id="sess_no_dup",
        candidate_answer=f"My answer is {unique_candidate_phrase}",
    ):
        pass

    assert len(recorded_messages) == 2
    system_msg = recorded_messages[0]
    user_msg = recorded_messages[1]

    assert system_msg["role"] == "system"
    assert user_msg["role"] == "user"
    assert unique_candidate_phrase in user_msg["content"]
    # Candidate phrase must NOT be present in system prompt
    assert unique_candidate_phrase not in system_msg["content"]


@pytest.mark.asyncio
async def test_runner_loads_prior_interviewer_response_in_turn_2(tmp_path: Path):
    """Verify that in turn 2, turn 1's interviewer response is loaded into recent turns in system prompt."""
    ws = SessionWorkspace(session_id="sess_prior_turns", root_path=tmp_path / "sess_prior_turns")
    ws.initialize()

    state_mgr = SessionStateManager()
    state_mgr.create_session(
        session_id="sess_prior_turns",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="s1",
    )
    director = StageDirector(stages=[{"stage_id": "s1", "target_turns": 5}], state_mgr=state_mgr)

    turn1_question = "Explain ACID properties in relational databases."
    turn2_messages = []

    # Turn 1 LLM produces turn1_question
    mock_llm_turn1 = MockLLMClient(canned_tokens=[turn1_question])
    runtime_agent = RuntimeAgent(client=mock_llm_turn1, workspace=ws)
    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=runtime_agent,
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )

    async for _ in runner.run_turn(
        session_id="sess_prior_turns",
        candidate_answer="My background is backend engineering.",
    ):
        pass

    # Turn 2: inspect messages passed to LLM
    class InspectingMockLLM(MockLLMClient):
        async def stream_chat(self, messages, tools=None, **kwargs):
            turn2_messages.extend(messages)
            async for chunk in super().stream_chat(messages, tools, **kwargs):
                yield chunk

    mock_llm_turn2 = InspectingMockLLM(canned_tokens=["Great explanation of ACID."])
    runner.runtime_agent = RuntimeAgent(client=mock_llm_turn2, workspace=ws)

    async for _ in runner.run_turn(
        session_id="sess_prior_turns",
        candidate_answer="ACID stands for Atomicity, Consistency, Isolation, Durability.",
    ):
        pass

    assert len(turn2_messages) == 2
    system_prompt = turn2_messages[0]["content"]
    # Turn 1 interviewer question MUST be present in turn 2's system prompt conversation history!
    assert turn1_question in system_prompt
    assert "Atomicity, Consistency, Isolation" in turn2_messages[1]["content"]



@pytest.mark.asyncio
async def test_runner_derives_rubric_target_and_version_from_retrieved_rubric(tmp_path: Path):
    """Verify assessment jobs carry the active stage's retrieved criteria and rubric version."""
    ws = SessionWorkspace(session_id="sess_rubric", root_path=tmp_path / "sess_rubric")
    ws.initialize()
    ws.write_artifact(
        "context/rubric.json",
        artifact_type="rubric",
        payload={"rubric_version": "backend_senior_v1", "competencies": []},
    )

    state_mgr = SessionStateManager()
    stages = [
        {
            "stage_id": "technical",
            "target_turns": 2,
            "criteria": ["be_distributed_systems", "be_data_storage"],
        }
    ]
    director = StageDirector(stages=stages, state_mgr=state_mgr)
    state_mgr.create_session(
        session_id="sess_rubric",
        initial_status=LifecycleStatus.STAGE_ACTIVE,
        current_stage_id="technical",
    )
    queue = InMemoryAssessmentQueue()
    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=RuntimeAgent(client=MockLLMClient(canned_tokens=["Next."]), workspace=ws),
        context_assembler=ContextAssembler(max_tokens=1500),
        stage_director=director,
        assessment_queue=queue,
        state_mgr=state_mgr,
    )

    async for _ in runner.run_turn(session_id="sess_rubric", candidate_answer="Use idempotency keys."):
        pass
    async for _ in runner.run_turn(
        session_id="sess_rubric",
        candidate_answer="Add a composite index.",
        rubric_target=["explicit_target"],
        rubric_version="rubric_v9",
    ):
        pass

    derived, explicit = await queue.dequeue(), await queue.dequeue()
    assert derived["rubric_target"] == ["be_distributed_systems", "be_data_storage"]
    assert derived["rubric_version"] == "backend_senior_v1"
    assert derived["job_id"] == "sess_rubric:turn_001:backend_senior_v1"
    assert explicit["rubric_target"] == ["explicit_target"]
    assert explicit["rubric_version"] == "rubric_v9"


@pytest.mark.asyncio
async def test_runner_treats_empty_completion_as_retryable_failure(tmp_path: Path):
    ws = SessionWorkspace(session_id="sess_empty", root_path=tmp_path / "sess_empty")
    ws.initialize()
    state_mgr = SessionStateManager()
    director = StageDirector(stages=[{"stage_id": "s1", "target_turns": 3}], state_mgr=state_mgr)
    state_mgr.create_session("sess_empty", initial_status=LifecycleStatus.STAGE_ACTIVE, current_stage_id="s1")
    runner = InteractiveSubagentRunner(
        workspace=ws,
        runtime_agent=RuntimeAgent(client=MockLLMClient(canned_tokens=["   "]), workspace=ws),
        context_assembler=ContextAssembler(),
        stage_director=director,
        state_mgr=state_mgr,
    )
    events = [e async for e in runner.run_turn(session_id="sess_empty", candidate_answer="hello there")]
    assert events[-1].payload.get("code") == "TURN_FAILED" and events[-1].payload["cause"] == "EMPTY_COMPLETION"
    assert state_mgr.get_state("sess_empty").current_turn == 0
    assert not ws.exists("turns/turn_1_interviewer.json")
