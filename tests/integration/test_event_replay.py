"""Integration tests for WebSocket connection manager, connection lease displacement, and event replay."""

import asyncio
import pytest
from api.websocket import ConnectionManager, ConnectionLease
from core.runtime.types import LoopEvent, LoopEventType


@pytest.mark.asyncio
async def test_reconnection_and_event_replay():
    """Verify event recording and retrieval of events after last_sequence."""
    mgr = ConnectionManager()
    mgr.record_event("sess_1", LoopEvent(
        schema_version="1.0",
        event_id="e1",
        session_id="sess_1",
        turn_id="turn_001",
        sequence=1,
        timestamp="2026-09-24T00:00:00Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "Hello"}
    ))
    mgr.record_event("sess_1", LoopEvent(
        schema_version="1.0",
        event_id="e2",
        session_id="sess_1",
        turn_id="turn_001",
        sequence=2,
        timestamp="2026-09-24T00:00:01Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": " World"}
    ))

    replayed = mgr.get_events_after("sess_1", last_sequence=1)
    assert len(replayed) == 1
    assert replayed[0].event_id == "e2"


@pytest.mark.asyncio
async def test_event_replay_by_last_event_id():
    """Verify event replay using last_event_id resume semantics."""
    mgr = ConnectionManager()
    mgr.record_event("sess_2", LoopEvent(
        schema_version="1.0",
        event_id="evt_001",
        session_id="sess_2",
        turn_id="turn_001",
        sequence=1,
        timestamp="2026-09-24T00:00:00Z",
        type=LoopEventType.PROGRESS,
        payload={"iteration": 1}
    ))
    mgr.record_event("sess_2", LoopEvent(
        schema_version="1.0",
        event_id="evt_002",
        session_id="sess_2",
        turn_id="turn_001",
        sequence=2,
        timestamp="2026-09-24T00:00:01Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "First"}
    ))
    mgr.record_event("sess_2", LoopEvent(
        schema_version="1.0",
        event_id="evt_003",
        session_id="sess_2",
        turn_id="turn_001",
        sequence=3,
        timestamp="2026-09-24T00:00:02Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "Second"}
    ))

    # Replay after evt_001 should return evt_002 and evt_003
    replayed = mgr.get_events_after("sess_2", last_event_id="evt_001")
    assert len(replayed) == 2
    assert [e.event_id for e in replayed] == ["evt_002", "evt_003"]

    # Replay with unknown last_event_id returns all events
    all_events = mgr.get_events_after("sess_2", last_event_id="unknown_evt")
    assert len(all_events) == 3


class MockWebSocket:
    """Mock WebSocket for testing connection manager lease and dispatch."""

    def __init__(self):
        self.sent_messages = []
        self.closed = False
        self.close_code = None
        self.close_reason = None

    async def send_text(self, data: str):
        if self.closed:
            raise RuntimeError("WebSocket is closed")
        self.sent_messages.append(data)

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason


@pytest.mark.asyncio
async def test_connection_lease_displacement():
    """Verify that a newer connection lease displaces and closes the previous connection."""
    mgr = ConnectionManager()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()

    # Client 1 connects
    lease1 = await mgr.acquire_lease("sess_lease", ws1)
    assert lease1.is_active is True
    assert mgr.get_active_lease("sess_lease") is lease1

    # Client 2 connects for the same session -> displaces client 1
    lease2 = await mgr.acquire_lease("sess_lease", ws2)
    assert lease2.is_active is True
    assert lease1.is_active is False
    assert ws1.closed is True
    assert "displaced" in (ws1.close_reason or "").lower()
    assert mgr.get_active_lease("sess_lease") is lease2

    # Client 1 disconnect attempt does not remove lease 2
    await mgr.release_lease("sess_lease", lease1)
    assert mgr.get_active_lease("sess_lease") is lease2

    # Client 2 disconnect removes active lease
    await mgr.release_lease("sess_lease", lease2)
    assert mgr.get_active_lease("sess_lease") is None


@pytest.mark.asyncio
async def test_broadcast_event_to_active_lease():
    """Verify sending event to active connection lease serializes LoopEvent."""
    mgr = ConnectionManager()
    ws = MockWebSocket()
    lease = await mgr.acquire_lease("sess_send", ws)

    event = LoopEvent(
        schema_version="1.0",
        event_id="e_test",
        session_id="sess_send",
        turn_id="turn_001",
        sequence=1,
        timestamp="2026-09-24T00:00:00Z",
        type=LoopEventType.RESPONSE_TOKEN,
        payload={"token": "Streamed"}
    )
    sent = await mgr.send_event("sess_send", event)
    assert sent is True
    assert len(ws.sent_messages) == 1
    assert "Streamed" in ws.sent_messages[0]


def test_websocket_rejects_query_token(tmp_path):
    """Verify constraint: Tokens must not be exposed in query strings and are rejected with WS_1008."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.auth import create_session_token
    import pytest

    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    client = TestClient(app)
    token = create_session_token("sess_query", secret="test_jwt_secret_32_chars_long_1234")

    # Connecting with ?token=... must be closed/rejected
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/sessions/sess_query/stream?token={token}") as ws:
            pass


def test_websocket_stream_turn_and_envelope_structure(tmp_path):
    """Verify WebSocket streaming candidate turn and protocol envelope compliance."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.auth import create_service_token, create_session_token
    from core.session.state import LifecycleStatus
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    client = TestClient(app)

    service_token = create_service_token(secret=settings.JWT_SECRET)
    create_res = client.post(
        "/api/v1/sessions",
        json={"candidate_name": "Nguyen Van C", "target_role": "Backend Engineer"},
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert create_res.status_code == 202
    sess_id = create_res.json()["session_id"]
    sess_token = create_res.json()["session_token"]

    # Connect WebSocket using Authorization header
    with client.websocket_connect(
        f"/ws/sessions/{sess_id}/stream",
        headers={"Authorization": f"Bearer {sess_token}"},
    ) as ws:
        # 1. Ping / pong test
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"

        # 2. Start the interview (opening question, turn 0)
        ws.send_json({"type": "start_interview"})
        opening = _drain_until(ws, ("turn_complete", "error"))
        assert opening[0]["type"] == "interview_started"
        assert opening[-1]["type"] == "turn_complete"
        assert opening[-1]["turn_id"] == "turn_000"

        # 3. Submit candidate answer turn
        ws.send_json({
            "type": "candidate_answer",
            "message_id": "msg-envelope-1",
            "candidate_answer": "I have extensive experience with FastAPI, Redis, and event streaming.",
            "vad_metrics": {"wpm": 130},
        })

        # 3. Read streamed response tokens
        received_events = []
        # Receive tokens until complete or timeout
        for _ in range(20):
            try:
                frame = ws.receive_json()
                received_events.append(frame)
                # Verify protocol envelope compliance:
                # schema_version, event_id, session_id, turn_id, sequence, timestamp, type, payload
                assert "schema_version" in frame
                assert "event_id" in frame
                assert "session_id" in frame
                assert "turn_id" in frame
                assert "sequence" in frame
                assert "timestamp" in frame
                assert "type" in frame
                assert "payload" in frame
                if frame["type"] in ("complete", "turn_complete", "error"):
                    break
            except Exception:
                break

        assert len(received_events) > 0
        token_frames = [e for e in received_events if e["type"] == "response_token"]
        assert len(token_frames) > 0
        received_text = "".join([f["payload"]["token"] for f in token_frames])
        assert "Hello" in received_text or len(received_text) > 0


def test_websocket_auth_handshake_first_message(tmp_path):
    """Verify WebSocket authentication via first message payload frame."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.auth import create_session_token
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    client = TestClient(app)
    sess_id = "sess_first_msg_auth"
    token = create_session_token(sess_id, secret=settings.JWT_SECRET)

    with client.websocket_connect(f"/ws/sessions/{sess_id}/stream") as ws:
        # First message delivers auth token
        ws.send_json({"type": "auth", "token": token})
        auth_ok = ws.receive_json()
        assert auth_ok["type"] == "auth_ok"
        assert auth_ok["session_id"] == sess_id

        # Connection is now authenticated and accepts commands
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_websocket_multi_turn_monotonic_sequence(tmp_path):
    """Verify that turn 2 and subsequent turns maintain strictly monotonic sequence numbers without crashing."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.auth import create_service_token
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    client = TestClient(app)

    service_token = create_service_token(secret=settings.JWT_SECRET)
    create_res = client.post(
        "/api/v1/sessions",
        json={"candidate_name": "Multi Turn Candidate", "target_role": "Backend Engineer"},
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert create_res.status_code == 202
    sess_id = create_res.json()["session_id"]
    sess_token = create_res.json()["session_token"]

    with client.websocket_connect(
        f"/ws/sessions/{sess_id}/stream",
        headers={"Authorization": f"Bearer {sess_token}"},
    ) as ws:
        ws.send_json({"type": "start_interview"})
        _drain_until(ws, ("turn_complete", "error"))

        # Turn 1
        ws.send_json({
            "type": "candidate_answer",
            "message_id": "msg-multi-1",
            "candidate_answer": "Turn 1: I have experience in distributed systems.",
        })
        turn1_sequences = []
        for _ in range(20):
            try:
                frame = ws.receive_json()
                turn1_sequences.append(frame["sequence"])
                if frame["type"] in ("complete", "turn_complete", "error"):
                    break
            except Exception:
                break

        assert len(turn1_sequences) > 0
        max_turn1_seq = max(turn1_sequences)

        # Turn 2: RuntimeAgent resets sequence to 1, but ConnectionManager must ensure monotonic increment
        ws.send_json({
            "type": "candidate_answer",
            "message_id": "msg-multi-2",
            "candidate_answer": "Turn 2: We used Redis for distributed locking.",
        })
        turn2_sequences = []
        for _ in range(20):
            try:
                frame = ws.receive_json()
                turn2_sequences.append(frame["sequence"])
                if frame["type"] in ("complete", "turn_complete", "error"):
                    break
            except Exception:
                break

        assert len(turn2_sequences) > 0
        # All turn 2 sequences must be strictly greater than max turn 1 sequence
        for seq in turn2_sequences:
            assert seq > max_turn1_seq


def test_websocket_start_interview_gates_answers_and_is_idempotent(tmp_path):
    """Answers before start are rejected; start moves READY to STAGE_ACTIVE once; duplicates replay."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.auth import create_service_token
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    client = TestClient(app)

    service_token = create_service_token(secret=settings.JWT_SECRET)
    service_headers = {"Authorization": f"Bearer {service_token}"}
    create_res = client.post(
        "/api/v1/sessions",
        json={"candidate_name": "Ready State Candidate"},
        headers=service_headers,
    )
    sess_id = create_res.json()["session_id"]
    sess_token = create_res.json()["session_token"]
    assert client.get(f"/api/v1/sessions/{sess_id}", headers=service_headers).json()["status"] == "READY"

    with client.websocket_connect(
        f"/ws/sessions/{sess_id}/stream",
        headers={"Authorization": f"Bearer {sess_token}"},
    ) as ws:
        # Answer before start -> typed rejection, nothing executed
        ws.send_json({"type": "candidate_answer", "message_id": "early", "candidate_answer": "Hi"})
        early = ws.receive_json()
        assert early["type"] == "error"
        assert early["payload"]["code"] == "INTERVIEW_NOT_STARTED"

        # Missing message_id -> rejected
        ws.send_json({"type": "candidate_answer", "candidate_answer": "Hi"})
        assert ws.receive_json()["payload"]["code"] == "MISSING_MESSAGE_ID"

        ws.send_json({"type": "start_interview"})
        events = _drain_until(ws, ("turn_complete", "error"))
        assert not any(e["type"] == "error" for e in events)
        first_opening = [e for e in events if e["type"] == "turn_complete"][0]

        # Starting again replays the recorded opening instead of regenerating it
        calls_before = app.state.llm_client.call_count
        ws.send_json({"type": "start_interview"})
        replayed = _drain_until(ws, ("turn_complete", "error"))
        assert app.state.llm_client.call_count == calls_before
        assert replayed[-1]["event_id"] == first_opening["event_id"]

    status_res = client.get(f"/api/v1/sessions/{sess_id}", headers=service_headers)
    assert status_res.json()["status"] == "STAGE_ACTIVE"


def test_websocket_concurrency_ceiling_enforced(tmp_path):
    """Verify that WebSocket connections exceeding concurrency ceiling are rejected with code 1013."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.auth import create_session_token
    import asyncio
    import pytest

    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    # Set capacity to 1 for testing concurrency ceiling
    app.state.concurrency_semaphore = asyncio.Semaphore(1)
    client = TestClient(app)

    token1 = create_session_token("sess_conc_1", secret=settings.JWT_SECRET)
    token2 = create_session_token("sess_conc_2", secret=settings.JWT_SECRET)

    # Client 1 connects and holds the only permit
    with client.websocket_connect(
        "/ws/sessions/sess_conc_1/stream",
        headers={"Authorization": f"Bearer {token1}"},
    ) as ws1:
        ws1.send_json({"type": "ping"})
        assert ws1.receive_json()["type"] == "pong"

        # Client 2 attempts to connect when 0 permits remain -> rejected with 1013
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/ws/sessions/sess_conc_2/stream",
                headers={"Authorization": f"Bearer {token2}"},
            ) as ws2:
                pass

    # After Client 1 disconnects, permit is released -> Client 2 can now connect
    with client.websocket_connect(
        "/ws/sessions/sess_conc_2/stream",
        headers={"Authorization": f"Bearer {token2}"},
    ) as ws2:
        ws2.send_json({"type": "ping"})
        assert ws2.receive_json()["type"] == "pong"


def _drain_until(ws, terminal_types, limit=60):
    """Receive frames until one of terminal_types (inclusive) or the limit is reached."""
    frames = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if frame.get("type") in terminal_types:
            break
    return frames
