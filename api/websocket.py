"""WebSocket connection manager, connection lease lifecycle, event replay, and streaming endpoint."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from api.auth import AuthError, verify_service_token, verify_session_token
from api.schemas import CandidateAnswerMessage
from core.runtime.agent import sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType
from core.session.events import EventReplayer, SessionEventStore

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["websocket"])


class ConnectionLease:
    """Represents an active WebSocket connection lease for a specific session.

    Attributes:
        session_id: Target session identifier.
        websocket: Active WebSocket instance.
        lease_id: Unique identifier for this connection lease.
        connected_at: Unix timestamp when lease was granted.
        is_active: Whether lease is currently held and valid.
    """

    def __init__(self, session_id: str, websocket: Any) -> None:
        self.session_id = session_id
        self.websocket = websocket
        self.lease_id = f"lease_{uuid.uuid4().hex[:16]}"
        self.connected_at = time.time()
        self.is_active = True

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the underlying WebSocket connection and mark lease inactive."""
        self.is_active = False
        try:
            if hasattr(self.websocket, "close"):
                await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.debug("Error closing displaced websocket: %s", sanitize_secrets(str(e)))


class ConnectionManager:
    """Coordinates active WebSocket connection leases, event replay, and message dispatch.

    Enforces that at most one active connection lease is held per session. A newer
    connection immediately displaces and disconnects any existing lease.
    """

    def __init__(self, event_store: Optional[SessionEventStore] = None) -> None:
        self.event_store = event_store or SessionEventStore()
        self._leases: Dict[str, ConnectionLease] = {}
        self._lock = asyncio.Lock()

    def record_event(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Record an event to the session event store.

        Args:
            session_id: Target session identifier.
            event: Event to persist.

        Returns:
            LoopEvent: The recorded event.
        """
        return self.event_store.append(session_id, event)

    def get_events_after(
        self,
        session_id: str,
        last_sequence: Optional[int] = None,
        last_event_id: Optional[str] = None,
    ) -> List[LoopEvent]:
        """Retrieve events recorded strictly after a given sequence number or event_id.

        Args:
            session_id: Target session identifier.
            last_sequence: If provided, slice events strictly where sequence > last_sequence.
            last_event_id: If provided, slice events occurring after this event_id.

        Returns:
            List[LoopEvent]: Sliced event list.
        """
        all_events = self.event_store.get_events(session_id)
        if not all_events:
            return []

        # 1. Slicing by last_sequence takes priority if specified
        if last_sequence is not None:
            return [e for e in all_events if e.sequence > last_sequence]

        # 2. Slicing by last_event_id
        if last_event_id is not None:
            found_idx = -1
            for idx, evt in enumerate(all_events):
                if evt.event_id == last_event_id:
                    found_idx = idx
                    break

            if found_idx != -1:
                return all_events[found_idx + 1 :]
            # If unknown last_event_id, return full sequence
            return list(all_events)

        return list(all_events)

    async def acquire_lease(self, session_id: str, websocket: Any) -> ConnectionLease:
        """Acquire a connection lease for a session, displacing any existing active lease.

        Args:
            session_id: Target session identifier.
            websocket: Connecting WebSocket instance.

        Returns:
            ConnectionLease: Newly granted connection lease.
        """
        async with self._lock:
            existing = self._leases.get(session_id)
            if existing is not None and existing.is_active:
                logger.info(
                    "Displacing existing connection lease '%s' for session '%s'",
                    existing.lease_id,
                    session_id,
                )
                existing.is_active = False
                await existing.close(
                    code=4001,
                    reason="Displaced by newer connection lease",
                )

            new_lease = ConnectionLease(session_id=session_id, websocket=websocket)
            self._leases[session_id] = new_lease
            return new_lease

    async def release_lease(self, session_id: str, lease: ConnectionLease) -> None:
        """Release a connection lease if held by the requesting lease instance.

        Args:
            session_id: Target session identifier.
            lease: The lease instance being released.
        """
        async with self._lock:
            current = self._leases.get(session_id)
            if current is not None and current.lease_id == lease.lease_id:
                lease.is_active = False
                del self._leases[session_id]

    def get_active_lease(self, session_id: str) -> Optional[ConnectionLease]:
        """Get the current active lease for a session, if any."""
        lease = self._leases.get(session_id)
        if lease is not None and lease.is_active:
            return lease
        return None

    async def send_event(self, session_id: str, event: LoopEvent) -> bool:
        """Send an event to the currently active connection lease for the session.

        Args:
            session_id: Target session identifier.
            event: LoopEvent envelope to transmit.

        Returns:
            bool: True if transmitted, False if no active lease or send failed.
        """
        lease = self.get_active_lease(session_id)
        if lease is None:
            return False

        try:
            serialized = event.model_dump_json()
            await lease.websocket.send_text(serialized)
            return True
        except Exception as e:
            logger.warning(
                "Failed to send event to session '%s': %s",
                session_id,
                sanitize_secrets(str(e)),
            )
            return False


async def authenticate_websocket(websocket: WebSocket, session_id: str) -> bool:
    """Authenticate incoming WebSocket connection using header or initial auth frame.

    Security constraints strictly enforced:
    - Tokens MUST NOT be passed or accepted via query parameters.
    - Token values must never be logged.
    """
    # 1. Reject token in query parameters
    query_params = dict(websocket.query_params)
    if "token" in query_params or "jwt" in query_params or "access_token" in query_params:
        logger.warning("Rejected WebSocket connection attempt containing token in query parameters")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Tokens in query parameters are strictly forbidden",
        )
        return False

    # 2. Check Authorization header
    auth_header = websocket.headers.get("authorization", "")
    token: Optional[str] = None

    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    # 3. Check Sec-WebSocket-Protocol (e.g. bearer, <token>)
    if not token:
        sec_protocol = websocket.headers.get("sec-websocket-protocol", "")
        if sec_protocol:
            for part in sec_protocol.split(","):
                part = part.strip()
                if part.startswith("token."):
                    token = part[6:]
                    break
                elif part.startswith("Bearer "):
                    token = part[7:]
                    break

    # 4. If token found in headers, validate immediately before accept
    if token:
        try:
            # Check session token scoped to session_id
            verify_session_token(token, expected_session_id=session_id)
            return True
        except AuthError:
            try:
                # Fallback to service token
                verify_service_token(token)
                return True
            except AuthError:
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid authentication token",
                )
                return False

    # 5. Fallback: Accept connection and await initial {"type": "auth", "token": "..."} message
    await websocket.accept()
    try:
        raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        msg_data = json.loads(raw_msg)
        if msg_data.get("type") == "auth" and msg_data.get("token"):
            auth_token = str(msg_data["token"]).strip()
            try:
                verify_session_token(auth_token, expected_session_id=session_id)
                await websocket.send_text(json.dumps({"type": "auth_ok", "session_id": session_id}))
                return True
            except AuthError:
                try:
                    verify_service_token(auth_token)
                    await websocket.send_text(json.dumps({"type": "auth_ok", "session_id": session_id}))
                    return True
                except AuthError:
                    pass
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        pass

    await websocket.close(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Authentication timeout or invalid credentials",
    )
    return False


@ws_router.websocket("/ws/sessions/{session_id}/stream")
async def websocket_session_stream(websocket: WebSocket, session_id: str) -> None:
    """WebSocket streaming gateway endpoint for real-time interview turns and event replay.

    Replay semantics:
    - Client can supply `last_sequence` or `last_event_id` in initial message to receive missed events.
    - When `candidate_answer` message is received, dispatches to InteractiveSubagentRunner.
    """
    app = websocket.app
    connection_mgr: ConnectionManager = getattr(
        app.state,
        "connection_mgr",
        None,
    ) or ConnectionManager()

    # Authenticate
    is_authenticated = await authenticate_websocket(websocket, session_id)
    if not is_authenticated:
        return

    # If already accepted in step 5, websocket is open. If not, accept now.
    if websocket.client_state.name != "CONNECTED":
        await websocket.accept()

    # Acquire exclusive connection lease
    lease = await connection_mgr.acquire_lease(session_id, websocket)

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON frame"})
                )
                continue

            msg_type = payload.get("type", "")

            # Ping / keep-alive
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                continue

            # Replay request
            if msg_type == "replay":
                last_seq = payload.get("last_sequence")
                last_eid = payload.get("last_event_id")
                missed = connection_mgr.get_events_after(
                    session_id,
                    last_sequence=last_seq,
                    last_event_id=last_eid,
                )
                for evt in missed:
                    await websocket.send_text(evt.model_dump_json())
                continue

            # Candidate answer turn
            if msg_type in ("candidate_answer", "turn_submit"):
                candidate_msg = CandidateAnswerMessage.model_validate(payload)
                answer_text = candidate_msg.get_answer_text()
                vad_metrics = candidate_msg.get_vad_metrics()
                expected_turn = candidate_msg.get_expected_turn()

                # Replay any pending events if requested in frame
                if candidate_msg.last_sequence is not None or candidate_msg.last_event_id is not None:
                    missed = connection_mgr.get_events_after(
                        session_id,
                        last_sequence=candidate_msg.last_sequence,
                        last_event_id=candidate_msg.last_event_id,
                    )
                    for evt in missed:
                        await websocket.send_text(evt.model_dump_json())

                # Resolve runner for this session
                runner_resolver = getattr(app.state, "get_runner", None)
                if runner_resolver is None:
                    # Emit error event
                    err_event = LoopEvent(
                        schema_version="1.0",
                        event_id=f"evt_{uuid.uuid4().hex[:16]}",
                        session_id=session_id,
                        turn_id=f"turn_{(expected_turn or 1):03d}",
                        sequence=1,
                        timestamp=time.time(),
                        type=LoopEventType.ERROR,
                        payload={"error": "Interactive runner not configured for session", "code": "RUNNER_UNAVAILABLE"},
                    )
                    connection_mgr.record_event(session_id, err_event)
                    await websocket.send_text(err_event.model_dump_json())
                    continue

                runner = await runner_resolver(session_id)
                if runner is None:
                    err_event = LoopEvent(
                        schema_version="1.0",
                        event_id=f"evt_{uuid.uuid4().hex[:16]}",
                        session_id=session_id,
                        turn_id=f"turn_{(expected_turn or 1):03d}",
                        sequence=1,
                        timestamp=time.time(),
                        type=LoopEventType.ERROR,
                        payload={"error": f"Session '{session_id}' not ready or runner unavailable", "code": "NOT_READY"},
                    )
                    connection_mgr.record_event(session_id, err_event)
                    await websocket.send_text(err_event.model_dump_json())
                    continue

                # Stream turn execution
                try:
                    async for event in runner.run_turn(
                        session_id=session_id,
                        candidate_answer=answer_text,
                        vad_metrics=vad_metrics,
                        expected_turn=expected_turn,
                    ):
                        connection_mgr.record_event(session_id, event)
                        # Send only if lease is still active
                        if lease.is_active:
                            await websocket.send_text(event.model_dump_json())
                        else:
                            break
                except Exception as e:
                    logger.error("Error executing turn for '%s': %s", session_id, sanitize_secrets(str(e)))
                    err_evt = LoopEvent(
                        schema_version="1.0",
                        event_id=f"evt_{uuid.uuid4().hex[:16]}",
                        session_id=session_id,
                        turn_id=f"turn_{(expected_turn or 1):03d}",
                        sequence=999,
                        timestamp=time.time(),
                        type=LoopEventType.ERROR,
                        payload={"error": sanitize_secrets(str(e)), "code": "TURN_ERROR"},
                    )
                    connection_mgr.record_event(session_id, err_evt)
                    if lease.is_active:
                        await websocket.send_text(err_evt.model_dump_json())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for session '%s'", session_id)
    except Exception as e:
        logger.warning("WebSocket error for session '%s': %s", session_id, sanitize_secrets(str(e)))
    finally:
        await connection_mgr.release_lease(session_id, lease)
