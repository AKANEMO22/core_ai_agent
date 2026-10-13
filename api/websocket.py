"""WebSocket gateway: connection leases, event publishing, replay/resync and turn dispatch."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from api.auth import AuthError, verify_service_token, verify_session_token
from api.schemas import CandidateAnswerMessage
from core.observability import session_ref
from core.runtime.agent import sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType
from core.session.events import SessionEventStore

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["websocket"])

# Strong references to turn tasks: a turn keeps running after its connection drops.
_INFLIGHT_TASKS: Set[asyncio.Task] = set()


class ConnectionLease:
    """An active WebSocket connection lease for a session.

    Sends are serialized with a lock because turn streaming, background events (assessments,
    directives, report) and control replies may write to the same socket concurrently.
    """

    def __init__(self, session_id: str, websocket: Any) -> None:
        self.session_id = session_id
        self.websocket = websocket
        self.lease_id = f"lease_{uuid.uuid4().hex[:16]}"
        self.connected_at = time.time()
        self.is_active = True
        self._send_lock = asyncio.Lock()

    async def send_text(self, text: str) -> bool:
        """Send one frame if the lease is still active. Returns False on failure."""
        if not self.is_active:
            return False
        async with self._send_lock:
            try:
                await self.websocket.send_text(text)
                return True
            except Exception as e:
                logger.debug("WebSocket send failed: %s", sanitize_secrets(str(e)))
                return False

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the underlying WebSocket connection and mark lease inactive."""
        self.is_active = False
        try:
            if hasattr(self.websocket, "close"):
                await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.debug("Error closing displaced websocket: %s", sanitize_secrets(str(e)))


@dataclass
class ReplayResult:
    """Events to resend after a reconnect, or a request to resynchronize from a snapshot."""

    events: List[LoopEvent] = field(default_factory=list)
    resync_required: bool = False


class ConnectionManager:
    """Owns connection leases and is the single publisher of session events.

    `publish` records an event in the replayable store and pushes it to the session's active
    lease. At most one lease is active per session; a newer connection displaces the older one.
    """

    def __init__(self, event_store: Optional[SessionEventStore] = None) -> None:
        self.event_store = event_store or SessionEventStore()
        self._leases: Dict[str, ConnectionLease] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Recording & publishing
    # ------------------------------------------------------------------

    def record_event(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Synchronously record an event with a store-assigned monotonic sequence."""
        event.sequence = 0
        return self.event_store.append(session_id, event)

    async def record_event_async(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Record an event with a store-assigned monotonic sequence (Redis INCR when configured).

        Producer-local sequences (e.g. per-turn agent counters) are discarded so that the
        session stream has a single monotonic order.
        """
        event.sequence = 0
        return await self.event_store.append_async(session_id, event)

    async def publish(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Record an event and push it to the active lease, if any."""
        recorded = await self.record_event_async(session_id, event)
        await self.send_event(session_id, recorded)
        return recorded

    async def send_event(self, session_id: str, event: LoopEvent) -> bool:
        """Push an event to the session's active lease without recording it."""
        lease = self.get_active_lease(session_id)
        if lease is None:
            return False
        payload_json = getattr(event, "_cached_json", None) or event.model_dump_json()
        return await lease.send_text(payload_json)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def get_events_after(
        self,
        session_id: str,
        last_sequence: Optional[int] = None,
        last_event_id: Optional[str] = None,
    ) -> List[LoopEvent]:
        """Return cached events strictly after a sequence number or event_id."""
        return self._slice(self.event_store.get_events(session_id), last_sequence, last_event_id)

    async def get_events_after_async(
        self,
        session_id: str,
        last_sequence: Optional[int] = None,
        last_event_id: Optional[str] = None,
    ) -> List[LoopEvent]:
        """Return events strictly after a cursor, reloading from Redis when the cache is cold."""
        events = await self.event_store.get_events_async(session_id)
        return self._slice(events, last_sequence, last_event_id)

    @staticmethod
    def _slice(
        all_events: List[LoopEvent],
        last_sequence: Optional[int],
        last_event_id: Optional[str],
    ) -> List[LoopEvent]:
        if not all_events:
            return []
        if last_sequence is not None:
            return [e for e in all_events if e.sequence > last_sequence]
        if last_event_id is not None:
            for idx, evt in enumerate(all_events):
                if evt.event_id == last_event_id:
                    return all_events[idx + 1 :]
            return list(all_events)
        return list(all_events)

    async def replay(
        self,
        session_id: str,
        last_sequence: Optional[int] = None,
        last_event_id: Optional[str] = None,
    ) -> ReplayResult:
        """Compute what a reconnecting client needs.

        `resync_required` is set when the cursor points before the oldest retained event
        (trimmed or expired), so a gap-free replay is impossible.
        """
        events = await self.event_store.get_events_async(session_id)
        latest = await self.event_store.get_latest_sequence_async(session_id)
        if last_sequence is not None:
            if last_sequence > 0 and not events and latest >= last_sequence:
                return ReplayResult(resync_required=True)
            if events and last_sequence < events[0].sequence - 1:
                return ReplayResult(resync_required=True)
            return ReplayResult(events=self._slice(events, last_sequence, None))
        if last_event_id is not None:
            if events and not any(e.event_id == last_event_id for e in events):
                return ReplayResult(resync_required=True)
            return ReplayResult(events=self._slice(events, None, last_event_id))
        return ReplayResult(events=list(events))

    # ------------------------------------------------------------------
    # Leases
    # ------------------------------------------------------------------

    async def acquire_lease(self, session_id: str, websocket: Any) -> ConnectionLease:
        """Acquire a connection lease for a session, displacing any existing active lease."""
        async with self._lock:
            existing = self._leases.get(session_id)
            if existing is not None and existing.is_active:
                logger.info("Displacing connection lease for session %s", session_ref(session_id))
                existing.is_active = False
                await existing.close(code=4001, reason="Displaced by newer connection lease")
            new_lease = ConnectionLease(session_id=session_id, websocket=websocket)
            self._leases[session_id] = new_lease
            return new_lease

    async def release_lease(self, session_id: str, lease: ConnectionLease) -> None:
        """Release a connection lease if held by the requesting lease instance."""
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

    def active_connection_count(self) -> int:
        """Number of active leases (for metrics)."""
        return sum(1 for lease in self._leases.values() if lease.is_active)

    async def close_session(self, session_id: str, code: int = 4002, reason: str = "Session closed") -> None:
        """Close and drop the active lease of a session (cancellation, deletion)."""
        async with self._lock:
            lease = self._leases.pop(session_id, None)
        if lease is not None:
            await lease.close(code=code, reason=reason)


async def authenticate_websocket(websocket: WebSocket, session_id: str) -> bool:
    """Authenticate the WebSocket via Authorization header, subprotocol, or an initial auth frame.

    Security constraints:
    - Tokens are never accepted via query parameters.
    - Token values are never logged.
    - Revoked (deleted) sessions are rejected.
    """
    query_params = dict(websocket.query_params)
    if "token" in query_params or "jwt" in query_params or "access_token" in query_params:
        logger.warning("Rejected WebSocket connection attempt containing token in query parameters")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Tokens in query parameters are strictly forbidden",
        )
        return False

    revocations = getattr(websocket.app.state, "revocations", None)
    if revocations is not None and await revocations.is_revoked(session_id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Session revoked")
        return False

    def _verify(raw_token: str) -> bool:
        try:
            verify_session_token(raw_token, expected_session_id=session_id)
            return True
        except AuthError:
            try:
                verify_service_token(raw_token)
                return True
            except AuthError:
                return False

    token: Optional[str] = None
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    if not token:
        sec_protocol = websocket.headers.get("sec-websocket-protocol", "")
        for part in sec_protocol.split(","):
            part = part.strip()
            if part.startswith("token."):
                token = part[6:]
                break
            if part.startswith("Bearer "):
                token = part[7:]
                break

    if token:
        if _verify(token):
            return True
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid authentication token")
        return False

    # Fallback: accept and await an initial {"type": "auth", "token": "..."} frame.
    await websocket.accept()
    try:
        raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        msg_data = json.loads(raw_msg)
        if msg_data.get("type") == "auth" and msg_data.get("token"):
            if _verify(str(msg_data["token"]).strip()):
                await websocket.send_text(json.dumps({"type": "auth_ok", "session_id": session_id}))
                return True
    except Exception:
        pass

    await websocket.close(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Authentication timeout or invalid credentials",
    )
    return False


def _control_frame(session_id: str, event_type: LoopEventType, payload: Dict[str, Any], turn_id: str = "control") -> str:
    """Serialize an unrecorded, connection-local event."""
    return LoopEvent(
        schema_version="1.0",
        event_id=f"evt_{uuid.uuid4().hex[:16]}",
        session_id=session_id,
        turn_id=turn_id,
        sequence=0,
        timestamp=time.time(),
        type=event_type,
        payload=payload,
    ).model_dump_json()


async def _resync_snapshot(app: Any, session_id: str) -> Dict[str, Any]:
    """Latest durable turn snapshot sent with resync_required."""
    snapshot: Dict[str, Any] = {}
    state_store = getattr(app.state, "state_mgr", None)
    state = await state_store.aget(session_id) if state_store is not None else None
    if state is not None:
        snapshot.update(
            {
                "status": state.status.value,
                "state_version": state.state_version,
                "current_turn": state.current_turn,
                "current_stage_id": state.current_stage_id,
            }
        )
        workspace_mgr = getattr(app.state, "workspace_mgr", None)
        if workspace_mgr is not None:
            ws = workspace_mgr.get_or_create_workspace(session_id)
            rel = f"turns/turn_{state.current_turn}_interviewer.json"
            if ws.exists(rel):
                snapshot["last_interviewer_response"] = ws.read_json(rel).get("content")
    latest = await app.state.connection_mgr.event_store.get_latest_sequence_async(session_id)
    snapshot["latest_sequence"] = latest
    return snapshot


@ws_router.websocket("/ws/sessions/{session_id}/stream")
async def websocket_session_stream(websocket: WebSocket, session_id: str) -> None:
    """Real-time interview stream.

    Inbound message types: ping, replay, ack, start_interview, candidate_answer (alias
    turn_submit, requires message_id), cancel_generation.

    A disconnect does not cancel an in-flight turn: its events keep being recorded and are
    delivered to the next lease or by replay.
    """
    app = websocket.app
    connection_mgr: ConnectionManager = app.state.connection_mgr
    controller = getattr(app.state, "controller", None)
    settings = app.state.settings

    if not await authenticate_websocket(websocket, session_id):
        return

    reconnect_limiter = getattr(app.state, "reconnect_limiter", None)
    if reconnect_limiter is not None and not reconnect_limiter.acquire(session_id):
        # Closing before accept rejects the handshake (HTTP 403) when auth used headers.
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Reconnect rate limit exceeded")
        return

    semaphore = getattr(app.state, "concurrency_semaphore", None)
    semaphore_acquired = False
    if semaphore is not None:
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.5)
            semaphore_acquired = True
        except (asyncio.TimeoutError, Exception):
            logger.warning("Rejected WebSocket for session %s: connection ceiling reached", session_ref(session_id))
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Max concurrent sessions limit reached")
            return

    if websocket.client_state.name != "CONNECTED":
        await websocket.accept()

    lease = await connection_mgr.acquire_lease(session_id, websocket)
    metrics = getattr(app.state, "metrics", None)
    if metrics is not None:
        metrics.set_gauge("active_connections", connection_mgr.active_connection_count())

    async def _send_result(result: Any) -> None:
        for evt in result.replay_events:
            await lease.send_text(evt.model_dump_json())
        error_event = result.to_event(session_id)
        if error_event is not None:
            await lease.send_text(error_event.model_dump_json())

    async def _run_and_reply(coro: Any) -> None:
        try:
            await _send_result(await coro)
        except Exception as e:
            logger.error("Turn task failed for session %s: %s", session_ref(session_id), sanitize_secrets(str(e)))
            await lease.send_text(
                _control_frame(session_id, LoopEventType.ERROR, {"code": "INTERNAL_ERROR", "retryable": True})
            )

    def _spawn(coro: Any) -> None:
        task = asyncio.create_task(_run_and_reply(coro))
        _INFLIGHT_TASKS.add(task)
        task.add_done_callback(_INFLIGHT_TASKS.discard)

    async def _send_replay(last_sequence: Optional[int], last_event_id: Optional[str]) -> None:
        replay = await connection_mgr.replay(session_id, last_sequence, last_event_id)
        if replay.resync_required:
            await lease.send_text(
                _control_frame(session_id, LoopEventType.RESYNC_REQUIRED, await _resync_snapshot(app, session_id))
            )
            return
        for evt in replay.events:
            await lease.send_text(getattr(evt, "_cached_json", None) or evt.model_dump_json())

    try:
        while True:
            raw_text = await websocket.receive_text()
            if len(raw_text) > settings.MAX_INBOUND_FRAME_BYTES:
                await lease.send_text(_control_frame(session_id, LoopEventType.ERROR, {"code": "FRAME_TOO_LARGE"}))
                continue
            try:
                payload = json.loads(raw_text)
                if not isinstance(payload, dict):
                    raise ValueError("frame must be an object")
            except (json.JSONDecodeError, ValueError):
                await lease.send_text(json.dumps({"type": "error", "code": "INVALID_JSON", "message": "Invalid JSON frame"}))
                continue

            msg_type = payload.get("type", "")

            if msg_type == "ping":
                await lease.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
            elif msg_type == "ack":
                continue
            elif msg_type == "replay":
                await _send_replay(payload.get("last_sequence"), payload.get("last_event_id"))
            elif msg_type == "start_interview":
                if controller is None:
                    await lease.send_text(_control_frame(session_id, LoopEventType.ERROR, {"code": "RUNNER_UNAVAILABLE"}))
                    continue
                _spawn(controller.start_interview(session_id))
            elif msg_type in ("candidate_answer", "turn_submit"):
                try:
                    candidate_msg = CandidateAnswerMessage.model_validate(payload)
                except ValidationError:
                    await lease.send_text(_control_frame(session_id, LoopEventType.ERROR, {"code": "INVALID_MESSAGE"}))
                    continue
                if candidate_msg.last_sequence is not None or candidate_msg.last_event_id is not None:
                    await _send_replay(candidate_msg.last_sequence, candidate_msg.last_event_id)
                answer_text = candidate_msg.get_answer_text()
                if len(answer_text) > settings.MAX_ANSWER_CHARS:
                    await lease.send_text(_control_frame(session_id, LoopEventType.ERROR, {"code": "ANSWER_TOO_LONG"}))
                    continue
                if controller is None:
                    await lease.send_text(_control_frame(session_id, LoopEventType.ERROR, {"code": "RUNNER_UNAVAILABLE"}))
                    continue
                _spawn(
                    controller.submit_answer(
                        session_id=session_id,
                        message_id=candidate_msg.get_message_id(),
                        answer_text=answer_text,
                        vad_metrics=candidate_msg.get_vad_metrics(),
                        expected_turn=candidate_msg.get_expected_turn(),
                    )
                )
            elif msg_type == "cancel_generation":
                if controller is None or not controller.cancel_generation(session_id):
                    await lease.send_text(
                        _control_frame(session_id, LoopEventType.ERROR, {"code": "NO_ACTIVE_GENERATION"})
                    )
            else:
                await lease.send_text(
                    _control_frame(session_id, LoopEventType.ERROR, {"code": "UNKNOWN_MESSAGE_TYPE"})
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for session %s", session_ref(session_id))
    except Exception as e:
        logger.warning("WebSocket error for session %s: %s", session_ref(session_id), sanitize_secrets(str(e)))
    finally:
        try:
            await connection_mgr.release_lease(session_id, lease)
        finally:
            if semaphore_acquired and semaphore is not None:
                semaphore.release()
            if metrics is not None:
                metrics.set_gauge("active_connections", connection_mgr.active_connection_count())
