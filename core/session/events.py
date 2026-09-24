"""Session event store and replay primitives with monotonic sequence validation."""

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

from core.runtime.agent import sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType

logger = logging.getLogger(__name__)


class SessionEventError(Exception):
    """Base exception for session event store errors."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


class EventNotFoundError(SessionEventError):
    """Raised when an event ID cannot be found in the event store."""


class SequenceOrderingError(SessionEventError, ValueError):
    """Raised when an explicit event sequence violates monotonic ordering."""


class SessionEventStore:
    """In-memory and Redis-compatible append-only event log with monotonic sequence numbers.

    Attributes:
        redis_client: Optional redis-compatible client providing lpush/rpush/lrange.
        redis_key_prefix: Prefix used for Redis keys (default 'session:events:').
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        redis_key_prefix: str = "session:events:",
    ) -> None:
        self._redis = redis_client
        self._prefix = redis_key_prefix
        self._events: Dict[str, List[LoopEvent]] = defaultdict(list)
        self._sequences: Dict[str, int] = defaultdict(int)

    def append(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Append an event to the session event log, enforcing strictly monotonic sequence numbers.

        Args:
            session_id: Target session identifier.
            event: LoopEvent envelope to append.

        Returns:
            LoopEvent: The persisted LoopEvent with validated monotonic sequence.

        Raises:
            SequenceOrderingError: If explicit sequence <= current sequence for session.
        """
        current_seq = self._sequences[session_id]

        if event.sequence <= 0:
            # Auto-assign next monotonic sequence number
            next_seq = current_seq + 1
            event.sequence = next_seq
        else:
            # Explicit sequence provided: must be strictly greater than current_seq
            if event.sequence <= current_seq:
                raise SequenceOrderingError(
                    f"Event sequence {event.sequence} is non-monotonic for session "
                    f"'{session_id}'; current sequence is {current_seq}"
                )

        # Ensure session_id is consistent
        if event.session_id != session_id:
            event.session_id = session_id

        # Update in-memory sequence counter and append
        self._sequences[session_id] = event.sequence
        self._events[session_id].append(event)

        # Synchronize with Redis if client configured
        if self._redis is not None:
            try:
                redis_key = f"{self._prefix}{session_id}"
                serialized = event.model_dump_json()
                if hasattr(self._redis, "rpush"):
                    res = self._redis.rpush(redis_key, serialized)
                    if hasattr(res, "__await__"):
                        try:
                            import asyncio
                            loop = asyncio.get_running_loop()
                            loop.create_task(res)
                        except RuntimeError:
                            pass
            except Exception as e:
                logger.warning(
                    f"Failed to persist event to Redis: {sanitize_secrets(str(e))}"
                )

        return event

    async def append_async(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Asynchronously append an event to the session event log, awaiting Redis persistence."""
        current_seq = self._sequences[session_id]

        if event.sequence <= 0:
            next_seq = current_seq + 1
            event.sequence = next_seq
        else:
            if event.sequence <= current_seq:
                raise SequenceOrderingError(
                    f"Event sequence {event.sequence} is non-monotonic for session "
                    f"'{session_id}'; current sequence is {current_seq}"
                )

        if event.session_id != session_id:
            event.session_id = session_id

        self._sequences[session_id] = event.sequence
        self._events[session_id].append(event)
        serialized = event.model_dump_json()
        object.__setattr__(event, "_cached_json", serialized)

        if self._redis is not None and hasattr(self._redis, "rpush"):
            try:
                redis_key = f"{self._prefix}{session_id}"
                res = self._redis.rpush(redis_key, serialized)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:
                logger.warning(
                    f"Failed to persist event to Redis: {sanitize_secrets(str(e))}"
                )
        return event

    def append_event(
        self,
        session_id: str,
        turn_id: str,
        event_type: Union[str, LoopEventType],
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> LoopEvent:
        """Construct and append a new LoopEvent with auto-assigned sequence and timestamp.

        Args:
            session_id: Target session identifier.
            turn_id: Current turn identifier.
            event_type: Classifier from LoopEventType or string.
            payload: Structured event payload dictionary.
            event_id: Optional custom event identifier.

        Returns:
            LoopEvent: The newly appended LoopEvent.
        """
        typed_event_type = (
            LoopEventType(event_type)
            if isinstance(event_type, str)
            else event_type
        )
        eid = event_id or f"evt_{uuid.uuid4().hex[:16]}"
        next_seq = self._sequences[session_id] + 1

        event = LoopEvent(
            schema_version="1.0",
            event_id=eid,
            session_id=session_id,
            turn_id=turn_id,
            sequence=next_seq,
            timestamp=time.time(),
            type=typed_event_type,
            payload=payload or {},
        )
        return self.append(session_id, event)

    async def load_from_redis_async(self, session_id: str) -> List[LoopEvent]:
        """Asynchronously load historical events from Redis into in-memory store."""
        if self._redis is not None and hasattr(self._redis, "lrange"):
            try:
                redis_key = f"{self._prefix}{session_id}"
                res = self._redis.lrange(redis_key, 0, -1)
                raw_items = await res if hasattr(res, "__await__") else res
                loaded = [LoopEvent.model_validate_json(item) for item in raw_items]
                self._events[session_id] = loaded
                if loaded:
                    self._sequences[session_id] = max(e.sequence for e in loaded)
            except Exception as e:
                logger.warning(
                    f"Failed to load events from Redis: {sanitize_secrets(str(e))}"
                )
        return list(self._events.get(session_id, []))

    def get_events(self, session_id: str) -> List[LoopEvent]:
        """Retrieve all recorded events for a session in sequence order."""
        if self._redis is not None and not self._events.get(session_id):
            try:
                redis_key = f"{self._prefix}{session_id}"
                if hasattr(self._redis, "lrange"):
                    raw_items = self._redis.lrange(redis_key, 0, -1)
                    if not hasattr(raw_items, "__await__"):
                        loaded = [LoopEvent.model_validate_json(item) for item in raw_items]
                        self._events[session_id] = loaded
                        if loaded:
                            self._sequences[session_id] = max(e.sequence for e in loaded)
            except Exception as e:
                logger.warning(
                    f"Failed to read events from Redis: {sanitize_secrets(str(e))}"
                )

        return list(self._events.get(session_id, []))

    def get_event_by_id(self, session_id: str, event_id: str) -> Optional[LoopEvent]:
        """Find a single event by event_id within a session."""
        for evt in self._events.get(session_id, []):
            if evt.event_id == event_id:
                return evt
        return None

    def get_latest_sequence(self, session_id: str) -> int:
        """Return the highest recorded sequence number for a session (0 if empty)."""
        return self._sequences.get(session_id, 0)

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear recorded events for a specific session or all sessions."""
        if session_id:
            self._events.pop(session_id, None)
            self._sequences.pop(session_id, None)
        else:
            self._events.clear()
            self._sequences.clear()


class EventReplayer:
    """Replays and slices stored session events for streaming recovery and reconnects."""

    def __init__(self, store: SessionEventStore) -> None:
        self.store = store

    def replay(
        self,
        session_id: str,
        from_event_id: Optional[str] = None,
        from_sequence: Optional[int] = None,
        to_sequence: Optional[int] = None,
        event_types: Optional[List[Union[str, LoopEventType]]] = None,
        inclusive: bool = False,
        raise_if_not_found: bool = True,
    ) -> List[LoopEvent]:
        """Replay events for a session with slicing and filtering.

        Args:
            session_id: Target session identifier.
            from_event_id: Slice after this event_id (SSE Last-Event-ID resume semantics).
            from_sequence: Slice events starting at or after this sequence number.
            to_sequence: Slice events up to and including this sequence number.
            event_types: Optional list of event types to filter by.
            inclusive: If True and from_event_id given, include the from_event_id itself.
            raise_if_not_found: If True, raise EventNotFoundError if from_event_id is missing.

        Returns:
            List[LoopEvent]: Sliced and filtered list of events.

        Raises:
            EventNotFoundError: If from_event_id is not found and raise_if_not_found is True.
        """
        all_events = self.store.get_events(session_id)
        if not all_events:
            if from_event_id and raise_if_not_found:
                raise EventNotFoundError(
                    f"Event ID '{from_event_id}' not found in empty session '{session_id}'"
                )
            return []

        # 1. Slice by from_event_id if requested
        if from_event_id is not None:
            found_idx = -1
            for idx, evt in enumerate(all_events):
                if evt.event_id == from_event_id:
                    found_idx = idx
                    break

            if found_idx == -1:
                if raise_if_not_found:
                    raise EventNotFoundError(
                        f"Event ID '{from_event_id}' not found in session '{session_id}'"
                    )
            else:
                start_offset = found_idx if inclusive else found_idx + 1
                all_events = all_events[start_offset:]

        # 2. Slice by from_sequence
        if from_sequence is not None:
            all_events = [e for e in all_events if e.sequence >= from_sequence]

        # 3. Slice by to_sequence
        if to_sequence is not None:
            all_events = [e for e in all_events if e.sequence <= to_sequence]

        # 4. Filter by event_types
        if event_types is not None:
            allowed_types: Set[str] = {
                t.value if isinstance(t, LoopEventType) else str(t)
                for t in event_types
            }
            all_events = [
                e
                for e in all_events
                if (e.type.value if isinstance(e.type, LoopEventType) else str(e.type))
                in allowed_types
            ]

        return all_events

    async def replay_stream(
        self,
        session_id: str,
        from_event_id: Optional[str] = None,
        from_sequence: Optional[int] = None,
        to_sequence: Optional[int] = None,
        event_types: Optional[List[Union[str, LoopEventType]]] = None,
        inclusive: bool = False,
    ) -> AsyncIterator[LoopEvent]:
        """Asynchronously stream sliced events for SSE or WebSocket reconnection playback."""
        events = self.replay(
            session_id=session_id,
            from_event_id=from_event_id,
            from_sequence=from_sequence,
            to_sequence=to_sequence,
            event_types=event_types,
            inclusive=inclusive,
        )
        for evt in events:
            yield evt
