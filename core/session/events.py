"""Nhật ký LoopEvent theo phiên và công cụ phát lại cho kết nối bị gián đoạn.

Publisher ghi sự kiện từ runner/runtime; API streaming dùng EventReplayer để lấy
phần lịch sử sau event_id hoặc sequence của client. Đầu vào là session_id và event,
đầu ra là event đã gán sequence hoặc danh sách/luồng event được lọc.
Khi có Redis, append_async cấp sequence bằng INCR; lịch sử vẫn có giới hạn độ dài
và TTL nên replay chỉ bao phủ các sự kiện còn được lưu, không phải toàn bộ vĩnh viễn.
"""

import asyncio
import logging
import time
import uuid
from collections import OrderedDict, defaultdict
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
    """Append-only session event log with monotonic sequences, Redis persistence and retention.

    Redis layout (when a client is configured):
        {prefix}{session_id}        LIST of serialized LoopEvents (newest last), trimmed to max_length.
        {prefix}{session_id}:seq    INCR counter assigning monotonic sequences per session,
                                    so ordering survives process restarts.
    Both keys carry `ttl_seconds` and are refreshed on every append.

    The in-memory cache keeps at most `memory_max_sessions` sessions (LRU) and `max_length`
    events per session. With Redis, evicted sessions reload on demand through the async
    readers; without Redis the cache is the only copy and sessions are never evicted.

    Attributes:
        ttl_seconds: Retention for replayable events.
        max_length: Maximum retained events per session.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        redis_key_prefix: str = "session:events:",
        ttl_seconds: int = 86400,
        max_length: int = 20000,
        memory_max_sessions: int = 1000,
    ) -> None:
        self._redis = redis_client
        self._prefix = redis_key_prefix
        self.ttl_seconds = ttl_seconds
        self.max_length = max_length
        self.memory_max_sessions = memory_max_sessions
        self._events: "OrderedDict[str, List[LoopEvent]]" = OrderedDict()
        self._sequences: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _seq_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}:seq"

    def _cache_append(self, session_id: str, event: LoopEvent) -> None:
        bucket = self._events.get(session_id)
        if bucket is None:
            bucket = []
            self._events[session_id] = bucket
        bucket.append(event)
        if len(bucket) > self.max_length:
            del bucket[: len(bucket) - self.max_length]
        self._events.move_to_end(session_id)
        self._sequences[session_id] = max(self._sequences[session_id], event.sequence)
        # Chỉ loại cache theo LRU khi còn bản Redis; RAM-only cần giữ bản duy nhất.
        if self._redis is not None:
            while len(self._events) > self.memory_max_sessions:
                evicted, _ = self._events.popitem(last=False)
                self._sequences.pop(evicted, None)

    @staticmethod
    def _check_explicit_sequence(session_id: str, sequence: int, current: int) -> None:
        if sequence <= current:
            raise SequenceOrderingError(
                f"Event sequence {sequence} is non-monotonic for session "
                f"'{session_id}'; current sequence is {current}"
            )

    @staticmethod
    def _build(
        session_id: str,
        turn_id: str,
        event_type: Union[str, LoopEventType],
        payload: Optional[Dict[str, Any]],
        event_id: Optional[str],
    ) -> LoopEvent:
        typed_event_type = LoopEventType(event_type) if isinstance(event_type, str) else event_type
        return LoopEvent(
            schema_version="1.0",
            event_id=event_id or f"evt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=turn_id,
            sequence=0,
            timestamp=time.time(),
            type=typed_event_type,
            payload=payload or {},
        )

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def append(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Synchronously append an event to the in-memory log (tests and Redis-less mode).

        A non-positive sequence is auto-assigned; an explicit one must be strictly greater
        than the current sequence. With Redis configured, production code must use
        `append_async`, which assigns sequences atomically in Redis; this method then
        persists best-effort only.

        Raises:
            SequenceOrderingError: If an explicit sequence is not monotonic.
        """
        current_seq = self._sequences[session_id]
        if event.sequence <= 0:
            event.sequence = current_seq + 1
        else:
            self._check_explicit_sequence(session_id, event.sequence, current_seq)
        if event.session_id != session_id:
            event.session_id = session_id
        self._cache_append(session_id, event)

        if self._redis is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist(session_id, event, set_sequence=True))
            except RuntimeError:
                pass
        return event

    async def append_async(self, session_id: str, event: LoopEvent) -> LoopEvent:
        """Append an event, assigning its sequence atomically in Redis when configured.

        Raises:
            SequenceOrderingError: If an explicit sequence is not monotonic.
        """
        if event.session_id != session_id:
            event.session_id = session_id

        if self._redis is not None and session_id not in self._events:
            # Nạp lịch sử Redis trước append để cache sau restart không chỉ chứa sự kiện mới.
            await self.load_from_redis_async(session_id)

        if self._redis is None:
            current_seq = self._sequences[session_id]
            if event.sequence <= 0:
                event.sequence = current_seq + 1
            else:
                self._check_explicit_sequence(session_id, event.sequence, current_seq)
        elif event.sequence <= 0:
            event.sequence = int(await self._redis.incr(self._seq_key(session_id)))
        else:
            current = int(await self._redis.get(self._seq_key(session_id)) or 0)
            self._check_explicit_sequence(session_id, event.sequence, current)
            await self._redis.set(self._seq_key(session_id), event.sequence, ex=self.ttl_seconds)

        # Serialize sau khi có sequence để payload phát đi và bản lưu dùng cùng thứ tự.
        serialized = event.model_dump_json()
        object.__setattr__(event, "_cached_json", serialized)
        self._cache_append(session_id, event)
        if self._redis is not None:
            await self._persist(session_id, event, serialized=serialized)
        return event

    async def _persist(
        self,
        session_id: str,
        event: LoopEvent,
        serialized: Optional[str] = None,
        set_sequence: bool = False,
    ) -> None:
        """Push the event to the Redis list, trim it and refresh retention TTLs."""
        try:
            payload = serialized or event.model_dump_json()
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(self._list_key(session_id), payload)
                pipe.ltrim(self._list_key(session_id), -self.max_length, -1)
                pipe.expire(self._list_key(session_id), self.ttl_seconds)
                if set_sequence:
                    pipe.set(self._seq_key(session_id), event.sequence, ex=self.ttl_seconds)
                else:
                    pipe.expire(self._seq_key(session_id), self.ttl_seconds)
                await pipe.execute()
        except Exception as e:
            logger.warning("Failed to persist event to Redis: %s", sanitize_secrets(str(e)))

    def append_event(
        self,
        session_id: str,
        turn_id: str,
        event_type: Union[str, LoopEventType],
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> LoopEvent:
        """Construct and synchronously append a LoopEvent with an auto-assigned sequence."""
        return self.append(session_id, self._build(session_id, turn_id, event_type, payload, event_id))

    async def append_event_async(
        self,
        session_id: str,
        turn_id: str,
        event_type: Union[str, LoopEventType],
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> LoopEvent:
        """Construct and asynchronously append a LoopEvent with an auto-assigned sequence."""
        return await self.append_async(
            session_id, self._build(session_id, turn_id, event_type, payload, event_id)
        )

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    async def load_from_redis_async(self, session_id: str) -> List[LoopEvent]:
        """Reload a session's retained events from Redis into the cache."""
        if self._redis is not None:
            try:
                raw_items = await self._redis.lrange(self._list_key(session_id), 0, -1)
                loaded = [LoopEvent.model_validate_json(item) for item in raw_items]
                self._events[session_id] = loaded[-self.max_length :]
                self._events.move_to_end(session_id)
                if loaded:
                    self._sequences[session_id] = max(e.sequence for e in loaded)
            except Exception as e:
                logger.warning("Failed to load events from Redis: %s", sanitize_secrets(str(e)))
        return list(self._events.get(session_id, []))

    def get_events(self, session_id: str) -> List[LoopEvent]:
        """Return cached events for a session in sequence order."""
        return list(self._events.get(session_id, []))

    async def get_events_async(self, session_id: str) -> List[LoopEvent]:
        """Return events for a session, reloading from Redis when the cache is cold."""
        # Cache đã được nạp từ Redis trước append; replay lạnh mới cần tải lại lịch sử.
        if session_id not in self._events and self._redis is not None:
            return await self.load_from_redis_async(session_id)
        return self.get_events(session_id)

    async def get_turn_events(self, session_id: str, turn_id: str) -> List[LoopEvent]:
        """Return the retained events recorded for one turn."""
        return [e for e in await self.get_events_async(session_id) if e.turn_id == turn_id]

    def get_event_by_id(self, session_id: str, event_id: str) -> Optional[LoopEvent]:
        """Find a single cached event by event_id within a session."""
        for evt in self._events.get(session_id, []):
            if evt.event_id == event_id:
                return evt
        return None

    def get_latest_sequence(self, session_id: str) -> int:
        """Return the highest cached sequence number for a session (0 if empty)."""
        return self._sequences.get(session_id, 0)

    async def get_latest_sequence_async(self, session_id: str) -> int:
        """Return the highest assigned sequence, consulting Redis when configured."""
        if self._redis is not None:
            raw = await self._redis.get(self._seq_key(session_id))
            if raw is not None:
                return int(raw)
        return self.get_latest_sequence(session_id)

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear cached events for a specific session or all sessions."""
        if session_id:
            self._events.pop(session_id, None)
            self._sequences.pop(session_id, None)
        else:
            self._events.clear()
            self._sequences.clear()

    async def purge(self, session_id: str) -> None:
        """Delete a session's events from the cache and Redis (retention deletion)."""
        self.clear(session_id)
        if self._redis is not None:
            await self._redis.delete(self._list_key(session_id), self._seq_key(session_id))


class EventReplayer:
    """Replays and slices stored session events for streaming recovery and reconnects."""

    def __init__(self, store: SessionEventStore) -> None:
        self.store = store

    # Replay cắt trên lịch sử đã lưu; không chạy lại agent hay tạo câu trả lời mới.
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
