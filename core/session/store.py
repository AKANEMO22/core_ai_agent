"""Redis-backed session state store with optimistic compare-and-set transitions."""

import json
import logging
from typing import Any, Dict, List, Optional

from redis.exceptions import WatchError

from core.session.state import (
    LifecycleStatus,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionState,
    InvalidStateTransitionError,
    build_transition,
)

logger = logging.getLogger(__name__)


class RedisSessionStateStore:
    """Production `SessionStateStore` persisting lifecycle state in Redis.

    Keys:
        session:{id}:state             JSON snapshot, refreshed TTL on every write.
        sessions:status:{STATUS}       Set index used by recovery sweepers.

    Transitions use WATCH/MULTI: the write commits only if the key was not modified
    between the version check and EXEC, which makes CAS safe across processes.
    """

    def __init__(
        self,
        redis_client: Any,
        ttl_seconds: int = 7 * 24 * 3600,
        key_prefix: str = "session:",
        index_prefix: str = "sessions:status:",
    ) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.index_prefix = index_prefix

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}:state"

    def _index(self, status: LifecycleStatus) -> str:
        return f"{self.index_prefix}{status.value}"

    @staticmethod
    def _decode(raw: Any) -> SessionState:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return SessionState.from_dict(json.loads(raw))

    @staticmethod
    def _encode(state: SessionState) -> str:
        return json.dumps(state.to_dict(), sort_keys=True)

    async def acreate(
        self,
        session_id: str,
        initial_status: LifecycleStatus = LifecycleStatus.INITIALIZED,
        metadata: Optional[Dict[str, Any]] = None,
        **fields: Any,
    ) -> SessionState:
        """Register a new session at version 1; fails if the identifier already exists."""
        state = SessionState(
            session_id=session_id,
            status=initial_status,
            state_version=1,
            metadata=dict(metadata or {}),
            **fields,
        )
        created = await self.redis.set(
            self._key(session_id), self._encode(state), nx=True, ex=self.ttl_seconds
        )
        if not created:
            raise SessionAlreadyExistsError(f"Session '{session_id}' already exists")
        await self.redis.sadd(self._index(initial_status), session_id)
        return state

    async def aget(self, session_id: str) -> Optional[SessionState]:
        """Return the stored state or None if absent or expired."""
        raw = await self.redis.get(self._key(session_id))
        return self._decode(raw) if raw is not None else None

    async def atransition(
        self,
        state: SessionState,
        expected_version: int,
        new_status: Optional[LifecycleStatus] = None,
        **updates: Any,
    ) -> SessionState:
        """Apply a transition atomically if the stored version equals expected_version."""
        key = self._key(state.session_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                # WATCH/MULTI giữ phép đọc-version và ghi state trong cùng một CAS trên Redis.
                await pipe.watch(key)
                raw = await pipe.get(key)
                if raw is None:
                    raise SessionNotFoundError(
                        f"Session '{state.session_id}' not found in state store"
                    )
                current = self._decode(raw)
                updated = build_transition(current, expected_version, new_status, **updates)

                pipe.multi()
                pipe.set(key, self._encode(updated), ex=self.ttl_seconds)
                if updated.status != current.status:
                    pipe.srem(self._index(current.status), state.session_id)
                    pipe.sadd(self._index(updated.status), state.session_id)
                await pipe.execute()
                return updated
            except WatchError as e:
                raise InvalidStateTransitionError(
                    f"CAS mismatch for session '{state.session_id}': state changed concurrently"
                ) from e

    async def adelete(self, session_id: str) -> None:
        """Delete the state key and remove index membership (idempotent)."""
        current = await self.aget(session_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(self._key(session_id))
            if current is not None:
                pipe.srem(self._index(current.status), session_id)
            await pipe.execute()

    async def alist_by_status(self, status: LifecycleStatus) -> List[str]:
        """Return session ids indexed under status, pruning entries whose state expired."""
        members = await self.redis.smembers(self._index(status))
        result: List[str] = []
        for member in members:
            session_id = member.decode("utf-8") if isinstance(member, bytes) else str(member)
            current = await self.aget(session_id)
            if current is None or current.status != status:
                await self.redis.srem(self._index(status), session_id)
                continue
            result.append(session_id)
        return sorted(result)
