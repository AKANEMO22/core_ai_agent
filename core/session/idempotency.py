"""Idempotency management, payload conflict detection, per-session locks, and token bucket rate limiting."""

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from core.runtime.agent import sanitize_secrets

logger = logging.getLogger(__name__)


class IdempotencyError(Exception):
    """Base exception for idempotency violations."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


class IdempotencyConflictError(IdempotencyError):
    """Raised when an idempotency key is reused with a differing payload (HTTP 409 Conflict)."""


class IdempotencyInFlightError(IdempotencyError):
    """Raised when an identical request is still being processed by another caller."""


def compute_payload_hash(payload: Any) -> str:
    """Compute a deterministic SHA-256 hash for an arbitrary payload.

    Args:
        payload: Input payload (dict, list, str, bytes, or primitive).

    Returns:
        str: Hexadecimal SHA-256 digest string.
    """
    if payload is None:
        raw_bytes = b""
    elif isinstance(payload, bytes):
        raw_bytes = payload
    elif isinstance(payload, str):
        raw_bytes = payload.encode("utf-8")
    elif isinstance(payload, (dict, list)):
        try:
            raw_bytes = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raw_bytes = str(payload).encode("utf-8")
    else:
        raw_bytes = str(payload).encode("utf-8")

    return hashlib.sha256(raw_bytes).hexdigest()


@dataclass
class CachedResult:
    """Represents a cached idempotent operation outcome."""

    result: Any
    payload_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if cached entry has exceeded its time-to-live."""
        if self.expires_at is None:
            return False
        current_time = now if now is not None else time.time()
        return current_time >= self.expires_at


class TokenBucket:
    """Per-session token bucket rate limiter supporting bursts up to capacity."""

    def __init__(self, rate: float = 5.0, capacity: float = 10.0) -> None:
        """Initialize TokenBucket.

        Args:
            rate: Replenishment rate in tokens per second.
            capacity: Maximum token capacity burst limit.
        """
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._buckets: Dict[str, Tuple[float, float]] = {}  # session_id -> (tokens, last_refreshed_at)
        self._lock = threading.Lock()

    def acquire(self, session_id: str, tokens: float = 1.0) -> bool:
        """Attempt to acquire tokens for a session.

        Args:
            session_id: Target session identifier.
            tokens: Number of tokens requested.

        Returns:
            bool: True if tokens were acquired, False if capacity was exceeded.
        """
        now = time.monotonic()
        with self._lock:
            if session_id not in self._buckets:
                current_tokens = self.capacity
                last_time = now
            else:
                current_tokens, last_time = self._buckets[session_id]
                elapsed = max(0.0, now - last_time)
                # Refill tokens up to capacity
                current_tokens = min(self.capacity, current_tokens + (elapsed * self.rate))
                last_time = now

            if current_tokens >= tokens:
                current_tokens -= tokens
                self._buckets[session_id] = (current_tokens, last_time)
                return True

            # Insufficient tokens
            self._buckets[session_id] = (current_tokens, last_time)
            return False

    def reset(self, session_id: Optional[str] = None) -> None:
        """Reset token state for a specific session or all sessions."""
        with self._lock:
            if session_id:
                self._buckets.pop(session_id, None)
            else:
                self._buckets.clear()


class IdempotencyManager:
    """Coordinates idempotency caching, payload hash validation, and per-session concurrency locks."""

    def __init__(
        self,
        default_ttl_seconds: Optional[int] = 86400,
        rate_limiter: Optional[TokenBucket] = None,
    ) -> None:
        self._default_ttl = default_ttl_seconds
        self.rate_limiter = rate_limiter or TokenBucket()
        self._cache: Dict[str, CachedResult] = {}
        self._in_flight: Dict[str, float] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._lock_guard = threading.Lock()

    def _get_or_create_session_lock(self, session_id: str) -> asyncio.Lock:
        """Thread-safely obtain or instantiate an asyncio.Lock for a session."""
        with self._lock_guard:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    @contextlib.asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncIterator[None]:
        """Async context manager serializing operations for a specific session.

        Operations across different sessions run concurrently without contention.
        """
        lock = self._get_or_create_session_lock(session_id)
        async with lock:
            yield

    def record(
        self,
        key: str,
        result: Any,
        payload: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Cache an idempotent result and record its payload hash for future conflict checks.

        Args:
            key: Unique idempotency key or message_id.
            result: Result payload to store.
            payload: Optional request payload used to compute SHA-256 conflict detection hash.
            ttl_seconds: Optional custom expiration TTL in seconds.

        Raises:
            IdempotencyConflictError: If key is already recorded with a differing payload hash.
        """
        now = time.time()
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None
        new_hash = compute_payload_hash(payload) if payload is not None else None

        existing = self._cache.get(key)
        if existing is not None and not existing.is_expired(now):
            # If payload hashes are present and mismatch, detect 409 conflict
            if (
                existing.payload_hash is not None
                and new_hash is not None
                and existing.payload_hash != new_hash
            ):
                raise IdempotencyConflictError(
                    f"Idempotency payload mismatch for key '{key}': "
                    f"existing hash {existing.payload_hash[:8]}... != incoming hash {new_hash[:8]}..."
                )

        self._cache[key] = CachedResult(
            result=result,
            payload_hash=new_hash,
            created_at=now,
            expires_at=expires_at,
        )
        self._in_flight.pop(key, None)

    def get(self, key: str, payload: Optional[Any] = None) -> Optional[Any]:
        """Retrieve a cached result if present and not expired, validating payload consistency.

        Args:
            key: Unique idempotency key or message_id.
            payload: Optional request payload to verify against cached payload hash.

        Returns:
            Optional[Any]: Cached result if valid, else None.

        Raises:
            IdempotencyConflictError: If key is present but incoming payload hash mismatches.
        """
        now = time.time()
        cached = self._cache.get(key)
        if cached is None:
            return None

        if cached.is_expired(now):
            del self._cache[key]
            return None

        if payload is not None and cached.payload_hash is not None:
            incoming_hash = compute_payload_hash(payload)
            if incoming_hash != cached.payload_hash:
                raise IdempotencyConflictError(
                    f"Idempotency payload mismatch for key '{key}': "
                    f"cached hash {cached.payload_hash[:8]}... != incoming hash {incoming_hash[:8]}..."
                )

        return cached.result

    def start_operation(self, key: str, payload: Optional[Any] = None) -> bool:
        """Mark an operation as in-flight.

        Args:
            key: Unique operation key.
            payload: Optional operation payload.

        Returns:
            bool: True if operation successfully marked in-flight; False if already completed or in-flight.

        Raises:
            IdempotencyConflictError: If key is already cached with a conflicting payload hash.
        """
        # Check completed cache first
        cached = self._cache.get(key)
        if cached is not None and not cached.is_expired():
            if payload is not None and cached.payload_hash is not None:
                incoming_hash = compute_payload_hash(payload)
                if incoming_hash != cached.payload_hash:
                    raise IdempotencyConflictError(
                        f"Idempotency conflict for key '{key}': payload hash mismatch"
                    )
            return False

        if key in self._in_flight:
            return False

        self._in_flight[key] = time.time()
        return True

    def finish_operation(
        self,
        key: str,
        result: Any,
        payload: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Mark an in-flight operation as finished and cache its result."""
        self.record(key, result=result, payload=payload, ttl_seconds=ttl_seconds)

    def is_in_flight(self, key: str) -> bool:
        """Check whether an operation key is currently marked in-flight."""
        return key in self._in_flight

    def clear(self) -> None:
        """Clear all cached entries and locks."""
        self._cache.clear()
        self._in_flight.clear()
        with self._lock_guard:
            self._session_locks.clear()

    # ------------------------------------------------------------------
    # Async store API (shared with RedisIdempotencyStore)
    # ------------------------------------------------------------------

    async def abegin(self, key: str, payload: Optional[Any] = None) -> Tuple[bool, Optional[Any]]:
        """Claim a key for execution or return the cached result.

        Returns:
            (True, None) when the caller must execute and then `arecord`;
            (False, result) when a completed result is cached.

        Raises:
            IdempotencyConflictError: Key reused with a different payload.
            IdempotencyInFlightError: Identical request still executing.
        """
        cached = self.get(key, payload=payload)
        if cached is not None:
            return False, cached
        if not self.start_operation(key, payload=payload):
            raise IdempotencyInFlightError(f"Request with idempotency key '{key}' is in progress")
        return True, None

    async def arecord(
        self,
        key: str,
        result: Any,
        payload: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store the completed result for a claimed key."""
        self.record(key, result=result, payload=payload, ttl_seconds=ttl_seconds)

    async def arelease(self, key: str) -> None:
        """Release a claimed key after a failed execution so it can be retried."""
        self._in_flight.pop(key, None)


class RedisIdempotencyStore:
    """Redis-backed idempotency records shared across processes and restarts.

    Each key stores JSON {"state": "in_flight" | "done", "payload_hash", "result"} with a TTL.
    In-flight claims use SET NX with a short lease so a crashed caller does not block retries.
    """

    def __init__(
        self,
        redis_client: Any,
        ttl_seconds: int = 86400,
        in_flight_ttl_seconds: int = 120,
        prefix: str = "idempotency:",
    ) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.in_flight_ttl_seconds = in_flight_ttl_seconds
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def abegin(self, key: str, payload: Optional[Any] = None) -> Tuple[bool, Optional[Any]]:
        """Claim a key for execution or return the cached result (see IdempotencyManager.abegin)."""
        payload_hash = compute_payload_hash(payload) if payload is not None else None
        marker = json.dumps({"state": "in_flight", "payload_hash": payload_hash})
        claimed = await self.redis.set(
            self._key(key), marker, nx=True, ex=self.in_flight_ttl_seconds
        )
        if claimed:
            return True, None

        raw = await self.redis.get(self._key(key))
        if raw is None:
            # Expired between SET NX and GET: retry the claim once.
            claimed = await self.redis.set(
                self._key(key), marker, nx=True, ex=self.in_flight_ttl_seconds
            )
            if claimed:
                return True, None
            raw = await self.redis.get(self._key(key))
            if raw is None:
                raise IdempotencyInFlightError(f"Request with idempotency key '{key}' is in progress")

        record = json.loads(raw)
        stored_hash = record.get("payload_hash")
        if payload_hash is not None and stored_hash is not None and stored_hash != payload_hash:
            raise IdempotencyConflictError(
                f"Idempotency payload mismatch for key '{key}': "
                f"stored hash {stored_hash[:8]}... != incoming hash {payload_hash[:8]}..."
            )
        if record.get("state") == "done":
            return False, record.get("result")
        raise IdempotencyInFlightError(f"Request with idempotency key '{key}' is in progress")

    async def arecord(
        self,
        key: str,
        result: Any,
        payload: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store the completed result for a claimed key."""
        record = {
            "state": "done",
            "payload_hash": compute_payload_hash(payload) if payload is not None else None,
            "result": result,
        }
        await self.redis.set(
            self._key(key), json.dumps(record, default=str), ex=ttl_seconds or self.ttl_seconds
        )

    async def arelease(self, key: str) -> None:
        """Release an in-flight claim after a failed execution (completed records are kept)."""
        raw = await self.redis.get(self._key(key))
        if raw is not None and json.loads(raw).get("state") == "in_flight":
            await self.redis.delete(self._key(key))

    async def apurge_prefix(self, prefix: str) -> int:
        """Delete every record whose key starts with prefix (used by session deletion)."""
        deleted = 0
        async for redis_key in self.redis.scan_iter(match=f"{self._key(prefix)}*"):
            deleted += int(await self.redis.delete(redis_key))
        return deleted


@dataclass
class LedgerEntry:
    """State of one client message in the per-session message ledger."""

    message_id: str
    turn_id: str
    status: str  # "in_progress" | "completed"
    payload_hash: Optional[str] = None


class MessageLedger:
    """Per-session mapping of client `message_id` to the turn it produced.

    Key `session:{id}:messages` is a Redis hash `message_id -> JSON LedgerEntry` with a TTL.
    Guarantees that a repeated `message_id` never invokes the LLM twice: completed messages
    are answered by replaying the recorded turn events.
    """

    def __init__(self, redis_client: Any, ttl_seconds: int = 86400) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def get(self, session_id: str, message_id: str) -> Optional[LedgerEntry]:
        """Return the ledger entry for a message, or None."""
        raw = await self.redis.hget(self._key(session_id), message_id)
        if raw is None:
            return None
        data = json.loads(raw)
        return LedgerEntry(message_id=message_id, **data)

    async def begin(
        self, session_id: str, message_id: str, turn_id: str, payload_hash: Optional[str]
    ) -> Optional[LedgerEntry]:
        """Register a message as in progress; return the existing entry if already registered.

        Raises:
            IdempotencyConflictError: Existing entry has a different payload hash.
        """
        value = json.dumps({"turn_id": turn_id, "status": "in_progress", "payload_hash": payload_hash})
        created = await self.redis.hsetnx(self._key(session_id), message_id, value)
        await self.redis.expire(self._key(session_id), self.ttl_seconds)
        if created:
            return None
        existing = await self.get(session_id, message_id)
        if (
            existing is not None
            and existing.payload_hash
            and payload_hash
            and existing.payload_hash != payload_hash
        ):
            raise IdempotencyConflictError(
                f"message_id '{message_id}' was reused with a different payload"
            )
        return existing

    async def complete(self, session_id: str, message_id: str) -> None:
        """Mark a message's turn as completed."""
        entry = await self.get(session_id, message_id)
        if entry is None:
            return
        value = json.dumps(
            {"turn_id": entry.turn_id, "status": "completed", "payload_hash": entry.payload_hash}
        )
        await self.redis.hset(self._key(session_id), message_id, value)

    async def discard(self, session_id: str, message_id: str) -> None:
        """Forget a message whose turn failed so the client can retry it."""
        await self.redis.hdel(self._key(session_id), message_id)

    async def purge(self, session_id: str) -> None:
        """Delete the session's ledger."""
        await self.redis.delete(self._key(session_id))


class RedisLease:
    """Expiring mutual-exclusion lease stored in Redis (SET NX PX + token-checked release)."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    async def acquire(self, key: str, ttl_ms: int) -> Optional[str]:
        """Try to take the lease; return its token or None if held by someone else."""
        token = uuid.uuid4().hex
        acquired = await self.redis.set(key, token, nx=True, px=ttl_ms)
        return token if acquired else None

    async def release(self, key: str, token: str) -> bool:
        """Release the lease only if it is still held with this token."""
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                current = await pipe.get(key)
                if current is None or (
                    current.decode("utf-8") if isinstance(current, bytes) else current
                ) != token:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
                return True
            except Exception:
                return False
