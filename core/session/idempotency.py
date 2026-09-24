"""Idempotency management, payload conflict detection, per-session locks, and token bucket rate limiting."""

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
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
