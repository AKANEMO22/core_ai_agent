"""Per-session cancellation tokens for in-flight generations and delegated tasks."""

import asyncio
from typing import Dict, Optional, Set


class CancellationRegistry:
    """Tracks the cancellation token of each session's active generation and its background tasks.

    - `token_for(session_id)` returns the token for the next/current generation.
    - `cancel_generation` sets it (barge-in); the next generation gets a fresh token.
    - `cancel_session` additionally cancels every registered background task (orchestrator
      cycles, delegations) and marks the session so new work is refused.
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, asyncio.Event] = {}
        self._tasks: Dict[str, Set[asyncio.Task]] = {}
        self._cancelled_sessions: Set[str] = set()
        self._reasons: Dict[str, str] = {}

    def token_for(self, session_id: str) -> asyncio.Event:
        """Return the live token for a session, creating a fresh one if the last was used."""
        token = self._tokens.get(session_id)
        if token is None or token.is_set():
            token = asyncio.Event()
            if session_id in self._cancelled_sessions:
                token.set()
            self._tokens[session_id] = token
        return token

    def cancel_generation(self, session_id: str, reason: str = "barge_in") -> bool:
        """Cancel the in-flight generation. Returns True if one was running."""
        token = self._tokens.get(session_id)
        if token is None or token.is_set():
            return False
        self._reasons[session_id] = reason
        token.set()
        return True

    def reason(self, session_id: str) -> Optional[str]:
        """Reason recorded by the last cancellation, if any."""
        return self._reasons.get(session_id)

    def finish_generation(self, session_id: str, token: asyncio.Event) -> None:
        """Forget a completed generation's token."""
        if self._tokens.get(session_id) is token:
            self._tokens.pop(session_id, None)
        self._reasons.pop(session_id, None)

    def register_task(self, session_id: str, task: asyncio.Task) -> None:
        """Track a background task so session cancellation can stop it."""
        bucket = self._tasks.setdefault(session_id, set())
        bucket.add(task)
        task.add_done_callback(lambda t: bucket.discard(t))

    def cancel_session(self, session_id: str) -> int:
        """Cancel the generation and all tracked tasks; refuse new work. Returns tasks cancelled."""
        self._cancelled_sessions.add(session_id)
        self.cancel_generation(session_id, reason="session_cancelled")
        cancelled = 0
        for task in list(self._tasks.get(session_id, set())):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    async def cancel_session_and_wait(self, session_id: str, timeout: float = 5.0) -> int:
        """Cancel like `cancel_session`, then wait for the tracked tasks to finish unwinding."""
        tasks = [t for t in self._tasks.get(session_id, set()) if not t.done()]
        cancelled = self.cancel_session(session_id)
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        return cancelled

    def is_session_cancelled(self, session_id: str) -> bool:
        """True after cancel_session."""
        return session_id in self._cancelled_sessions

    def forget(self, session_id: str) -> None:
        """Drop all state for a session (after deletion or terminal completion)."""
        self._tokens.pop(session_id, None)
        self._tasks.pop(session_id, None)
        self._reasons.pop(session_id, None)
        self._cancelled_sessions.discard(session_id)
