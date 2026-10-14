"""Per-session blackboard in Redis: shared state written by delegated agents, read by the orchestrator.

Bảng trao đổi tạm thời theo session trong Redis, được orchestration dùng để
lưu coverage, cờ guard và kết quả phân tích phụ dưới dạng JSON. set/get phục vụ
luồng chạy; persist sao chép snapshot sang workspace để truy vết sau phiên.
TTL thuộc cả hash của session và được gia hạn sau mỗi lần ghi.
"""

import json
from typing import Any, Dict, Optional

from core.workspace.session import SessionWorkspace


class Blackboard:
    """Redis hash `session:{id}:blackboard` with JSON values.

    Keys used by the runtime: coverage, pending_follow_ups, flags, claims, last_cycle,
    directive:<turn>, delegation:<id>.
    """

    def __init__(self, redis_client: Any, ttl_seconds: int = 7 * 24 * 3600) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(session_id: str) -> str:
        return f"session:{session_id}:blackboard"

    async def set(self, session_id: str, field: str, value: Any) -> None:
        await self.redis.hset(self.key(session_id), field, json.dumps(value, default=str))
        await self.redis.expire(self.key(session_id), self.ttl_seconds)

    async def get(self, session_id: str, field: str, default: Any = None) -> Any:
        raw = await self.redis.hget(self.key(session_id), field)
        return json.loads(raw) if raw is not None else default

    async def merge_flags(self, session_id: str, flags: Dict[str, Any]) -> Dict[str, Any]:
        # Đây là đọc-sửa-ghi, không phải merge nguyên tử; None nghĩa là agent chưa có kết luận mới.
        current = await self.get(session_id, "flags", {}) or {}
        current.update({k: v for k, v in flags.items() if v is not None})
        await self.set(session_id, "flags", current)
        return current

    async def snapshot(self, session_id: str) -> Dict[str, Any]:
        raw = await self.redis.hgetall(self.key(session_id))
        return {k: json.loads(v) for k, v in raw.items()}

    async def persist(self, session_id: str, workspace: SessionWorkspace) -> None:
        """Write the blackboard snapshot as an artifact (dialogue completion, audit)."""
        # Redis phục vụ trao đổi tức thời; artifact này giữ bản chụp để audit sau phiên.
        workspace.write_artifact(
            "orchestration/blackboard.json", artifact_type="blackboard", payload=await self.snapshot(session_id)
        )

    async def purge(self, session_id: str) -> None:
        await self.redis.delete(self.key(session_id))
