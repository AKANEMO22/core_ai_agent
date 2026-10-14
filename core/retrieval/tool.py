"""`retrieve` runtime tool exposing the retrieval port to control-plane agents.

Đưa retrieval port thành tool tên retrieve trong vòng chạy agent. search nhận
collection/query/competency/top_k, ghép filter cố định của session rồi gọi
adapter dưới deadline; trả status cùng content, nguồn và metadata được cho phép.
Expected_points bị ẩn mặc định để câu trả lời mẫu không lọt vào prompt speaker;
chỉ bên có quyền evaluator mới nên bật include_expected_points.
"""

import asyncio
from typing import Any, Dict, Iterable, Optional

from core.retrieval.port import RetrievalClient, RetrievalCollection, RetrievalError, RetrievalQuery
from core.runtime.tooling import RuntimeTool

DEFAULT_ALLOWED_COLLECTIONS = (
    RetrievalCollection.QUESTION_BANK.value,
    RetrievalCollection.RUBRIC.value,
    RetrievalCollection.COMPANY_KNOWLEDGE.value,
)

# Metadata fields an agent may see. Expected answers stay out of interviewer-facing tools.
VISIBLE_METADATA = {
    "question_id",
    "competency",
    "stage_type",
    "difficulty",
    "follow_ups",
    "name",
    "topic",
    "confidence",
}


class RetrieveTool(RuntimeTool):
    """Search allowlisted knowledge collections with a bounded `top_k` and a hard deadline.

    Attributes:
        client: Retrieval backend adapter.
        allowed_collections: Collections the calling agent may query.
        max_top_k: Upper bound on results per call.
        deadline_seconds: Per-call deadline.
        fixed_filters: Filters forced on every call (e.g. the session's role family and level).
    """

    def __init__(
        self,
        client: RetrievalClient,
        allowed_collections: Iterable[str] = DEFAULT_ALLOWED_COLLECTIONS,
        max_top_k: int = 5,
        deadline_seconds: float = 2.0,
        fixed_filters: Optional[Dict[str, Any]] = None,
        include_expected_points: bool = False,
    ) -> None:
        self.client = client
        self.allowed_collections = tuple(allowed_collections)
        self.max_top_k = max_top_k
        self.deadline_seconds = deadline_seconds
        self.fixed_filters = dict(fixed_filters or {})
        self.include_expected_points = include_expected_points
        super().__init__(
            name="retrieve",
            description=(
                "Search interview knowledge (question bank, rubric, company knowledge). "
                "Use for follow-up questions or to check a competency definition."
            ),
            func=self.search,
            param_schema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "enum": list(self.allowed_collections)},
                    "query": {"type": "string", "minLength": 2, "maxLength": 500},
                    "competency": {"type": "string", "description": "Optional competency id filter"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": self.max_top_k},
                },
                "required": ["collection", "query"],
                "additionalProperties": False,
            },
        )

    async def search(
        self,
        collection: str,
        query: str,
        competency: Optional[str] = None,
        top_k: int = 3,
        **_: Any,
    ) -> Dict[str, Any]:
        """Execute one bounded retrieval and return content plus a safe metadata subset."""
        if collection not in self.allowed_collections:
            return {"status": "denied", "error": f"collection '{collection}' is not allowed"}
        # Role/level lấy từ session, competency từ lần gọi tool; không nhận tùy ý mọi metadata filter.
        filters = dict(self.fixed_filters)
        if competency:
            filters["competency"] = competency
        try:
            chunks = await asyncio.wait_for(
                self.client.search(
                    RetrievalQuery(
                        collection=collection,
                        text=query,
                        filters=filters,
                        top_k=max(1, min(int(top_k), self.max_top_k)),
                    )
                ),
                timeout=self.deadline_seconds,
            )
        except asyncio.TimeoutError:
            return {"status": "error", "error": "retrieval timed out"}
        except RetrievalError as e:
            return {"status": "error", "error": str(e)}

        # Lọc đầu ra sau retrieval: backend có thể trả nhiều metadata hơn agent được phép nhìn.
        visible = set(VISIBLE_METADATA)
        if self.include_expected_points:
            visible.add("expected_points")
        return {
            "status": "ok",
            "results": [
                {
                    "id": c.chunk_id,
                    "content": c.content,
                    "metadata": {k: v for k, v in c.metadata.items() if k in visible},
                    "source": c.source,
                }
                for c in chunks
            ],
        }
