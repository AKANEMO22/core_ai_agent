"""Retrieval port, mock adapter, and interview knowledge composition.

Cổng import chung cho retrieval: công bố query/chunk/protocol, mock adapter
và dịch vụ ghép knowledge pack. Profiler dùng InterviewKnowledgeService;
agent ở control plane có thể dùng RetrieveTool từ module tool. Backend thật
được chọn qua factory, giúp logic phỏng vấn không phụ thuộc trực tiếp index RAG.
"""

from core.retrieval.knowledge import (
    STAGE_TYPES,
    InterviewKnowledgeService,
    classify_stage_type,
    ground_stage_plan,
    infer_level,
    infer_role_family,
)
from core.retrieval.mock import MockRetrievalClient
from core.retrieval.port import (
    RetrievalClient,
    RetrievalCollection,
    RetrievalError,
    RetrievalQuery,
    RetrievedChunk,
)

__all__ = [
    "STAGE_TYPES",
    "InterviewKnowledgeService",
    "MockRetrievalClient",
    "RetrievalClient",
    "RetrievalCollection",
    "RetrievalError",
    "RetrievalQuery",
    "RetrievedChunk",
    "classify_stage_type",
    "ground_stage_plan",
    "infer_level",
    "infer_role_family",
]
